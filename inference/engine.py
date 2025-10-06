"""
SignGlove 추론 엔진

모델 로딩, 전처리, 추론, 후처리를 통합한 고수준 API
"""

import torch
import numpy as np
from pathlib import Path
from typing import Union, List, Dict

from .models.mscsgru_inference import MSCSGRUInference
from .utils.preprocessor import InferencePreprocessor
from .utils.postprocessor import InferencePostprocessor


class SignGloveInference:
    """
    SignGlove 통합 추론 엔진
    
    모델 로딩부터 예측 결과 출력까지 모든 과정을 관리
    사용하기 쉬운 고수준 API 제공
    """
    
    def __init__(
        self,
        model_path: str,
        model_type: str = 'MSCSGRU',
        input_size: int = 8,
        hidden_size: int = 64,
        classes: int = 24,
        cnn_filters: int = 32,
        dropout: float = 0.3,
        target_timesteps: int = 87,
        device: str = None,
        class_names: List[str] = None,
        scaler_path: str = 'best_model/scaler.pkl',
        single_predict_device: str = 'cpu',
        enable_dtw: bool = False
    ):
        """
        Args:
            model_path: 체크포인트 파일 경로
            model_type: 모델 타입 (현재 'MSCSGRU'만 지원)
            input_size: 입력 채널 수
            hidden_size: 히든 사이즈
            classes: 클래스 수
            cnn_filters: CNN 필터 수
            dropout: 드롭아웃 비율
            target_timesteps: 타임스텝 길이
            device: 디바이스 ('cuda', 'cpu', None=자동)
            class_names: 클래스 이름 리스트
        """
        self.model_path = Path(model_path)
        self.model_type = model_type
        self.target_timesteps = target_timesteps
        
        # 디바이스 설정
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"🚀 SignGlove 추론 엔진 초기화...")
        print(f"  디바이스: {self.device}")
        
        # 모델 로딩
        self.model = self._load_model(
            input_size=input_size,
            hidden_size=hidden_size,
            classes=classes,
            cnn_filters=cnn_filters,
            dropout=dropout
        )
        self.model.to(self.device)
        
        # 전처리기 초기화 (훈련 시 저장된 StandardScaler 강제 사용)
        try:
            self.preprocessor = InferencePreprocessor.load_scaler(
                scaler_path,
                target_timesteps=target_timesteps,
                n_channels=input_size
            )
            print(f"  Scaler loaded from: {scaler_path}")
        except Exception as e:
            # 안전장치: 로드 실패 시 명시적으로 예외 전파해 무결성 보장
            raise FileNotFoundError(
                f"StandardScaler file not found or invalid at '{scaler_path}'. "
                f"Train-time scaler must be provided. Original error: {e}"
            )
        
        # 후처리기 초기화
        self.postprocessor = InferencePostprocessor(class_names=class_names)

        # 옵션 저장
        self.single_predict_device = single_predict_device or 'cpu'
        self.enable_dtw = bool(enable_dtw)
        
        print(f"✅ 초기화 완료!")
        print(f"  모델: {self.model_type}")
        print(f"  파라미터 수: {self.model.count_parameters():,}")
        print(f"  클래스 수: {classes}")
    
    def _load_model(self, **model_kwargs) -> MSCSGRUInference:
        """모델 로딩"""
        if self.model_type == 'MSCSGRU':
            model = MSCSGRUInference.from_checkpoint(
                str(self.model_path),
                **model_kwargs
            )
        else:
            raise ValueError(f"지원하지 않는 모델 타입: {self.model_type}")
        
        return model
    
    def predict_single(
        self,
        raw_data: Union[np.ndarray, List[List[float]]],
        top_k: int = 5,
        return_all_info: bool = True
    ) -> Dict:
        """
        단일 샘플 예측
        
        Args:
            raw_data: 원시 센서 데이터 (timesteps, channels)
            top_k: 상위 K개 클래스 반환
            return_all_info: True이면 모든 정보 반환, False이면 최상위 예측만
        
        Returns:
            result: 예측 결과 딕셔너리
        """
        # 전처리
        x = self.preprocessor.preprocess_single(raw_data, normalize=True)
        # 단일 샘플은 latency 최소화를 위해 기본적으로 CPU에서 처리
        run_device = torch.device(self.single_predict_device)
        x = x.to(run_device)
        
        # 추론 (필요 시 임시로 모델을 해당 디바이스로 이동)
        original_device = next(self.model.parameters()).device
        if original_device != run_device:
            self.model.to(run_device)
        logits = self.model.predict(x)
        if original_device != run_device:
            self.model.to(original_device)
        
        # 후처리
        if return_all_info:
            result = self.postprocessor.format_single_prediction(logits, top_k=top_k)
        else:
            predicted_class, confidence = self.postprocessor.logits_to_class(logits)
            result = {
                'predicted_class': self.postprocessor.class_names[predicted_class.item()],
                'confidence': float(confidence.item())
            }
        
        return result
    
    def predict_batch(
        self,
        raw_data_list: List[Union[np.ndarray, List[List[float]]]],
        top_k: int = 5
    ) -> List[Dict]:
        """
        배치 예측
        
        Args:
            raw_data_list: 원시 센서 데이터 리스트
            top_k: 상위 K개 클래스 반환
        
        Returns:
            results: 예측 결과 리스트
        """
        # 전처리
        x = self.preprocessor.preprocess_batch(raw_data_list, normalize=True)
        x = x.to(self.device)
        
        # 추론
        logits = self.model.predict(x)
        
        # 후처리
        results = self.postprocessor.format_batch_predictions(logits, top_k=top_k)
        
        return results
    
    def predict_with_details(
        self,
        raw_data: Union[np.ndarray, List[List[float]]]
    ) -> Dict:
        """
        상세 정보를 포함한 예측
        
        Args:
            raw_data: 원시 센서 데이터 (timesteps, channels)
        
        Returns:
            result: 상세 예측 결과
                - predicted_class: 예측 클래스
                - confidence: 예측 확률
                - top_k_predictions: 상위 K개 예측
                - all_class_probabilities: 모든 클래스의 확률
                - input_shape: 입력 데이터 shape
        """
        # 입력 정보
        if isinstance(raw_data, list):
            raw_data = np.array(raw_data)
        input_shape = raw_data.shape
        
        # 전처리
        x = self.preprocessor.preprocess_single(raw_data, normalize=True)
        x = x.to(self.device)
        
        # 추론
        logits = self.model.predict(x)
        
        # 후처리
        result = self.postprocessor.format_single_prediction(logits, top_k=5)
        
        # 모든 클래스의 확률 추가
        all_probs = self.postprocessor.get_class_probabilities(logits)
        result['all_class_probabilities'] = all_probs
        result['input_shape'] = input_shape
        
        return result
    
    def get_model_info(self) -> Dict:
        """
        모델 정보 반환
        
        Returns:
            info: 모델 정보 딕셔너리
        """
        info = self.model.get_model_info()
        info.update({
            'device': str(self.device),
            'target_timesteps': self.target_timesteps,
            'model_path': str(self.model_path),
            'class_names': self.postprocessor.class_names
        })
        return info
    
    def print_prediction(self, prediction: Dict):
        """
        예측 결과를 보기 좋게 출력
        
        Args:
            prediction: predict_single 또는 predict_with_details의 반환값
        """
        self.postprocessor.print_prediction(prediction)


# 편의 함수
def load_inference_engine(
    model_path: str,
    device: str = None,
    **kwargs
) -> SignGloveInference:
    """
    추론 엔진 로딩 편의 함수
    
    Args:
        model_path: 체크포인트 파일 경로
        device: 디바이스
        **kwargs: SignGloveInference 초기화 인자
    
    Returns:
        engine: 추론 엔진
    """
    return SignGloveInference(model_path=model_path, device=device, **kwargs)


# 테스트 코드
if __name__ == "__main__":
    print("🧪 SignGloveInference 테스트...")
    
    # 더미 모델로 테스트 (실제 체크포인트가 없는 경우)
    print("\n⚠️  실제 체크포인트가 필요합니다.")
    print("테스트를 위해서는 다음과 같이 사용하세요:")
    print("""
    # 추론 엔진 초기화
    engine = SignGloveInference(
        model_path='best_model/best_model.ckpt',
        model_type='MSCSGRU',
        device='cpu'
    )
    
    # 단일 샘플 예측
    raw_data = np.random.randn(87, 8)  # 테스트 데이터
    result = engine.predict_single(raw_data)
    engine.print_prediction(result)
    
    # 배치 예측
    raw_data_list = [np.random.randn(87, 8) for _ in range(5)]
    results = engine.predict_batch(raw_data_list)
    
    # 모델 정보
    info = engine.get_model_info()
    print(info)
    """)

