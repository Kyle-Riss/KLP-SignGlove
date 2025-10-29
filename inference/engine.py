"""
SignGlove 추론 엔진

모델 로딩, 전처리, 추론, 후처리를 통합한 고수준 API
"""

import torch
import numpy as np
from pathlib import Path
from typing import Union, List, Dict, Optional
import warnings

from .models.mscsgru_inference import MSCSGRUInference
from .models.ms3dgru_inference import MS3DGRUInference
from .models.ms3dstackedgru_inference import MS3DStackedGRUInference
from .models.gru_inference import GRUInference
from .utils.preprocessor import InferencePreprocessor
from .utils.postprocessor import InferencePostprocessor


class SignGloveInference:
    """
    SignGlove 통합 추론 엔진
    
    모델 로딩부터 예측 결과 출력까지 모든 과정을 관리
    사용하기 쉬운 고수준 API 제공
    
    Example:
        >>> engine = SignGloveInference(
        ...     model_path='best_model.ckpt',
        ...     model_type='MS3DGRU',
        ...     device='cpu'
        ... )
        >>> result = engine.predict_single(sensor_data)
    """
    
    # 모델별 기본 설정
    MODEL_CONFIGS = {
        'GRU': {
            'class': GRUInference,
            'default_params': {'layers': 2, 'dropout': 0.2}
        },
        'MS3DGRU': {
            'class': MS3DGRUInference,
            'default_params': {'cnn_filters': 32, 'dropout': 0.1}
        },
        'MS3DStackedGRU': {
            'class': MS3DStackedGRUInference,
            'default_params': {'cnn_filters': 32, 'dropout': 0.05}
        },
        'MSCSGRU': {
            'class': MSCSGRUInference,
            'default_params': {'cnn_filters': 32, 'dropout': 0.3}
        }
    }
    
    def __init__(
        self,
        model_path: str,
        model_type: str = 'MS3DGRU',
        input_size: int = 8,
        hidden_size: int = 64,
        classes: int = 24,
        cnn_filters: Optional[int] = None,
        dropout: Optional[float] = None,
        target_timesteps: int = 87,
        device: Optional[str] = None,
        class_names: Optional[List[str]] = None,
        scaler_path: Optional[str] = None,
        single_predict_device: str = 'cpu',
        enable_dtw: bool = False
    ):
        """
        Args:
            model_path: 체크포인트 파일 경로
            model_type: 모델 타입 ('GRU', 'MS3DGRU', 'MS3DStackedGRU', 'MSCSGRU')
            input_size: 입력 채널 수 (default: 8)
            hidden_size: 히든 사이즈 (default: 64)
            classes: 클래스 수 (default: 24)
            cnn_filters: CNN 필터 수 (None이면 모델별 기본값 사용)
            dropout: 드롭아웃 비율 (None이면 모델별 기본값 사용)
            target_timesteps: 타임스텝 길이 (default: 87)
            device: 디바이스 ('cuda', 'cpu', None=자동)
            class_names: 클래스 이름 리스트
            scaler_path: StandardScaler 파일 경로
            single_predict_device: 단일 예측 시 사용할 디바이스
            enable_dtw: DTW 사용 여부 (현재 미구현)
        """
        self.model_path = Path(model_path)
        self.model_type = model_type
        self.target_timesteps = target_timesteps
        
        # 모델 타입 검증
        if model_type not in self.MODEL_CONFIGS:
            raise ValueError(
                f"지원하지 않는 모델 타입: {model_type}. "
                f"지원 모델: {list(self.MODEL_CONFIGS.keys())}"
            )
        
        # 디바이스 설정
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"🚀 SignGlove 추론 엔진 초기화...")
        print(f"  모델 타입: {model_type}")
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
        
        # 전처리기 초기화
        self.preprocessor = self._init_preprocessor(
            scaler_path=scaler_path,
            target_timesteps=target_timesteps,
            input_size=input_size
        )
        
        # 후처리기 초기화
        self.postprocessor = InferencePostprocessor(class_names=class_names)

        # 옵션 저장
        self.single_predict_device = single_predict_device or 'cpu'
        self.enable_dtw = bool(enable_dtw)
        
        print(f"✅ 초기화 완료!")
        print(f"  파라미터 수: {self.model.count_parameters():,}")
        print(f"  클래스 수: {classes}")
    
    def _load_checkpoint_state_dict(self, checkpoint_path: Path) -> dict:
        """
        체크포인트에서 state_dict 로드 (공통 로직)
        
        Args:
            checkpoint_path: 체크포인트 파일 경로
            
        Returns:
            state_dict: 정제된 state_dict
        """
        try:
            checkpoint = torch.load(str(checkpoint_path), map_location='cpu')
        except Exception as e:
            raise RuntimeError(f"체크포인트 로드 실패: {checkpoint_path}\n오류: {e}")
        
        # state_dict 추출
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # 'model.' 접두사 제거
        cleaned_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('model.'):
                cleaned_key = key[6:]  # 'model.' 제거
                cleaned_state_dict[cleaned_key] = value
            else:
                cleaned_state_dict[key] = value
        
        return cleaned_state_dict
    
    def _load_model(self, **model_kwargs):
        """
        모델 로딩 (개선된 버전 - 중복 코드 제거)
        
        Args:
            **model_kwargs: 모델 초기화 인자
            
        Returns:
            model: 로드된 모델
        """
        model_config = self.MODEL_CONFIGS[self.model_type]
        model_class = model_config['class']
        default_params = model_config['default_params']
        
        # 모델별 파라미터 준비 (기본값 사용)
        final_params = {
            'input_size': model_kwargs.get('input_size', 8),
            'hidden_size': model_kwargs.get('hidden_size', 64),
            'classes': model_kwargs.get('classes', 24),
        }
        
        # 모델별 특수 파라미터 추가
        for param_name, default_value in default_params.items():
            # 사용자가 명시적으로 제공한 값이 있으면 사용, 없으면 기본값
            user_value = model_kwargs.get(param_name)
            final_params[param_name] = user_value if user_value is not None else default_value
        
        # MSCSGRU는 from_checkpoint 사용
        if self.model_type == 'MSCSGRU':
            model = model_class.from_checkpoint(
                str(self.model_path),
                **final_params
            )
        else:
            # 다른 모델들은 직접 로드
            model = model_class(**final_params)
            
            # 체크포인트 로드
            if self.model_path.exists():
                state_dict = self._load_checkpoint_state_dict(self.model_path)
                try:
                    model.load_state_dict(state_dict, strict=False)
                except Exception as e:
                    warnings.warn(
                        f"체크포인트 로드 중 일부 파라미터 불일치: {e}\n"
                        f"모델이 초기화된 가중치로 실행됩니다."
                    )
            else:
                warnings.warn(
                    f"체크포인트 파일을 찾을 수 없습니다: {self.model_path}\n"
                    f"모델이 초기화된 가중치로 실행됩니다."
                )
            
            model.eval()
        
        return model
    
    def _init_preprocessor(
        self,
        scaler_path: Optional[str],
        target_timesteps: int,
        input_size: int
    ) -> InferencePreprocessor:
        """
        전처리기 초기화 (개선된 버전 - 명확한 경고)
        
        Args:
            scaler_path: Scaler 파일 경로
            target_timesteps: 타겟 타임스텝
            input_size: 입력 채널 수
            
        Returns:
            preprocessor: 전처리기 인스턴스
        """
        # 스케일러 경로 결정
        if scaler_path is None:
            scaler_path = str(self.model_path.parent / 'scaler.pkl')
        
        # 스케일러 로드 시도
        try:
            preprocessor = InferencePreprocessor.load_scaler(
                scaler_path,
                target_timesteps=target_timesteps,
                n_channels=input_size
            )
            print(f"  ✅ Scaler 로드 성공: {scaler_path}")
        except FileNotFoundError:
            warnings.warn(
                f"⚠️  Scaler 파일을 찾을 수 없습니다: {scaler_path}\n"
                f"   정규화 없이 추론을 진행합니다. 성능이 저하될 수 있습니다.\n"
                f"   훈련 시 사용한 scaler.pkl 파일을 제공하는 것을 권장합니다."
            )
            preprocessor = InferencePreprocessor(
                target_timesteps=target_timesteps,
                n_channels=input_size,
                scaler=None
            )
        
        return preprocessor
    
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
                - predicted_class: 예측된 클래스명
                - predicted_class_idx: 예측된 클래스 인덱스
                - confidence: 예측 확률
                - top_k_predictions: 상위 K개 예측 리스트
        """
        # 전처리
        x = self.preprocessor.preprocess_single(raw_data, normalize=True)
        
        # 단일 샘플은 latency 최소화를 위해 지정된 디바이스에서 처리
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
    model_type: str = 'MS3DGRU',
    device: Optional[str] = None,
    **kwargs
) -> SignGloveInference:
    """
    추론 엔진 로딩 편의 함수
    
    Args:
        model_path: 체크포인트 파일 경로
        model_type: 모델 타입
        device: 디바이스
        **kwargs: SignGloveInference 초기화 인자
    
    Returns:
        engine: 추론 엔진
        
    Example:
        >>> engine = load_inference_engine(
        ...     'best_model.ckpt',
        ...     model_type='MS3DGRU',
        ...     device='cpu'
        ... )
    """
    return SignGloveInference(
        model_path=model_path,
        model_type=model_type,
        device=device,
        **kwargs
    )
