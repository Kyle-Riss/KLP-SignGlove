import torch
import torch.nn as nn
import numpy as np
import h5py
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

# 모델 import
import sys
sys.path.append('../models')
from deep_learning import DeepLearningPipeline

# 개선된 모델 import
sys.path.append('..')
from improved_training import ImprovedModel
from simple_robust_training import SimpleRobustModel

# 라벨 매퍼 import
sys.path.append('../training')
from label_mapping import KSLLabelMapper

class SignGloveInference:
    """
    SignGlove 실시간 한국수어 인식 추론 시스템
    """
    def __init__(self, model_path: str, config_path: str = None, device: str = 'auto'):
        """
        Args:
            model_path: 훈련된 모델 파일 경로
            config_path: 설정 파일 경로 (선택사항)
            device: 사용할 디바이스 ('cpu', 'cuda', 'auto')
        """
        self.model_path = model_path
        self.device = self._setup_device(device)
        self.config = self._load_config(config_path)
        
        # 간단한 테스트 모델 사용 (개발 중)
        if 'simple_test_model.pth' in model_path:
            self.config['hidden_dim'] = 64
            self.config['num_layers'] = 2
        self.model = None
        self.label_mapper = None
        
        # 로깅 설정
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # 모델 로드
        self._load_model()
        
        # 실시간 추론을 위한 버퍼
        self.sensor_buffer = []
        self.buffer_size = self.config.get('sequence_length', 20)
        self.confidence_threshold = self.config.get('confidence_threshold', 0.7)
        
        self.logger.info(f"SignGlove 추론 시스템 초기화 완료 (디바이스: {self.device})")
    
    def _setup_device(self, device: str) -> torch.device:
        """디바이스 설정"""
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        return torch.device(device)
    
    def _load_config(self, config_path: str) -> Dict:
        """설정 파일 로드"""
        default_config = {
            'input_features': 8,
            'sequence_length': 20,
            'num_classes': 24,
            'hidden_dim': 48,
            'num_layers': 1,
            'dropout': 0.3,
            'confidence_threshold': 0.7,
            'smoothing_window': 5,
            'prediction_delay': 0.1  # 초
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                user_config = json.load(f)
                default_config.update(user_config)
        
        return default_config
    
    def _load_model(self):
        """훈련된 모델 로드"""
        try:
            # 모델 초기화 (모델 타입에 따라)
            if 'simple_robust_model.pth' in self.model_path:
                self.model = SimpleRobustModel(
                    input_features=self.config['input_features'],
                    sequence_length=self.config['sequence_length'],
                    num_classes=self.config['num_classes'],
                    hidden_dim=64,
                    num_layers=2,
                    dropout=self.config['dropout']
                )
            elif 'improved_model.pth' in self.model_path:
                self.model = ImprovedModel(
                    input_features=self.config['input_features'],
                    sequence_length=self.config['sequence_length'],
                    num_classes=self.config['num_classes'],
                    hidden_dim=128,
                    num_layers=3,
                    dropout=self.config['dropout']
                )
            else:
                self.model = DeepLearningPipeline(
                    input_features=self.config['input_features'],
                    sequence_length=self.config['sequence_length'],
                    num_classes=self.config['num_classes'],
                    hidden_dim=self.config['hidden_dim'],
                    num_layers=self.config['num_layers'],
                    dropout=self.config['dropout']
                )
            
            # 모델 가중치 로드
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            # 라벨 매퍼 로드
            if 'label_mapper' in checkpoint:
                self.label_mapper = checkpoint['label_mapper']
            else:
                # KSLLabelMapper 사용 (훈련 시와 동일한 매퍼)
                ksl_mapper = KSLLabelMapper()
                self.label_mapper = {i: ksl_mapper.get_class_name(i) for i in range(24)}
            
            self.logger.info(f"모델 로드 완료: {self.model_path}")
            self.logger.info(f"총 파라미터 수: {sum(p.numel() for p in self.model.parameters()):,}")
            
        except Exception as e:
            self.logger.error(f"모델 로드 실패: {e}")
            raise
    
    def preprocess_sensor_data(self, sensor_data: np.ndarray) -> torch.Tensor:
        """
        센서 데이터 전처리
        
        Args:
            sensor_data: (sequence_length, 8) 형태의 센서 데이터
            
        Returns:
            전처리된 텐서: (1, sequence_length, 8)
        """
        # 데이터 정규화 (0-1 범위로)
        sensor_data = (sensor_data - sensor_data.min()) / (sensor_data.max() - sensor_data.min() + 1e-8)
        
        # 텐서 변환 및 차원 추가
        tensor_data = torch.FloatTensor(sensor_data).unsqueeze(0)  # (1, seq_len, 8)
        
        return tensor_data
    
    def predict_single(self, sensor_data: np.ndarray) -> Dict:
        """
        단일 센서 데이터에 대한 예측
        
        Args:
            sensor_data: (sequence_length, 8) 형태의 센서 데이터
            
        Returns:
            예측 결과 딕셔너리
        """
        with torch.no_grad():
            # 데이터 전처리
            input_tensor = self.preprocess_sensor_data(sensor_data).to(self.device)
            
            # 모델 추론
            outputs = self.model(input_tensor)
            logits = outputs['class_logits']
            
            # 확률 계산
            probabilities = torch.softmax(logits, dim=1)
            confidence, predicted_idx = torch.max(probabilities, dim=1)
            
            # 결과 생성
            predicted_label = self.label_mapper[predicted_idx.item()]
            confidence_score = confidence.item()
            
            return {
                'predicted_label': predicted_label,
                'confidence': confidence_score,
                'probabilities': probabilities.cpu().numpy()[0],
                'attention_weights': outputs.get('attention_weights', None)
            }
    
    def add_sensor_data(self, sensor_data: np.ndarray):
        """
        실시간 센서 데이터 추가
        
        Args:
            sensor_data: (8,) 형태의 단일 센서 데이터
        """
        self.sensor_buffer.append(sensor_data)
        
        # 버퍼 크기 유지
        if len(self.sensor_buffer) > self.buffer_size:
            self.sensor_buffer.pop(0)
    
    def predict_realtime(self) -> Optional[Dict]:
        """
        실시간 예측 (버퍼가 가득 찼을 때만)
        
        Returns:
            예측 결과 또는 None (버퍼가 부족한 경우)
        """
        if len(self.sensor_buffer) < self.buffer_size:
            return None
        
        # 버퍼를 numpy 배열로 변환
        sensor_sequence = np.array(self.sensor_buffer)
        
        # 예측 수행
        result = self.predict_single(sensor_sequence)
        
        # 신뢰도 임계값 확인
        if result['confidence'] < self.confidence_threshold:
            result['predicted_label'] = 'UNKNOWN'
        
        return result
    
    def predict_batch(self, sensor_data_list: List[np.ndarray]) -> List[Dict]:
        """
        배치 예측
        
        Args:
            sensor_data_list: 센서 데이터 리스트
            
        Returns:
            예측 결과 리스트
        """
        results = []
        for sensor_data in sensor_data_list:
            result = self.predict_single(sensor_data)
            results.append(result)
        return results
    
    def get_model_info(self) -> Dict:
        """모델 정보 반환"""
        return {
            'model_path': self.model_path,
            'device': str(self.device),
            'config': self.config,
            'total_parameters': sum(p.numel() for p in self.model.parameters()),
            'label_mapper': self.label_mapper
        }
    
    def reset_buffer(self):
        """센서 버퍼 초기화"""
        self.sensor_buffer.clear()
        self.logger.info("센서 버퍼 초기화 완료")

class SignGloveDemo:
    """
    SignGlove 데모 시스템 (시뮬레이션용)
    """
    def __init__(self, inference_system: SignGloveInference):
        self.inference = inference_system
        self.logger = logging.getLogger(__name__)
    
    def simulate_sensor_data(self, duration: float = 10.0, sample_rate: float = 50.0) -> List[np.ndarray]:
        """
        시뮬레이션 센서 데이터 생성
        
        Args:
            duration: 시뮬레이션 시간 (초)
            sample_rate: 샘플링 레이트 (Hz)
            
        Returns:
            시뮬레이션 센서 데이터 리스트
        """
        num_samples = int(duration * sample_rate)
        sequence_length = self.inference.config['sequence_length']
        
        # 랜덤 센서 데이터 생성
        sensor_data_list = []
        for i in range(num_samples):
            # 8축 센서 데이터 시뮬레이션 (flex1-5, pitch, roll, yaw)
            sensor_data = np.random.rand(8)
            sensor_data_list.append(sensor_data)
        
        return sensor_data_list
    
    def run_demo(self, duration: float = 10.0):
        """
        데모 실행
        
        Args:
            duration: 데모 실행 시간 (초)
        """
        self.logger.info(f"SignGlove 데모 시작 (지속시간: {duration}초)")
        
        # 시뮬레이션 데이터 생성
        sensor_data_list = self.simulate_sensor_data(duration)
        
        # 실시간 시뮬레이션
        start_time = time.time()
        prediction_count = 0
        
        for i, sensor_data in enumerate(sensor_data_list):
            # 센서 데이터 추가
            self.inference.add_sensor_data(sensor_data)
            
            # 실시간 예측
            result = self.inference.predict_realtime()
            
            if result:
                prediction_count += 1
                elapsed_time = time.time() - start_time
                
                print(f"[{elapsed_time:.2f}s] 예측: {result['predicted_label']} "
                      f"(신뢰도: {result['confidence']:.3f})")
                
                # 예측 간격 조절
                time.sleep(self.inference.config['prediction_delay'])
        
        self.logger.info(f"데모 완료: 총 {prediction_count}개 예측 수행")

def main():
    """메인 함수"""
    # 모델 경로 설정
    model_path = "../cross_validation_model.pth"
    
    # 추론 시스템 초기화
    try:
        inference_system = SignGloveInference(model_path)
        
        # 모델 정보 출력
        model_info = inference_system.get_model_info()
        print("=== SignGlove 추론 시스템 정보 ===")
        print(f"모델 경로: {model_info['model_path']}")
        print(f"디바이스: {model_info['device']}")
        print(f"총 파라미터 수: {model_info['total_parameters']:,}")
        print(f"라벨 수: {len(model_info['label_mapper'])}")
        print("라벨:", list(model_info['label_mapper'].values()))
        
        # 데모 실행
        demo = SignGloveDemo(inference_system)
        demo.run_demo(duration=5.0)
        
    except Exception as e:
        print(f"오류 발생: {e}")
        return

if __name__ == "__main__":
    main()
