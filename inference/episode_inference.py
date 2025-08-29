#!/usr/bin/env python3
"""
Episode 데이터 전용 추론 파이프라인
균형잡힌 모델과 호환되는 Episode 데이터 형태로 추론
"""

import os
import sys
import time
import torch
import torch.nn.functional as F
import numpy as np
import logging
from collections import deque
from typing import Optional, Dict, List, Union
from dataclasses import dataclass

# 프로젝트 루트 추가
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.deep_learning import DeepLearningPipeline
from training.label_mapping import KSLLabelMapper

@dataclass
class EpisodeSensorReading:
    """Episode 센서 데이터"""
    timestamp: float
    flex_data: List[float]  # 5개 flex 센서
    orientation_data: List[float]  # 3개 orientation (pitch, roll, yaw)
    source: str = "episode"
    
    def to_array(self) -> np.ndarray:
        """numpy 배열로 변환"""
        return np.concatenate([self.flex_data, self.orientation_data])

@dataclass
class EpisodeInferenceResult:
    """Episode 추론 결과"""
    predicted_class: str
    confidence: float
    processing_time: float
    timestamp: float
    correct: bool = False
    expected_class: str = ""

class EpisodeInferencePipeline:
    """Episode 데이터 전용 추론 파이프라인"""
    
    def __init__(self, model_path: str, window_size: int = 20):
        self.model_path = model_path
        self.window_size = window_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 라벨 매퍼
        self.label_mapper = KSLLabelMapper()
        self.class_names = [self.label_mapper.get_class_name(i) for i in range(self.label_mapper.get_num_classes())]
        
        # 모델 로드
        self.model = self._load_model()
        
        # 데이터 버퍼
        self.data_buffer = deque(maxlen=window_size * 2)
        
        # 로깅
        self.logger = logging.getLogger(__name__)
        
        print(f"🚀 Episode 추론 파이프라인 초기화 완료")
        print(f"  장치: {self.device}")
        print(f"  윈도우 크기: {window_size}")
        print(f"  클래스 수: {len(self.class_names)}")
    
    def _load_model(self) -> DeepLearningPipeline:
        """모델 로드"""
        try:
            print(f"📥 모델 로드 중: {self.model_path}")
            
            # 모델 초기화
            model = DeepLearningPipeline(
                input_features=8,
                sequence_length=self.window_size,
                num_classes=self.label_mapper.get_num_classes(),
                hidden_dim=128,
                num_layers=2,
                dropout=0.3
            ).to(self.device)
            
            # 체크포인트 로드
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
            
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            elif hasattr(checkpoint, 'eval'):
                model = checkpoint
            else:
                model.load_state_dict(checkpoint)
            
            model.eval()
            print(f"✅ 모델 로드 완료")
            return model
            
        except Exception as e:
            print(f"❌ 모델 로드 실패: {e}")
            raise
    
    def add_sensor_data(self, flex_data: List[float], orientation_data: List[float]) -> bool:
        """센서 데이터 추가"""
        try:
            # 데이터 검증
            if len(flex_data) != 5 or len(orientation_data) != 3:
                print(f"⚠️ 잘못된 데이터 크기: flex={len(flex_data)}, orientation={len(orientation_data)}")
                return False
            
            # SensorReading 생성
            reading = EpisodeSensorReading(
                timestamp=time.time(),
                flex_data=flex_data,
                orientation_data=orientation_data
            )
            
            # 버퍼에 추가
            self.data_buffer.append(reading)
            return True
            
        except Exception as e:
            print(f"❌ 데이터 추가 실패: {e}")
            return False
    
    def predict_single(self, expected_class: str = "") -> Optional[EpisodeInferenceResult]:
        """단일 추론 수행"""
        try:
            # 충분한 데이터 확인
            if len(self.data_buffer) < self.window_size:
                return None
            
            # 윈도우 데이터 생성
            window_data = self._create_window()
            
            # 추론 수행
            start_time = time.time()
            prediction = self._perform_inference(window_data)
            inference_time = time.time() - start_time
            
            if prediction is None:
                return None
            
            # 정확성 계산
            is_correct = expected_class and prediction['class'] == expected_class
            
            # 결과 생성
            result = EpisodeInferenceResult(
                predicted_class=prediction['class'],
                confidence=prediction['confidence'],
                processing_time=inference_time,
                timestamp=time.time(),
                correct=is_correct,
                expected_class=expected_class
            )
            
            return result
            
        except Exception as e:
            print(f"❌ 추론 실패: {e}")
            return None
    
    def _create_window(self) -> np.ndarray:
        """추론용 윈도우 생성"""
        # 최근 데이터로 윈도우 생성
        recent_data = list(self.data_buffer)[-self.window_size:]
        
        # 윈도우 데이터를 2D 배열로 변환
        window_array = np.array([
            reading.to_array() for reading in recent_data
        ])
        
        # 배치 차원 추가 (1, window_size, 8)
        window_array = window_array.reshape(1, self.window_size, 8)
        
        return window_array
    
    def _perform_inference(self, window_data: np.ndarray) -> Optional[Dict]:
        """실제 추론 수행"""
        try:
            # 텐서 변환
            input_tensor = torch.FloatTensor(window_data).to(self.device)
            
            # 추론
            with torch.no_grad():
                output = self.model(input_tensor)
                
                # 출력 형태에 따른 처리
                if isinstance(output, dict):
                    logits = output['class_logits']
                else:
                    logits = output
                
                # 확률 계산
                probabilities = F.softmax(logits, dim=1)
                
                # 최대 확률과 클래스
                max_prob, predicted_class = torch.max(probabilities, 1)
                
                confidence = max_prob.item()
                class_idx = predicted_class.item()
                
                return {
                    'class': self.class_names[class_idx],
                    'confidence': confidence,
                    'probabilities': probabilities.cpu().numpy(),
                    'metadata': {
                        'class_index': class_idx,
                        'all_probabilities': probabilities.cpu().numpy().tolist()
                    }
                }
                
        except Exception as e:
            print(f"❌ 모델 추론 실패: {e}")
            return None
    
    def get_performance_stats(self) -> Dict:
        """성능 통계 반환"""
        return {
            'window_size': self.window_size,
            'buffer_size': len(self.data_buffer),
            'device': str(self.device),
            'model_path': self.model_path
        }

def create_episode_inference_pipeline(model_path: str, config: Dict = None) -> EpisodeInferencePipeline:
    """Episode 추론 파이프라인 생성"""
    if config is None:
        config = {}
    
    window_size = config.get('window_size', 20)
    
    return EpisodeInferencePipeline(
        model_path=model_path,
        window_size=window_size
    )
