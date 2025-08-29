"""
Unified SignGlove 추론 시스템 - 상세 설계
GitHub SignGlove_HW/unified 저장소 구조를 참고한 통합 추론 파이프라인
"""

import torch
import torch.nn.functional as F
import numpy as np
import time
import threading
import queue
import asyncio
from collections import deque, OrderedDict
from typing import Dict, List, Optional, Tuple, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
import json
from pathlib import Path
import os

# 프로젝트 모듈들
from models.deep_learning import DeepLearningPipeline
from preprocessing.normalization import SensorNormalization
from preprocessing.filters import apply_low_pass_filter

class SensorType(Enum):
    """센서 타입 정의"""
    FLEX = "flex"
    IMU = "imu"
    ORIENTATION = "orientation"

class InferenceMode(Enum):
    """추론 모드 정의"""
    REALTIME = "realtime"
    BATCH = "batch"
    HYBRID = "hybrid"

@dataclass
class SensorReading:
    """센서 읽기 데이터 구조 - 상보필터 전용"""
    timestamp: float
    flex_data: List[float] = field(default_factory=list)  # 5개 flex 센서
    orientation_data: List[float] = field(default_factory=list)  # 3개 오일러각 (상보필터)
    raw_data: Optional[List[float]] = None
    source: str = "unknown"
    
    def to_unified_array(self) -> np.ndarray:
        """통합 배열로 변환 (flex5 + orientation3) - 상보필터 데이터만 사용"""
        unified = []
        
        # Flex 센서 데이터 (5개)
        if self.flex_data and len(self.flex_data) >= 5:
            unified.extend(self.flex_data[:5])
        else:
            unified.extend([800.0] * 5)  # 기본값
        
        # 상보필터 각도 데이터 (3개)
        if self.orientation_data and len(self.orientation_data) >= 3:
            unified.extend(self.orientation_data[:3])
        else:
            unified.extend([0.0, 0.0, 0.0])  # 기본값
        
        return np.array(unified, dtype=np.float32)

@dataclass
class InferenceResult:
    """추론 결과 구조"""
    predicted_class: str
    confidence: float
    processing_time: float
    timestamp: float
    stability_score: float = 0.0
    metadata: Dict = field(default_factory=dict)
    correct: bool = False  # 예측이 정확한지 여부
    expected_class: str = ""  # 예상 클래스

class UnifiedInferencePipeline:
    """
    통합 SignGlove 추론 파이프라인 - 상보필터 전용
    - 상보필터 센서 데이터 지원 (Flex + Orientation)
    - 적응형 전처리 및 정규화
    - 멀티스레드 실시간 처리
    - 지능형 안정성 체크
    - 성능 모니터링 및 최적화
    """
    
    def __init__(self, 
                 model_path: str = 'best_dl_model.pth',
                 config: Optional[Dict] = None,
                 mode: InferenceMode = InferenceMode.REALTIME):
        """
        Args:
            model_path: 학습된 모델 파일 경로
            config: 설정 딕셔너리
            mode: 추론 모드
        """
        self.mode = mode
        self.config = self._load_config(config)
        
        # 로깅 설정
        self._setup_logging()
        
        # 장치 설정
        self.device = self._setup_device()
        
        # 전처리 파이프라인
        self.normalizer = SensorNormalization(
            method=self.config['preprocessing']['normalization_method']
        )
        
        # 클래스 매핑
        self.class_names = self.config['classes']['names']
        self.num_classes = len(self.class_names)
        
        # 모델 로드
        self.model = self._load_model(model_path)
        
        # 버퍼 및 큐 시스템
        self._setup_buffers()
        
        # 실시간 처리 상태
        self.is_running = False
        self.inference_thread = None
        self.processing_thread = None
        
        # 성능 모니터링
        self._setup_performance_monitoring()
        
        # 콜백 시스템
        self.callbacks = {
            'prediction': [],
            'data_received': [],
            'error': [],
            'performance': []
        }
        
        self.logger.info(f"UnifiedInferencePipeline 초기화 완료 - 모드: {mode.value} (상보필터 전용)")
    
    def _load_config(self, config: Optional[Dict]) -> Dict:
        """설정 로드"""
        default_config = {
            'preprocessing': {
                'normalization_method': 'minmax',
                'noise_reduction': True,
                'window_size': 20,
                'stride': 5
            },
            'inference': {
                'confidence_threshold': 0.7,
                'stability_window': 5,
                'max_predictions_per_second': 100
            },
            'performance': {
                'target_latency_ms': 10.0,
                'max_fps': 200,
                'buffer_size': 1000
            },
            'classes': {
                'names': ['ㄱ', 'ㄴ', 'ㄷ', 'ㄹ', 'ㅁ']
            }
        }
        
        if config:
            # 중첩 딕셔너리 업데이트
            for key, value in config.items():
                if key in default_config and isinstance(value, dict):
                    default_config[key].update(value)
                else:
                    default_config[key] = value
        
        return default_config
    
    def _setup_logging(self):
        """로깅 설정"""
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def _setup_device(self) -> torch.device:
        """장치 설정"""
        if torch.cuda.is_available():
            device = torch.device('cuda')
            self.logger.info("CUDA 장치 사용")
        else:
            device = torch.device('cpu')
            self.logger.info("CPU 장치 사용")
        return device

    def _load_model(self, model_path: str) -> torch.nn.Module:
        """모델 로드"""
        try:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")
            
            # 모델 로드 (state_dict 또는 전체 모델)
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                # state_dict 형태
                model = DeepLearningPipeline(
                    input_features=8,
                    sequence_length=self.config['preprocessing']['window_size'],
                    num_classes=self.num_classes,
                    hidden_dim=128,
                    num_layers=2,
                    dropout=0.3
                ).to(self.device)
                model.load_state_dict(checkpoint['state_dict'])
            elif hasattr(checkpoint, 'eval'):
                # 전체 모델 객체
                model = checkpoint
            else:
                # state_dict만 있는 경우
                model = DeepLearningPipeline(
                    input_features=8,
                    sequence_length=self.config['preprocessing']['window_size'],
                    num_classes=self.num_classes,
                    hidden_dim=128,
                    num_layers=2,
                    dropout=0.3
                ).to(self.device)
                model.load_state_dict(checkpoint)
            
            model.eval()
            self.logger.info(f"모델 로드 완료: {model_path}")
            return model
            
        except Exception as e:
            self.logger.error(f"모델 로드 실패: {e}")
            raise

    def _setup_buffers(self):
        """버퍼 및 큐 시스템 설정"""
        buffer_size = self.config['performance']['buffer_size']
        
        # 데이터 버퍼
        self.raw_data_buffer = deque(maxlen=buffer_size)
        self.processed_data_buffer = deque(maxlen=buffer_size)
        
        # 추론 큐
        self.inference_queue = queue.Queue(maxsize=100)
        
        # 예측 이력
        self.prediction_history = deque(maxlen=50)
        
        # 안정성 체크 버퍼
        self.stability_buffer = deque(maxlen=self.config['inference']['stability_window'])

    def _setup_performance_monitoring(self):
        """성능 모니터링 설정"""
        self.performance_metrics = {
            'inference_times': deque(maxlen=100),
            'preprocessing_times': deque(maxlen=100),
            'total_frames': 0,
            'successful_predictions': 0,
            'start_time': time.time()
        }

    def add_sensor_data(self, 
                       sensor_reading: Union[SensorReading, Dict, List[float]],
                       source: str = "external") -> bool:
        """
        센서 데이터 추가 - 상보필터 데이터만 지원
        
        Args:
            sensor_reading: 센서 데이터 (SensorReading, Dict, 또는 List)
            source: 데이터 소스
            
        Returns:
            성공 여부
        """
        try:
            # 데이터 타입에 따른 파싱
            if isinstance(sensor_reading, SensorReading):
                reading = sensor_reading
            elif isinstance(sensor_reading, dict):
                reading = self._parse_dict_sensor_data(sensor_reading, source)
            elif isinstance(sensor_reading, list):
                reading = self._parse_list_sensor_data(sensor_reading, source)
            else:
                raise ValueError(f"지원하지 않는 데이터 타입: {type(sensor_reading)}")
            
            # 전처리
            processed_reading = self._preprocess_sensor_reading(reading)
            
            # 버퍼에 추가
            self.raw_data_buffer.append(reading)
            self.processed_data_buffer.append(processed_reading)
            
            # 성능 메트릭 업데이트
            self.performance_metrics['total_frames'] += 1
            
            # 콜백 호출
            self._trigger_callbacks('data_received', {
                'type': 'data_addition',
                'timestamp': reading.timestamp,
                'source': source,
                'data_size': len(processed_reading.to_unified_array())
            })
            
            return True
            
        except Exception as e:
            self.logger.error(f"센서 데이터 추가 실패: {e}")
            self._trigger_callbacks('error', {
                'type': 'data_addition', 
                'error': str(e)
            })
            return False

    def _parse_dict_sensor_data(self, data: Dict, source: str) -> SensorReading:
        """딕셔너리 형태 센서 데이터 파싱 - 상보필터 전용"""
        reading = SensorReading(
            timestamp=data.get('timestamp', time.time()),
            source=source
        )
        
        # Flex 데이터 추출 (flex_data 키가 있으면 사용, 없으면 개별 키에서 추출)
        if 'flex_data' in data:
            reading.flex_data = data['flex_data']
        else:
            flex_data = []
            for i in range(1, 6):
                flex_key = f'flex{i}'
                if flex_key in data:
                    flex_data.append(data[flex_key])
            
            if flex_data:
                reading.flex_data = flex_data
        
        # Orientation 데이터 추출 (orientation_data 키가 있으면 사용, 없으면 개별 키에서 추출)
        if 'orientation_data' in data:
            reading.orientation_data = data['orientation_data']
        else:
            orientation_data = []
            for key in ['pitch', 'roll', 'yaw']:
                if key in data:
                    orientation_data.append(data[key])
            
            if orientation_data:
                reading.orientation_data = orientation_data
        
        return reading

    def _parse_list_sensor_data(self, data: List[float], source: str) -> SensorReading:
        """리스트 형태 센서 데이터 파싱 - 상보필터 전용"""
        try:
            if len(data) != 8:
                raise ValueError(f"센서 데이터는 8개 값이어야 합니다. 받은 값: {len(data)}")
            
            # 안전한 슬라이싱
            flex_data = data[:5] if len(data) >= 5 else data + [800.0] * (5 - len(data))
            orientation_data = data[5:8] if len(data) >= 8 else [0.0] * 3
            
            reading = SensorReading(
                timestamp=time.time(),
                flex_data=flex_data,  # Flex 센서 5개
                orientation_data=orientation_data,  # 각도 데이터 3개
                source=source
            )
            
            return reading
            
        except Exception as e:
            self.logger.error(f"리스트 데이터 파싱 실패: {e}, 데이터: {data}")
            # 기본값으로 fallback
            return SensorReading(
                timestamp=time.time(),
                flex_data=[800.0] * 5,
                orientation_data=[0.0] * 3,
                source=source
            )

    def _preprocess_sensor_reading(self, reading: SensorReading) -> SensorReading:
        """센서 읽기 데이터 전처리 - 상보필터 전용"""
        start_time = time.time()
        
        try:
            # 통합 배열로 변환
            unified_data = reading.to_unified_array()
            
            # 노이즈 감소
            if self.config['preprocessing']['noise_reduction']:
                unified_data = self._apply_noise_reduction(unified_data)
            
            # 정규화 제거 (학습과 동일하게 원본 데이터 사용)
            # normalized_data = self._normalize_sensor_data(unified_data)
            
            # 전처리된 reading 생성 (원본 데이터 사용)
            flex_data = unified_data[:5].tolist() if len(unified_data) >= 5 else unified_data.tolist() + [0.0] * (5 - len(unified_data))
            orientation_data = unified_data[5:8].tolist() if len(unified_data) >= 8 else [0.0] * 3
            
            processed_reading = SensorReading(
                timestamp=reading.timestamp,
                flex_data=flex_data,
                orientation_data=orientation_data,
                raw_data=unified_data.tolist(),
                source=reading.source
            )
            
            # 전처리 시간 기록
            processing_time = time.time() - start_time
            self.performance_metrics['preprocessing_times'].append(processing_time)
            
            return processed_reading
            
        except Exception as e:
            self.logger.error(f"전처리 실패: {e}")
            return reading

    def _apply_noise_reduction(self, data: np.ndarray) -> np.ndarray:
        """노이즈 감소 필터 적용"""
        if len(self.processed_data_buffer) > 0:
            # 이전 데이터와의 가중 평균 (간단한 저역 통과 필터)
            prev_data = np.array(self.processed_data_buffer[-1].to_unified_array())
            alpha = 0.7  # 현재 데이터 가중치
            data = alpha * data + (1 - alpha) * prev_data
        
        return data

    def _normalize_sensor_data(self, data: np.ndarray) -> np.ndarray:
        """센서 데이터 정규화 - 학습과 동일한 방식"""
        try:
            # Flex 센서와 Orientation 센서를 분리하여 정규화
            flex_data = data[:5]  # Flex 센서 5개
            orientation_data = data[5:8]  # Orientation 센서 3개
            
            # Flex 센서 정규화 (700-900 범위를 0-1로)
            flex_normalized = np.clip((flex_data - 700) / 200, 0, 1)
            
            # Orientation 센서 정규화 (-180~180 범위를 -1~1로)
            orientation_normalized = np.clip(orientation_data / 180, -1, 1)
            
            # 통합
            normalized_data = np.concatenate([flex_normalized, orientation_normalized])
            
            return normalized_data
            
        except Exception as e:
            self.logger.error(f"정규화 실패: {e}")
            return data

    def predict_single(self, force_predict: bool = False, expected_class: str = "") -> Optional[InferenceResult]:
        """
        단일 추론 수행 - 상보필터 데이터만 사용
        
        Args:
            force_predict: 강제 추론 여부
            expected_class: 예상 클래스 (정확성 계산용)
            
        Returns:
            추론 결과 또는 None
        """
        try:
            # 충분한 데이터가 있는지 확인
            if len(self.processed_data_buffer) < self.config['preprocessing']['window_size']:
                if not force_predict:
                    return None
            
            # 윈도우 데이터 생성
            window_data = self._create_inference_window()
            
            # 추론 수행
            start_time = time.time()
            prediction = self._perform_inference(window_data)
            inference_time = time.time() - start_time
            
            if prediction is None:
                return None
            
            # 정확성 계산
            is_correct = expected_class and prediction['class'] == expected_class
            
            # 결과 생성
            result = InferenceResult(
                predicted_class=prediction['class'],
                confidence=prediction['confidence'],
                processing_time=inference_time,
                timestamp=time.time(),
                stability_score=self._calculate_stability_score(prediction['class']),
                metadata=prediction.get('metadata', {}),
                correct=is_correct,
                expected_class=expected_class
            )
            
            # 예측 이력 업데이트
            self.prediction_history.append(result)
            self.stability_buffer.append(result.predicted_class)
            
            # 성능 메트릭 업데이트
            self.performance_metrics['inference_times'].append(inference_time)
            self.performance_metrics['successful_predictions'] += 1
            
            # 콜백 호출
            self._trigger_callbacks('prediction', {
                'type': 'prediction',
                'result': result,
                'timestamp': result.timestamp
            })
            
            return result
            
        except Exception as e:
            self.logger.error(f"추론 실패: {e}")
            self._trigger_callbacks('error', {
                'type': 'inference',
                'error': str(e)
            })
            return None

    def _create_inference_window(self) -> np.ndarray:
        """추론용 윈도우 데이터 생성"""
        window_size = self.config['preprocessing']['window_size']
        
        # 최근 데이터로 윈도우 생성
        recent_data = list(self.processed_data_buffer)[-window_size:]
        
        # 윈도우 데이터를 2D 배열로 변환
        window_array = np.array([
            reading.to_unified_array() for reading in recent_data
        ])
        
        # 배치 차원 추가
        window_array = window_array.reshape(1, window_size, -1)
        
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
                
                # 신뢰도 임계값 체크
                if confidence < self.config['inference']['confidence_threshold']:
                    return None
                
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
            self.logger.error(f"모델 추론 실패: {e}")
            return None

    def _calculate_stability_score(self, predicted_class: str) -> float:
        """안정성 점수 계산"""
        if len(self.stability_buffer) < 2:
            return 0.0
        
        # 최근 예측들에서 같은 클래스 비율 계산
        recent_predictions = list(self.stability_buffer)
        same_class_count = recent_predictions.count(predicted_class)
        
        stability_score = same_class_count / len(recent_predictions)
        return stability_score

    def get_stable_prediction(self, min_stability: float = 0.8) -> Optional[InferenceResult]:
        """안정적인 예측 결과 반환"""
        if len(self.prediction_history) == 0:
            return None
        
        latest_result = self.prediction_history[-1]
        
        if latest_result.stability_score >= min_stability:
            return latest_result
        
        return None

    def get_performance_stats(self) -> Dict:
        """성능 통계 반환"""
        stats = {
            'total_frames': self.performance_metrics['total_frames'],
            'successful_predictions': self.performance_metrics['successful_predictions'],
            'window_size': self.config['preprocessing']['window_size'],
            'confidence_threshold': self.config['inference']['confidence_threshold'],
            'target_latency_ms': self.config['performance']['target_latency_ms']
        }
        
        # 평균 추론 시간
        if self.performance_metrics['inference_times']:
            avg_inference_time = np.mean(self.performance_metrics['inference_times'])
            stats['avg_inference_time_ms'] = avg_inference_time * 1000
            stats['fps'] = 1.0 / avg_inference_time if avg_inference_time > 0 else 0
        
        # 평균 전처리 시간
        if self.performance_metrics['preprocessing_times']:
            avg_preprocessing_time = np.mean(self.performance_metrics['preprocessing_times'])
            stats['avg_preprocessing_time_ms'] = avg_preprocessing_time * 1000
        
        # 실행 시간
        elapsed_time = time.time() - self.performance_metrics['start_time']
        stats['elapsed_time'] = elapsed_time
        
        return stats

    def _trigger_callbacks(self, callback_type: str, data: Dict):
        """콜백 함수 호출"""
        if callback_type in self.callbacks:
            for callback in self.callbacks[callback_type]:
                try:
                    callback(data)
                except Exception as e:
                    self.logger.error(f"콜백 실행 실패: {e}")

    def add_callback(self, callback_type: str, callback: Callable):
        """콜백 함수 추가"""
        if callback_type in self.callbacks:
            self.callbacks[callback_type].append(callback)

    def start_realtime_inference(self, prediction_callback: Optional[Callable] = None):
        """실시간 추론 시작"""
        if self.is_running:
            self.logger.warning("이미 실행 중입니다.")
            return
        
        if prediction_callback:
            self.add_callback('prediction', prediction_callback)
        
        self.is_running = True
        self.inference_thread = threading.Thread(target=self._inference_loop)
        self.inference_thread.daemon = True
        self.inference_thread.start()
        
        self.logger.info("실시간 추론 시작")

    def stop_realtime_inference(self):
        """실시간 추론 중지"""
        self.is_running = False
        
        if self.inference_thread:
            self.inference_thread.join(timeout=1.0)
        
        self.logger.info("실시간 추론 중지")

    def _inference_loop(self):
        """추론 루프"""
        while self.is_running:
            try:
                # 추론 수행
                result = self.predict_single()
                
                if result:
                    # 성능 제한 체크
                    max_fps = self.config['performance']['max_fps']
                    if max_fps > 0:
                        time.sleep(1.0 / max_fps)
                
            except Exception as e:
                self.logger.error(f"추론 루프 오류: {e}")
                time.sleep(0.01)

def create_unified_inference_pipeline(config_path: Optional[str] = None, 
                                     model_path: str = 'best_dl_model.pth',
                                     config: Optional[Dict] = None) -> UnifiedInferencePipeline:
    """통합 추론 파이프라인 생성 - 상보필터 전용"""
    
    # 설정 로드
    file_config = None
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            file_config = json.load(f)
    
    # 전달된 config와 파일 config 병합
    if file_config and config:
        # 중첩 딕셔너리 업데이트
        for key, value in config.items():
            if key in file_config and isinstance(value, dict):
                file_config[key].update(value)
            else:
                file_config[key] = value
        final_config = file_config
    elif config:
        final_config = config
    elif file_config:
        final_config = file_config
    else:
        final_config = None
    
    return UnifiedInferencePipeline(
        model_path=model_path,
        config=final_config,
        mode=InferenceMode.REALTIME
    )

if __name__ == "__main__":
    # 테스트 코드
    pipeline = create_unified_inference_pipeline()
    print("UnifiedInferencePipeline 생성 완료 (상보필터 전용)")
    
    # 성능 통계 출력
    stats = pipeline.get_performance_stats()
    print(f"초기 성능 통계: {stats}")
