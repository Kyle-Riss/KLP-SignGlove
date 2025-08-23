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

# 프로젝트 모듈들
from models.deep_learning import DeepLearningPipeline
from preprocessing.normalization import SensorNormalization
from preprocessing.filters import apply_low_pass_filter
# Madgwick 어댑터 제거됨

class SensorType(Enum):
    """센서 타입 정의"""
    FLEX = "flex"
    IMU = "imu"
    ORIENTATION = "orientation"
    # MADGWICK = "madgwick"  # 제거됨

class InferenceMode(Enum):
    """추론 모드 정의"""
    REALTIME = "realtime"
    BATCH = "batch"
    SIMULATION = "simulation"
    HYBRID = "hybrid"

@dataclass
class SensorReading:
    """센서 읽기 데이터 구조"""
    timestamp: float
    flex_data: List[float] = field(default_factory=list)  # 5개 flex 센서
    imu_data: List[float] = field(default_factory=list)   # 6개 IMU (accel + gyro)
    orientation_data: List[float] = field(default_factory=list)  # 3개 오일러각
    madgwick_data: Optional[Dict] = None  # Madgwick 필터 데이터
    raw_data: Optional[List[float]] = None
    source: str = "unknown"
    
    def to_unified_array(self) -> np.ndarray:
        """통합 배열로 변환 (flex5 + orientation3)"""
        unified = []
        
        # Flex 센서 데이터 (5개)
        if self.flex_data and len(self.flex_data) >= 5:
            unified.extend(self.flex_data[:5])
        else:
            unified.extend([800.0] * 5)  # 기본값
        
        # 오리엔테이션 데이터 (3개)
        if self.orientation_data and len(self.orientation_data) >= 3:
            unified.extend(self.orientation_data[:3])
        elif self.madgwick_data:
            # Madgwick에서 오리엔테이션 추출
            unified.extend([
                self.madgwick_data.get('pitch', 0.0),
                self.madgwick_data.get('roll', 0.0),
                self.madgwick_data.get('yaw', 0.0)
            ])
        else:
            unified.extend([0.0, 0.0, 0.0])  # 기본값
        
        return np.array(unified, dtype=np.float32)

@dataclass
class InferenceResult:
    """추론 결과 구조"""
    predicted_class: str
    predicted_idx: int
    confidence: float
    probabilities: np.ndarray
    timestamp: float
    processing_time: float
    source_data: SensorReading
    attention_weights: Optional[np.ndarray] = None
    stability_score: float = 0.0
    metadata: Dict = field(default_factory=dict)

class UnifiedInferencePipeline:
    """
    통합 SignGlove 추론 파이프라인
    - 다중 센서 타입 지원 (Flex + IMU + Madgwick)
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
        
        # Madgwick 어댑터 제거됨
        # self.madgwick_adapter = MadgwickDataAdapter()
        
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
        
        self.logger.info(f"UnifiedInferencePipeline 초기화 완료 - 모드: {mode.value}")
    
    def _load_config(self, config: Optional[Dict]) -> Dict:
        """설정 로드"""
        default_config = {
            'window_size': 20,
            'stride': 5,
            'confidence_threshold': 0.7,
            'stability_threshold': 0.7,
            'buffer_size': 100,
            'preprocessing': {
                'normalization_method': 'minmax',
                'filter_type': 'lowpass',
                'filter_cutoff': 10.0,
                'noise_reduction': True
            },
            'performance': {
                'max_fps': 200,
                'target_latency': 0.01,  # 10ms
                'memory_limit': 200  # MB
            },
            'classes': {
                'names': ['ㄱ', 'ㄴ', 'ㄷ', 'ㄹ', 'ㅁ']
            },
            'stability': {
                'history_length': 5,
                'consensus_threshold': 0.7,
                'confidence_decay': 0.95
            }
        }
        
        if config:
            # 딥 머지
            self._deep_merge(default_config, config)
        
        return default_config
    
    def _deep_merge(self, base: Dict, override: Dict):
        """딕셔너리 깊은 병합"""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value
    
    def _setup_logging(self):
        """로깅 시스템 설정"""
        self.logger = logging.getLogger(f"UnifiedInference-{id(self)}")
        self.logger.setLevel(logging.INFO)
        
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def _setup_device(self) -> torch.device:
        """추론 장치 설정"""
        if torch.cuda.is_available():
            device = torch.device('cuda')
            self.logger.info(f"CUDA 장치 사용: {torch.cuda.get_device_name()}")
        else:
            device = torch.device('cpu')
            self.logger.info("CPU 장치 사용")
        
        return device
    
    def _load_model(self, model_path: str) -> torch.nn.Module:
        """모델 로드 및 최적화"""
        try:
            # 모델 파일에서 클래스 수 확인
            if Path(model_path).exists():
                state_dict = torch.load(model_path, map_location=self.device)
                # 마지막 분류층의 출력 크기로 클래스 수 추정
                if 'classifier.3.weight' in state_dict:
                    num_classes = state_dict['classifier.3.weight'].shape[0]
                    self.logger.info(f"모델에서 감지된 클래스 수: {num_classes}")
                else:
                    num_classes = self.num_classes
                    self.logger.warning(f"클래스 수를 추정할 수 없어 기본값 사용: {num_classes}")
            else:
                num_classes = self.num_classes
                self.logger.warning(f"모델 파일 없음: {model_path}, 기본 클래스 수 사용: {num_classes}")
            
            model = DeepLearningPipeline(
                input_features=8,
                sequence_length=self.config['window_size'],
                num_classes=num_classes,
                hidden_dim=128,
                num_layers=2,
                dropout=0.3
            )
            
            if Path(model_path).exists():
                state_dict = torch.load(model_path, map_location=self.device)
                model.load_state_dict(state_dict)
                self.logger.info(f"모델 로드 성공: {model_path}")
            else:
                self.logger.warning(f"모델 파일 없음: {model_path}, 기본 초기화 모델 사용")
            
            model.to(self.device)
            model.eval()
            
            # 모델 최적화 (if available)
            if hasattr(torch.jit, 'script'):
                try:
                    # JIT 컴파일 시도
                    dummy_input = torch.randn(1, self.config['window_size'], 8).to(self.device)
                    model = torch.jit.trace(model, dummy_input)
                    self.logger.info("모델 JIT 최적화 완료")
                except Exception as e:
                    self.logger.warning(f"JIT 최적화 실패: {e}")
            
            return model
            
        except Exception as e:
            self.logger.error(f"모델 로드 오류: {e}")
            raise
    
    def _setup_buffers(self):
        """버퍼 및 큐 시스템 설정"""
        buffer_size = self.config['buffer_size']
        
        # 원시 데이터 버퍼
        self.raw_data_buffer = deque(maxlen=buffer_size)
        
        # 전처리된 데이터 버퍼
        self.processed_data_buffer = deque(maxlen=buffer_size)
        
        # 예측 이력
        self.prediction_history = deque(maxlen=self.config['stability']['history_length'] * 2)
        
        # 실시간 처리 큐들
        self.data_queue = queue.Queue(maxsize=buffer_size)
        self.result_queue = queue.Queue(maxsize=50)
        self.error_queue = queue.Queue(maxsize=20)
        
        # 센서 타입별 특화 버퍼
        self.sensor_buffers = {
            SensorType.FLEX: deque(maxlen=buffer_size),
            SensorType.IMU: deque(maxlen=buffer_size),
            SensorType.ORIENTATION: deque(maxlen=buffer_size),
            # SensorType.MADGWICK: deque(maxlen=buffer_size)  # 제거됨
        }
    
    def _setup_performance_monitoring(self):
        """성능 모니터링 설정"""
        self.performance_metrics = {
            'inference_times': deque(maxlen=100),
            'preprocessing_times': deque(maxlen=100),
            'total_processing_times': deque(maxlen=100),
            'frame_count': 0,
            'error_count': 0,
            'start_time': time.time()
        }
        
        # 성능 통계
        self.stats = {
            'fps': 0.0,
            'avg_latency': 0.0,
            'memory_usage': 0.0,
            'cpu_usage': 0.0
        }
    
    def add_sensor_data(self, 
                       sensor_reading: Union[SensorReading, Dict, List[float]],
                       source: str = "external") -> bool:
        """
        센서 데이터 추가 (다양한 형태 지원)
        
        Args:
            sensor_reading: 센서 데이터 (다양한 형태)
            source: 데이터 소스
            
        Returns:
            성공 여부
        """
        try:
            # 데이터 타입에 따른 변환
            if isinstance(sensor_reading, list):
                # 리스트 형태 [flex1-5, pitch, roll, yaw]
                reading = SensorReading(
                    timestamp=time.time(),
                    flex_data=sensor_reading[:5] if len(sensor_reading) >= 5 else [800.0] * 5,
                    orientation_data=sensor_reading[5:8] if len(sensor_reading) >= 8 else [0.0] * 3,
                    source=source
                )
            elif isinstance(sensor_reading, dict):
                # 딕셔너리 형태 (Madgwick 등)
                reading = self._parse_dict_sensor_data(sensor_reading, source)
            elif isinstance(sensor_reading, SensorReading):
                reading = sensor_reading
            else:
                raise ValueError(f"지원하지 않는 센서 데이터 타입: {type(sensor_reading)}")
            
            # 원시 데이터 버퍼에 추가
            self.raw_data_buffer.append(reading)
            
            # 전처리
            processed_reading = self._preprocess_sensor_reading(reading)
            self.processed_data_buffer.append(processed_reading)
            
            # 센서 타입별 버퍼에 추가
            self._update_sensor_buffers(reading)
            
            # 실시간 모드에서 큐에 추가
            if self.is_running and self.mode == InferenceMode.REALTIME:
                try:
                    self.data_queue.put(processed_reading, block=False)
                except queue.Full:
                    self.logger.warning("데이터 큐가 가득 찼습니다. 오래된 데이터를 제거합니다.")
                    try:
                        self.data_queue.get_nowait()
                        self.data_queue.put(processed_reading, block=False)
                    except queue.Empty:
                        pass
            
            # 콜백 호출
            self._trigger_callbacks('data_received', processed_reading)
            
            return True
            
        except Exception as e:
            self.logger.error(f"센서 데이터 추가 실패: {e}")
            self._trigger_callbacks('error', {'type': 'data_addition', 'error': str(e)})
            return False
    
    def _parse_dict_sensor_data(self, data: Dict, source: str) -> SensorReading:
        """딕셔너리 형태 센서 데이터 파싱"""
        reading = SensorReading(
            timestamp=data.get('timestamp', time.time()),
            source=source
        )
        
        # Madgwick 데이터 형태 체크
        if 'pitch' in data and 'roll' in data and 'yaw' in data:
            reading.madgwick_data = {
                'pitch': data.get('pitch', 0.0),
                'roll': data.get('roll', 0.0),
                'yaw': data.get('yaw', 0.0)
            }
            
            # 가속도 데이터가 있다면 추가
            if 'ax' in data:
                reading.madgwick_data.update({
                    'ax': data.get('ax', 0.0),
                    'ay': data.get('ay', 0.0),
                    'az': data.get('az', 0.0)
                })
        
        # Flex 데이터 추출
        flex_data = []
        for i in range(1, 6):
            flex_key = f'flex{i}'
            if flex_key in data:
                flex_data.append(data[flex_key])
        
        if flex_data:
            reading.flex_data = flex_data
        
        # 오리엔테이션 데이터 (Madgwick에서 추출하거나 직접)
        orientation_data = []
        for key in ['pitch', 'roll', 'yaw']:
            if key in data:
                orientation_data.append(data[key])
        
        if orientation_data:
            reading.orientation_data = orientation_data
        
        return reading
    
    def _preprocess_sensor_reading(self, reading: SensorReading) -> SensorReading:
        """센서 읽기 데이터 전처리"""
        start_time = time.time()
        
        try:
            # 통합 배열로 변환
            unified_data = reading.to_unified_array()
            
            # 노이즈 감소
            if self.config['preprocessing']['noise_reduction']:
                unified_data = self._apply_noise_reduction(unified_data)
            
            # 정규화
            normalized_data = self._normalize_sensor_data(unified_data)
            
            # 전처리된 reading 생성
            processed_reading = SensorReading(
                timestamp=reading.timestamp,
                flex_data=normalized_data[:5].tolist(),
                orientation_data=normalized_data[5:8].tolist(),
                madgwick_data=reading.madgwick_data,
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
        """센서 데이터 정규화"""
        normalized_data = data.copy()
        
        # Flex 센서 정규화 (0-1024 -> 0-1)
        normalized_data[:5] = np.clip(normalized_data[:5] / 1024.0, 0, 1)
        
        # 오리엔테이션 정규화 (-180~180 -> -1~1)
        normalized_data[5:8] = np.clip(normalized_data[5:8] / 180.0, -1, 1)
        
        return normalized_data
    
    def _update_sensor_buffers(self, reading: SensorReading):
        """센서 타입별 버퍼 업데이트"""
        if reading.flex_data:
            self.sensor_buffers[SensorType.FLEX].append(reading.flex_data)
        
        if reading.imu_data:
            self.sensor_buffers[SensorType.IMU].append(reading.imu_data)
        
        if reading.orientation_data:
            self.sensor_buffers[SensorType.ORIENTATION].append(reading.orientation_data)
        
        if reading.madgwick_data:
            self.sensor_buffers[SensorType.MADGWICK].append(reading.madgwick_data)
    
    def predict_single(self, force_predict: bool = False) -> Optional[InferenceResult]:
        """
        단일 예측 수행
        
        Args:
            force_predict: 신뢰도 임계값 무시
            
        Returns:
            추론 결과 또는 None
        """
        if len(self.processed_data_buffer) < self.config['window_size']:
            return None
        
        start_time = time.time()
        
        try:
            # 윈도우 데이터 생성
            window_data = self._create_window_data()
            
            # 모델 추론
            inference_start = time.time()
            model_output = self._run_model_inference(window_data)
            inference_time = time.time() - inference_start
            
            # 결과 생성
            result = self._create_inference_result(
                model_output, 
                inference_time, 
                self.processed_data_buffer[-1]
            )
            
            # 신뢰도 체크
            if not force_predict and result.confidence < self.config['confidence_threshold']:
                return None
            
            # 예측 이력에 추가
            self.prediction_history.append(result)
            
            # 성능 메트릭 업데이트
            total_time = time.time() - start_time
            self._update_performance_metrics(inference_time, total_time)
            
            return result
            
        except Exception as e:
            self.logger.error(f"예측 실패: {e}")
            self._trigger_callbacks('error', {'type': 'prediction', 'error': str(e)})
            return None
    
    def _create_window_data(self) -> np.ndarray:
        """윈도우 데이터 생성"""
        window_size = self.config['window_size']
        recent_data = list(self.processed_data_buffer)[-window_size:]
        
        window_data = np.array([
            reading.to_unified_array() for reading in recent_data
        ])
        
        return window_data
    
    def _run_model_inference(self, window_data: np.ndarray) -> Dict:
        """모델 추론 실행"""
        with torch.no_grad():
            # 텐서 변환
            x = torch.FloatTensor(window_data).unsqueeze(0).to(self.device)
            
            # 모델 예측
            outputs = self.model(x)
            
            return outputs
    
    def _create_inference_result(self, 
                               model_output: Dict, 
                               inference_time: float,
                               source_data: SensorReading) -> InferenceResult:
        """추론 결과 객체 생성"""
        logits = model_output['class_logits']
        probabilities = F.softmax(logits, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)
        
        predicted_class = self.class_names[predicted_idx.item()]
        confidence_value = confidence.item()
        
        result = InferenceResult(
            predicted_class=predicted_class,
            predicted_idx=predicted_idx.item(),
            confidence=confidence_value,
            probabilities=probabilities.cpu().numpy()[0],
            timestamp=time.time(),
            processing_time=inference_time,
            source_data=source_data
        )
        
        # Attention weights 추가
        if 'attention_weights' in model_output:
            result.attention_weights = model_output['attention_weights'].cpu().numpy()[0]
        
        # 안정성 점수 계산
        result.stability_score = self._calculate_stability_score(result)
        
        return result
    
    def _calculate_stability_score(self, current_result: InferenceResult) -> float:
        """안정성 점수 계산"""
        if len(self.prediction_history) < 2:
            return 0.0
        
        history_length = min(5, len(self.prediction_history))
        recent_predictions = list(self.prediction_history)[-history_length:]
        
        # 클래스 일관성 체크
        same_class_count = sum(
            1 for pred in recent_predictions 
            if pred.predicted_class == current_result.predicted_class
        )
        
        class_consistency = same_class_count / history_length
        
        # 신뢰도 일관성 체크
        confidences = [pred.confidence for pred in recent_predictions]
        confidence_std = np.std(confidences) if len(confidences) > 1 else 0
        confidence_consistency = max(0, 1 - confidence_std)
        
        # 종합 안정성 점수
        stability_score = (class_consistency * 0.7 + confidence_consistency * 0.3)
        
        return stability_score
    
    def _update_performance_metrics(self, inference_time: float, total_time: float):
        """성능 메트릭 업데이트"""
        self.performance_metrics['inference_times'].append(inference_time)
        self.performance_metrics['total_processing_times'].append(total_time)
        self.performance_metrics['frame_count'] += 1
        
        # FPS 계산
        if len(self.performance_metrics['total_processing_times']) > 0:
            avg_time = np.mean(list(self.performance_metrics['total_processing_times']))
            self.stats['fps'] = 1.0 / avg_time if avg_time > 0 else 0
            self.stats['avg_latency'] = avg_time
    
    def get_stable_prediction(self, 
                            history_length: Optional[int] = None) -> Optional[InferenceResult]:
        """안정적인 예측 결과 반환"""
        if history_length is None:
            history_length = self.config['stability']['history_length']
        
        if len(self.prediction_history) < history_length:
            return None
        
        recent_predictions = list(self.prediction_history)[-history_length:]
        
        # 클래스별 투표
        class_votes = {}
        total_confidence = 0
        
        for pred in recent_predictions:
            class_name = pred.predicted_class
            weight = pred.confidence * pred.stability_score
            class_votes[class_name] = class_votes.get(class_name, 0) + weight
            total_confidence += pred.confidence
        
        # 최고 득표 클래스
        if not class_votes:
            return None
        
        winner_class = max(class_votes, key=class_votes.get)
        winner_score = class_votes[winner_class]
        total_votes = sum(class_votes.values())
        
        consensus_ratio = winner_score / total_votes if total_votes > 0 else 0
        avg_confidence = total_confidence / history_length
        
        # 안정성 임계값 체크
        consensus_threshold = self.config['stability']['consensus_threshold']
        
        if consensus_ratio >= consensus_threshold:
            # 최근 해당 클래스 예측 찾기
            latest_prediction = None
            for pred in reversed(recent_predictions):
                if pred.predicted_class == winner_class:
                    latest_prediction = pred
                    break
            
            if latest_prediction:
                # 안정적인 결과 생성
                stable_result = InferenceResult(
                    predicted_class=winner_class,
                    predicted_idx=latest_prediction.predicted_idx,
                    confidence=avg_confidence,
                    probabilities=latest_prediction.probabilities,
                    timestamp=time.time(),
                    processing_time=latest_prediction.processing_time,
                    source_data=latest_prediction.source_data,
                    attention_weights=latest_prediction.attention_weights,
                    stability_score=consensus_ratio,
                    metadata={
                        'is_stable': True,
                        'consensus_ratio': consensus_ratio,
                        'history_length': history_length,
                        'class_votes': class_votes
                    }
                )
                
                return stable_result
        
        return None
    
    def start_realtime_inference(self, 
                                prediction_callback: Optional[Callable] = None,
                                stable_prediction_callback: Optional[Callable] = None):
        """실시간 추론 시작"""
        if self.is_running:
            self.logger.warning("이미 실시간 추론이 실행 중입니다.")
            return
        
        self.is_running = True
        
        # 콜백 등록
        if prediction_callback:
            self.add_callback('prediction', prediction_callback)
        
        def inference_loop():
            """추론 루프"""
            self.logger.info("실시간 추론 루프 시작")
            
            while self.is_running:
                try:
                    # 데이터 대기
                    sensor_reading = self.data_queue.get(timeout=0.1)
                    
                    # 단일 예측
                    result = self.predict_single()
                    
                    if result:
                        # 일반 예측 콜백
                        self._trigger_callbacks('prediction', result)
                        
                        # 결과 큐에 추가
                        try:
                            self.result_queue.put(result, block=False)
                        except queue.Full:
                            # 큐 정리
                            try:
                                self.result_queue.get_nowait()
                                self.result_queue.put(result, block=False)
                            except queue.Empty:
                                pass
                    
                    # 안정적인 예측 체크
                    if stable_prediction_callback:
                        stable_result = self.get_stable_prediction()
                        if stable_result:
                            stable_prediction_callback(stable_result)
                
                except queue.Empty:
                    continue
                except Exception as e:
                    self.logger.error(f"추론 루프 오류: {e}")
                    self.performance_metrics['error_count'] += 1
                    self._trigger_callbacks('error', {
                        'type': 'inference_loop', 
                        'error': str(e)
                    })
                    continue
            
            self.logger.info("실시간 추론 루프 종료")
        
        # 백그라운드 스레드 시작
        self.inference_thread = threading.Thread(
            target=inference_loop, 
            daemon=True,
            name="UnifiedInference"
        )
        self.inference_thread.start()
        
        self.logger.info("실시간 추론 시작됨")
    
    def stop_realtime_inference(self):
        """실시간 추론 중지"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # 스레드 종료 대기
        if self.inference_thread:
            self.inference_thread.join(timeout=2.0)
            if self.inference_thread.is_alive():
                self.logger.warning("추론 스레드가 정상적으로 종료되지 않았습니다.")
        
        self.logger.info("실시간 추론 중지됨")
    
    def add_callback(self, event_type: str, callback: Callable):
        """콜백 함수 추가"""
        if event_type in self.callbacks:
            self.callbacks[event_type].append(callback)
        else:
            self.logger.warning(f"알 수 없는 이벤트 타입: {event_type}")
    
    def _trigger_callbacks(self, event_type: str, data):
        """콜백 함수들 호출"""
        if event_type in self.callbacks:
            for callback in self.callbacks[event_type]:
                try:
                    callback(data)
                except Exception as e:
                    self.logger.error(f"콜백 실행 오류 ({event_type}): {e}")
    
    def get_performance_stats(self) -> Dict:
        """상세 성능 통계 반환"""
        metrics = self.performance_metrics
        
        stats = {
            'fps': self.stats['fps'],
            'avg_latency_ms': self.stats['avg_latency'] * 1000,
            'total_frames': metrics['frame_count'],
            'error_count': metrics['error_count'],
            'uptime_seconds': time.time() - metrics['start_time'],
            'buffer_utilization': {
                'raw_data': len(self.raw_data_buffer) / self.config['buffer_size'],
                'processed_data': len(self.processed_data_buffer) / self.config['buffer_size'],
                'prediction_history': len(self.prediction_history) / (self.config['stability']['history_length'] * 2)
            }
        }
        
        # 상세 타이밍 통계
        if metrics['inference_times']:
            inference_times = list(metrics['inference_times'])
            stats['inference_timing'] = {
                'avg_ms': np.mean(inference_times) * 1000,
                'min_ms': np.min(inference_times) * 1000,
                'max_ms': np.max(inference_times) * 1000,
                'std_ms': np.std(inference_times) * 1000
            }
        
        if metrics['preprocessing_times']:
            prep_times = list(metrics['preprocessing_times'])
            stats['preprocessing_timing'] = {
                'avg_ms': np.mean(prep_times) * 1000,
                'min_ms': np.min(prep_times) * 1000,
                'max_ms': np.max(prep_times) * 1000
            }
        
        return stats
    
    def clear_buffers(self):
        """모든 버퍼 초기화"""
        self.raw_data_buffer.clear()
        self.processed_data_buffer.clear()
        self.prediction_history.clear()
        
        # 큐 비우기
        self._clear_queue(self.data_queue)
        self._clear_queue(self.result_queue)
        self._clear_queue(self.error_queue)
        
        # 센서 버퍼 초기화
        for buffer in self.sensor_buffers.values():
            buffer.clear()
        
        self.logger.info("모든 버퍼 초기화 완료")
    
    def _clear_queue(self, q: queue.Queue):
        """큐 비우기"""
        while not q.empty():
            try:
                q.get_nowait()
            except queue.Empty:
                break
    
    def save_performance_log(self, filepath: str):
        """성능 로그 저장"""
        log_data = {
            'timestamp': time.time(),
            'config': self.config,
            'performance_stats': self.get_performance_stats(),
            'prediction_history': [
                {
                    'predicted_class': pred.predicted_class,
                    'confidence': pred.confidence,
                    'stability_score': pred.stability_score,
                    'timestamp': pred.timestamp
                } for pred in list(self.prediction_history)
            ]
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"성능 로그 저장: {filepath}")

# 사용 예시 및 팩토리 함수들
def create_unified_inference_pipeline(config_path: Optional[str] = None, 
                                     model_path: str = 'best_dl_model.pth') -> UnifiedInferencePipeline:
    """통합 추론 파이프라인 생성 팩토리"""
    config = None
    if config_path and Path(config_path).exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
    
    return UnifiedInferencePipeline(
        model_path=model_path,
        config=config,
        mode=InferenceMode.REALTIME
    )

# def create_madgwick_compatible_pipeline(model_path: str = 'best_dl_model.pth') -> UnifiedInferencePipeline:
#     """Madgwick 데이터 호환 파이프라인 생성 - 제거됨"""
#     config = {
#         'preprocessing': {
#             'normalization_method': 'minmax',
#             'madgwick_integration': True,
#             'synthetic_flex_generation': True
#         },
#         'performance': {
#             'target_latency': 0.005,  # 5ms for Madgwick data
#             'max_fps': 300
#         }
#     }
#     
#     return UnifiedInferencePipeline(
#         model_path=model_path,
#         config=config,
#         mode=InferenceMode.HYBRID
#     )

if __name__ == "__main__":
    # 테스트 코드
    pipeline = create_unified_inference_pipeline()
    print("UnifiedInferencePipeline 생성 완료")
    
    # 성능 통계 출력
    stats = pipeline.get_performance_stats()
    print(f"초기 성능 통계: {stats}")
