import torch
import torch.nn.functional as F
import numpy as np
import time
import threading
import queue
from collections import deque
from typing import Dict, List, Optional, Tuple, Callable
import sys
import os

# 모델 임포트
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.deep_learning import DeepLearningPipeline
from preprocessing.normalization import normalize_sensor_data
from preprocessing.filters import apply_low_pass_filter

class RealTimeInferencePipeline:
    """
    실시간 한국수어 인식 파이프라인
    - 센서 데이터 실시간 수집
    - 전처리 및 윈도우링
    - 딥러닝 모델 추론
    - 결과 후처리 및 출력
    """
    
    def __init__(self, model_path: str = 'best_dl_model.pth', 
                 window_size: int = 20, stride: int = 5, 
                 confidence_threshold: float = 0.7,
                 device: str = 'auto'):
        """
        Args:
            model_path: 학습된 모델 파일 경로
            window_size: 추론용 윈도우 크기
            stride: 윈도우 슬라이딩 간격
            confidence_threshold: 예측 신뢰도 임계값
            device: 추론 장치 ('auto', 'cpu', 'cuda')
        """
        self.window_size = window_size
        self.stride = stride
        self.confidence_threshold = confidence_threshold
        
        # 장치 설정
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"실시간 추론 시스템 초기화 - 장치: {self.device}")
        
        # 모델 로드
        self.model = self._load_model(model_path)
        
        # 클래스 매핑
        self.class_names = ['ㄱ', 'ㄴ', 'ㄷ', 'ㄹ', 'ㅁ']
        
        # 데이터 버퍼
        self.data_buffer = deque(maxlen=100)  # 최대 100개 샘플 보관
        self.prediction_history = deque(maxlen=10)  # 예측 이력
        
        # 실시간 처리 상태
        self.is_running = False
        self.data_queue = queue.Queue()
        self.result_queue = queue.Queue()
        
        # 성능 모니터링
        self.inference_times = deque(maxlen=50)
        self.frame_count = 0
        
        # 콜백 함수들
        self.prediction_callback = None
        self.data_callback = None
        
    def _load_model(self, model_path: str) -> torch.nn.Module:
        """학습된 모델 로드"""
        try:
            model = DeepLearningPipeline(
                input_features=8,
                sequence_length=self.window_size,
                num_classes=5,
                hidden_dim=128,
                num_layers=2,
                dropout=0.3
            )
            
            if os.path.exists(model_path):
                state_dict = torch.load(model_path, map_location=self.device)
                model.load_state_dict(state_dict)
                print(f"모델 로드 성공: {model_path}")
            else:
                print(f"경고: 모델 파일을 찾을 수 없습니다: {model_path}")
                print("기본 초기화된 모델을 사용합니다.")
            
            model.to(self.device)
            model.eval()
            return model
            
        except Exception as e:
            print(f"모델 로드 오류: {e}")
            raise
    
    def add_sensor_data(self, sensor_data: List[float]) -> None:
        """
        센서 데이터 추가
        Args:
            sensor_data: [flex1, flex2, flex3, flex4, flex5, pitch, roll, yaw]
        """
        if len(sensor_data) != 8:
            raise ValueError(f"센서 데이터는 8개 값이어야 합니다. 받은 값: {len(sensor_data)}")
        
        # 데이터 전처리
        processed_data = self._preprocess_sensor_data(sensor_data)
        
        # 버퍼에 추가
        self.data_buffer.append(processed_data)
        
        # 데이터 콜백 호출
        if self.data_callback:
            self.data_callback(processed_data)
        
        # 실시간 모드에서 큐에 추가
        if self.is_running:
            try:
                self.data_queue.put(processed_data, block=False)
            except queue.Full:
                pass  # 큐가 가득 찬 경우 무시
    
    def _preprocess_sensor_data(self, sensor_data: List[float]) -> np.ndarray:
        """센서 데이터 전처리"""
        data = np.array(sensor_data, dtype=np.float32)
        
        # 노이즈 필터링 (간단한 이동평균)
        if len(self.data_buffer) > 0:
            prev_data = np.array(self.data_buffer[-1])
            data = 0.7 * data + 0.3 * prev_data  # 간단한 저역 통과 필터
        
        # 정규화 (센서별 적절한 범위로)
        # Flex 센서: 0-1024 -> 0-1
        data[:5] = data[:5] / 1024.0
        
        # IMU: -180~180 -> -1~1
        data[5:] = data[5:] / 180.0
        
        return data
    
    def predict_single(self, force_predict: bool = False) -> Optional[Dict]:
        """
        단일 예측 수행
        Args:
            force_predict: 신뢰도 임계값을 무시하고 강제 예측
        Returns:
            예측 결과 딕셔너리 또는 None
        """
        if len(self.data_buffer) < self.window_size:
            return None
        
        # 최근 윈도우 데이터 추출
        window_data = np.array(list(self.data_buffer)[-self.window_size:])
        
        # 추론 수행
        start_time = time.time()
        result = self._inference(window_data)
        inference_time = time.time() - start_time
        
        self.inference_times.append(inference_time)
        self.frame_count += 1
        
        # 신뢰도 체크
        if not force_predict and result['confidence'] < self.confidence_threshold:
            return None
        
        # 예측 이력에 추가
        self.prediction_history.append(result)
        
        return result
    
    def _inference(self, window_data: np.ndarray) -> Dict:
        """모델 추론 수행"""
        with torch.no_grad():
            # 텐서 변환
            x = torch.FloatTensor(window_data).unsqueeze(0).to(self.device)  # (1, seq_len, features)
            
            # 모델 예측
            outputs = self.model(x)
            logits = outputs['class_logits']
            
            # 소프트맥스 적용
            probabilities = F.softmax(logits, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)
            
            predicted_class = self.class_names[predicted_idx.item()]
            confidence_value = confidence.item()
            
            # 결과 구성
            result = {
                'predicted_class': predicted_class,
                'predicted_idx': predicted_idx.item(),
                'confidence': confidence_value,
                'probabilities': probabilities.cpu().numpy()[0],
                'timestamp': time.time(),
                'all_class_probs': {
                    self.class_names[i]: probabilities[0][i].item() 
                    for i in range(len(self.class_names))
                }
            }
            
            # Attention weights가 있다면 추가
            if 'attention_weights' in outputs:
                result['attention_weights'] = outputs['attention_weights'].cpu().numpy()[0]
            
            return result
    
    def get_stable_prediction(self, history_length: int = 5) -> Optional[Dict]:
        """
        안정적인 예측 결과 반환 (최근 예측들의 일관성 체크)
        """
        if len(self.prediction_history) < history_length:
            return None
        
        recent_predictions = list(self.prediction_history)[-history_length:]
        
        # 최빈값 계산
        class_counts = {}
        total_confidence = 0
        
        for pred in recent_predictions:
            class_name = pred['predicted_class']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
            total_confidence += pred['confidence']
        
        # 가장 많이 예측된 클래스
        most_common_class = max(class_counts, key=class_counts.get)
        stability_ratio = class_counts[most_common_class] / history_length
        avg_confidence = total_confidence / history_length
        
        # 안정성 체크 (70% 이상 일치해야 함)
        if stability_ratio >= 0.7 and avg_confidence >= self.confidence_threshold:
            return {
                'predicted_class': most_common_class,
                'confidence': avg_confidence,
                'stability': stability_ratio,
                'timestamp': time.time()
            }
        
        return None
    
    def start_realtime_inference(self, prediction_callback: Callable = None):
        """실시간 추론 시작"""
        self.is_running = True
        self.prediction_callback = prediction_callback
        
        def inference_loop():
            while self.is_running:
                try:
                    # 데이터 대기 (타임아웃 있음)
                    sensor_data = self.data_queue.get(timeout=0.1)
                    
                    # 예측 수행
                    result = self.predict_single()
                    
                    if result and self.prediction_callback:
                        self.prediction_callback(result)
                    
                    # 결과 큐에 추가
                    if result:
                        try:
                            self.result_queue.put(result, block=False)
                        except queue.Full:
                            # 큐가 가득 찬 경우 가장 오래된 것 제거
                            try:
                                self.result_queue.get_nowait()
                                self.result_queue.put(result, block=False)
                            except queue.Empty:
                                pass
                
                except queue.Empty:
                    continue
                except Exception as e:
                    print(f"추론 루프 오류: {e}")
                    continue
        
        # 백그라운드 스레드 시작
        self.inference_thread = threading.Thread(target=inference_loop, daemon=True)
        self.inference_thread.start()
        
        print("실시간 추론 시작됨")
    
    def stop_realtime_inference(self):
        """실시간 추론 중지"""
        self.is_running = False
        if hasattr(self, 'inference_thread'):
            self.inference_thread.join(timeout=1.0)
        print("실시간 추론 중지됨")
    
    def get_performance_stats(self) -> Dict:
        """성능 통계 반환"""
        if len(self.inference_times) == 0:
            return {}
        
        inference_times = list(self.inference_times)
        
        return {
            'avg_inference_time': np.mean(inference_times),
            'max_inference_time': np.max(inference_times),
            'min_inference_time': np.min(inference_times),
            'fps': 1.0 / np.mean(inference_times) if np.mean(inference_times) > 0 else 0,
            'total_frames': self.frame_count,
            'buffer_size': len(self.data_buffer),
            'prediction_history_size': len(self.prediction_history)
        }
    
    def set_confidence_threshold(self, threshold: float):
        """신뢰도 임계값 설정"""
        self.confidence_threshold = max(0.0, min(1.0, threshold))
        print(f"신뢰도 임계값 설정: {self.confidence_threshold}")
    
    def clear_buffers(self):
        """버퍼 초기화"""
        self.data_buffer.clear()
        self.prediction_history.clear()
        
        # 큐 비우기
        while not self.data_queue.empty():
            try:
                self.data_queue.get_nowait()
            except queue.Empty:
                break
        
        while not self.result_queue.empty():
            try:
                self.result_queue.get_nowait()
            except queue.Empty:
                break
        
        print("버퍼 초기화 완료")
    
    def get_latest_results(self, max_count: int = 10) -> List[Dict]:
        """최근 결과들 반환"""
        results = []
        count = 0
        
        while count < max_count and not self.result_queue.empty():
            try:
                result = self.result_queue.get_nowait()
                results.append(result)
                count += 1
            except queue.Empty:
                break
        
        return results
