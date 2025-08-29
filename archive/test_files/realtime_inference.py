#!/usr/bin/env python3
"""
실시간 SignGlove 추론 시스템
과적합 방지 모델을 사용한 실시간 한국어 자음/모음 인식
"""

import torch
import torch.nn as nn
import numpy as np
import time
import json
import sys
from pathlib import Path
from collections import deque
import threading
import queue
from sklearn.preprocessing import StandardScaler

# 과적합 방지 모델 클래스 import
sys.path.append('.')
from anti_overfitting_gru import AntiOverfittingGRUModel

class RealtimeInference:
    """실시간 추론 시스템"""
    
    def __init__(self, model_path='best_anti_overfitting_model.pth', sequence_length=20, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.sequence_length = sequence_length
        self.class_names = ['ㄱ', 'ㄴ', 'ㄷ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅅ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ',
                           'ㅏ', 'ㅑ', 'ㅓ', 'ㅕ', 'ㅗ', 'ㅛ', 'ㅜ', 'ㅠ', 'ㅡ', 'ㅣ']
        
        # 데이터 버퍼
        self.data_buffer = deque(maxlen=sequence_length)
        self.scaler = StandardScaler()
        
        # 모델 로드
        self.model = self.load_model(model_path)
        
        # 추론 설정
        self.confidence_threshold = 0.3  # 신뢰도 임계값
        self.prediction_history = deque(maxlen=10)  # 예측 히스토리
        
        # 실시간 처리 설정
        self.is_running = False
        self.data_queue = queue.Queue()
        self.prediction_queue = queue.Queue()
        
        print(f"🚀 실시간 추론 시스템 초기화 완료 (디바이스: {device})")
        print(f"📊 모델 파라미터 수: {sum(p.numel() for p in self.model.parameters()):,}개")
    
    def load_model(self, model_path):
        """모델 로드"""
        model = AntiOverfittingGRUModel(
            input_size=8,
            hidden_size=32,
            num_layers=1,
            num_classes=24,
            dropout=0.0  # 추론 시에는 드롭아웃 비활성화
        )
        
        if Path(model_path).exists():
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"📁 모델 로드: {model_path}")
        else:
            print(f"⚠️ 모델 파일을 찾을 수 없습니다: {model_path}")
            return None
        
        model.to(self.device)
        model.eval()
        return model
    
    def preprocess_data(self, sensor_data):
        """센서 데이터 전처리"""
        # 데이터 정규화
        if len(self.data_buffer) == 0:
            # 첫 번째 데이터는 스케일러를 초기화
            self.scaler.fit(sensor_data.reshape(1, -1))
        
        normalized_data = self.scaler.transform(sensor_data.reshape(1, -1)).flatten()
        
        # 버퍼에 추가
        self.data_buffer.append(normalized_data)
        
        # 시퀀스가 충분히 쌓이면 추론 준비
        if len(self.data_buffer) >= self.sequence_length:
            sequence = np.array(list(self.data_buffer))
            return torch.FloatTensor(sequence).unsqueeze(0)  # (1, sequence_length, 8)
        
        return None
    
    def predict(self, sequence):
        """단일 시퀀스 예측"""
        if self.model is None:
            return None, 0.0
        
        with torch.no_grad():
            sequence = sequence.to(self.device)
            output = self.model(sequence)
            probabilities = torch.softmax(output, dim=1)
            
            # 최고 확률과 클래스
            max_prob, predicted_class = torch.max(probabilities, 1)
            
            return predicted_class.item(), max_prob.item()
    
    def get_prediction_with_confidence(self, sequence):
        """신뢰도를 고려한 예측"""
        predicted_class, confidence = self.predict(sequence)
        
        if confidence >= self.confidence_threshold:
            class_name = self.class_names[predicted_class]
            
            # 예측 히스토리에 추가
            self.prediction_history.append({
                'class': class_name,
                'confidence': confidence,
                'timestamp': time.time()
            })
            
            return class_name, confidence
        else:
            return None, confidence
    
    def simulate_sensor_data(self):
        """센서 데이터 시뮬레이션 (테스트용)"""
        # 실제 센서 데이터를 시뮬레이션
        # 8개 센서 값 (가속도계, 자이로스코프 등)
        sensor_data = np.random.normal(0, 1, 8)
        return sensor_data
    
    def realtime_inference_loop(self):
        """실시간 추론 루프"""
        print("🔄 실시간 추론 시작...")
        print("📝 사용법:")
        print("  - 'q': 종료")
        print("  - 'r': 신뢰도 임계값 조정")
        print("  - 'h': 예측 히스토리 출력")
        print("  - 'c': 현재 설정 출력")
        print("=" * 50)
        
        self.is_running = True
        last_prediction_time = 0
        prediction_cooldown = 1.0  # 1초 쿨다운
        
        try:
            while self.is_running:
                # 센서 데이터 시뮬레이션 (실제로는 센서에서 받아옴)
                sensor_data = self.simulate_sensor_data()
                
                # 데이터 전처리
                sequence = self.preprocess_data(sensor_data)
                
                if sequence is not None:
                    current_time = time.time()
                    
                    # 쿨다운 체크
                    if current_time - last_prediction_time >= prediction_cooldown:
                        # 예측 수행
                        predicted_class, confidence = self.get_prediction_with_confidence(sequence)
                        
                        if predicted_class:
                            print(f"🎯 예측: {predicted_class} (신뢰도: {confidence:.3f})")
                            last_prediction_time = current_time
                        else:
                            print(f"❓ 신뢰도 부족: {confidence:.3f} < {self.confidence_threshold}")
                            last_prediction_time = current_time
                
                # 사용자 입력 체크 (비동기)
                if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                    user_input = input().strip().lower()
                    
                    if user_input == 'q':
                        print("🛑 추론 종료")
                        self.is_running = False
                        break
                    elif user_input == 'r':
                        try:
                            new_threshold = float(input("새로운 신뢰도 임계값 (0.0-1.0): "))
                            if 0.0 <= new_threshold <= 1.0:
                                self.confidence_threshold = new_threshold
                                print(f"✅ 신뢰도 임계값이 {new_threshold:.3f}로 변경되었습니다.")
                            else:
                                print("❌ 0.0과 1.0 사이의 값을 입력하세요.")
                        except ValueError:
                            print("❌ 유효한 숫자를 입력하세요.")
                    elif user_input == 'h':
                        self.show_prediction_history()
                    elif user_input == 'c':
                        self.show_current_settings()
                
                time.sleep(0.1)  # 100ms 간격
                
        except KeyboardInterrupt:
            print("\n🛑 사용자에 의해 중단되었습니다.")
        finally:
            self.is_running = False
    
    def show_prediction_history(self):
        """예측 히스토리 출력"""
        print("\n📊 예측 히스토리:")
        print("-" * 40)
        
        if not self.prediction_history:
            print("아직 예측 기록이 없습니다.")
            return
        
        for i, pred in enumerate(reversed(list(self.prediction_history)), 1):
            timestamp = time.strftime('%H:%M:%S', time.localtime(pred['timestamp']))
            print(f"{i:2d}. {pred['class']} (신뢰도: {pred['confidence']:.3f}) - {timestamp}")
    
    def show_current_settings(self):
        """현재 설정 출력"""
        print("\n⚙️ 현재 설정:")
        print("-" * 40)
        print(f"신뢰도 임계값: {self.confidence_threshold:.3f}")
        print(f"시퀀스 길이: {self.sequence_length}")
        print(f"디바이스: {self.device}")
        print(f"버퍼 크기: {len(self.data_buffer)}/{self.sequence_length}")
    
    def start_inference(self):
        """추론 시작"""
        if self.model is None:
            print("❌ 모델이 로드되지 않았습니다.")
            return
        
        # 실시간 추론 시작
        self.realtime_inference_loop()

def main():
    """메인 함수"""
    print("🚀 SignGlove 실시간 추론 시스템")
    print("=" * 50)
    
    # 실시간 추론 시스템 초기화
    inference_system = RealtimeInference(
        model_path='best_anti_overfitting_model.pth',
        sequence_length=20
    )
    
    # 추론 시작
    inference_system.start_inference()

if __name__ == "__main__":
    import select
    main()



