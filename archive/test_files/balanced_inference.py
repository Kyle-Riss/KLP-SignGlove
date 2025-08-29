#!/usr/bin/env python3
"""
균형잡힌 SignGlove 추론 시스템
모든 클래스가 골고루 나올 수 있도록 개선된 추론
"""

import torch
import torch.nn as nn
import numpy as np
import time
import json
import sys
from pathlib import Path
from collections import deque, Counter
from sklearn.preprocessing import StandardScaler
import random

# 과적합 방지 모델 클래스 import
sys.path.append('.')
from anti_overfitting_gru import AntiOverfittingGRUModel

class BalancedInference:
    """균형잡힌 추론 시스템"""
    
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
        self.confidence_threshold = 0.2  # 더 낮은 임계값
        self.prediction_history = deque(maxlen=100)
        
        # 균형 조정 설정
        self.class_weights = np.ones(24)  # 클래스별 가중치
        self.recent_predictions = deque(maxlen=50)  # 최근 예측 기록
        self.diversity_boost = 0.1  # 다양성 부스트
        
        # 실시간 처리 설정
        self.is_running = False
        
        print(f"⚖️ 균형잡힌 추론 시스템 초기화 완료 (디바이스: {device})")
        print(f"📊 모델 파라미터 수: {sum(p.numel() for p in self.model.parameters()):,}개")
    
    def load_model(self, model_path):
        """모델 로드"""
        model = AntiOverfittingGRUModel(
            input_size=8,
            hidden_size=32,
            num_layers=1,
            num_classes=24,
            dropout=0.0
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
            self.scaler.fit(sensor_data.reshape(1, -1))
        
        normalized_data = self.scaler.transform(sensor_data.reshape(1, -1)).flatten()
        self.data_buffer.append(normalized_data)
        
        # 시퀀스가 충분히 쌓이면 추론 준비
        if len(self.data_buffer) >= self.sequence_length:
            sequence = np.array(list(self.data_buffer))
            return torch.FloatTensor(sequence).unsqueeze(0)
        
        return None
    
    def update_class_weights(self):
        """클래스 가중치 업데이트"""
        if len(self.recent_predictions) < 10:
            return
        
        # 최근 예측 빈도 계산
        prediction_counts = Counter(self.recent_predictions)
        
        # 각 클래스의 빈도 계산
        for i, class_name in enumerate(self.class_names):
            count = prediction_counts.get(class_name, 0)
            frequency = count / len(self.recent_predictions)
            
            # 빈도가 높으면 가중치 감소, 낮으면 가중치 증가
            if frequency > 0.1:  # 10% 이상 나오면
                self.class_weights[i] = max(0.5, 1.0 - frequency * 2)
            else:
                self.class_weights[i] = min(2.0, 1.0 + (0.05 - frequency) * 10)
    
    def predict_with_diversity(self, sequence):
        """다양성을 고려한 예측"""
        if self.model is None:
            return None, 0.0, []
        
        with torch.no_grad():
            sequence = sequence.to(self.device)
            output = self.model(sequence)
            probabilities = torch.softmax(output, dim=1)
            
            # 모든 클래스의 확률
            all_probs = probabilities.cpu().numpy().flatten()
            
            # 클래스 가중치 적용
            weighted_probs = all_probs * self.class_weights
            
            # 다양성 부스트 적용 (최근에 많이 나온 클래스는 확률 감소)
            if len(self.recent_predictions) > 0:
                recent_counts = Counter(self.recent_predictions)
                for i, class_name in enumerate(self.class_names):
                    recent_freq = recent_counts.get(class_name, 0) / len(self.recent_predictions)
                    weighted_probs[i] *= (1.0 - recent_freq * self.diversity_boost)
            
            # 정규화
            weighted_probs = np.maximum(weighted_probs, 0)
            if np.sum(weighted_probs) > 0:
                weighted_probs = weighted_probs / np.sum(weighted_probs)
            
            # 최고 확률과 클래스
            predicted_class = np.argmax(weighted_probs)
            confidence = weighted_probs[predicted_class]
            
            return predicted_class, confidence, weighted_probs
    
    def get_prediction_with_confidence(self, sequence):
        """신뢰도를 고려한 예측"""
        predicted_class, confidence, all_probs = self.predict_with_diversity(sequence)
        
        if confidence >= self.confidence_threshold:
            class_name = self.class_names[predicted_class]
            
            # 예측 히스토리에 추가
            self.prediction_history.append({
                'class': class_name,
                'confidence': confidence,
                'timestamp': time.time(),
                'all_probabilities': all_probs.tolist()
            })
            
            # 최근 예측에 추가
            self.recent_predictions.append(class_name)
            
            # 클래스 가중치 업데이트
            self.update_class_weights()
            
            return class_name, confidence, all_probs
        else:
            return None, confidence, all_probs
    
    def simulate_diverse_sensor_data(self):
        """다양한 센서 데이터 시뮬레이션"""
        # 각 클래스별로 다른 패턴의 센서 데이터 생성
        class_patterns = {
            'ㄱ': [1.0, 0.5, 0.3, 0.8, 0.2, 0.7, 0.4, 0.6],
            'ㄴ': [0.3, 1.0, 0.7, 0.2, 0.9, 0.4, 0.8, 0.1],
            'ㄷ': [0.8, 0.2, 1.0, 0.6, 0.3, 0.9, 0.1, 0.7],
            'ㄹ': [0.4, 0.9, 0.1, 1.0, 0.7, 0.2, 0.8, 0.5],
            'ㅁ': [0.7, 0.1, 0.8, 0.3, 1.0, 0.6, 0.2, 0.9],
            'ㅂ': [0.2, 0.8, 0.4, 0.9, 0.1, 1.0, 0.7, 0.3],
            'ㅅ': [0.9, 0.3, 0.6, 0.1, 0.8, 0.2, 1.0, 0.4],
            'ㅇ': [0.5, 0.7, 0.2, 0.4, 0.6, 0.1, 0.9, 1.0],
            'ㅈ': [0.6, 0.4, 0.9, 0.7, 0.2, 0.8, 0.3, 0.1],
            'ㅊ': [0.1, 0.6, 0.5, 0.2, 0.9, 0.3, 0.7, 0.8],
            'ㅋ': [0.8, 0.1, 0.7, 0.5, 0.4, 0.9, 0.2, 0.6],
            'ㅌ': [0.3, 0.9, 0.2, 0.8, 0.5, 0.1, 0.6, 0.7],
            'ㅍ': [0.7, 0.2, 0.8, 0.1, 0.6, 0.4, 0.9, 0.3],
            'ㅎ': [0.4, 0.7, 0.1, 0.6, 0.8, 0.2, 0.5, 0.9],
            'ㅏ': [0.9, 0.4, 0.6, 0.3, 0.1, 0.8, 0.7, 0.2],
            'ㅑ': [0.2, 0.8, 0.3, 0.7, 0.9, 0.1, 0.4, 0.6],
            'ㅓ': [0.6, 0.1, 0.9, 0.4, 0.2, 0.7, 0.8, 0.3],
            'ㅕ': [0.8, 0.3, 0.2, 0.9, 0.7, 0.4, 0.1, 0.5],
            'ㅗ': [0.1, 0.6, 0.8, 0.2, 0.4, 0.9, 0.3, 0.7],
            'ㅛ': [0.5, 0.9, 0.1, 0.6, 0.3, 0.2, 0.8, 0.4],
            'ㅜ': [0.7, 0.2, 0.4, 0.8, 0.1, 0.6, 0.9, 0.3],
            'ㅠ': [0.3, 0.5, 0.9, 0.1, 0.8, 0.7, 0.2, 0.6],
            'ㅡ': [0.6, 0.8, 0.2, 0.5, 0.9, 0.3, 0.1, 0.7],
            'ㅣ': [0.9, 0.1, 0.7, 0.3, 0.5, 0.8, 0.4, 0.2]
        }
        
        # 랜덤하게 클래스 선택 (균등 분포)
        selected_class = random.choice(self.class_names)
        base_pattern = class_patterns[selected_class]
        
        # 노이즈 추가
        noise = np.random.normal(0, 0.1, 8)
        sensor_data = np.array(base_pattern) + noise
        
        return sensor_data
    
    def realtime_inference_loop(self):
        """실시간 추론 루프"""
        print("🔄 균형잡힌 실시간 추론 시작...")
        print("📝 사용법:")
        print("  - 'q': 종료")
        print("  - 'r': 신뢰도 임계값 조정")
        print("  - 'h': 예측 히스토리 출력")
        print("  - 'c': 현재 설정 출력")
        print("  - 's': 클래스별 통계 출력")
        print("  - 'd': 다양성 부스트 조정")
        print("=" * 50)
        
        self.is_running = True
        last_prediction_time = 0
        prediction_cooldown = 1.0
        
        try:
            while self.is_running:
                # 다양한 센서 데이터 시뮬레이션
                sensor_data = self.simulate_diverse_sensor_data()
                
                # 데이터 전처리
                sequence = self.preprocess_data(sensor_data)
                
                if sequence is not None:
                    current_time = time.time()
                    
                    # 쿨다운 체크
                    if current_time - last_prediction_time >= prediction_cooldown:
                        # 예측 수행
                        predicted_class, confidence, all_probs = self.get_prediction_with_confidence(sequence)
                        
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
                    elif user_input == 's':
                        self.show_class_statistics()
                    elif user_input == 'd':
                        try:
                            new_boost = float(input("새로운 다양성 부스트 (0.0-1.0): "))
                            if 0.0 <= new_boost <= 1.0:
                                self.diversity_boost = new_boost
                                print(f"✅ 다양성 부스트가 {new_boost:.3f}로 변경되었습니다.")
                            else:
                                print("❌ 0.0과 1.0 사이의 값을 입력하세요.")
                        except ValueError:
                            print("❌ 유효한 숫자를 입력하세요.")
                
                time.sleep(0.1)
                
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
        print(f"다양성 부스트: {self.diversity_boost:.3f}")
        print(f"시퀀스 길이: {self.sequence_length}")
        print(f"디바이스: {self.device}")
        print(f"버퍼 크기: {len(self.data_buffer)}/{self.sequence_length}")
        print(f"최근 예측 수: {len(self.recent_predictions)}")
    
    def show_class_statistics(self):
        """클래스별 통계 출력"""
        print("\n📈 클래스별 통계:")
        print("-" * 40)
        
        if not self.recent_predictions:
            print("아직 예측 기록이 없습니다.")
            return
        
        # 최근 예측 빈도 계산
        prediction_counts = Counter(self.recent_predictions)
        total_predictions = len(self.recent_predictions)
        
        print(f"총 예측 수: {total_predictions}")
        print("\n클래스별 빈도:")
        
        for class_name in self.class_names:
            count = prediction_counts.get(class_name, 0)
            frequency = count / total_predictions
            weight = self.class_weights[self.class_names.index(class_name)]
            print(f"{class_name:2s}: {count:2d}회 ({frequency*100:5.1f}%) [가중치: {weight:.2f}]")
        
        # 다양성 점수 계산
        unique_classes = len(prediction_counts)
        diversity_score = unique_classes / 24  # 24개 클래스 중 몇 개가 나왔는지
        print(f"\n다양성 점수: {diversity_score:.3f} ({unique_classes}/24 클래스)")
    
    def start_inference(self):
        """추론 시작"""
        if self.model is None:
            print("❌ 모델이 로드되지 않았습니다.")
            return
        
        # 실시간 추론 시작
        self.realtime_inference_loop()

def main():
    """메인 함수"""
    print("⚖️ 균형잡힌 SignGlove 추론 시스템")
    print("=" * 50)
    
    # 균형잡힌 추론 시스템 초기화
    inference_system = BalancedInference(
        model_path='best_anti_overfitting_model.pth',
        sequence_length=20
    )
    
    # 추론 시작
    inference_system.start_inference()

if __name__ == "__main__":
    import select
    main()



