import torch
import torch.nn as nn
import numpy as np
import h5py
import os
import time
import threading
from collections import deque
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

print('🚀 실시간 추론 시스템 시작')

# 한글 폰트 설정
try:
    plt.rcParams['font.family'] = 'NanumGothic'
except:
    plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

class ImprovedGRU(nn.Module):
    """개선된 GRU 모델"""
    def __init__(self, input_size=8, hidden_size=64, num_layers=2, num_classes=24, dropout=0.3):
        super(ImprovedGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # GRU 레이어
        self.gru = nn.GRU(input_size, hidden_size, num_layers, 
                          batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        # 드롭아웃
        self.dropout = nn.Dropout(dropout)
        
        # 출력 레이어
        self.fc = nn.Linear(hidden_size, num_classes)
        
        # 가중치 초기화
        self._init_weights()
    
    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        
        out, _ = self.gru(x, h0)
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out

class AdvancedPreprocessor:
    """고급 전처리 클래스"""
    def __init__(self):
        self.flex_scalers = [StandardScaler() for _ in range(5)]
        self.orientation_scalers = [StandardScaler() for _ in range(3)]
        self.is_fitted = False
        
    def fit(self, data_list):
        """전처리 파라미터 학습"""
        print('🔧 전처리 파라미터 학습 중...')
        
        # Flex 센서별로 스케일러 학습
        for sensor_idx in range(5):
            sensor_data = []
            for data in data_list:
                sensor_values = data[:, sensor_idx]
                # 0값 제외하고 학습
                non_zero_values = sensor_values[sensor_values > 0]
                if len(non_zero_values) > 0:
                    sensor_data.extend(non_zero_values)
            
            if len(sensor_data) > 0:
                sensor_data = np.array(sensor_data).reshape(-1, 1)
                self.flex_scalers[sensor_idx].fit(sensor_data)
        
        # Orientation 센서별로 스케일러 학습
        for sensor_idx in range(3):
            sensor_data = []
            for data in data_list:
                sensor_values = data[:, sensor_idx + 5]
                sensor_data.extend(sensor_values)
            
            sensor_data = np.array(sensor_data).reshape(-1, 1)
            self.orientation_scalers[sensor_idx].fit(sensor_data)
        
        self.is_fitted = True
        print('✅ 전처리 파라미터 학습 완료')
    
    def transform_single(self, data):
        """단일 데이터 변환 (실시간용)"""
        if not self.is_fitted:
            raise ValueError("전처리 파라미터를 먼저 학습해야 합니다.")
        
        processed = data.copy()
        
        # Flex 센서 처리 (0값 처리 + 정규화)
        for sensor_idx in range(5):
            sensor_values = data[:, sensor_idx]
            
            # 0값을 해당 센서의 평균값으로 대체
            mean_val = np.mean(sensor_values[sensor_values > 0])
            if np.isnan(mean_val):
                mean_val = 500  # 기본값
            
            sensor_values[sensor_values == 0] = mean_val
            
            # 정규화
            sensor_values_normalized = self.flex_scalers[sensor_idx].transform(
                sensor_values.reshape(-1, 1)
            ).flatten()
            
            processed[:, sensor_idx] = sensor_values_normalized
        
        # Orientation 센서 처리 (정규화)
        for sensor_idx in range(3):
            sensor_values = data[:, sensor_idx + 5]
            sensor_values_normalized = self.orientation_scalers[sensor_idx].transform(
                sensor_values.reshape(-1, 1)
            ).flatten()
            processed[:, sensor_idx + 5] = sensor_values_normalized
        
        return processed

class RealtimeInferenceSystem:
    """실시간 추론 시스템"""
    def __init__(self, model_path, data_dir, window_size=300, update_interval=0.1):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.window_size = window_size
        self.update_interval = update_interval
        self.class_names = []
        
        # 모델 로드
        self.model = self.load_model(model_path)
        
        # 전처리기 로드
        self.preprocessor = self.load_preprocessor(data_dir)
        
        # 실시간 데이터 버퍼
        self.data_buffer = deque(maxlen=window_size)
        self.prediction_history = deque(maxlen=100)
        self.confidence_history = deque(maxlen=100)
        
        # 실시간 추론 상태
        self.is_running = False
        self.current_prediction = None
        self.current_confidence = 0.0
        
        print(f'✅ 실시간 추론 시스템 초기화 완료')
        print(f'  - 디바이스: {self.device}')
        print(f'  - 윈도우 크기: {window_size}')
        print(f'  - 업데이트 간격: {update_interval}초')
    
    def load_model(self, model_path):
        """모델 로드"""
        print('🤖 모델 로드 중...')
        model = ImprovedGRU(input_size=8, hidden_size=64, num_layers=2, num_classes=24, dropout=0.3)
        
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        print('✅ 모델 로드 완료')
        return model
    
    def load_preprocessor(self, data_dir):
        """전처리기 로드"""
        print('🔧 전처리기 로드 중...')
        
        # 클래스 이름 로드
        self.class_names = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
        
        # 샘플 데이터로 전처리기 학습
        sample_data = []
        for class_name in self.class_names[:5]:  # 처음 5개 클래스만 사용
            class_path = os.path.join(data_dir, class_name)
            for session in sorted(os.listdir(class_path))[:2]:  # 처음 2개 세션만 사용
                session_path = os.path.join(class_path, session)
                for file_name in sorted(os.listdir(session_path))[:3]:  # 처음 3개 파일만 사용
                    if file_name.endswith('.h5'):
                        file_path = os.path.join(session_path, file_name)
                        try:
                            with h5py.File(file_path, 'r') as f:
                                sensor_data = f['sensor_data'][:]
                                if sensor_data.shape[0] >= 20:
                                    sample_data.append(sensor_data)
                        except:
                            continue
        
        preprocessor = AdvancedPreprocessor()
        preprocessor.fit(sample_data)
        
        print('✅ 전처리기 로드 완료')
        return preprocessor
    
    def add_sensor_data(self, sensor_data):
        """센서 데이터 추가"""
        if len(sensor_data.shape) == 1:
            sensor_data = sensor_data.reshape(1, -1)
        
        self.data_buffer.append(sensor_data)
    
    def predict(self):
        """현재 데이터로 예측"""
        if len(self.data_buffer) < self.window_size:
            return None, 0.0
        
        # 버퍼의 데이터를 하나의 시퀀스로 결합
        sequence = np.vstack(list(self.data_buffer))
        
        # 전처리
        processed_sequence = self.preprocessor.transform_single(sequence)
        
        # 모델 예측
        with torch.no_grad():
            input_tensor = torch.FloatTensor(processed_sequence).unsqueeze(0).to(self.device)
            outputs = self.model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            predicted_class = predicted.item()
            confidence_value = confidence.item()
            
            if predicted_class < len(self.class_names):
                predicted_label = self.class_names[predicted_class]
            else:
                predicted_label = f"Class_{predicted_class}"
        
        return predicted_label, confidence_value
    
    def start_realtime_inference(self):
        """실시간 추론 시작"""
        print('🚀 실시간 추론 시작...')
        self.is_running = True
        
        def inference_loop():
            while self.is_running:
                if len(self.data_buffer) >= self.window_size:
                    prediction, confidence = self.predict()
                    if prediction is not None:
                        self.current_prediction = prediction
                        self.current_confidence = confidence
                        self.prediction_history.append(prediction)
                        self.confidence_history.append(confidence)
                        
                        # 실시간 출력
                        print(f'\r🔍 예측: {prediction} | 신뢰도: {confidence:.3f} | 버퍼: {len(self.data_buffer)}/{self.window_size}', end='')
                
                time.sleep(self.update_interval)
        
        # 별도 스레드에서 추론 실행
        self.inference_thread = threading.Thread(target=inference_loop)
        self.inference_thread.start()
    
    def stop_realtime_inference(self):
        """실시간 추론 중지"""
        print('\n🛑 실시간 추론 중지...')
        self.is_running = False
        if hasattr(self, 'inference_thread'):
            self.inference_thread.join()
    
    def simulate_realtime_data(self, test_data_path, duration=30):
        """테스트 데이터로 실시간 시뮬레이션"""
        print(f'🎬 실시간 데이터 시뮬레이션 시작 (지속시간: {duration}초)...')
        
        # 테스트 데이터 로드
        test_data = []
        with h5py.File(test_data_path, 'r') as f:
            test_data = f['sensor_data'][:]
        
        # 실시간 추론 시작
        self.start_realtime_inference()
        
        # 데이터 시뮬레이션
        start_time = time.time()
        frame_idx = 0
        
        try:
            while time.time() - start_time < duration:
                if frame_idx < len(test_data):
                    # 한 프레임씩 데이터 추가
                    self.add_sensor_data(test_data[frame_idx])
                    frame_idx += 1
                
                time.sleep(0.033)  # ~30 FPS
        
        except KeyboardInterrupt:
            print('\n⏹️  사용자에 의해 중단됨')
        
        finally:
            self.stop_realtime_inference()
            
            # 결과 요약
            print(f'\n📊 시뮬레이션 결과:')
            print(f'  - 총 프레임: {frame_idx}')
            print(f'  - 예측 횟수: {len(self.prediction_history)}')
            if self.prediction_history:
                print(f'  - 최종 예측: {self.current_prediction}')
                print(f'  - 평균 신뢰도: {np.mean(self.confidence_history):.3f}')
    
    def visualize_realtime_results(self):
        """실시간 결과 시각화"""
        if not self.prediction_history:
            print('❌ 시각화할 데이터가 없습니다.')
            return
        
        print('📈 실시간 결과 시각화 생성 중...')
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Realtime Inference Results', fontsize=16, fontweight='bold')
        
        # 1. 예측 분포
        ax1 = axes[0, 0]
        prediction_counts = {}
        for pred in self.prediction_history:
            prediction_counts[pred] = prediction_counts.get(pred, 0) + 1
        
        if prediction_counts:
            classes = list(prediction_counts.keys())
            counts = list(prediction_counts.values())
            bars = ax1.bar(classes, counts, alpha=0.7, color='skyblue')
            ax1.set_title('Prediction Distribution')
            ax1.set_xlabel('Predicted Class')
            ax1.set_ylabel('Count')
            ax1.tick_params(axis='x', rotation=45)
            
            for bar, count in zip(bars, counts):
                ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
                        str(count), ha='center', va='bottom')
        
        # 2. 신뢰도 변화
        ax2 = axes[0, 1]
        if self.confidence_history:
            ax2.plot(list(self.confidence_history), alpha=0.7, color='red')
            ax2.set_title('Confidence Over Time')
            ax2.set_xlabel('Prediction Index')
            ax2.set_ylabel('Confidence')
            ax2.set_ylim(0, 1)
            ax2.grid(True, alpha=0.3)
        
        # 3. 신뢰도 분포
        ax3 = axes[1, 0]
        if self.confidence_history:
            ax3.hist(list(self.confidence_history), bins=20, alpha=0.7, color='green', edgecolor='black')
            ax3.set_title('Confidence Distribution')
            ax3.set_xlabel('Confidence')
            ax3.set_ylabel('Frequency')
            ax3.grid(True, alpha=0.3)
        
        # 4. 성능 요약
        ax4 = axes[1, 1]
        if self.confidence_history:
            metrics = ['Avg Confidence', 'Max Confidence', 'Min Confidence', 'Std Confidence']
            values = [
                np.mean(self.confidence_history),
                np.max(self.confidence_history),
                np.min(self.confidence_history),
                np.std(self.confidence_history)
            ]
            colors = ['blue', 'green', 'red', 'orange']
            
            bars = ax4.bar(metrics, values, color=colors, alpha=0.7)
            ax4.set_title('Performance Summary')
            ax4.set_ylabel('Value')
            ax4.set_ylim(0, 1)
            
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{value:.3f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig('realtime_inference_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print('✅ 시각화 완료: realtime_inference_results.png')

def main():
    """메인 함수"""
    # 설정
    model_path = 'improved_preprocessing_model.pth'
    data_dir = '../SignGlove/external/SignGlove_HW/datasets/unified'
    
    # 실시간 추론 시스템 초기화
    inference_system = RealtimeInferenceSystem(
        model_path=model_path,
        data_dir=data_dir,
        window_size=300,
        update_interval=0.1
    )
    
    # 테스트 데이터로 시뮬레이션
    test_data_path = '../SignGlove/external/SignGlove_HW/datasets/unified/ㄱ/1/episode_20250819_190506_ㄱ_1.h5'
    
    if os.path.exists(test_data_path):
        print(f'🎬 테스트 데이터로 시뮬레이션 시작: {test_data_path}')
        inference_system.simulate_realtime_data(test_data_path, duration=10)
        
        # 결과 시각화
        inference_system.visualize_realtime_results()
    else:
        print(f'❌ 테스트 데이터를 찾을 수 없습니다: {test_data_path}')
        print('📁 사용 가능한 데이터 파일을 찾아보겠습니다...')
        
        # 사용 가능한 데이터 파일 찾기
        available_files = []
        for class_name in sorted(os.listdir(data_dir)):
            class_path = os.path.join(data_dir, class_name)
            if os.path.isdir(class_path):
                for session in sorted(os.listdir(class_path)):
                    session_path = os.path.join(class_path, session)
                    if os.path.isdir(session_path):
                        for file_name in sorted(os.listdir(session_path)):
                            if file_name.endswith('.h5'):
                                file_path = os.path.join(session_path, file_name)
                                available_files.append(file_path)
                                break
                        break
                break
        
        if available_files:
            test_data_path = available_files[0]
            print(f'✅ 사용 가능한 테스트 데이터: {test_data_path}')
            inference_system.simulate_realtime_data(test_data_path, duration=10)
            inference_system.visualize_realtime_results()
        else:
            print('❌ 사용 가능한 데이터 파일이 없습니다.')

if __name__ == "__main__":
    main()
