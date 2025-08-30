import torch
import torch.nn as nn
import numpy as np
import h5py
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, RobustScaler
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

print('🔧 개선된 전처리 모델 시작')

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
    
    def transform(self, data_list):
        """데이터 변환"""
        if not self.is_fitted:
            raise ValueError("전처리 파라미터를 먼저 학습해야 합니다.")
        
        print('🔄 데이터 변환 중...')
        processed_data = []
        
        for data in data_list:
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
            
            processed_data.append(processed)
        
        print(f'✅ {len(processed_data)}개 샘플 변환 완료')
        return processed_data

def load_and_preprocess_data(data_dir, max_samples_per_class=25):
    """데이터 로드 및 전처리"""
    print(f'📂 데이터 로드 중: {data_dir}')
    
    all_data = []
    all_labels = []
    class_names = []
    
    for class_idx, class_name in enumerate(sorted(os.listdir(data_dir))):
        class_path = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_path):
            continue
            
        class_names.append(class_name)
        print(f'  - 클래스 {class_name} 로딩 중...')
        
        sample_count = 0
        for session in sorted(os.listdir(class_path)):
            if sample_count >= max_samples_per_class:
                break
            session_path = os.path.join(class_path, session)
            if not os.path.isdir(session_path):
                continue
                
            for file_name in sorted(os.listdir(session_path)):
                if sample_count >= max_samples_per_class:
                    break
                if file_name.endswith('.h5'):
                    file_path = os.path.join(session_path, file_name)
                    try:
                        with h5py.File(file_path, 'r') as f:
                            sensor_data = f['sensor_data'][:]
                            if sensor_data.shape[0] >= 20:
                                all_data.append(sensor_data)
                                all_labels.append(class_idx)
                                sample_count += 1
                    except Exception as e:
                        print(f"    경고: {file_path} 로드 실패 - {e}")
    
    print(f'✅ 총 {len(all_data)}개 샘플, {len(class_names)}개 클래스 로드 완료')
    return all_data, all_labels, class_names

def train_improved_model(model, train_data, train_labels, val_data, val_labels, device, epochs=100):
    """개선된 모델 훈련"""
    print('🎯 모델 훈련 시작...')
    
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    train_losses = []
    val_accuracies = []
    best_val_acc = 0
    patience_counter = 0
    max_patience = 20
    
    for epoch in range(epochs):
        # 훈련
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for data, label in zip(train_data, train_labels):
            data_tensor = torch.FloatTensor(data).unsqueeze(0).to(device)
            label_tensor = torch.LongTensor([label]).to(device)
            
            optimizer.zero_grad()
            outputs = model(data_tensor)
            loss = criterion(outputs, label_tensor)
            loss.backward()
            
            # 그래디언트 클리핑
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += 1
            if predicted.item() == label:
                correct += 1
        
        train_acc = correct / total
        avg_loss = total_loss / len(train_data)
        
        # 검증
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for data, label in zip(val_data, val_labels):
                data_tensor = torch.FloatTensor(data).unsqueeze(0).to(device)
                outputs = model(data_tensor)
                _, predicted = torch.max(outputs, 1)
                val_total += 1
                if predicted.item() == label:
                    val_correct += 1
        
        val_acc = val_correct / val_total
        train_losses.append(avg_loss)
        val_accuracies.append(val_acc)
        
        # 스케줄러 업데이트
        scheduler.step(avg_loss)
        
        # 조기 종료
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= max_patience:
            print(f'  조기 종료: {epoch+1} 에포크')
            break
        
        if (epoch + 1) % 10 == 0:
            print(f'  Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f}, Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}')
    
    return train_losses, val_accuracies, best_val_acc

def evaluate_model(model, test_data, test_labels, class_names, device):
    """모델 평가"""
    print('🧪 모델 평가 중...')
    
    model.eval()
    all_predictions = []
    all_true_labels = []
    all_confidences = []
    
    with torch.no_grad():
        for data, label in zip(test_data, test_labels):
            data_tensor = torch.FloatTensor(data).unsqueeze(0).to(device)
            outputs = model(data_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            all_predictions.append(predicted.item())
            all_true_labels.append(label)
            all_confidences.append(confidence.item())
    
    accuracy = accuracy_score(all_true_labels, all_predictions)
    report = classification_report(all_true_labels, all_predictions, 
                                 target_names=class_names, output_dict=True)
    
    print(f'✅ 테스트 정확도: {accuracy:.4f} ({accuracy*100:.2f}%)')
    
    # 예측 분포 확인
    pred_counter = Counter(all_predictions)
    print(f'\n📊 예측 분포 (상위 5개):')
    for pred_class, count in pred_counter.most_common(5):
        class_name = class_names[pred_class]
        print(f'  - {class_name}: {count}개 ({count/len(all_predictions)*100:.1f}%)')
    
    return accuracy, report, all_confidences

def visualize_results(train_losses, val_accuracies, test_accuracy, confidences, class_names):
    """결과 시각화"""
    print('📈 결과 시각화 생성 중...')
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Improved Preprocessing Model Results', fontsize=16, fontweight='bold')
    
    # 1. 훈련 손실
    ax1 = axes[0, 0]
    ax1.plot(train_losses, label='Train Loss', color='blue', alpha=0.7)
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True, alpha=0.3)
    
    # 2. 검증 정확도
    ax2 = axes[0, 1]
    ax2.plot(val_accuracies, label='Validation Accuracy', color='red', alpha=0.7)
    ax2.set_title('Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)
    
    # 3. 예측 신뢰도 분포
    ax3 = axes[1, 0]
    ax3.hist(confidences, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    ax3.set_title('Prediction Confidence Distribution')
    ax3.set_xlabel('Confidence')
    ax3.set_ylabel('Frequency')
    ax3.grid(True, alpha=0.3)
    
    # 4. 성능 요약
    ax4 = axes[1, 1]
    metrics = ['Test Accuracy', 'Max Val Accuracy', 'Avg Confidence']
    values = [test_accuracy, max(val_accuracies), np.mean(confidences)]
    colors = ['green', 'blue', 'orange']
    
    bars = ax4.bar(metrics, values, color=colors, alpha=0.7)
    ax4.set_title('Performance Summary')
    ax4.set_ylabel('Score')
    ax4.set_ylim(0, 1)
    
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('improved_preprocessing_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print('✅ 시각화 완료: improved_preprocessing_results.png')

def main():
    """메인 함수"""
    # 데이터 경로
    data_dir = '../SignGlove/external/SignGlove_HW/datasets/unified'
    
    # GPU 사용 가능 여부 확인
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'🖥️  사용 디바이스: {device}')
    
    # 1. 데이터 로드
    all_data, all_labels, class_names = load_and_preprocess_data(data_dir, max_samples_per_class=25)
    
    # 2. 데이터 분할 (라벨 범위 확인 및 수정)
    print(f'라벨 범위: {min(all_labels)} ~ {max(all_labels)}')
    print(f'클래스 수: {len(class_names)}')
    
    # 라벨이 0부터 시작하는지 확인
    if min(all_labels) != 0 or max(all_labels) >= len(class_names):
        print('⚠️  라벨 범위 문제 발견! 라벨 재매핑 중...')
        # 라벨 재매핑
        unique_labels = sorted(set(all_labels))
        label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
        all_labels = [label_mapping[label] for label in all_labels]
        print(f'재매핑 후 라벨 범위: {min(all_labels)} ~ {max(all_labels)}')
    
    train_data, temp_data, train_labels, temp_labels = train_test_split(
        all_data, all_labels, test_size=0.4, random_state=42, stratify=all_labels
    )
    val_data, test_data, val_labels, test_labels = train_test_split(
        temp_data, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
    )
    
    print(f'📊 데이터 분할: 훈련={len(train_data)}, 검증={len(val_data)}, 테스트={len(test_data)}')
    
    # 3. 고급 전처리 적용
    preprocessor = AdvancedPreprocessor()
    preprocessor.fit(train_data)
    
    train_data_processed = preprocessor.transform(train_data)
    val_data_processed = preprocessor.transform(val_data)
    test_data_processed = preprocessor.transform(test_data)
    
    # 4. 모델 생성
    model = ImprovedGRU(input_size=8, hidden_size=64, num_layers=2, num_classes=24, dropout=0.3)
    print(f'🤖 모델 생성 완료: {sum(p.numel() for p in model.parameters())} 파라미터')
    
    # 5. 모델 훈련
    train_losses, val_accuracies, best_val_acc = train_improved_model(
        model, train_data_processed, train_labels, val_data_processed, val_labels, device, epochs=100
    )
    
    # 6. 모델 평가
    test_accuracy, report, confidences = evaluate_model(
        model, test_data_processed, test_labels, class_names, device
    )
    
    # 7. 결과 출력
    print(f'\n🎉 최종 결과:')
    print(f'  - 테스트 정확도: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)')
    print(f'  - 최고 검증 정확도: {best_val_acc:.4f}')
    print(f'  - 평균 신뢰도: {np.mean(confidences):.4f}')
    
    # 8. 시각화
    visualize_results(train_losses, val_accuracies, test_accuracy, confidences, class_names)
    
    # 9. 모델 저장
    torch.save({
        'model_state_dict': model.state_dict(),
        'test_accuracy': test_accuracy,
        'best_val_acc': best_val_acc,
        'preprocessor': preprocessor
    }, 'improved_preprocessing_model.pth')
    print(f'💾 모델 저장 완료: improved_preprocessing_model.pth')

if __name__ == "__main__":
    main()
