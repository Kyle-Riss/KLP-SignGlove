import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import h5py
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

print('🚀 최적화된 GRU 모델 구축 및 훈련 시작')

# 센서 이름
sensor_names = ['Flex1', 'Flex2', 'Flex3', 'Flex4', 'Flex5', 'Pitch', 'Roll', 'Yaw']

class DataCleaner:
    """1순위 데이터 정제 클래스"""
    
    def __init__(self):
        self.cleaned_data = {}
        self.removed_samples = {}
        self.normalized_samples = {}
        
    def clean_data(self, data_dir):
        """1순위 정제: 범위 오류 제거, 극단적 변동성 정규화"""
        print('🔧 데이터 정제 중...')
        
        for class_name in sorted(os.listdir(data_dir)):
            class_dir = os.path.join(data_dir, class_name)
            if os.path.isdir(class_dir):
                class_data = []
                class_paths = []
                self.removed_samples[class_name] = 0
                self.normalized_samples[class_name] = 0
                
                for session in ['1', '2', '3', '4', '5']:
                    session_dir = os.path.join(class_dir, session)
                    if os.path.exists(session_dir):
                        for file_name in os.listdir(session_dir):
                            if file_name.endswith('.h5'):
                                file_path = os.path.join(session_dir, file_name)
                                try:
                                    with h5py.File(file_path, 'r') as f:
                                        sensor_data = f['sensor_data'][:]  # (300, 8)
                                        
                                        # 데이터 정제
                                        cleaned_data = self._clean_single_sample(sensor_data, class_name)
                                        if cleaned_data is not None:
                                            # 마지막 20 프레임 사용
                                            data = cleaned_data[-20:]
                                            class_data.append(data)
                                            class_paths.append(file_path)
                                            
                                except Exception as e:
                                    print(f"Error loading {file_path}: {e}")
                
                if class_data:
                    self.cleaned_data[class_name] = {
                        'data': np.array(class_data),
                        'paths': class_paths
                    }
        
        print(f'✅ 정제 완료: {sum(self.removed_samples.values())}개 제거, {sum(self.normalized_samples.values())}개 정규화')
        return self.cleaned_data
    
    def _clean_single_sample(self, sensor_data, class_name):
        """단일 샘플 정제"""
        # 1. 범위 오류 검사 및 제거
        for i, sensor_name in enumerate(sensor_names):
            sensor_values = sensor_data[:, i]
            
            # Flex 센서 범위 검사 (0-1000)
            if i < 5:  # Flex 센서
                if np.any(sensor_values < 0) or np.any(sensor_values > 1000):
                    self.removed_samples[class_name] += 1
                    return None  # 샘플 제거
            
            # Orientation 센서 범위 검사
            else:  # Pitch, Roll, Yaw
                if np.any(sensor_values < -180) or np.any(sensor_values > 180):
                    self.removed_samples[class_name] += 1
                    return None  # 샘플 제거
        
        # 2. 극단적 변동성 정규화
        cleaned_data = sensor_data.copy()
        for i, sensor_name in enumerate(sensor_names):
            sensor_values = sensor_data[:, i]
            std_val = np.std(sensor_values)
            
            # Flex 센서 극단적 변동성 정규화 (std > 200)
            if i < 5 and std_val > 200:
                # Z-score 정규화 후 스케일링
                mean_val = np.mean(sensor_values)
                normalized = (sensor_values - mean_val) / std_val
                # 적절한 범위로 스케일링 (표준편차 100으로)
                cleaned_data[:, i] = mean_val + normalized * 100
                self.normalized_samples[class_name] += 1
        
        return cleaned_data

class CleanedH5Dataset(torch.utils.data.Dataset):
    """정제된 데이터셋"""
    
    def __init__(self, cleaned_data, sequence_length=20, use_augmentation=True):
        self.sequence_length = sequence_length
        self.use_augmentation = use_augmentation
        self.data = []
        self.labels = []
        self.class_names = []
        
        for class_name in sorted(cleaned_data.keys()):
            self.class_names.append(class_name)
            class_idx = len(self.class_names) - 1
            class_data = cleaned_data[class_name]['data']
            
            for i, data in enumerate(class_data):
                self.data.append(data)
                self.labels.append(class_idx)
                
                if self.use_augmentation:
                    # 기본 증강 (보수적)
                    augmented_data = self._basic_augmentation(data)
                    for aug_data in augmented_data:
                        self.data.append(aug_data)
                        self.labels.append(class_idx)
    
    def _basic_augmentation(self, data):
        """기본 증강 (보수적)"""
        augmented = []
        
        # 1. 노이즈 추가 (작은 수준)
        for noise_level in [0.005, 0.01]:
            aug_data = data + np.random.normal(0, noise_level, data.shape)
            augmented.append(aug_data)
        
        # 2. 시간 이동 (작은 수준)
        for shift in [1, 2]:
            aug_data = np.roll(data, shift, axis=0)
            augmented.append(aug_data)
        
        # 3. 스케일링 (작은 수준)
        for scale in [0.99, 1.01]:
            aug_data = data * scale
            augmented.append(aug_data)
        
        return augmented
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.data[idx]), torch.LongTensor([self.labels[idx]])

class FinalOptimizedGRU(nn.Module):
    """최종 최적화된 GRU 모델"""
    
    def __init__(self, input_size=8, hidden_size=48, num_layers=2, num_classes=24, dropout=0.5):
        super(FinalOptimizedGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # GRU 레이어
        self.gru = nn.GRU(input_size, hidden_size, num_layers, 
                         batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        # Dropout 레이어
        self.dropout = nn.Dropout(dropout)
        
        # 출력 레이어
        self.fc = nn.Linear(hidden_size, num_classes)
        
        # 가중치 초기화
        self._init_weights()
    
    def _init_weights(self):
        """Xavier 가중치 초기화"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # 초기 은닉 상태 초기화
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        
        # GRU 순전파
        out, _ = self.gru(x, h0)
        
        # 마지막 시퀀스 출력 사용
        out = self.dropout(out[:, -1, :])
        
        # 분류 레이어
        out = self.fc(out)
        
        return out

def train_final_model(model, train_loader, val_loader, epochs=150, device='cuda'):
    """최종 모델 훈련"""
    model.to(device)
    
    # 손실 함수 및 옵티마이저
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0)
    
    # 학습률 스케줄러 (Plateau - 최고 성능 전략)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True, min_lr=1e-6
    )
    
    # 조기 종료 설정
    best_val_acc = 0
    patience_counter = 0
    max_patience = 30
    
    # 훈련 기록
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    learning_rates = []
    
    print(f'🚀 최종 모델 훈련 시작 (최대 {epochs} 에포크)')
    print(f'📊 모델 파라미터 수: {sum(p.numel() for p in model.parameters()):,}')
    
    for epoch in range(epochs):
        # 훈련 단계
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.squeeze().to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            
            # 그래디언트 클리핑
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += batch_y.size(0)
            train_correct += (predicted == batch_y).sum().item()
        
        # 검증 단계
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.squeeze().to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                val_total += batch_y.size(0)
                val_correct += (predicted == batch_y).sum().item()
        
        # 평균 계산
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total
        
        # 기록 저장
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        learning_rates.append(optimizer.param_groups[0]['lr'])
        
        # 스케줄러 업데이트
        scheduler.step(val_loss)
        
        # 진행 상황 출력
        if epoch % 20 == 0 or epoch < 10:
            print(f'Epoch {epoch:3d}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
            print(f'           Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%')
            print(f'           LR: {optimizer.param_groups[0]["lr"]:.6f}, Gap: {train_acc - val_acc:.2f}%')
        
        # 모델 저장 및 조기 종료
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'train_accuracies': train_accuracies,
                'val_accuracies': val_accuracies,
                'learning_rates': learning_rates
            }, 'final_optimized_gru_model.pth')
            print(f'💾 모델 저장: Epoch {epoch}, 검증 정확도 {val_acc:.2f}%')
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                print(f'🛑 조기 종료: {epoch} 에포크에서 {max_patience} 에포크 동안 개선 없음')
                break
    
    print(f'✅ 훈련 완료! 최고 검증 정확도: {best_val_acc:.2f}%')
    return train_losses, val_losses, train_accuracies, val_accuracies, learning_rates, best_val_acc

def evaluate_final_model(model, test_loader, class_names, device='cuda'):
    """최종 모델 평가"""
    model.eval()
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.squeeze().to(device)
            outputs = model(batch_x)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    # 정확도 계산
    accuracy = 100 * sum(1 for x, y in zip(all_predictions, all_labels) if x == y) / len(all_labels)
    
    # 분류 리포트
    report = classification_report(all_labels, all_predictions, 
                                 target_names=class_names, output_dict=True)
    
    # 혼동 행렬
    cm = confusion_matrix(all_labels, all_predictions)
    
    return all_predictions, all_labels, all_probabilities, accuracy, report, cm

def visualize_training_results(train_losses, val_losses, train_accuracies, val_accuracies, learning_rates):
    """훈련 결과 시각화"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('최적화된 GRU 모델 훈련 결과', fontsize=16, fontweight='bold')
    
    # 손실 함수
    axes[0, 0].plot(train_losses, label='Train Loss', color='blue', alpha=0.7)
    axes[0, 0].plot(val_losses, label='Val Loss', color='red', alpha=0.7)
    axes[0, 0].set_title('손실 함수')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 정확도
    axes[0, 1].plot(train_accuracies, label='Train Acc', color='blue', alpha=0.7)
    axes[0, 1].plot(val_accuracies, label='Val Acc', color='red', alpha=0.7)
    axes[0, 1].set_title('정확도')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 학습률 변화
    axes[1, 0].plot(learning_rates, color='green', alpha=0.7)
    axes[1, 0].set_title('학습률 변화')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Learning Rate')
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 정확도 차이 (과적합 확인)
    acc_gap = [train - val for train, val in zip(train_accuracies, val_accuracies)]
    axes[1, 1].plot(acc_gap, color='orange', alpha=0.7)
    axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[1, 1].set_title('정확도 차이 (Train - Val)')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy Gap (%)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('final_model_training_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def visualize_confusion_matrix(cm, class_names, accuracy):
    """혼동 행렬 시각화"""
    plt.figure(figsize=(12, 10))
    
    # 정규화된 혼동 행렬
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # 히트맵 생성
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    
    plt.title(f'혼동 행렬 (정확도: {accuracy:.2f}%)')
    plt.xlabel('예측 레이블')
    plt.ylabel('실제 레이블')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig('final_model_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

def print_model_summary(model, accuracy, report):
    """모델 성능 요약 출력"""
    print(f'\n{"="*80}')
    print(f'🏆 최종 최적화된 GRU 모델 성능 요약')
    print(f'{"="*80}')
    
    # 모델 정보
    param_count = sum(p.numel() for p in model.parameters())
    print(f'📊 모델 정보:')
    print(f'  - 파라미터 수: {param_count:,}')
    print(f'  - 모델 크기: {param_count * 4 / 1024:.1f} KB')
    
    # 성능 정보
    print(f'\n📈 성능 정보:')
    print(f'  - 전체 정확도: {accuracy:.2f}%')
    print(f'  - 매크로 평균 F1: {report["macro avg"]["f1-score"]:.3f}')
    print(f'  - 가중 평균 F1: {report["weighted avg"]["f1-score"]:.3f}')
    
    # 클래스별 성능 (상위 5개, 하위 5개)
    class_f1_scores = []
    for class_name in report.keys():
        if class_name not in ['macro avg', 'weighted avg', 'micro avg']:
            if isinstance(report[class_name], dict) and 'f1-score' in report[class_name]:
                class_f1_scores.append((class_name, report[class_name]['f1-score']))
    
    class_f1_scores.sort(key=lambda x: x[1], reverse=True)
    
    print(f'\n🏅 상위 5개 클래스:')
    for i, (class_name, f1_score) in enumerate(class_f1_scores[:5]):
        print(f'  {i+1}. {class_name}: F1={f1_score:.3f}')
    
    print(f'\n⚠️ 하위 5개 클래스:')
    for i, (class_name, f1_score) in enumerate(class_f1_scores[-5:]):
        print(f'  {i+1}. {class_name}: F1={f1_score:.3f}')
    
    print(f'\n💾 저장된 파일:')
    print(f'  - final_optimized_gru_model.pth: 훈련된 모델')
    print(f'  - final_model_training_results.png: 훈련 과정')
    print(f'  - final_model_confusion_matrix.png: 혼동 행렬')

# 메인 실행
if __name__ == "__main__":
    # 1. 데이터 정제
    data_dir = '../SignGlove/external/SignGlove_HW/datasets/unified'
    cleaner = DataCleaner()
    cleaned_data = cleaner.clean_data(data_dir)
    
    # 2. 데이터셋 생성
    dataset = CleanedH5Dataset(cleaned_data, use_augmentation=True)
    print(f'\n📈 증강 후 총 샘플 수: {len(dataset)}')
    
    # 3. 데이터 분할
    train_data, temp_data, train_labels, temp_labels = train_test_split(
        dataset.data, dataset.labels, test_size=0.3, random_state=42, stratify=dataset.labels
    )
    
    val_data, test_data, val_labels, test_labels = train_test_split(
        temp_data, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
    )
    
    # 4. 데이터로더 생성
    train_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(train_data), torch.LongTensor(train_labels)
    )
    val_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(val_data), torch.LongTensor(val_labels)
    )
    test_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(test_data), torch.LongTensor(test_labels)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    print(f'\n📊 데이터 분할:')
    print(f'  훈련: {len(train_dataset)}')
    print(f'  검증: {len(val_dataset)}')
    print(f'  테스트: {len(test_dataset)}')
    
    # 5. 디바이스 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\n🖥️ 사용 디바이스: {device}')
    
    # 6. 최적화된 모델 생성
    model = FinalOptimizedGRU(
        input_size=8,           # 센서 수
        hidden_size=48,         # Medium 복잡도 (최적 균형점)
        num_layers=2,           # 2층 (효율성과 성능의 균형)
        num_classes=24,         # 클래스 수
        dropout=0.5             # 과적합 방지
    )
    
    # 7. 모델 훈련
    train_losses, val_losses, train_accuracies, val_accuracies, learning_rates, best_val_acc = train_final_model(
        model, train_loader, val_loader, epochs=150, device=device
    )
    
    # 8. 훈련 결과 시각화
    visualize_training_results(train_losses, val_losses, train_accuracies, val_accuracies, learning_rates)
    
    # 9. 최고 성능 모델 로드 및 평가
    checkpoint = torch.load('final_optimized_gru_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    predictions, labels, probabilities, test_accuracy, report, cm = evaluate_final_model(
        model, test_loader, dataset.class_names, device
    )
    
    # 10. 혼동 행렬 시각화
    visualize_confusion_matrix(cm, dataset.class_names, test_accuracy)
    
    # 11. 모델 성능 요약 출력
    print_model_summary(model, test_accuracy, report)
    
    print(f'\n🎉 최종 최적화된 GRU 모델 구축 완료!')
    print(f'🚀 현실적으로 사용 가능한 한국 수어 인식 모델이 준비되었습니다!')
