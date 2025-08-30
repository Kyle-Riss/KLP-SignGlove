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
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

print('🔍 과적합 및 이상치 분석 시작')

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

class GRUModel(nn.Module):
    """GRU 모델"""
    
    def __init__(self, input_size=8, hidden_size=64, num_layers=2, num_classes=24, dropout=0.6):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.gru = nn.GRU(input_size, hidden_size, num_layers, 
                         batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        # x: (batch_size, sequence_length, input_size)
        batch_size = x.size(0)
        
        # GRU forward pass
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        out, _ = self.gru(x, h0)
        
        # Use the last output
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out

def train_model_with_monitoring(model, train_loader, val_loader, epochs=100, device='cuda'):
    """과적합 모니터링이 포함된 모델 훈련"""
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    best_val_acc = 0
    patience_counter = 0
    max_patience = 20
    
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    print(f'🚀 GRU 모델 훈련 시작 (과적합 모니터링)')
    
    for epoch in range(epochs):
        # 훈련
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
        
        # 검증
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
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        
        scheduler.step(val_loss)
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch:3d}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
            print(f'           Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%')
            print(f'           Gap: {train_acc - val_acc:.2f}%')
        
        # 조기 종료
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), 'gru_model_monitored.pth')
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                print(f'🛑 조기 종료: {epoch} 에포크에서 {max_patience} 에포크 동안 개선 없음')
                break
    
    print(f'✅ 훈련 완료! 최고 검증 정확도: {best_val_acc:.2f}%')
    return train_losses, val_losses, train_accuracies, val_accuracies

def analyze_overfitting(train_losses, val_losses, train_accuracies, val_accuracies):
    """과적합 분석"""
    print('\n🔍 과적합 분석:')
    print('=' * 80)
    
    # 손실 함수 분석
    final_train_loss = train_losses[-1]
    final_val_loss = val_losses[-1]
    loss_gap = final_train_loss - final_val_loss
    
    # 정확도 분석
    final_train_acc = train_accuracies[-1]
    final_val_acc = val_accuracies[-1]
    acc_gap = final_train_acc - final_val_acc
    
    print(f'📊 최종 손실:')
    print(f'  훈련 손실: {final_train_loss:.4f}')
    print(f'  검증 손실: {final_val_loss:.4f}')
    print(f'  손실 차이: {loss_gap:.4f} {"(과적합 의심)" if loss_gap < -0.1 else "(정상)"}')
    
    print(f'\n📊 최종 정확도:')
    print(f'  훈련 정확도: {final_train_acc:.2f}%')
    print(f'  검증 정확도: {final_val_acc:.2f}%')
    print(f'  정확도 차이: {acc_gap:.2f}% {"(과적합 의심)" if acc_gap > 5 else "(정상)"}')
    
    # 과적합 판단
    overfitting_score = 0
    if loss_gap < -0.1:
        overfitting_score += 1
        print(f'⚠️ 손실 함수에서 과적합 징후 발견')
    
    if acc_gap > 5:
        overfitting_score += 1
        print(f'⚠️ 정확도에서 과적합 징후 발견')
    
    if overfitting_score >= 2:
        print(f'🚨 과적합이 의심됩니다!')
    elif overfitting_score == 1:
        print(f'⚠️ 약간의 과적합 징후가 있습니다.')
    else:
        print(f'✅ 과적합 징후가 없습니다.')
    
    return overfitting_score

def detect_outliers(model, test_loader, device='cuda'):
    """이상치 탐지"""
    print('\n🔍 이상치 탐지:')
    print('=' * 80)
    
    model.eval()
    all_predictions = []
    all_labels = []
    all_confidences = []
    all_features = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.squeeze().to(device)
            outputs = model(batch_x)
            
            # 예측 및 신뢰도
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
            all_confidences.extend(confidence.cpu().numpy())
            
            # 특징 추출 (마지막 GRU 출력)
            batch_size = batch_x.size(0)
            h0 = torch.zeros(2, batch_size, 64).to(device)
            out, _ = model.gru(batch_x, h0)
            features = out[:, -1, :].cpu().numpy()  # 마지막 시간 스텝의 특징
            all_features.extend(features)
    
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_confidences = np.array(all_confidences)
    all_features = np.array(all_features)
    
    # 1. 낮은 신뢰도 샘플 탐지
    low_confidence_threshold = 0.7
    low_confidence_samples = np.sum(all_confidences < low_confidence_threshold)
    low_confidence_rate = low_confidence_samples / len(all_confidences) * 100
    
    print(f'📊 신뢰도 분석:')
    print(f'  평균 신뢰도: {np.mean(all_confidences):.3f}')
    print(f'  낮은 신뢰도 샘플 (<0.7): {low_confidence_samples}개 ({low_confidence_rate:.1f}%)')
    
    # 2. 잘못 분류된 샘플의 신뢰도 분석
    incorrect_predictions = all_predictions != all_labels
    incorrect_confidences = all_confidences[incorrect_predictions]
    
    if len(incorrect_confidences) > 0:
        print(f'  잘못 분류된 샘플 평균 신뢰도: {np.mean(incorrect_confidences):.3f}')
        print(f'  높은 신뢰도로 잘못 분류된 샘플 (>0.8): {np.sum(incorrect_confidences > 0.8)}개')
    
    # 3. 특징 공간에서의 이상치 탐지
    print(f'\n📊 특징 공간 분석:')
    
    # t-SNE로 차원 축소
    if len(all_features) > 1000:
        # 샘플링
        indices = np.random.choice(len(all_features), 1000, replace=False)
        features_sample = all_features[indices]
        labels_sample = all_labels[indices]
    else:
        features_sample = all_features
        labels_sample = all_labels
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    features_2d = tsne.fit_transform(features_sample)
    
    # 이상치 탐지 (거리 기반)
    from sklearn.neighbors import LocalOutlierFactor
    lof = LocalOutlierFactor(contamination=0.1)
    outlier_labels = lof.fit_predict(features_2d)
    outliers = np.sum(outlier_labels == -1)
    
    print(f'  t-SNE 차원 축소 완료')
    print(f'  탐지된 이상치: {outliers}개 ({outliers/len(features_sample)*100:.1f}%)')
    
    return all_confidences, features_2d, labels_sample, outlier_labels

def visualize_analysis(train_losses, val_losses, train_accuracies, val_accuracies, 
                      confidences, features_2d, labels, outlier_labels):
    """분석 결과 시각화"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('과적합 및 이상치 분석', fontsize=16, fontweight='bold')
    
    # 1. 손실 함수
    axes[0, 0].plot(train_losses, label='Train Loss', color='blue')
    axes[0, 0].plot(val_losses, label='Val Loss', color='red')
    axes[0, 0].set_title('손실 함수')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 정확도
    axes[0, 1].plot(train_accuracies, label='Train Accuracy', color='blue')
    axes[0, 1].plot(val_accuracies, label='Val Accuracy', color='red')
    axes[0, 1].set_title('정확도')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 과적합 지표 (정확도 차이)
    acc_gap = np.array(train_accuracies) - np.array(val_accuracies)
    axes[0, 2].plot(acc_gap, label='Train-Val Gap', color='orange')
    axes[0, 2].axhline(y=5, color='red', linestyle='--', alpha=0.7, label='과적합 임계값')
    axes[0, 2].set_title('과적합 지표 (정확도 차이)')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Accuracy Gap (%)')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. 신뢰도 분포
    axes[1, 0].hist(confidences, bins=30, alpha=0.7, color='green')
    axes[1, 0].axvline(x=0.7, color='red', linestyle='--', alpha=0.7, label='임계값 (0.7)')
    axes[1, 0].set_title('예측 신뢰도 분포')
    axes[1, 0].set_xlabel('신뢰도')
    axes[1, 0].set_ylabel('빈도')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. t-SNE 시각화
    scatter = axes[1, 1].scatter(features_2d[:, 0], features_2d[:, 1], 
                                c=labels, cmap='tab20', alpha=0.6, s=20)
    axes[1, 1].set_title('t-SNE 특징 공간')
    axes[1, 1].set_xlabel('t-SNE 1')
    axes[1, 1].set_ylabel('t-SNE 2')
    plt.colorbar(scatter, ax=axes[1, 1])
    
    # 6. 이상치 시각화
    normal_points = outlier_labels == 1
    outlier_points = outlier_labels == -1
    
    axes[1, 2].scatter(features_2d[normal_points, 0], features_2d[normal_points, 1], 
                       c='blue', alpha=0.6, s=20, label='정상')
    axes[1, 2].scatter(features_2d[outlier_points, 0], features_2d[outlier_points, 1], 
                       c='red', alpha=0.8, s=30, label='이상치')
    axes[1, 2].set_title('이상치 탐지')
    axes[1, 2].set_xlabel('t-SNE 1')
    axes[1, 2].set_ylabel('t-SNE 2')
    axes[1, 2].legend()
    
    plt.tight_layout()
    plt.savefig('overfitting_outlier_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

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
    
    # 5. 모델 생성 및 훈련 (과적합 모니터링)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\n🖥️ 사용 디바이스: {device}')
    
    model = GRUModel(input_size=8, hidden_size=64, num_layers=2, num_classes=24, dropout=0.6)
    print(f'📊 모델 파라미터 수: {sum(p.numel() for p in model.parameters()):,}')
    
    # 훈련 (과적합 모니터링)
    train_losses, val_losses, train_accuracies, val_accuracies = train_model_with_monitoring(
        model, train_loader, val_loader, epochs=100, device=device
    )
    
    # 6. 과적합 분석
    overfitting_score = analyze_overfitting(train_losses, val_losses, train_accuracies, val_accuracies)
    
    # 7. 이상치 탐지
    model.load_state_dict(torch.load('gru_model_monitored.pth'))
    confidences, features_2d, labels, outlier_labels = detect_outliers(model, test_loader, device)
    
    # 8. 시각화
    visualize_analysis(train_losses, val_losses, train_accuracies, val_accuracies,
                      confidences, features_2d, labels, outlier_labels)
    
    # 9. 최종 결론
    print(f'\n{"="*80}')
    print(f'📋 최종 분석 결론')
    print(f'{"="*80}')
    
    if overfitting_score >= 2:
        print(f'🚨 과적합이 확인되었습니다!')
        print(f'   - 더 강한 정규화 필요')
        print(f'   - 데이터 증강 증가 고려')
        print(f'   - 모델 복잡도 감소 고려')
    elif overfitting_score == 1:
        print(f'⚠️ 약간의 과적합 징후가 있습니다.')
        print(f'   - 모니터링 필요')
        print(f'   - 필요시 정규화 강화')
    else:
        print(f'✅ 과적합 징후가 없습니다.')
        print(f'   - 모델이 적절히 일반화됨')
    
    print(f'\n💾 저장된 파일:')
    print(f'  - gru_model_monitored.pth: 모니터링된 모델')
    print(f'  - overfitting_outlier_analysis.png: 분석 결과')
