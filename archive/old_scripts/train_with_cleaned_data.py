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

print('🚀 정제된 데이터로 모델 훈련 시작')

# 센서 이름
sensor_names = ['Flex1', 'Flex2', 'Flex3', 'Flex4', 'Flex5', 'Pitch', 'Roll', 'Yaw']

class DataCleaner:
    """1순위 데이터 정제 클래스 (이전과 동일)"""
    
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

class ImprovedMLP(nn.Module):
    """개선된 MLP 모델"""
    
    def __init__(self, input_size=160, hidden_sizes=[64, 32, 16], num_classes=24, dropout=0.6):
        super(ImprovedMLP, self).__init__()
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, num_classes))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        return self.network(x)

def train_model(model, train_loader, val_loader, epochs=100, device='cuda'):
    """모델 훈련"""
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    best_val_acc = 0
    patience_counter = 0
    max_patience = 20
    
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    print(f'🚀 모델 훈련 시작 (최대 {epochs} 에포크)')
    
    for epoch in range(epochs):
        # 훈련
        model.train()
        train_loss = 0
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
        
        # 검증
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.squeeze().to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        val_acc = 100 * correct / total
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        scheduler.step(val_loss)
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch:3d}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # 조기 종료
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), 'cleaned_mlp_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                print(f'🛑 조기 종료: {epoch} 에포크에서 {max_patience} 에포크 동안 개선 없음')
                break
    
    print(f'✅ 훈련 완료! 최고 검증 정확도: {best_val_acc:.2f}%')
    return train_losses, val_losses, val_accuracies

def evaluate_model(model, test_loader, device='cuda'):
    """모델 평가"""
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.squeeze().to(device)
            outputs = model(batch_x)
            _, predicted = torch.max(outputs.data, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
    
    # 분류 리포트
    print('\n📊 분류 성능 리포트:')
    print('=' * 80)
    print(classification_report(all_labels, all_predictions, 
                              target_names=[f'Class_{i}' for i in range(24)]))
    
    # 혼동 행렬
    cm = confusion_matrix(all_labels, all_predictions)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('혼동 행렬 (정제된 데이터)')
    plt.xlabel('예측')
    plt.ylabel('실제')
    plt.savefig('cleaned_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return all_predictions, all_labels

# 메인 실행
if __name__ == "__main__":
    # 1. 데이터 정제
    data_dir = '../SignGlove/external/SignGlove_HW/datasets/unified'
    cleaner = DataCleaner()
    cleaned_data = cleaner.clean_data(data_dir)
    
    print(f'\n📊 정제 후 데이터 통계:')
    total_samples = 0
    for class_name, data_info in cleaned_data.items():
        samples = len(data_info['data'])
        total_samples += samples
        removed = cleaner.removed_samples.get(class_name, 0)
        normalized = cleaner.normalized_samples.get(class_name, 0)
        
        print(f'  {class_name}: {samples}개 샘플', end='')
        if removed > 0:
            print(f' (제거: {removed}개)', end='')
        if normalized > 0:
            print(f' (정규화: {normalized}개)', end='')
        print()
    
    print(f'  총 샘플 수: {total_samples}')
    
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
    
    # 5. 모델 생성 및 훈련
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\n🖥️ 사용 디바이스: {device}')
    
    model = ImprovedMLP(input_size=160, hidden_sizes=[64, 32, 16], num_classes=24, dropout=0.6)
    print(f'📊 모델 파라미터 수: {sum(p.numel() for p in model.parameters()):,}')
    
    train_losses, val_losses, val_accuracies = train_model(
        model, train_loader, val_loader, epochs=100, device=device
    )
    
    # 6. 모델 평가
    model.load_state_dict(torch.load('cleaned_mlp_model.pth'))
    predictions, labels = evaluate_model(model, test_loader, device)
    
    # 7. 훈련 과정 시각화
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('손실 함수')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.plot(val_accuracies, label='Val Accuracy')
    plt.title('검증 정확도')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    plt.plot(train_losses, label='Train Loss', alpha=0.7)
    plt.plot(val_losses, label='Val Loss', alpha=0.7)
    plt.plot(val_accuracies, label='Val Accuracy', alpha=0.7)
    plt.title('전체 훈련 과정')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('cleaned_training_process.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f'\n🎉 정제된 데이터로 훈련 완료!')
    print(f'💾 저장된 파일:')
    print(f'  - cleaned_mlp_model.pth: 최고 성능 모델')
    print(f'  - cleaned_confusion_matrix.png: 혼동 행렬')
    print(f'  - cleaned_training_process.png: 훈련 과정')
