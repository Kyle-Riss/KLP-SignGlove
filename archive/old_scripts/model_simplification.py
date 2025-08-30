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

print('🔧 모델 구조 단순화 비교 시작')

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

class SimpleGRU(nn.Module):
    """매우 단순한 GRU 모델 (1층, 작은 유닛)"""
    
    def __init__(self, input_size=8, hidden_size=16, num_classes=24, dropout=0.3):
        super(SimpleGRU, self).__init__()
        self.hidden_size = hidden_size
        
        self.gru = nn.GRU(input_size, hidden_size, num_layers=1, 
                         batch_first=True, dropout=0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(1, batch_size, self.hidden_size).to(x.device)
        out, _ = self.gru(x, h0)
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out

class LightGRU(nn.Module):
    """가벼운 GRU 모델 (1층, 중간 유닛)"""
    
    def __init__(self, input_size=8, hidden_size=32, num_classes=24, dropout=0.4):
        super(LightGRU, self).__init__()
        self.hidden_size = hidden_size
        
        self.gru = nn.GRU(input_size, hidden_size, num_layers=1, 
                         batch_first=True, dropout=0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(1, batch_size, self.hidden_size).to(x.device)
        out, _ = self.gru(x, h0)
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out

class MediumGRU(nn.Module):
    """중간 복잡도 GRU 모델 (2층, 중간 유닛)"""
    
    def __init__(self, input_size=8, hidden_size=48, num_layers=2, num_classes=24, dropout=0.5):
        super(MediumGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.gru = nn.GRU(input_size, hidden_size, num_layers, 
                         batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        out, _ = self.gru(x, h0)
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out

class ComplexGRU(nn.Module):
    """복잡한 GRU 모델 (3층, 큰 유닛) - 과적합 위험"""
    
    def __init__(self, input_size=8, hidden_size=128, num_layers=3, num_classes=24, dropout=0.6):
        super(ComplexGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.gru = nn.GRU(input_size, hidden_size, num_layers, 
                         batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        out, _ = self.gru(x, h0)
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out

class UltraSimpleGRU(nn.Module):
    """초단순 GRU 모델 (1층, 매우 작은 유닛)"""
    
    def __init__(self, input_size=8, hidden_size=8, num_classes=24, dropout=0.2):
        super(UltraSimpleGRU, self).__init__()
        self.hidden_size = hidden_size
        
        self.gru = nn.GRU(input_size, hidden_size, num_layers=1, 
                         batch_first=True, dropout=0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(1, batch_size, self.hidden_size).to(x.device)
        out, _ = self.gru(x, h0)
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out

def train_model_with_monitoring(model, train_loader, val_loader, model_name, epochs=100, device='cuda'):
    """모델 훈련 및 모니터링"""
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    best_val_acc = 0
    patience_counter = 0
    max_patience = 20
    
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    print(f'🚀 {model_name} 훈련 시작')
    
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
        
        if epoch % 20 == 0:
            print(f'Epoch {epoch:3d}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
            print(f'           Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%')
            print(f'           Gap: {train_acc - val_acc:.2f}%')
        
        # 조기 종료
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), f'{model_name.lower()}_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                print(f'🛑 조기 종료: {epoch} 에포크에서 {max_patience} 에포크 동안 개선 없음')
                break
    
    print(f'✅ {model_name} 훈련 완료! 최고 검증 정확도: {best_val_acc:.2f}%')
    return train_losses, val_losses, train_accuracies, val_accuracies, best_val_acc

def evaluate_model(model, test_loader, model_name, device='cuda'):
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
    
    # 정확도 계산
    accuracy = 100 * sum(1 for x, y in zip(all_predictions, all_labels) if x == y) / len(all_labels)
    
    return all_predictions, all_labels, accuracy

def analyze_overfitting(train_acc, val_acc, train_loss, val_loss):
    """과적합 분석"""
    acc_gap = train_acc - val_acc
    loss_gap = val_loss - train_loss
    
    # 과적합 판단 기준
    overfitting_score = 0
    
    if acc_gap > 10:  # 정확도 차이가 10% 이상
        overfitting_score += 2
    elif acc_gap > 5:  # 정확도 차이가 5% 이상
        overfitting_score += 1
    
    if loss_gap > 0.5:  # 손실 차이가 0.5 이상
        overfitting_score += 2
    elif loss_gap > 0.2:  # 손실 차이가 0.2 이상
        overfitting_score += 1
    
    if overfitting_score >= 3:
        return "높음", overfitting_score
    elif overfitting_score >= 1:
        return "보통", overfitting_score
    else:
        return "낮음", overfitting_score

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
    
    # 5. 모델 정의 (복잡도 순)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\n🖥️ 사용 디바이스: {device}')
    
    models = {
        'UltraSimple': UltraSimpleGRU(input_size=8, hidden_size=8, num_classes=24, dropout=0.2),
        'Simple': SimpleGRU(input_size=8, hidden_size=16, num_classes=24, dropout=0.3),
        'Light': LightGRU(input_size=8, hidden_size=32, num_classes=24, dropout=0.4),
        'Medium': MediumGRU(input_size=8, hidden_size=48, num_layers=2, num_classes=24, dropout=0.5),
        'Complex': ComplexGRU(input_size=8, hidden_size=128, num_layers=3, num_classes=24, dropout=0.6)
    }
    
    # 파라미터 수 및 복잡도 분석
    print(f'\n📊 모델 복잡도 분석:')
    print(f'{"모델":<12} {"레이어":<6} {"유닛":<6} {"파라미터":<10} {"복잡도":<8}')
    print(f'{"-"*50}')
    
    complexity_info = {}
    for name, model in models.items():
        param_count = sum(p.numel() for p in model.parameters())
        
        if name == 'UltraSimple':
            complexity = "매우 낮음"
        elif name == 'Simple':
            complexity = "낮음"
        elif name == 'Light':
            complexity = "보통"
        elif name == 'Medium':
            complexity = "높음"
        else:
            complexity = "매우 높음"
        
        complexity_info[name] = {
            'param_count': param_count,
            'complexity': complexity
        }
        
        # 레이어 수와 유닛 수 추출
        if hasattr(model, 'num_layers'):
            layers = model.num_layers
        else:
            layers = 1
        
        units = model.hidden_size
        
        print(f'{name:<12} {layers:<6} {units:<6} {param_count:<10,} {complexity:<8}')
    
    # 6. 모델 훈련 및 평가
    results = {}
    
    for model_name, model in models.items():
        print(f'\n{"="*60}')
        print(f'🎯 {model_name} 모델 훈련 및 평가')
        print(f'{"="*60}')
        
        # 훈련
        train_losses, val_losses, train_accuracies, val_accuracies, best_val_acc = train_model_with_monitoring(
            model, train_loader, val_loader, model_name, epochs=100, device=device
        )
        
        # 평가
        model.load_state_dict(torch.load(f'{model_name.lower()}_model.pth'))
        predictions, labels, test_accuracy = evaluate_model(model, test_loader, model_name, device)
        
        # 과적합 분석
        final_train_acc = train_accuracies[-1]
        final_val_acc = val_accuracies[-1]
        final_train_loss = train_losses[-1]
        final_val_loss = val_losses[-1]
        
        overfitting_level, overfitting_score = analyze_overfitting(
            final_train_acc, final_val_acc, final_train_loss, final_val_loss
        )
        
        # 결과 저장
        results[model_name] = {
            'best_val_acc': best_val_acc,
            'test_accuracy': test_accuracy,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accuracies': train_accuracies,
            'val_accuracies': val_accuracies,
            'overfitting_level': overfitting_level,
            'overfitting_score': overfitting_score,
            'param_count': complexity_info[model_name]['param_count'],
            'complexity': complexity_info[model_name]['complexity']
        }
    
    # 7. 결과 비교 및 시각화
    print(f'\n{"="*80}')
    print(f'🏆 모델 구조 단순화 성능 비교')
    print(f'{"="*80}')
    
    # 성능 비교 테이블
    print(f'{"모델":<12} {"복잡도":<8} {"파라미터":<10} {"검증 정확도":<12} {"테스트 정확도":<12} {"과적합":<8}')
    print(f'{"-"*80}')
    
    for model_name, result in results.items():
        print(f'{model_name:<12} {result["complexity"]:<8} {result["param_count"]:<10,} '
              f'{result["best_val_acc"]:<12.2f} {result["test_accuracy"]:<12.2f} {result["overfitting_level"]:<8}')
    
    # 최고 성능 모델 찾기
    best_model = max(results.items(), key=lambda x: x[1]['test_accuracy'])
    print(f'\n🏆 최고 성능 모델: {best_model[0]} (테스트 정확도: {best_model[1]["test_accuracy"]:.2f}%)')
    
    # 최적 복잡도 모델 찾기 (성능 대비 파라미터 효율성)
    efficiency_scores = {}
    for name, result in results.items():
        efficiency = result['test_accuracy'] / (result['param_count'] / 1000)  # 정확도 / (파라미터/1000)
        efficiency_scores[name] = efficiency
    
    best_efficiency = max(efficiency_scores.items(), key=lambda x: x[1])
    print(f'⚡ 최적 효율성 모델: {best_efficiency[0]} (효율성 점수: {best_efficiency[1]:.2f})')
    
    # 훈련 과정 시각화
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('모델 복잡도별 훈련 과정 비교', fontsize=16, fontweight='bold')
    
    colors = {'UltraSimple': 'red', 'Simple': 'orange', 'Light': 'yellow', 
              'Medium': 'green', 'Complex': 'blue'}
    
    for i, model_name in enumerate(['UltraSimple', 'Simple', 'Light', 'Medium', 'Complex']):
        result = results[model_name]
        row = i // 3
        col = i % 3
        
        # 손실 함수
        axes[row, col].plot(result['train_losses'], label='Train Loss', color=colors[model_name], alpha=0.7)
        axes[row, col].plot(result['val_losses'], label='Val Loss', color=colors[model_name], linestyle='--', alpha=0.7)
        axes[row, col].set_title(f'{model_name} 손실 함수')
        axes[row, col].set_xlabel('Epoch')
        axes[row, col].set_ylabel('Loss')
        axes[row, col].legend()
        axes[row, col].grid(True, alpha=0.3)
    
    # 마지막 자리에 효율성 비교
    model_names = list(results.keys())
    efficiencies = [efficiency_scores[name] for name in model_names]
    
    axes[1, 2].bar(model_names, efficiencies, color=[colors[name] for name in model_names], alpha=0.7)
    axes[1, 2].set_title('효율성 비교 (정확도/파라미터)')
    axes[1, 2].set_xlabel('모델')
    axes[1, 2].set_ylabel('효율성 점수')
    axes[1, 2].tick_params(axis='x', rotation=45)
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('model_simplification_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 성능 vs 복잡도 비교
    plt.figure(figsize=(12, 8))
    
    # 서브플롯 1: 정확도 비교
    plt.subplot(2, 2, 1)
    model_names = list(results.keys())
    val_accuracies = [results[name]['best_val_acc'] for name in model_names]
    test_accuracies = [results[name]['test_accuracy'] for name in model_names]
    
    x = np.arange(len(model_names))
    width = 0.35
    
    bars1 = plt.bar(x - width/2, val_accuracies, width, label='검증 정확도', color='skyblue', alpha=0.7)
    bars2 = plt.bar(x + width/2, test_accuracies, width, label='테스트 정확도', color='lightcoral', alpha=0.7)
    
    plt.xlabel('모델 복잡도')
    plt.ylabel('정확도 (%)')
    plt.title('모델 복잡도별 성능 비교')
    plt.xticks(x, model_names, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 값 표시
    for bar in bars1:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=8)
    
    for bar in bars2:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=8)
    
    # 서브플롯 2: 파라미터 수 비교
    plt.subplot(2, 2, 2)
    param_counts = [results[name]['param_count'] for name in model_names]
    
    bars = plt.bar(model_names, param_counts, color=[colors[name] for name in model_names], alpha=0.7)
    plt.xlabel('모델 복잡도')
    plt.ylabel('파라미터 수')
    plt.title('모델 복잡도별 파라미터 수')
    plt.xticks(rotation=45)
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    
    # 값 표시
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height * 1.1,
                f'{height:,}', ha='center', va='bottom', fontsize=8)
    
    # 서브플롯 3: 과적합 점수 비교
    plt.subplot(2, 2, 3)
    overfitting_scores = [results[name]['overfitting_score'] for name in model_names]
    
    bars = plt.bar(model_names, overfitting_scores, color=[colors[name] for name in model_names], alpha=0.7)
    plt.xlabel('모델 복잡도')
    plt.ylabel('과적합 점수')
    plt.title('모델 복잡도별 과적합 점수')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # 값 표시
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height}', ha='center', va='bottom', fontsize=8)
    
    # 서브플롯 4: 효율성 비교
    plt.subplot(2, 2, 4)
    efficiencies = [efficiency_scores[name] for name in model_names]
    
    bars = plt.bar(model_names, efficiencies, color=[colors[name] for name in model_names], alpha=0.7)
    plt.xlabel('모델 복잡도')
    plt.ylabel('효율성 점수')
    plt.title('모델 복잡도별 효율성')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # 값 표시
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.1f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('model_simplification_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f'\n🎉 모델 구조 단순화 비교 완료!')
    print(f'💾 저장된 파일:')
    print(f'  - ultrasimple_model.pth, simple_model.pth, light_model.pth, medium_model.pth, complex_model.pth: 각 모델')
    print(f'  - model_simplification_comparison.png: 훈련 과정 비교')
    print(f'  - model_simplification_analysis.png: 종합 분석')
