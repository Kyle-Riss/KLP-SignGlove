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

print('🔧 학습률 최적화 비교 시작')

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

class OptimizedGRU(nn.Module):
    """최적화된 GRU 모델 (Medium 복잡도)"""
    
    def __init__(self, input_size=8, hidden_size=48, num_layers=2, num_classes=24, dropout=0.5):
        super(OptimizedGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.gru = nn.GRU(input_size, hidden_size, num_layers, 
                         batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
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

def train_model_with_lr_strategy(model, train_loader, val_loader, model_name, 
                                lr_strategy='fixed', initial_lr=0.001, epochs=100, device='cuda'):
    """다양한 학습률 전략으로 모델 훈련"""
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    
    # 학습률 전략 설정
    if lr_strategy == 'fixed':
        optimizer = torch.optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=0)
        scheduler = None
        print(f'📊 {model_name}: 고정 학습률 (lr={initial_lr})')
        
    elif lr_strategy == 'step':
        optimizer = torch.optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=0)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
        print(f'📊 {model_name}: Step LR (lr={initial_lr}, step=30, gamma=0.5)')
        
    elif lr_strategy == 'plateau':
        optimizer = torch.optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=0)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5, verbose=True)
        print(f'📊 {model_name}: Plateau LR (lr={initial_lr}, patience=10, factor=0.5)')
        
    elif lr_strategy == 'cosine':
        optimizer = torch.optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=0)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
        print(f'📊 {model_name}: Cosine LR (lr={initial_lr}, T_max={epochs})')
        
    elif lr_strategy == 'warmup':
        optimizer = torch.optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=0)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=initial_lr, epochs=epochs, 
                                                       steps_per_epoch=len(train_loader))
        print(f'📊 {model_name}: OneCycle LR (max_lr={initial_lr})')
        
    elif lr_strategy == 'low_lr':
        optimizer = torch.optim.AdamW(model.parameters(), lr=initial_lr*0.1, weight_decay=0)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=15, factor=0.7, verbose=True)
        print(f'📊 {model_name}: 낮은 LR (lr={initial_lr*0.1}, patience=15)')
    
    best_val_acc = 0
    patience_counter = 0
    max_patience = 25
    
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    learning_rates = []
    
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
            
            # OneCycleLR의 경우 매 스텝마다 스케줄러 업데이트
            if lr_strategy == 'warmup' and scheduler is not None:
                scheduler.step()
            
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
        
        # 현재 학습률 기록
        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)
        
        # 스케줄러 업데이트 (OneCycleLR 제외)
        if scheduler is not None and lr_strategy != 'warmup':
            if lr_strategy == 'plateau' or lr_strategy == 'low_lr':
                scheduler.step(val_loss)
            else:
                scheduler.step()
        
        if epoch % 20 == 0:
            print(f'Epoch {epoch:3d}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
            print(f'           Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%')
            print(f'           LR: {current_lr:.6f}, Gap: {train_acc - val_acc:.2f}%')
        
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
    return train_losses, val_losses, train_accuracies, val_accuracies, learning_rates, best_val_acc

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

def analyze_training_stability(train_losses, val_losses, learning_rates):
    """훈련 안정성 분석"""
    # 손실 변동성 계산
    train_loss_std = np.std(train_losses[-20:])  # 마지막 20 에포크
    val_loss_std = np.std(val_losses[-20:])
    
    # 학습률 변화 분석
    lr_changes = np.diff(learning_rates)
    lr_stability = np.std(lr_changes)
    
    # 안정성 점수 (낮을수록 안정적)
    stability_score = train_loss_std + val_loss_std + lr_stability * 10
    
    if stability_score < 0.1:
        return "매우 안정적", stability_score
    elif stability_score < 0.3:
        return "안정적", stability_score
    elif stability_score < 0.5:
        return "보통", stability_score
    else:
        return "불안정", stability_score

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
    
    # 5. 학습률 전략 정의
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\n🖥️ 사용 디바이스: {device}')
    
    lr_strategies = {
        'Fixed': {'strategy': 'fixed', 'lr': 0.001},
        'Step': {'strategy': 'step', 'lr': 0.001},
        'Plateau': {'strategy': 'plateau', 'lr': 0.001},
        'Cosine': {'strategy': 'cosine', 'lr': 0.001},
        'OneCycle': {'strategy': 'warmup', 'lr': 0.001},
        'LowLR': {'strategy': 'low_lr', 'lr': 0.001}
    }
    
    # 6. 모델 훈련 및 평가
    results = {}
    
    for strategy_name, config in lr_strategies.items():
        print(f'\n{"="*60}')
        print(f'🎯 {strategy_name} 학습률 전략 훈련 및 평가')
        print(f'{"="*60}')
        
        # 모델 생성
        model = OptimizedGRU(input_size=8, hidden_size=48, num_layers=2, num_classes=24, dropout=0.5)
        
        # 훈련
        train_losses, val_losses, train_accuracies, val_accuracies, learning_rates, best_val_acc = train_model_with_lr_strategy(
            model, train_loader, val_loader, strategy_name,
            lr_strategy=config['strategy'], initial_lr=config['lr'],
            epochs=100, device=device
        )
        
        # 평가
        model.load_state_dict(torch.load(f'{strategy_name.lower()}_model.pth'))
        predictions, labels, test_accuracy = evaluate_model(model, test_loader, strategy_name, device)
        
        # 훈련 안정성 분석
        stability_level, stability_score = analyze_training_stability(train_losses, val_losses, learning_rates)
        
        # 결과 저장
        results[strategy_name] = {
            'best_val_acc': best_val_acc,
            'test_accuracy': test_accuracy,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accuracies': train_accuracies,
            'val_accuracies': val_accuracies,
            'learning_rates': learning_rates,
            'stability_level': stability_level,
            'stability_score': stability_score,
            'config': config
        }
    
    # 7. 결과 비교 및 시각화
    print(f'\n{"="*80}')
    print(f'🏆 학습률 전략 성능 비교')
    print(f'{"="*80}')
    
    # 성능 비교 테이블
    print(f'{"전략":<12} {"초기 LR":<10} {"검증 정확도":<12} {"테스트 정확도":<12} {"안정성":<8}')
    print(f'{"-"*70}')
    
    for strategy_name, result in results.items():
        config = result['config']
        print(f'{strategy_name:<12} {config["lr"]:<10.0e} {result["best_val_acc"]:<12.2f} '
              f'{result["test_accuracy"]:<12.2f} {result["stability_level"]:<8}')
    
    # 최고 성능 전략 찾기
    best_strategy = max(results.items(), key=lambda x: x[1]['test_accuracy'])
    print(f'\n🏆 최고 성능 전략: {best_strategy[0]} (테스트 정확도: {best_strategy[1]["test_accuracy"]:.2f}%)')
    
    # 최고 안정성 전략 찾기
    best_stability = min(results.items(), key=lambda x: x[1]['stability_score'])
    print(f'⚡ 최고 안정성 전략: {best_stability[0]} (안정성 점수: {best_stability[1]["stability_score"]:.3f})')
    
    # 훈련 과정 시각화
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('학습률 전략별 훈련 과정 비교', fontsize=16, fontweight='bold')
    
    colors = {'Fixed': 'blue', 'Step': 'red', 'Plateau': 'green', 
              'Cosine': 'orange', 'OneCycle': 'purple', 'LowLR': 'brown'}
    
    for i, strategy_name in enumerate(['Fixed', 'Step', 'Plateau', 'Cosine', 'OneCycle', 'LowLR']):
        result = results[strategy_name]
        row = i // 3
        col = i % 3
        
        # 손실 함수
        axes[row, col].plot(result['train_losses'], label='Train Loss', color=colors[strategy_name], alpha=0.7)
        axes[row, col].plot(result['val_losses'], label='Val Loss', color=colors[strategy_name], linestyle='--', alpha=0.7)
        axes[row, col].set_title(f'{strategy_name} 손실 함수')
        axes[row, col].set_xlabel('Epoch')
        axes[row, col].set_ylabel('Loss')
        axes[row, col].legend()
        axes[row, col].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('learning_rate_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 학습률 변화 시각화
    plt.figure(figsize=(15, 10))
    
    # 서브플롯 1: 학습률 변화
    plt.subplot(2, 2, 1)
    for strategy_name, result in results.items():
        plt.plot(result['learning_rates'], label=strategy_name, color=colors[strategy_name], alpha=0.7)
    
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('학습률 변화 비교')
    plt.legend()
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    
    # 서브플롯 2: 정확도 비교
    plt.subplot(2, 2, 2)
    strategy_names = list(results.keys())
    val_accuracies = [results[name]['best_val_acc'] for name in strategy_names]
    test_accuracies = [results[name]['test_accuracy'] for name in strategy_names]
    
    x = np.arange(len(strategy_names))
    width = 0.35
    
    bars1 = plt.bar(x - width/2, val_accuracies, width, label='검증 정확도', color='skyblue', alpha=0.7)
    bars2 = plt.bar(x + width/2, test_accuracies, width, label='테스트 정확도', color='lightcoral', alpha=0.7)
    
    plt.xlabel('학습률 전략')
    plt.ylabel('정확도 (%)')
    plt.title('학습률 전략별 성능 비교')
    plt.xticks(x, strategy_names, rotation=45)
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
    
    # 서브플롯 3: 안정성 점수 비교
    plt.subplot(2, 2, 3)
    stability_scores = [results[name]['stability_score'] for name in strategy_names]
    
    bars = plt.bar(strategy_names, stability_scores, color=[colors[name] for name in strategy_names], alpha=0.7)
    plt.xlabel('학습률 전략')
    plt.ylabel('안정성 점수 (낮을수록 안정적)')
    plt.title('학습률 전략별 안정성 비교')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # 값 표시
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    # 서브플롯 4: 수렴 속도 비교 (손실 감소율)
    plt.subplot(2, 2, 4)
    convergence_scores = []
    for strategy_name in strategy_names:
        result = results[strategy_name]
        # 초기 손실 대비 최종 손실 감소율
        initial_loss = result['val_losses'][0]
        final_loss = result['val_losses'][-1]
        convergence_rate = (initial_loss - final_loss) / initial_loss * 100
        convergence_scores.append(convergence_rate)
    
    bars = plt.bar(strategy_names, convergence_scores, color=[colors[name] for name in strategy_names], alpha=0.7)
    plt.xlabel('학습률 전략')
    plt.ylabel('수렴률 (%)')
    plt.title('학습률 전략별 수렴 속도 비교')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # 값 표시
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('learning_rate_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f'\n🎉 학습률 최적화 비교 완료!')
    print(f'💾 저장된 파일:')
    print(f'  - fixed_model.pth, step_model.pth, plateau_model.pth, cosine_model.pth, onecycle_model.pth, lowlr_model.pth: 각 모델')
    print(f'  - learning_rate_comparison.png: 훈련 과정 비교')
    print(f'  - learning_rate_analysis.png: 종합 분석')
