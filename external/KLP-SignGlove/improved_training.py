import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import h5py
import os
from pathlib import Path
import sys
import random
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append('models')
from deep_learning import DeepLearningPipeline

class ImprovedDataLoader:
    """개선된 데이터 로더"""
    
    def __init__(self, data_path, sequence_length=20, augment=True):
        self.data_path = Path(data_path)
        self.sequence_length = sequence_length
        self.augment = augment
        self.scaler = StandardScaler()
        
    def load_all_data(self):
        """전체 데이터셋 로드"""
        print("📊 전체 데이터셋 로딩 중...")
        
        data = []
        labels = []
        class_names = ['ㄱ', 'ㄴ', 'ㄷ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅅ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ',
                       'ㅏ', 'ㅑ', 'ㅓ', 'ㅕ', 'ㅗ', 'ㅛ', 'ㅜ', 'ㅠ', 'ㅡ', 'ㅣ']
        
        class_counts = defaultdict(int)
        
        for class_idx, class_name in enumerate(class_names):
            class_dir = self.data_path / class_name
            if not class_dir.exists():
                continue
                
            print(f"  {class_name} 클래스 로딩 중...")
            
            for sub_dir in class_dir.iterdir():
                if sub_dir.is_dir():
                    for h5_file in sub_dir.glob("*.h5"):
                        try:
                            with h5py.File(h5_file, 'r') as f:
                                sensor_data = f['sensor_data'][:]
                                
                                # 시퀀스로 분할
                                for i in range(0, len(sensor_data), self.sequence_length // 2):  # 50% 오버랩
                                    if i + self.sequence_length <= len(sensor_data):
                                        sequence = sensor_data[i:i+self.sequence_length]
                                        data.append(sequence)
                                        labels.append(class_idx)
                                        class_counts[class_name] += 1
                        except Exception as e:
                            print(f"⚠️ 파일 로드 오류 {h5_file}: {e}")
                            continue
        
        print(f"\n📊 데이터 로딩 완료:")
        for class_name in class_names:
            if class_name in class_counts:
                print(f"  {class_name}: {class_counts[class_name]}개 시퀀스")
        
        return np.array(data), np.array(labels)
    
    def augment_data(self, data, labels):
        """데이터 증강"""
        if not self.augment:
            return data, labels
        
        print("🔄 데이터 증강 중...")
        
        augmented_data = []
        augmented_labels = []
        
        # 클래스별 샘플 수 계산
        class_counts = defaultdict(int)
        for label in labels:
            class_counts[label] += 1
        
        # 목표 샘플 수 (가장 많은 클래스의 80%)
        max_samples = max(class_counts.values())
        target_samples = int(max_samples * 0.8)
        
        for class_idx in range(24):
            current_count = class_counts[class_idx]
            if current_count < target_samples:
                needed = target_samples - current_count
                print(f"  클래스 {class_idx}: {current_count} → {target_samples} (+{needed})")
                
                # 해당 클래스의 데이터 찾기
                class_indices = np.where(labels == class_idx)[0]
                
                for _ in range(needed):
                    # 랜덤하게 원본 데이터 선택
                    idx = np.random.choice(class_indices)
                    original_sequence = data[idx].copy()
                    
                    # 증강 기법 적용
                    augmented_sequence = self._apply_augmentation(original_sequence)
                    
                    augmented_data.append(augmented_sequence)
                    augmented_labels.append(class_idx)
        
        # 원본 데이터와 증강 데이터 결합
        if len(augmented_data) > 0:
            augmented_data_array = np.array(augmented_data)
            final_data = np.vstack([data, augmented_data_array])
            final_labels = np.concatenate([labels, np.array(augmented_labels)])
        else:
            final_data = data
            final_labels = labels
        
        print(f"📊 증강 후 데이터: {len(final_data)}개 시퀀스")
        return final_data, final_labels
    
    def _apply_augmentation(self, sequence):
        """단일 시퀀스에 증강 적용"""
        augmented = sequence.copy()
        
        # 1. 노이즈 추가 (가우시안)
        if random.random() < 0.3:
            noise = np.random.normal(0, 0.01, augmented.shape)
            augmented += noise
        
        # 2. 시간 이동 (Time shifting)
        if random.random() < 0.3:
            shift = random.randint(-2, 2)
            if shift > 0:
                augmented = np.roll(augmented, shift, axis=0)
                augmented[:shift] = augmented[shift]
            elif shift < 0:
                augmented = np.roll(augmented, shift, axis=0)
                augmented[shift:] = augmented[shift-1]
        
        # 3. 스케일링 (Scale)
        if random.random() < 0.2:
            scale_factor = random.uniform(0.9, 1.1)
            augmented *= scale_factor
        
        # 4. 마스킹 (Random masking)
        if random.random() < 0.1:
            mask_ratio = random.uniform(0.05, 0.15)
            mask_size = int(self.sequence_length * mask_ratio)
            mask_start = random.randint(0, self.sequence_length - mask_size)
            augmented[mask_start:mask_start+mask_size] = 0
        
        return augmented
    
    def preprocess_data(self, data, labels):
        """데이터 전처리"""
        print("🔧 데이터 전처리 중...")
        
        # 1. 정규화
        original_shape = data.shape
        data_reshaped = data.reshape(-1, data.shape[-1])
        data_normalized = self.scaler.fit_transform(data_reshaped)
        data = data_normalized.reshape(original_shape)
        
        # 2. 클래스별 샘플 수 확인
        class_counts = defaultdict(int)
        for label in labels:
            class_counts[label] += 1
        
        print(f"📊 전처리 후 클래스별 샘플 수:")
        for class_idx in range(24):
            if class_idx in class_counts:
                print(f"  클래스 {class_idx}: {class_counts[class_idx]}개")
        
        return data, labels

class ImprovedModel(nn.Module):
    """개선된 모델 아키텍처"""
    
    def __init__(self, input_features=8, sequence_length=20, num_classes=24, 
                 hidden_dim=128, num_layers=3, dropout=0.3):
        super(ImprovedModel, self).__init__()
        
        self.input_features = input_features
        self.sequence_length = sequence_length
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        
        # 1. 더 깊은 CNN 레이어
        self.conv1 = nn.Conv1d(input_features, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(256, 512, kernel_size=3, padding=1)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(512)
        
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(dropout)
        
        # 2. 양방향 LSTM
        self.lstm = nn.LSTM(512, hidden_dim, num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0,
                           bidirectional=True)
        
        # 3. 어텐션 메커니즘
        self.attention = nn.Linear(hidden_dim * 2, 1)
        
        # 4. 분류 헤드
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # 5. 특징 추출기
        self.feature_extractor = nn.Linear(hidden_dim * 2, 64)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # CNN 특징 추출
        x = x.transpose(1, 2)  # (batch, features, sequence)
        
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.dropout(x)
        
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.dropout(x)
        
        x = torch.relu(self.bn3(self.conv3(x)))
        x = self.dropout(x)
        
        x = torch.relu(self.bn4(self.conv4(x)))
        x = self.dropout(x)
        
        # LSTM을 위한 차원 변환
        x = x.transpose(1, 2)  # (batch, sequence, features)
        
        # LSTM 처리
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # 어텐션 메커니즘
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        attended_output = torch.sum(attention_weights * lstm_out, dim=1)
        
        # 분류
        class_logits = self.classifier(attended_output)
        
        # 특징 추출
        features = self.feature_extractor(attended_output)
        
        return {
            'class_logits': class_logits,
            'features': features,
            'attention_weights': attention_weights
        }

class ImprovedTrainer:
    """개선된 훈련기"""
    
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        
        # 1. 옵티마이저 (AdamW 사용)
        self.optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        
        # 2. 학습률 스케줄러
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=5
        )
        
        # 3. 손실 함수 (Focal Loss + Label Smoothing)
        self.criterion = self._create_criterion()
        
        # 4. 조기 종료
        self.best_val_acc = 0
        self.patience = 15
        self.patience_counter = 0
        
    def _create_criterion(self):
        """개선된 손실 함수 생성"""
        # Focal Loss 구현
        class FocalLoss(nn.Module):
            def __init__(self, alpha=1, gamma=2, label_smoothing=0.1):
                super(FocalLoss, self).__init__()
                self.alpha = alpha
                self.gamma = gamma
                self.label_smoothing = label_smoothing
                
            def forward(self, inputs, targets):
                ce_loss = nn.functional.cross_entropy(
                    inputs, targets, 
                    label_smoothing=self.label_smoothing,
                    reduction='none'
                )
                pt = torch.exp(-ce_loss)
                focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
                return focal_loss.mean()
        
        return FocalLoss(alpha=1, gamma=2, label_smoothing=0.1)
    
    def train_epoch(self, train_loader):
        """한 에포크 훈련"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            
            output = self.model(data)
            loss = self.criterion(output['class_logits'], target)
            
            loss.backward()
            
            # 그래디언트 클리핑
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            
            pred = output['class_logits'].argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            
            if batch_idx % 50 == 0:
                print(f"  Batch {batch_idx}/{len(train_loader)} | "
                      f"Loss: {loss.item():.4f} | "
                      f"Acc: {100.*correct/total:.2f}%")
        
        return total_loss / len(train_loader), correct / total
    
    def validate_epoch(self, val_loader):
        """검증"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                loss = self.criterion(output['class_logits'], target)
                
                total_loss += loss.item()
                
                pred = output['class_logits'].argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
        
        return total_loss / len(val_loader), correct / total
    
    def train(self, train_loader, val_loader, epochs=100):
        """전체 훈련 과정"""
        print(f"🚀 개선된 모델 훈련 시작 (디바이스: {self.device})")
        print(f"📊 훈련 데이터: {len(train_loader.dataset)}개")
        print(f"📊 검증 데이터: {len(val_loader.dataset)}개")
        print(f"🔧 모델 파라미터: {sum(p.numel() for p in self.model.parameters()):,}개")
        
        history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'lr': []
        }
        
        for epoch in range(epochs):
            print(f"\n🔄 Epoch {epoch+1}/{epochs}")
            
            # 훈련
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # 검증
            val_loss, val_acc = self.validate_epoch(val_loader)
            
            # 학습률 스케줄러 업데이트
            self.scheduler.step(val_acc)
            
            # 히스토리 저장
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['lr'].append(self.optimizer.param_groups[0]['lr'])
            
            print(f"Epoch {epoch+1:3d} | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
                  f"LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # 조기 종료 체크
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.patience_counter = 0
                # 최고 모델 저장
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_accuracy': val_acc,
                    'epoch': epoch
                }, 'improved_model.pth')
                print(f"🎯 새로운 최고 정확도: {val_acc:.4f}")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    print(f"🛑 조기 종료: {epoch+1} 에포크에서 중단")
                    break
        
        return history

def main():
    """메인 훈련 함수"""
    print("🎯 개선된 SignGlove 모델 훈련 시작!")
    
    # 1. 데이터 로드
    data_loader = ImprovedDataLoader(
        data_path='/home/billy/25-1kp/SignGlove/external/SignGlove_HW/datasets/unified',
        sequence_length=20,
        augment=True
    )
    
    data, labels = data_loader.load_all_data()
    
    if len(data) == 0:
        print("❌ 데이터를 로드할 수 없습니다!")
        return
    
    # 2. 데이터 증강
    data, labels = data_loader.augment_data(data, labels)
    
    # 3. 데이터 전처리
    data, labels = data_loader.preprocess_data(data, labels)
    
    # 4. 데이터 분할
    X_train, X_val, y_train, y_val = train_test_split(
        data, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # 5. 데이터로더 생성
    train_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_train), torch.LongTensor(y_train)
    )
    val_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_val), torch.LongTensor(y_val)
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=32, shuffle=True, num_workers=2
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=32, shuffle=False, num_workers=2
    )
    
    # 6. 모델 생성
    model = ImprovedModel(
        input_features=8,
        sequence_length=20,
        num_classes=24,
        hidden_dim=128,
        num_layers=3,
        dropout=0.3
    )
    
    # 7. 훈련
    trainer = ImprovedTrainer(model)
    history = trainer.train(train_loader, val_loader, epochs=100)
    
    # 8. 결과 시각화
    plot_training_history(history)
    
    print(f"\n🎉 훈련 완료! 최고 검증 정확도: {trainer.best_val_acc:.4f}")

def plot_training_history(history):
    """훈련 히스토리 시각화"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 손실 그래프
    axes[0, 0].plot(history['train_loss'], label='Train Loss')
    axes[0, 0].plot(history['val_loss'], label='Val Loss')
    axes[0, 0].set_title('Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # 정확도 그래프
    axes[0, 1].plot(history['train_acc'], label='Train Acc')
    axes[0, 1].plot(history['val_acc'], label='Val Acc')
    axes[0, 1].set_title('Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # 학습률 그래프
    axes[1, 0].plot(history['lr'])
    axes[1, 0].set_title('Learning Rate')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Learning Rate')
    axes[1, 0].grid(True)
    
    # 정확도 차이 그래프
    acc_diff = [t - v for t, v in zip(history['train_acc'], history['val_acc'])]
    axes[1, 1].plot(acc_diff)
    axes[1, 1].set_title('Train-Val Accuracy Difference')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy Difference')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('improved_training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()
