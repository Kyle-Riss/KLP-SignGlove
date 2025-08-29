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

class SimpleRobustDataLoader:
    """간단하고 안정적인 데이터 로더"""
    
    def __init__(self, data_path, sequence_length=20):
        self.data_path = Path(data_path)
        self.sequence_length = sequence_length
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
                                
                                # 시퀀스로 분할 (오버랩 없이)
                                for i in range(0, len(sensor_data), self.sequence_length):
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

class SimpleRobustModel(nn.Module):
    """간단하고 안정적인 모델 아키텍처"""
    
    def __init__(self, input_features=8, sequence_length=20, num_classes=24, 
                 hidden_dim=64, num_layers=2, dropout=0.2):
        super(SimpleRobustModel, self).__init__()
        
        self.input_features = input_features
        self.sequence_length = sequence_length
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        
        # 1. 간단한 CNN 레이어 (과적합 방지)
        self.conv1 = nn.Conv1d(input_features, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        
        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(64)
        
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(dropout)
        
        # 2. 단순한 LSTM (양방향 사용하지 않음)
        self.lstm = nn.LSTM(64, hidden_dim, num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        # 3. 간단한 분류 헤드
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
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
        
        # LSTM을 위한 차원 변환
        x = x.transpose(1, 2)  # (batch, sequence, features)
        
        # LSTM 처리
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # 마지막 시퀀스 출력 사용
        last_output = lstm_out[:, -1, :]
        
        # 분류
        class_logits = self.classifier(last_output)
        
        return {
            'class_logits': class_logits,
            'features': last_output
        }

class SimpleRobustTrainer:
    """간단하고 안정적인 훈련기"""
    
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        
        # 1. 옵티마이저 (SGD 사용 - 더 안정적)
        self.optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.001)
        
        # 2. 학습률 스케줄러
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.5)
        
        # 3. 손실 함수 (단순한 CrossEntropy)
        self.criterion = nn.CrossEntropyLoss()
        
        # 4. 조기 종료
        self.best_val_acc = 0
        self.patience = 10
        self.patience_counter = 0
        
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
            
            if batch_idx % 100 == 0:
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
    
    def train(self, train_loader, val_loader, epochs=50):
        """전체 훈련 과정"""
        print(f"🚀 간단하고 안정적인 모델 훈련 시작 (디바이스: {self.device})")
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
            self.scheduler.step()
            
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
                }, 'simple_robust_model.pth')
                print(f"🎯 새로운 최고 정확도: {val_acc:.4f}")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    print(f"🛑 조기 종료: {epoch+1} 에포크에서 중단")
                    break
        
        return history

def main():
    """메인 훈련 함수"""
    print("🎯 간단하고 안정적인 SignGlove 모델 훈련 시작!")
    
    # 1. 데이터 로드
    data_loader = SimpleRobustDataLoader(
        data_path='/home/billy/25-1kp/SignGlove/external/SignGlove_HW/datasets/unified',
        sequence_length=20
    )
    
    data, labels = data_loader.load_all_data()
    
    if len(data) == 0:
        print("❌ 데이터를 로드할 수 없습니다!")
        return
    
    # 2. 데이터 전처리
    data, labels = data_loader.preprocess_data(data, labels)
    
    # 3. 데이터 분할
    X_train, X_val, y_train, y_val = train_test_split(
        data, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # 4. 데이터로더 생성
    train_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_train), torch.LongTensor(y_train)
    )
    val_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_val), torch.LongTensor(y_val)
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=64, shuffle=True, num_workers=2
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=64, shuffle=False, num_workers=2
    )
    
    # 5. 모델 생성
    model = SimpleRobustModel(
        input_features=8,
        sequence_length=20,
        num_classes=24,
        hidden_dim=64,
        num_layers=2,
        dropout=0.2
    )
    
    # 6. 훈련
    trainer = SimpleRobustTrainer(model)
    history = trainer.train(train_loader, val_loader, epochs=50)
    
    # 7. 결과 시각화
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
    plt.savefig('simple_robust_training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()



