#!/usr/bin/env python3
"""
GRU 기반 SignGlove 모델
데이터가 적은 상황에 최적화된 모델
"""

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
from torch.utils.data import Dataset, DataLoader

class SignGloveDataset(Dataset):
    """SignGlove 데이터셋"""
    
    def __init__(self, data_path, sequence_length=20, augment=True, transform=None):
        self.data_path = Path(data_path)
        self.sequence_length = sequence_length
        self.augment = augment
        self.transform = transform
        self.scaler = StandardScaler()
        
        # 데이터 로드
        self.data, self.labels = self.load_data()
        
        # 데이터 정규화
        self.normalize_data()
        
    def load_data(self):
        """데이터 로드"""
        print("📊 GRU 모델용 데이터 로딩 중...")
        
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
                                
                                # 시퀀스로 분할 (50% 오버랩)
                                for i in range(0, len(sensor_data), self.sequence_length // 2):
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
    
    def normalize_data(self):
        """데이터 정규화"""
        print("🔧 데이터 정규화 중...")
        
        # 데이터 형태 변경: (samples, sequence, features) -> (samples, features)
        original_shape = self.data.shape
        data_reshaped = self.data.reshape(-1, self.data.shape[-1])
        
        # 정규화
        data_normalized = self.scaler.fit_transform(data_reshaped)
        
        # 원래 형태로 복원
        self.data = data_normalized.reshape(original_shape)
        
        print(f"✅ 정규화 완료: 범위 [{self.data.min():.3f}, {self.data.max():.3f}]")
    
    def augment_sequence(self, sequence):
        """시퀀스 증강"""
        if not self.augment:
            return sequence
        
        # 가우시안 노이즈 추가 (매우 작은 노이즈)
        noise = np.random.normal(0, 0.01, sequence.shape)
        augmented = sequence + noise
        
        return augmented
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sequence = self.data[idx]
        label = self.labels[idx]
        
        # 증강 적용
        if self.augment:
            sequence = self.augment_sequence(sequence)
        
        # 변환 적용
        if self.transform:
            sequence = self.transform(sequence)
        
        return torch.FloatTensor(sequence), torch.LongTensor([label])

class GRUSignGloveModel(nn.Module):
    """GRU 기반 SignGlove 모델"""
    
    def __init__(self, input_size=8, hidden_size=64, num_layers=2, num_classes=24, dropout=0.3):
        super(GRUSignGloveModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # GRU 레이어
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # 어텐션 메커니즘
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2,  # bidirectional
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
        
        # 분류 레이어
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )
        
        # 배치 정규화
        self.batch_norm = nn.BatchNorm1d(hidden_size * 2)
        
        # 파라미터 수 계산
        total_params = sum(p.numel() for p in self.parameters())
        print(f"📊 GRU 모델 파라미터 수: {total_params:,}개")
    
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        batch_size = x.size(0)
        
        # GRU 처리
        gru_out, _ = self.gru(x)
        # gru_out shape: (batch_size, sequence_length, hidden_size * 2)
        
        # 어텐션 적용
        attn_out, _ = self.attention(gru_out, gru_out, gru_out)
        
        # 시퀀스 평균 풀링
        pooled = torch.mean(attn_out, dim=1)
        # pooled shape: (batch_size, hidden_size * 2)
        
        # 배치 정규화
        pooled = self.batch_norm(pooled)
        
        # 분류
        output = self.classifier(pooled)
        
        return output

class GRUTrainer:
    """GRU 모델 훈련기"""
    
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=5
        )
        
        print(f"🚀 GRU 훈련기 초기화 완료 (디바이스: {device})")
    
    def train_epoch(self, train_loader):
        """한 에포크 훈련"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.squeeze().to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            
            # 그래디언트 클리핑
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            if batch_idx % 10 == 0:
                print(f"  배치 {batch_idx}/{len(train_loader)}: "
                      f"손실 {loss.item():.4f}, 정확도 {100.*correct/total:.2f}%")
        
        return total_loss / len(train_loader), 100. * correct / total
    
    def validate(self, val_loader):
        """검증"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.squeeze().to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        return total_loss / len(val_loader), 100. * correct / total
    
    def train(self, train_loader, val_loader, epochs=50, early_stopping_patience=10):
        """전체 훈련"""
        print(f"🎯 GRU 모델 훈련 시작 (에포크: {epochs})")
        print("=" * 60)
        
        best_val_acc = 0
        patience_counter = 0
        train_losses = []
        train_accs = []
        val_losses = []
        val_accs = []
        
        for epoch in range(epochs):
            print(f"\n📈 에포크 {epoch+1}/{epochs}")
            print("-" * 40)
            
            # 훈련
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # 검증
            val_loss, val_acc = self.validate(val_loader)
            
            # 학습률 스케줄링
            self.scheduler.step(val_acc)
            
            # 결과 저장
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            val_losses.append(val_loss)
            val_accs.append(val_acc)
            
            print(f"✅ 훈련: 손실 {train_loss:.4f}, 정확도 {train_acc:.2f}%")
            print(f"✅ 검증: 손실 {val_loss:.4f}, 정확도 {val_acc:.2f}%")
            
            # 최고 성능 모델 저장
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                torch.save(self.model.state_dict(), 'best_gru_model.pth')
                print(f"🏆 새로운 최고 성능! 정확도: {val_acc:.2f}%")
            else:
                patience_counter += 1
                print(f"⏳ 조기 종료 카운터: {patience_counter}/{early_stopping_patience}")
            
            # 조기 종료
            if patience_counter >= early_stopping_patience:
                print(f"🛑 조기 종료: {early_stopping_patience} 에포크 동안 개선 없음")
                break
        
        # 훈련 결과 시각화
        self.plot_training_history(train_losses, train_accs, val_losses, val_accs)
        
        return best_val_acc
    
    def plot_training_history(self, train_losses, train_accs, val_losses, val_accs):
        """훈련 히스토리 시각화"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # 손실 그래프
        ax1.plot(train_losses, label='훈련 손실', color='blue')
        ax1.plot(val_losses, label='검증 손실', color='red')
        ax1.set_title('GRU 모델 훈련 손실')
        ax1.set_xlabel('에포크')
        ax1.set_ylabel('손실')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 정확도 그래프
        ax2.plot(train_accs, label='훈련 정확도', color='blue')
        ax2.plot(val_accs, label='검증 정확도', color='red')
        ax2.set_title('GRU 모델 훈련 정확도')
        ax2.set_xlabel('에포크')
        ax2.set_ylabel('정확도 (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('gru_training_history.png', dpi=300, bbox_inches='tight')
        print("📊 훈련 히스토리가 'gru_training_history.png'에 저장되었습니다.")

def main():
    """메인 함수"""
    print("🚀 GRU 기반 SignGlove 모델 시작")
    print("=" * 60)
    
    # 데이터 로드
    dataset = SignGloveDataset(
        data_path="../SignGlove_HW/datasets/unified",
        sequence_length=20,
        augment=True
    )
    
    # 데이터 분할
    train_data, val_data, train_labels, val_labels = train_test_split(
        dataset.data, dataset.labels, test_size=0.2, random_state=42, stratify=dataset.labels
    )
    
    # 데이터셋 생성
    train_dataset = SignGloveDataset(
        data_path="../SignGlove_HW/datasets/unified",
        sequence_length=20,
        augment=True
    )
    train_dataset.data = train_data
    train_dataset.labels = train_labels
    
    val_dataset = SignGloveDataset(
        data_path="../SignGlove_HW/datasets/unified",
        sequence_length=20,
        augment=False
    )
    val_dataset.data = val_data
    val_dataset.labels = val_labels
    
    # 데이터 로더 생성
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)
    
    print(f"📊 훈련 데이터: {len(train_dataset)}개")
    print(f"📊 검증 데이터: {len(val_dataset)}개")
    
    # 모델 생성
    model = GRUSignGloveModel(
        input_size=8,
        hidden_size=64,
        num_layers=2,
        num_classes=24,
        dropout=0.3
    )
    
    # 훈련기 생성 및 훈련
    trainer = GRUTrainer(model)
    best_acc = trainer.train(train_loader, val_loader, epochs=50)
    
    print(f"\n🎉 훈련 완료! 최고 검증 정확도: {best_acc:.2f}%")

if __name__ == "__main__":
    main()
