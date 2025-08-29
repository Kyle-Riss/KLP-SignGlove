#!/usr/bin/env python3
"""
과적합 방지 GRU 모델
과적합을 줄이기 위한 다양한 기법들을 적용한 모델
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import h5py
import os
from pathlib import Path
import sys
from collections import defaultdict
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset, DataLoader
import random
import json
from sklearn.model_selection import train_test_split

class AntiOverfittingGRUModel(nn.Module):
    """과적합 방지 GRU 모델"""
    
    def __init__(self, input_size=8, hidden_size=32, num_layers=1, num_classes=24, dropout=0.5):
        super(AntiOverfittingGRUModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # 더 작은 GRU 레이어 (파라미터 수 감소)
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False  # 단방향으로 변경하여 파라미터 감소
        )
        
        # 더 작은 어텐션 메커니즘
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=2,  # 헤드 수 감소
            dropout=dropout,
            batch_first=True
        )
        
        # 더 간단한 분류 레이어
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )
        
        # 배치 정규화
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        
        # L2 정규화를 위한 weight decay 설정
        self.weight_decay = 0.1
        
        # 파라미터 수 계산
        total_params = sum(p.numel() for p in self.parameters())
        print(f"📊 과적합 방지 GRU 모델 파라미터 수: {total_params:,}개")
    
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        batch_size = x.size(0)
        
        # GRU 처리
        gru_out, _ = self.gru(x)
        # gru_out shape: (batch_size, sequence_length, hidden_size)
        
        # 어텐션 적용
        attn_out, _ = self.attention(gru_out, gru_out, gru_out)
        
        # 시퀀스 평균 풀링
        pooled = torch.mean(attn_out, dim=1)
        # pooled shape: (batch_size, hidden_size)
        
        # 배치 정규화
        pooled = self.batch_norm(pooled)
        
        # 분류
        output = self.classifier(pooled)
        
        return output

class AntiOverfittingDataset(Dataset):
    """과적합 방지 데이터셋"""
    
    def __init__(self, data_path, sequence_length=20, validation_type='user_split', random_seed=42):
        self.data_path = Path(data_path)
        self.sequence_length = sequence_length
        self.scaler = StandardScaler()
        
        # 클래스 목록
        self.class_names = ['ㄱ', 'ㄴ', 'ㄷ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅅ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ',
                           'ㅏ', 'ㅑ', 'ㅓ', 'ㅕ', 'ㅗ', 'ㅛ', 'ㅜ', 'ㅠ', 'ㅡ', 'ㅣ']
        
        # 랜덤 시드 설정
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        # 데이터 로드
        self.data, self.labels, self.class_indices = self.load_data(validation_type)
        
        # 데이터 정규화
        self.normalize_data()
        
    def load_data(self, validation_type):
        """검증 타입에 따른 데이터 로드"""
        print(f"📊 {validation_type} 데이터 로딩 중...")
        
        data = []
        labels = []
        class_indices = defaultdict(list)
        
        for class_idx, class_name in enumerate(self.class_names):
            print(f"  {class_name} 클래스 데이터 로딩 중...")
            
            class_dir = self.data_path / class_name
            if not class_dir.exists():
                continue
            
            class_data_count = 0
            
            for sub_dir in class_dir.iterdir():
                if sub_dir.is_dir():
                    h5_files = list(sub_dir.glob("*.h5"))
                    
                    if validation_type == 'user_split':
                        # 사용자별 분할 - 사용자 1-4만 훈련용
                        user_id = sub_dir.name
                        if user_id == '5':  # 테스트용 사용자 제외
                            continue
                    elif validation_type == 'time_split':
                        # 시간별 분할 - 처음 80%만 훈련용
                        h5_files.sort()
                        h5_files = h5_files[:int(len(h5_files) * 0.8)]
                    elif validation_type == 'no_augmentation':
                        # 증강 없는 데이터 - 랜덤하게 일부만 선택
                        random.shuffle(h5_files)
                        h5_files = h5_files[:int(len(h5_files) * 0.7)]
                    
                    for h5_file in h5_files:
                        try:
                            with h5py.File(h5_file, 'r') as f:
                                sensor_data = f['sensor_data'][:]
                                
                                # 시퀀스로 분할 (50% 오버랩 없이)
                                for i in range(0, len(sensor_data), self.sequence_length):
                                    if i + self.sequence_length <= len(sensor_data):
                                        sequence = sensor_data[i:i+self.sequence_length]
                                        data.append(sequence)
                                        labels.append(class_idx)
                                        class_indices[class_name].append(len(data) - 1)
                                        class_data_count += 1
                        except Exception as e:
                            print(f"⚠️ 파일 로드 오류 {h5_file}: {e}")
                            continue
            
            print(f"    {class_name}: {class_data_count}개 시퀀스")
        
        return np.array(data), np.array(labels), class_indices
    
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
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sequence = self.data[idx]
        label = self.labels[idx]
        return torch.FloatTensor(sequence), torch.LongTensor([label])

class AntiOverfittingTrainer:
    """과적합 방지 훈련기"""
    
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model
        self.device = device
        self.model.to(device)
        
        # 과적합 방지를 위한 설정
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(
            model.parameters(), 
            lr=0.001,  # 더 낮은 학습률
            weight_decay=0.1  # 강한 L2 정규화
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=3
        )
        
        print(f"🚀 과적합 방지 훈련기 초기화 완료 (디바이스: {device})")
    
    def train_epoch(self, train_loader, epoch):
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
            
            # L2 정규화 추가
            l2_lambda = 0.01
            l2_norm = sum(p.pow(2.0).sum() for p in self.model.parameters())
            loss = loss + l2_lambda * l2_norm
            
            loss.backward()
            
            # 그래디언트 클리핑
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            if batch_idx % 20 == 0:
                print(f'  배치 {batch_idx}/{len(train_loader)}: 손실={loss.item():.4f}')
        
        accuracy = 100. * correct / total
        avg_loss = total_loss / len(train_loader)
        
        return avg_loss, accuracy
    
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
        
        accuracy = 100. * correct / total
        avg_loss = total_loss / len(val_loader)
        
        return avg_loss, accuracy
    
    def train(self, train_loader, val_loader, epochs=50, early_stopping_patience=10):
        """전체 훈련 과정"""
        print("🚀 과적합 방지 훈련 시작")
        print("=" * 60)
        
        best_val_accuracy = 0
        patience_counter = 0
        train_losses = []
        train_accuracies = []
        val_losses = []
        val_accuracies = []
        
        for epoch in range(epochs):
            # 훈련
            train_loss, train_acc = self.train_epoch(train_loader, epoch)
            
            # 검증
            val_loss, val_acc = self.validate(val_loader)
            
            # 스케줄러 업데이트
            self.scheduler.step(val_acc)
            
            # 결과 저장
            train_losses.append(train_loss)
            train_accuracies.append(train_acc)
            val_losses.append(val_loss)
            val_accuracies.append(val_acc)
            
            print(f'에포크 {epoch+1}/{epochs}:')
            print(f'  훈련 - 손실: {train_loss:.4f}, 정확도: {train_acc:.2f}%')
            print(f'  검증 - 손실: {val_loss:.4f}, 정확도: {val_acc:.2f}%')
            
            # 최고 성능 모델 저장
            if val_acc > best_val_accuracy:
                best_val_accuracy = val_acc
                patience_counter = 0
                torch.save(self.model.state_dict(), 'best_anti_overfitting_model.pth')
                print(f'  ✅ 새로운 최고 검증 정확도: {val_acc:.2f}%')
            else:
                patience_counter += 1
                print(f'  ⏳ 조기 종료 카운터: {patience_counter}/{early_stopping_patience}')
            
            # 조기 종료
            if patience_counter >= early_stopping_patience:
                print(f'  🛑 조기 종료: {early_stopping_patience} 에포크 동안 개선 없음')
                break
            
            print()
        
        # 훈련 곡선 시각화
        self.plot_training_curves(train_losses, train_accuracies, val_losses, val_accuracies)
        
        return best_val_accuracy
    
    def plot_training_curves(self, train_losses, train_accuracies, val_losses, val_accuracies):
        """훈련 곡선 시각화"""
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='훈련 손실')
        plt.plot(val_losses, label='검증 손실')
        plt.xlabel('에포크')
        plt.ylabel('손실')
        plt.title('훈련 및 검증 손실')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(train_accuracies, label='훈련 정확도')
        plt.plot(val_accuracies, label='검증 정확도')
        plt.xlabel('에포크')
        plt.ylabel('정확도 (%)')
        plt.title('훈련 및 검증 정확도')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('anti_overfitting_training_curves.png', dpi=300, bbox_inches='tight')
        print("📊 훈련 곡선이 'anti_overfitting_training_curves.png'에 저장되었습니다.")

def main():
    """메인 함수"""
    print("🚀 과적합 방지 GRU 모델 훈련 시작")
    print("=" * 60)
    
    # 검증 타입들
    validation_types = ['user_split', 'time_split', 'no_augmentation']
    results = []
    
    for validation_type in validation_types:
        print(f"\n🔍 {validation_type} 훈련 시작")
        print("-" * 40)
        
        # 데이터 로드
        dataset = AntiOverfittingDataset(
            data_path="../SignGlove_HW/datasets/unified",
            sequence_length=20,
            validation_type=validation_type
        )
        
        # 훈련/검증 분할
        train_indices, val_indices = train_test_split(
            range(len(dataset)), 
            test_size=0.2, 
            random_state=42,
            stratify=dataset.labels
        )
        
        train_dataset = torch.utils.data.Subset(dataset, train_indices)
        val_dataset = torch.utils.data.Subset(dataset, val_indices)
        
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)
        
        print(f"📊 훈련 데이터: {len(train_dataset)}개 시퀀스")
        print(f"📊 검증 데이터: {len(val_dataset)}개 시퀀스")
        
        # 모델 생성
        model = AntiOverfittingGRUModel(
            input_size=8,
            hidden_size=32,  # 더 작은 hidden size
            num_layers=1,    # 더 적은 레이어
            num_classes=24,
            dropout=0.5      # 더 높은 드롭아웃
        )
        
        # 훈련기 생성
        trainer = AntiOverfittingTrainer(model)
        
        # 훈련
        best_val_accuracy = trainer.train(
            train_loader, 
            val_loader, 
            epochs=30,  # 더 적은 에포크
            early_stopping_patience=5
        )
        
        results.append({
            'validation_type': validation_type,
            'best_val_accuracy': best_val_accuracy,
            'model_params': sum(p.numel() for p in model.parameters())
        })
        
        print(f"✅ {validation_type} 훈련 완료: 최고 검증 정확도 {best_val_accuracy:.2f}%")
    
    # 결과 요약
    print("\n🎉 과적합 방지 훈련 완료!")
    print("=" * 60)
    
    print("\n📊 훈련 결과 요약:")
    print("-" * 40)
    
    for result in results:
        print(f"\n{result['validation_type']}:")
        print(f"  최고 검증 정확도: {result['best_val_accuracy']:.2f}%")
        print(f"  모델 파라미터 수: {result['model_params']:,}개")
    
    # 결과 저장
    with open('anti_overfitting_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n📁 결과가 'anti_overfitting_results.json'에 저장되었습니다.")

if __name__ == "__main__":
    main()



