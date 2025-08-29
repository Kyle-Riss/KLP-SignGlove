#!/usr/bin/env python3
"""
과적합 방지 모음 성능 개선 스크립트
더 견고하고 일반화 성능이 좋은 모델을 훈련
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import h5py
import os
from pathlib import Path
import sys
from collections import defaultdict
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from scipy.ndimage import gaussian_filter1d
import random
import json
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

class VowelRobustDataset(Dataset):
    """과적합 방지를 위한 견고한 데이터셋"""
    
    def __init__(self, data_path, sequence_length=20, apply_vowel_enhancement=True, augmentation=True):
        self.data_path = Path(data_path)
        self.sequence_length = sequence_length
        self.apply_vowel_enhancement = apply_vowel_enhancement
        self.augmentation = augmentation
        
        # 클래스 분류
        self.consonants = ['ㄱ', 'ㄴ', 'ㄷ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅅ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
        self.vowels = ['ㅏ', 'ㅑ', 'ㅓ', 'ㅕ', 'ㅗ', 'ㅛ', 'ㅜ', 'ㅠ', 'ㅡ', 'ㅣ']
        self.all_classes = self.consonants + self.vowels
        
        # 스케일러
        self.scaler = RobustScaler()
        
        # 데이터 로드
        self.data, self.labels, self.class_indices = self.load_data()
        
        # 모음 향상 전처리 적용
        if self.apply_vowel_enhancement:
            self.enhance_vowel_data()
        
        # 데이터 정규화
        self.normalize_data()
        
    def load_data(self):
        """데이터 로드"""
        print("📊 견고한 데이터셋 로딩 중...")
        
        data = []
        labels = []
        class_indices = defaultdict(list)
        
        for class_idx, class_name in enumerate(self.all_classes):
            print(f"  {class_name} 클래스 데이터 로딩 중...")
            
            class_dir = self.data_path / class_name
            if not class_dir.exists():
                continue
            
            class_data_count = 0
            
            for sub_dir in class_dir.iterdir():
                if sub_dir.is_dir():
                    h5_files = list(sub_dir.glob("*.h5"))
                    
                    for h5_file in h5_files:
                        try:
                            with h5py.File(h5_file, 'r') as f:
                                sensor_data = f['sensor_data'][:]
                                
                                # 시퀀스로 분할
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
    
    def enhance_vowel_data(self):
        """모음 데이터 향상 전처리 (더 보수적으로)"""
        print("🔧 모음 데이터 향상 전처리 적용 중...")
        
        enhanced_data = []
        
        for i, (sequence, label) in enumerate(zip(self.data, self.labels)):
            class_name = self.all_classes[label]
            
            if class_name in self.vowels:
                # 모음 데이터에 보수적인 향상 전처리 적용
                enhanced_sequence = self.apply_conservative_enhancement(sequence)
                enhanced_data.append(enhanced_sequence)
            else:
                # 자음 데이터는 그대로 유지
                enhanced_data.append(sequence)
        
        self.data = np.array(enhanced_data)
        print("✅ 모음 데이터 향상 전처리 완료")
    
    def apply_conservative_enhancement(self, sequence):
        """보수적인 모음 데이터 향상 기법"""
        # 1. 가벼운 노이즈 제거 (더 작은 sigma)
        sequence_filtered = gaussian_filter1d(sequence, sigma=0.3, axis=0)
        
        # 2. 가벼운 스무딩
        window_size = 2
        sequence_smoothed = np.zeros_like(sequence_filtered)
        for i in range(len(sequence_filtered)):
            start = max(0, i - window_size // 2)
            end = min(len(sequence_filtered), i + window_size // 2 + 1)
            sequence_smoothed[i] = np.mean(sequence_filtered[start:end], axis=0)
        
        # 3. 변동성 감소 (더 보수적으로)
        sequence_normalized = sequence_smoothed.copy()
        for sensor_idx in range(sequence_smoothed.shape[1]):
            sensor_data = sequence_smoothed[:, sensor_idx]
            mean_val = np.mean(sensor_data)
            std_val = np.std(sensor_data)
            if std_val > 0:
                # 변동성을 10%만 감소 (더 보수적)
                compression_factor = 0.9
                sequence_normalized[:, sensor_idx] = mean_val + (sensor_data - mean_val) * compression_factor
        
        return sequence_normalized
    
    def normalize_data(self):
        """데이터 정규화"""
        print("🔧 데이터 정규화 중...")
        
        original_shape = self.data.shape
        data_reshaped = self.data.reshape(-1, self.data.shape[-1])
        
        data_normalized = self.scaler.fit_transform(data_reshaped)
        self.data = data_normalized.reshape(original_shape)
        
        print(f"✅ 정규화 완료: 범위 [{self.data.min():.3f}, {self.data.max():.3f}]")
    
    def augment_data(self, sequence):
        """데이터 증강"""
        if not self.augmentation:
            return sequence
        
        # 랜덤 노이즈 추가
        if random.random() < 0.3:
            noise = np.random.normal(0, 0.01, sequence.shape)
            sequence = sequence + noise
        
        # 랜덤 시간 이동
        if random.random() < 0.2:
            shift = random.randint(-2, 2)
            if shift > 0:
                sequence = np.pad(sequence, ((shift, 0), (0, 0)), mode='edge')[:-shift]
            elif shift < 0:
                sequence = np.pad(sequence, ((0, -shift), (0, 0)), mode='edge')[-shift:]
        
        return sequence
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sequence = self.data[idx]
        label = self.labels[idx]
        
        # 데이터 증강 적용
        sequence = self.augment_data(sequence)
        
        return torch.FloatTensor(sequence), torch.LongTensor([label])

class VowelRobustGRU(nn.Module):
    """과적합 방지를 위한 견고한 GRU 모델"""
    
    def __init__(self, input_size=8, hidden_size=64, num_layers=2, num_classes=24, dropout=0.5):
        super(VowelRobustGRU, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # 더 작은 hidden size와 더 높은 dropout
        self.gru = nn.GRU(
            input_size, 
            hidden_size, 
            num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 더 많은 dropout과 정규화
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(hidden_size // 2),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout * 0.8),
            nn.BatchNorm1d(hidden_size // 4),
            nn.Linear(hidden_size // 4, num_classes)
        )
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # GRU 처리
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        out, _ = self.gru(x, h0)
        
        # 마지막 시퀀스 출력
        last_output = out[:, -1, :]
        
        # 분류
        output = self.classifier(last_output)
        
        return output

class VowelBalancedLoss(nn.Module):
    """균형잡힌 손실 함수"""
    
    def __init__(self, num_classes=24, vowel_weight=1.5):  # 가중치를 낮춤
        super(VowelBalancedLoss, self).__init__()
        
        # 모음 클래스 인덱스 (10개)
        vowel_indices = list(range(14, 24))  # 모음은 14-23 인덱스
        
        # 클래스 가중치 설정 (더 보수적으로)
        class_weights = torch.ones(num_classes)
        for idx in vowel_indices:
            class_weights[idx] = vowel_weight
        
        self.register_buffer('class_weights', class_weights)
    
    def forward(self, outputs, targets):
        return nn.functional.cross_entropy(outputs, targets.squeeze(), weight=self.class_weights)

def train_robust_model(model, train_loader, val_loader, num_epochs=30, learning_rate=0.0005):
    """견고한 모델 훈련"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # 더 보수적인 손실 함수
    criterion = VowelBalancedLoss(num_classes=24, vowel_weight=1.5)
    criterion = criterion.to(device)
    
    # 더 낮은 학습률과 더 강한 정규화
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.7)
    
    # 훈련 기록
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    vowel_accuracies = []
    
    # Early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    patience_limit = 8
    
    print(f"🚀 견고한 모음 모델 훈련 시작 (디바이스: {device})")
    print("=" * 60)
    
    for epoch in range(num_epochs):
        # 훈련
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        vowel_correct = 0
        vowel_total = 0
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += targets.size(0)
            train_correct += (predicted == targets.squeeze()).sum().item()
            
            # 모음 정확도 계산
            vowel_mask = (targets.squeeze() >= 14) & (targets.squeeze() < 24)
            if vowel_mask.sum() > 0:
                vowel_correct += (predicted[vowel_mask] == targets.squeeze()[vowel_mask]).sum().item()
                vowel_total += vowel_mask.sum().item()
        
        # 검증
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(device), targets.to(device)
                outputs = model(data)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += targets.size(0)
                val_correct += (predicted == targets.squeeze()).sum().item()
        
        # 평균 계산
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_accuracy = 100 * train_correct / train_total
        val_accuracy = 100 * val_correct / val_total
        vowel_accuracy = 100 * vowel_correct / vowel_total if vowel_total > 0 else 0
        
        # 스케줄러 업데이트
        scheduler.step(val_loss)
        
        # Early stopping 체크
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # 최고 모델 저장
            torch.save(model.state_dict(), 'best_robust_vowel_model.pth')
        else:
            patience_counter += 1
        
        # 기록 저장
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)
        vowel_accuracies.append(vowel_accuracy)
        
        # 출력
        if (epoch + 1) % 3 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}]")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")
            print(f"  Vowel Acc: {vowel_accuracy:.2f}%")
            print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
            print(f"  Patience: {patience_counter}/{patience_limit}")
            print("-" * 40)
        
        # Early stopping
        if patience_counter >= patience_limit:
            print(f"🛑 Early stopping at epoch {epoch+1}")
            break
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies,
        'vowel_accuracies': vowel_accuracies
    }

def evaluate_robust_performance(model, test_loader, class_names):
    """견고한 모델 성능 평가"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for data, targets in test_loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(targets.squeeze().cpu().numpy())
    
    # 모음 클래스만 추출
    vowels = ['ㅏ', 'ㅑ', 'ㅓ', 'ㅕ', 'ㅗ', 'ㅛ', 'ㅜ', 'ㅠ', 'ㅡ', 'ㅣ']
    vowel_indices = [class_names.index(v) for v in vowels]
    
    vowel_predictions = []
    vowel_targets = []
    
    for pred, target in zip(all_predictions, all_targets):
        if target in vowel_indices:
            vowel_predictions.append(pred)
            vowel_targets.append(target)
    
    # 모음 성능 리포트
    vowel_class_names = [class_names[i] for i in vowel_indices]
    report = classification_report(vowel_targets, vowel_predictions, 
                                 target_names=vowel_class_names, output_dict=True)
    
    return report, vowel_predictions, vowel_targets

def visualize_robust_results(history, vowel_report):
    """견고한 모델 결과 시각화"""
    print("📊 견고한 모델 결과 시각화")
    print("=" * 50)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('견고한 모음 모델 결과 (과적합 방지)', fontsize=16, fontweight='bold')
    
    # 1. 훈련 곡선
    axes[0, 0].plot(history['train_losses'], label='Train Loss', color='blue')
    axes[0, 0].plot(history['val_losses'], label='Val Loss', color='red')
    axes[0, 0].set_title('손실 함수')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 정확도 곡선
    axes[0, 1].plot(history['train_accuracies'], label='Train Acc', color='blue')
    axes[0, 1].plot(history['val_accuracies'], label='Val Acc', color='red')
    axes[0, 1].plot(history['vowel_accuracies'], label='Vowel Acc', color='green', linewidth=2)
    axes[0, 1].set_title('정확도')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 모음별 정확도
    vowels = ['ㅏ', 'ㅑ', 'ㅓ', 'ㅕ', 'ㅗ', 'ㅛ', 'ㅜ', 'ㅠ', 'ㅡ', 'ㅣ']
    vowel_accuracies = []
    
    for vowel in vowels:
        if vowel in vowel_report:
            vowel_accuracies.append(vowel_report[vowel]['precision'] * 100)
        else:
            vowel_accuracies.append(0)
    
    bars = axes[1, 0].bar(vowels, vowel_accuracies, color='lightcoral', alpha=0.7)
    axes[1, 0].set_title('모음별 정확도')
    axes[1, 0].set_xlabel('모음')
    axes[1, 0].set_ylabel('정확도 (%)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 값 표시
    for bar, acc in zip(bars, vowel_accuracies):
        axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                       f'{acc:.1f}%', ha='center', va='bottom')
    
    # 4. 모음 평균 성능
    overall_metrics = ['precision', 'recall', 'f1-score']
    overall_values = []
    
    for metric in overall_metrics:
        if 'weighted avg' in vowel_report:
            overall_values.append(vowel_report['weighted avg'][metric] * 100)
        else:
            overall_values.append(0)
    
    bars = axes[1, 1].bar(overall_metrics, overall_values, color=['lightblue', 'lightgreen', 'lightyellow'])
    axes[1, 1].set_title('모음 전체 성능')
    axes[1, 1].set_ylabel('값 (%)')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 값 표시
    for bar, val in zip(bars, overall_values):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                       f'{val:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('robust_vowel_results.png', dpi=300, bbox_inches='tight')
    print("📊 견고한 모델 결과 차트가 'robust_vowel_results.png'에 저장되었습니다.")

def main():
    """메인 함수"""
    print("🔧 견고한 모음 성능 개선 시작")
    print("=" * 50)
    
    # 1. 견고한 데이터셋 로드
    print("📊 견고한 데이터셋 로딩 중...")
    dataset = VowelRobustDataset(
        data_path="../SignGlove_HW/datasets/unified",
        sequence_length=20,
        apply_vowel_enhancement=True,
        augmentation=True
    )
    
    # 2. 데이터 분할
    train_indices, test_indices = train_test_split(
        range(len(dataset)), test_size=0.2, random_state=42, stratify=dataset.labels
    )
    train_indices, val_indices = train_test_split(
        train_indices, test_size=0.2, random_state=42, 
        stratify=[dataset.labels[i] for i in train_indices]
    )
    
    # 3. 데이터로더 생성
    train_loader = DataLoader(
        torch.utils.data.Subset(dataset, train_indices),
        batch_size=16, shuffle=True  # 더 작은 배치 사이즈
    )
    val_loader = DataLoader(
        torch.utils.data.Subset(dataset, val_indices),
        batch_size=16, shuffle=False
    )
    test_loader = DataLoader(
        torch.utils.data.Subset(dataset, test_indices),
        batch_size=16, shuffle=False
    )
    
    print(f"📊 데이터 분할 완료:")
    print(f"  훈련: {len(train_indices)}개")
    print(f"  검증: {len(val_indices)}개")
    print(f"  테스트: {len(test_indices)}개")
    
    # 4. 견고한 모델 생성
    model = VowelRobustGRU(
        input_size=8,
        hidden_size=64,  # 더 작은 hidden size
        num_layers=2,
        num_classes=24,
        dropout=0.5  # 더 높은 dropout
    )
    
    # 5. 모델 훈련
    history = train_robust_model(
        model, train_loader, val_loader, 
        num_epochs=30, learning_rate=0.0005  # 더 낮은 학습률
    )
    
    # 6. 최고 모델 로드
    model.load_state_dict(torch.load('best_robust_vowel_model.pth'))
    
    # 7. 모음 성능 평가
    print("\n📊 견고한 모델 모음 성능 평가")
    print("=" * 50)
    
    vowel_report, vowel_preds, vowel_targets = evaluate_robust_performance(
        model, test_loader, dataset.all_classes
    )
    
    # 8. 결과 출력
    print("🔍 모음별 성능:")
    vowels = ['ㅏ', 'ㅑ', 'ㅓ', 'ㅕ', 'ㅗ', 'ㅛ', 'ㅜ', 'ㅠ', 'ㅡ', 'ㅣ']
    for vowel in vowels:
        if vowel in vowel_report:
            precision = vowel_report[vowel]['precision'] * 100
            recall = vowel_report[vowel]['recall'] * 100
            f1 = vowel_report[vowel]['f1-score'] * 100
            print(f"  {vowel}: Precision={precision:.1f}%, Recall={recall:.1f}%, F1={f1:.1f}%")
    
    if 'weighted avg' in vowel_report:
        overall_precision = vowel_report['weighted avg']['precision'] * 100
        overall_recall = vowel_report['weighted avg']['recall'] * 100
        overall_f1 = vowel_report['weighted avg']['f1-score'] * 100
        print(f"\n📈 모음 전체 성능:")
        print(f"  Precision: {overall_precision:.1f}%")
        print(f"  Recall: {overall_recall:.1f}%")
        print(f"  F1-Score: {overall_f1:.1f}%")
    
    # 9. 결과 시각화
    visualize_robust_results(history, vowel_report)
    
    # 10. 모델 저장
    torch.save(model.state_dict(), 'robust_vowel_model.pth')
    print("\n💾 견고한 모음 모델이 'robust_vowel_model.pth'에 저장되었습니다.")
    
    print(f"\n✅ 견고한 모음 성능 개선 완료!")

if __name__ == "__main__":
    main()


