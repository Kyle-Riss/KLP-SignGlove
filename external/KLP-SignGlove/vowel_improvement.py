#!/usr/bin/env python3
"""
모음 성능 개선 스크립트
분석 결과를 바탕으로 모음 데이터의 변동성과 노이즈를 줄이고 성능을 개선
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

class VowelImprovedDataset(Dataset):
    """모음 개선용 데이터셋"""
    
    def __init__(self, data_path, sequence_length=20, apply_vowel_enhancement=True):
        self.data_path = Path(data_path)
        self.sequence_length = sequence_length
        self.apply_vowel_enhancement = apply_vowel_enhancement
        
        # 클래스 분류
        self.consonants = ['ㄱ', 'ㄴ', 'ㄷ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅅ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
        self.vowels = ['ㅏ', 'ㅑ', 'ㅓ', 'ㅕ', 'ㅗ', 'ㅛ', 'ㅜ', 'ㅠ', 'ㅡ', 'ㅣ']
        self.all_classes = self.consonants + self.vowels
        
        # 모음 전용 스케일러
        self.vowel_scaler = RobustScaler()  # 모음은 RobustScaler 사용
        self.consonant_scaler = StandardScaler()  # 자음은 StandardScaler 사용
        
        # 데이터 로드
        self.data, self.labels, self.class_indices = self.load_data()
        
        # 모음 향상 전처리 적용
        if self.apply_vowel_enhancement:
            self.enhance_vowel_data()
        
        # 데이터 정규화
        self.normalize_data()
        
    def load_data(self):
        """데이터 로드"""
        print("📊 모음/자음 데이터 로딩 중...")
        
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
        """모음 데이터 향상 전처리"""
        print("🔧 모음 데이터 향상 전처리 적용 중...")
        
        enhanced_data = []
        
        for i, (sequence, label) in enumerate(zip(self.data, self.labels)):
            class_name = self.all_classes[label]
            
            if class_name in self.vowels:
                # 모음 데이터에 향상 전처리 적용
                enhanced_sequence = self.apply_vowel_enhancement_techniques(sequence)
                enhanced_data.append(enhanced_sequence)
            else:
                # 자음 데이터는 그대로 유지
                enhanced_data.append(sequence)
        
        self.data = np.array(enhanced_data)
        print("✅ 모음 데이터 향상 전처리 완료")
    
    def apply_vowel_enhancement_techniques(self, sequence):
        """모음 데이터 향상 기법 적용"""
        # 1. 노이즈 제거 (가우시안 필터)
        sequence_filtered = gaussian_filter1d(sequence, sigma=0.5, axis=0)
        
        # 2. 신호 스무딩 (이동 평균)
        window_size = 3
        sequence_smoothed = np.zeros_like(sequence_filtered)
        for i in range(len(sequence_filtered)):
            start = max(0, i - window_size // 2)
            end = min(len(sequence_filtered), i + window_size // 2 + 1)
            sequence_smoothed[i] = np.mean(sequence_filtered[start:end], axis=0)
        
        # 3. 변동성 감소 (신호 정규화)
        sequence_normalized = sequence_smoothed.copy()
        for sensor_idx in range(sequence_smoothed.shape[1]):
            sensor_data = sequence_smoothed[:, sensor_idx]
            # 변동성 감소를 위한 압축
            mean_val = np.mean(sensor_data)
            std_val = np.std(sensor_data)
            if std_val > 0:
                # 변동성을 20% 감소
                compression_factor = 0.8
                sequence_normalized[:, sensor_idx] = mean_val + (sensor_data - mean_val) * compression_factor
        
        return sequence_normalized
    
    def normalize_data(self):
        """데이터 정규화 - 모음과 자음에 다른 스케일러 적용"""
        print("🔧 데이터 정규화 중...")
        
        # 모음과 자음 데이터 분리
        vowel_indices = []
        consonant_indices = []
        
        for i, label in enumerate(self.labels):
            class_name = self.all_classes[label]
            if class_name in self.vowels:
                vowel_indices.append(i)
            else:
                consonant_indices.append(i)
        
        # 모음 데이터 정규화 (RobustScaler)
        if vowel_indices:
            vowel_data = self.data[vowel_indices]
            original_shape = vowel_data.shape
            vowel_data_reshaped = vowel_data.reshape(-1, vowel_data.shape[-1])
            vowel_data_normalized = self.vowel_scaler.fit_transform(vowel_data_reshaped)
            self.data[vowel_indices] = vowel_data_normalized.reshape(original_shape)
        
        # 자음 데이터 정규화 (StandardScaler)
        if consonant_indices:
            consonant_data = self.data[consonant_indices]
            original_shape = consonant_data.shape
            consonant_data_reshaped = consonant_data.reshape(-1, consonant_data.shape[-1])
            consonant_data_normalized = self.consonant_scaler.fit_transform(consonant_data_reshaped)
            self.data[consonant_indices] = consonant_data_normalized.reshape(original_shape)
        
        print(f"✅ 정규화 완료: 범위 [{self.data.min():.3f}, {self.data.max():.3f}]")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sequence = self.data[idx]
        label = self.labels[idx]
        return torch.FloatTensor(sequence), torch.LongTensor([label])

class VowelEnhancedGRU(nn.Module):
    """모음 성능 향상을 위한 개선된 GRU 모델"""
    
    def __init__(self, input_size=8, hidden_size=128, num_layers=2, num_classes=24, dropout=0.3):
        super(VowelEnhancedGRU, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # 모음 클래스 가중치 적용을 위한 레이어
        self.vowel_attention = nn.Linear(hidden_size, 1)
        
        # GRU 레이어
        self.gru = nn.GRU(
            input_size, 
            hidden_size, 
            num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 모음 전용 분류 레이어
        self.vowel_classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 10)  # 모음 10개
        )
        
        # 자음 전용 분류 레이어
        self.consonant_classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 14)  # 자음 14개
        )
        
        # 전체 분류 레이어
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )
        
        # 모음 클래스 가중치
        self.vowel_weight = 2.0  # 모음에 더 높은 가중치
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # GRU 처리
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        out, _ = self.gru(x, h0)
        
        # 마지막 시퀀스 출력
        last_output = out[:, -1, :]
        
        # 모음 어텐션 가중치 계산
        vowel_attention_weights = torch.sigmoid(self.vowel_attention(last_output))
        
        # 전체 분류
        output = self.classifier(last_output)
        
        return output, vowel_attention_weights

class VowelWeightedLoss(nn.Module):
    """모음 가중치 손실 함수"""
    
    def __init__(self, num_classes=24, vowel_weight=2.0):
        super(VowelWeightedLoss, self).__init__()
        
        # 모음 클래스 인덱스 (10개)
        vowel_indices = list(range(14, 24))  # 모음은 14-23 인덱스
        
        # 클래스 가중치 설정
        class_weights = torch.ones(num_classes)
        for idx in vowel_indices:
            class_weights[idx] = vowel_weight
        
        self.register_buffer('class_weights', class_weights)
    
    def forward(self, outputs, targets):
        return nn.functional.cross_entropy(outputs, targets.squeeze(), weight=self.class_weights)

def train_vowel_enhanced_model(model, train_loader, val_loader, num_epochs=50, learning_rate=0.001):
    """모음 향상 모델 훈련"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # 모음 가중치 손실 함수
    criterion = VowelWeightedLoss(num_classes=24, vowel_weight=2.0)
    criterion = criterion.to(device)  # 손실 함수를 디바이스로 이동
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    # 훈련 기록
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    vowel_accuracies = []
    
    print(f"🚀 모음 향상 모델 훈련 시작 (디바이스: {device})")
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
            outputs, attention_weights = model(data)
            loss = criterion(outputs, targets)
            
            loss.backward()
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
                outputs, attention_weights = model(data)
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
        
        # 기록 저장
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)
        vowel_accuracies.append(vowel_accuracy)
        
        # 출력
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}]")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")
            print(f"  Vowel Acc: {vowel_accuracy:.2f}%")
            print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
            print("-" * 40)
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies,
        'vowel_accuracies': vowel_accuracies
    }

def evaluate_vowel_performance(model, test_loader, class_names):
    """모음 성능 평가"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for data, targets in test_loader:
            data, targets = data.to(device), targets.to(device)
            outputs, attention_weights = model(data)
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

def visualize_vowel_improvement(history, vowel_report):
    """모음 개선 결과 시각화"""
    print("📊 모음 개선 결과 시각화")
    print("=" * 50)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('모음 성능 개선 결과', fontsize=16, fontweight='bold')
    
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
    plt.savefig('vowel_improvement_results.png', dpi=300, bbox_inches='tight')
    print("📊 모음 개선 결과 차트가 'vowel_improvement_results.png'에 저장되었습니다.")

def main():
    """메인 함수"""
    print("🔧 모음 성능 개선 시작")
    print("=" * 50)
    
    # 1. 향상된 데이터셋 로드
    print("📊 향상된 데이터셋 로딩 중...")
    dataset = VowelImprovedDataset(
        data_path="../SignGlove_HW/datasets/unified",
        sequence_length=20,
        apply_vowel_enhancement=True
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
        batch_size=32, shuffle=True
    )
    val_loader = DataLoader(
        torch.utils.data.Subset(dataset, val_indices),
        batch_size=32, shuffle=False
    )
    test_loader = DataLoader(
        torch.utils.data.Subset(dataset, test_indices),
        batch_size=32, shuffle=False
    )
    
    print(f"📊 데이터 분할 완료:")
    print(f"  훈련: {len(train_indices)}개")
    print(f"  검증: {len(val_indices)}개")
    print(f"  테스트: {len(test_indices)}개")
    
    # 4. 모음 향상 모델 생성
    model = VowelEnhancedGRU(
        input_size=8,
        hidden_size=128,
        num_layers=2,
        num_classes=24,
        dropout=0.3
    )
    
    # 5. 모델 훈련
    history = train_vowel_enhanced_model(
        model, train_loader, val_loader, 
        num_epochs=50, learning_rate=0.001
    )
    
    # 6. 모음 성능 평가
    print("\n📊 모음 성능 평가")
    print("=" * 50)
    
    vowel_report, vowel_preds, vowel_targets = evaluate_vowel_performance(
        model, test_loader, dataset.all_classes
    )
    
    # 7. 결과 출력
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
    
    # 8. 결과 시각화
    visualize_vowel_improvement(history, vowel_report)
    
    # 9. 모델 저장
    torch.save(model.state_dict(), 'vowel_enhanced_model.pth')
    print("\n💾 모음 향상 모델이 'vowel_enhanced_model.pth'에 저장되었습니다.")
    
    print(f"\n✅ 모음 성능 개선 완료!")

if __name__ == "__main__":
    main()
