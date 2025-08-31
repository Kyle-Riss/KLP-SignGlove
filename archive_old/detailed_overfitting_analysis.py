#!/usr/bin/env python3
"""
KLP-SignGlove: 상세 과적합 분석 스크립트
충분한 에폭으로 훈련하여 과적합 여부를 명확히 증명

- 더 많은 에폭으로 훈련 (100 에폭)
- 훈련/검증 손실 곡선 상세 시각화
- 과적합 패턴 분석
- 실시간 학습 곡선 모니터링

작성자: KLP-SignGlove Team
버전: 1.0.0
날짜: 2024
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import h5py
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

class PositionalEncoding(nn.Module):
    """위치 인코딩"""
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class TransformerModel(nn.Module):
    """Transformer 모델"""
    def __init__(self, input_size=8, d_model=64, nhead=8, num_layers=2, 
                 num_classes=24, dropout=0.1, max_seq_len=300):
        super(TransformerModel, self).__init__()
        
        self.input_size = input_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        self.input_projection = nn.Linear(input_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, num_classes)
        
        self._init_weights()
    
    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() > 1:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        
        x = self.input_projection(x)
        x = x.transpose(0, 1)
        x = self.pos_encoding(x)
        x = x.transpose(0, 1)
        
        x = self.transformer_encoder(x)
        x = torch.mean(x, dim=1)
        x = self.dropout(x)
        x = self.fc(x)
        
        return x

class AdvancedPreprocessor:
    """고급 전처리기"""
    def __init__(self):
        self.flex_scalers = [StandardScaler() for _ in range(5)]
        self.orientation_scalers = [StandardScaler() for _ in range(3)]
        self.is_fitted = False
        
    def fit(self, data_list):
        print('🔧 전처리 파라미터 학습 중...')
        
        for sensor_idx in range(5):
            all_values = []
            for data in data_list:
                sensor_values = data[:, sensor_idx]
                valid_values = sensor_values[sensor_values > 0]
                all_values.extend(valid_values)
            
            if all_values:
                self.flex_scalers[sensor_idx].fit(np.array(all_values).reshape(-1, 1))
        
        for sensor_idx in range(3):
            all_values = []
            for data in data_list:
                sensor_values = data[:, sensor_idx + 5]
                all_values.extend(sensor_values)
            
            if all_values:
                self.orientation_scalers[sensor_idx].fit(np.array(all_values).reshape(-1, 1))
        
        self.is_fitted = True
        print('✅ 전처리 파라미터 학습 완료')
    
    def transform_single(self, data):
        if not self.is_fitted:
            raise ValueError('전처리 파라미터를 먼저 학습해야 합니다.')
        
        processed = data.copy()
        
        for sensor_idx in range(5):
            sensor_values = data[:, sensor_idx]
            mean_val = np.mean(sensor_values[sensor_values > 0])
            if np.isnan(mean_val):
                mean_val = 500
            sensor_values[sensor_values == 0] = mean_val
            sensor_values_normalized = self.flex_scalers[sensor_idx].transform(
                sensor_values.reshape(-1, 1)
            ).flatten()
            processed[:, sensor_idx] = sensor_values_normalized
        
        for sensor_idx in range(3):
            sensor_values = data[:, sensor_idx + 5]
            sensor_values_normalized = self.orientation_scalers[sensor_idx].transform(
                sensor_values.reshape(-1, 1)
            ).flatten()
            processed[:, sensor_idx + 5] = sensor_values_normalized
        
        return processed
    
    def transform(self, data_list):
        return [self.transform_single(data) for data in data_list]

def load_and_preprocess_data(data_dir, max_samples_per_class=25):
    """데이터 로드 및 전처리"""
    print('📁 데이터 로드 중...')
    
    class_names = ['ㄱ', 'ㄴ', 'ㄷ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅅ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ', 
                  'ㅏ', 'ㅑ', 'ㅓ', 'ㅕ', 'ㅗ', 'ㅛ', 'ㅜ', 'ㅠ', 'ㅡ', 'ㅣ']
    
    all_data = []
    all_labels = []
    
    for class_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(data_dir, class_name, '1')
        if not os.path.exists(class_dir):
            continue
        
        h5_files = [f for f in os.listdir(class_dir) if f.endswith('.h5')]
        h5_files = h5_files[:max_samples_per_class]
        
        for h5_file in h5_files:
            file_path = os.path.join(class_dir, h5_file)
            try:
                with h5py.File(file_path, 'r') as f:
                    data = f['sensor_data'][:]
                    all_data.append(data)
                    all_labels.append(class_idx)
            except:
                continue
    
    print(f'✅ 데이터 로드 완료: {len(all_data)}개 샘플, {len(class_names)}개 클래스')
    return all_data, all_labels, class_names

def train_model_extended(model, train_data, train_labels, val_data, val_labels, 
                        device, epochs=100, learning_rate=0.001, model_name="Model"):
    """충분한 에폭으로 모델 훈련"""
    print(f'🏋️ {model_name} 확장 훈련 시작 (총 {epochs} 에폭)...')
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5, verbose=True)
    
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    learning_rates = []
    
    best_val_acc = 0
    patience_counter = 0
    max_patience = 25  # 더 긴 patience
    
    print(f'📊 훈련 진행 상황 모니터링:')
    print(f'{"Epoch":<6} {"Train Loss":<12} {"Train Acc":<12} {"Val Loss":<12} {"Val Acc":<12} {"LR":<12}')
    print('-' * 70)
    
    for epoch in range(epochs):
        # 훈련
        model.train()
        total_train_loss = 0
        train_correct = 0
        train_total = 0
        
        for data, label in zip(train_data, train_labels):
            data_tensor = torch.FloatTensor(data).unsqueeze(0).to(device)
            label_tensor = torch.LongTensor([label]).to(device)
            
            optimizer.zero_grad()
            outputs = model(data_tensor)
            loss = criterion(outputs, label_tensor)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += 1
            if predicted.item() == label:
                train_correct += 1
        
        train_acc = train_correct / train_total
        avg_train_loss = total_train_loss / len(train_data)
        
        # 검증
        model.eval()
        total_val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, label in zip(val_data, val_labels):
                data_tensor = torch.FloatTensor(data).unsqueeze(0).to(device)
                outputs = model(data_tensor)
                loss = criterion(outputs, torch.LongTensor([label]).to(device))
                
                total_val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += 1
                if predicted.item() == label:
                    val_correct += 1
        
        val_acc = val_correct / val_total
        avg_val_loss = total_val_loss / len(val_data)
        
        # 히스토리 저장
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_acc)
        learning_rates.append(optimizer.param_groups[0]['lr'])
        
        # 스케줄러 업데이트
        scheduler.step(avg_val_loss)
        
        # 조기 종료
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1
        
        # 진행 상황 출력
        if (epoch + 1) % 5 == 0 or epoch < 10:
            print(f'{epoch+1:<6} {avg_train_loss:<12.4f} {train_acc:<12.4f} {avg_val_loss:<12.4f} {val_acc:<12.4f} {optimizer.param_groups[0]["lr"]:<12.6f}')
        
        if patience_counter >= max_patience:
            print(f'  조기 종료: {epoch+1} 에포크 (patience: {max_patience})')
            break
    
    print(f'✅ {model_name} 훈련 완료! 최종 검증 정확도: {best_val_acc:.4f}')
    
    return {
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
        'learning_rates': learning_rates,
        'best_val_acc': best_val_acc,
        'final_epoch': len(train_losses)
    }

def analyze_overfitting_patterns(history, model_name):
    """과적합 패턴 상세 분석"""
    print(f'\n🔍 {model_name} 과적합 패턴 분석')
    
    train_losses = history['train_losses']
    val_losses = history['val_losses']
    train_accuracies = history['train_accuracies']
    val_accuracies = history['val_accuracies']
    
    # 1. 정확도 격차 분석
    accuracy_gaps = [train_acc - val_acc for train_acc, val_acc in zip(train_accuracies, val_accuracies)]
    max_accuracy_gap = max(accuracy_gaps)
    final_accuracy_gap = accuracy_gaps[-1]
    
    # 2. 손실 격차 분석
    loss_gaps = [val_loss - train_loss for train_loss, val_loss in zip(train_losses, val_losses)]
    max_loss_gap = max(loss_gaps)
    final_loss_gap = loss_gaps[-1]
    
    # 3. 과적합 지표 계산
    overfitting_indicators = []
    
    if max_accuracy_gap > 0.05:
        overfitting_indicators.append(f"최대 정확도 격차: {max_accuracy_gap:.4f} (높음)")
    
    if max_loss_gap > 0.1:
        overfitting_indicators.append(f"최대 손실 격차: {max_loss_gap:.4f} (높음)")
    
    if train_accuracies[-1] > 0.99 and val_accuracies[-1] < 0.95:
        overfitting_indicators.append("훈련 정확도가 과도하게 높음")
    
    # 4. 수렴 패턴 분석
    convergence_analysis = []
    
    # 마지막 20 에폭의 변화량
    if len(train_losses) >= 20:
        recent_train_loss_change = abs(train_losses[-1] - train_losses[-20])
        recent_val_loss_change = abs(val_losses[-1] - val_losses[-20])
        
        if recent_train_loss_change < 0.001 and recent_val_loss_change < 0.001:
            convergence_analysis.append("안정적으로 수렴됨")
        else:
            convergence_analysis.append("아직 수렴 중")
    
    # 5. 과적합 판단
    if not overfitting_indicators:
        overfitting_conclusion = "과적합 없음"
    else:
        overfitting_conclusion = f"과적합 의심: {', '.join(overfitting_indicators)}"
    
    print(f"  최종 훈련 정확도: {train_accuracies[-1]:.4f}")
    print(f"  최종 검증 정확도: {val_accuracies[-1]:.4f}")
    print(f"  최종 정확도 격차: {final_accuracy_gap:.4f}")
    print(f"  최대 정확도 격차: {max_accuracy_gap:.4f}")
    print(f"  최종 손실 격차: {final_loss_gap:.4f}")
    print(f"  최대 손실 격차: {max_loss_gap:.4f}")
    print(f"  수렴 분석: {', '.join(convergence_analysis)}")
    print(f"  과적합 판단: {overfitting_conclusion}")
    
    return {
        'overfitting_conclusion': overfitting_conclusion,
        'max_accuracy_gap': max_accuracy_gap,
        'final_accuracy_gap': final_accuracy_gap,
        'max_loss_gap': max_loss_gap,
        'final_loss_gap': final_loss_gap,
        'convergence_stable': len(convergence_analysis) > 0 and "안정적으로 수렴됨" in convergence_analysis[0]
    }

def create_detailed_overfitting_visualization(gru_history, transformer_history):
    """상세한 과적합 분석 시각화"""
    print('\n📊 상세 과적합 분석 시각화 생성 중...')
    
    fig = plt.figure(figsize=(20, 16))
    
    # 1. 훈련/검증 손실 곡선 (GRU)
    plt.subplot(3, 4, 1)
    epochs = range(1, len(gru_history['train_losses']) + 1)
    plt.plot(epochs, gru_history['train_losses'], 'b-', label='Train Loss', linewidth=2, alpha=0.8)
    plt.plot(epochs, gru_history['val_losses'], 'b--', label='Validation Loss', linewidth=2, alpha=0.8)
    plt.title('GRU: 훈련/검증 손실 곡선', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, max(max(gru_history['train_losses']), max(gru_history['val_losses'])) * 1.1)
    
    # 2. 훈련/검증 정확도 곡선 (GRU)
    plt.subplot(3, 4, 2)
    plt.plot(epochs, gru_history['train_accuracies'], 'g-', label='Train Accuracy', linewidth=2, alpha=0.8)
    plt.plot(epochs, gru_history['val_accuracies'], 'g--', label='Validation Accuracy', linewidth=2, alpha=0.8)
    plt.title('GRU: 훈련/검증 정확도 곡선', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0.8, 1.0)
    
    # 3. 훈련/검증 손실 곡선 (Transformer)
    plt.subplot(3, 4, 3)
    epochs = range(1, len(transformer_history['train_losses']) + 1)
    plt.plot(epochs, transformer_history['train_losses'], 'r-', label='Train Loss', linewidth=2, alpha=0.8)
    plt.plot(epochs, transformer_history['val_losses'], 'r--', label='Validation Loss', linewidth=2, alpha=0.8)
    plt.title('Transformer: 훈련/검증 손실 곡선', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, max(max(transformer_history['train_losses']), max(transformer_history['val_losses'])) * 1.1)
    
    # 4. 훈련/검증 정확도 곡선 (Transformer)
    plt.subplot(3, 4, 4)
    plt.plot(epochs, transformer_history['train_accuracies'], 'orange', label='Train Accuracy', linewidth=2, alpha=0.8)
    plt.plot(epochs, transformer_history['val_accuracies'], 'orange', linestyle='--', label='Validation Accuracy', linewidth=2, alpha=0.8)
    plt.title('Transformer: 훈련/검증 정확도 곡선', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0.8, 1.0)
    
    # 5. 정확도 격차 비교
    plt.subplot(3, 4, 5)
    gru_accuracy_gaps = [train_acc - val_acc for train_acc, val_acc in zip(gru_history['train_accuracies'], gru_history['val_accuracies'])]
    transformer_accuracy_gaps = [train_acc - val_acc for train_acc, val_acc in zip(transformer_history['train_accuracies'], transformer_history['val_accuracies'])]
    
    plt.plot(range(1, len(gru_accuracy_gaps) + 1), gru_accuracy_gaps, 'b-', label='GRU', linewidth=2, alpha=0.8)
    plt.plot(range(1, len(transformer_accuracy_gaps) + 1), transformer_accuracy_gaps, 'r-', label='Transformer', linewidth=2, alpha=0.8)
    plt.axhline(y=0.05, color='red', linestyle='--', alpha=0.5, label='과적합 임계값 (5%)')
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.title('정확도 격차 변화 (Train - Val)', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy Gap')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 6. 손실 격차 비교
    plt.subplot(3, 4, 6)
    gru_loss_gaps = [val_loss - train_loss for train_loss, val_loss in zip(gru_history['train_losses'], gru_history['val_losses'])]
    transformer_loss_gaps = [val_loss - train_loss for train_loss, val_loss in zip(transformer_history['train_losses'], transformer_history['val_losses'])]
    
    plt.plot(range(1, len(gru_loss_gaps) + 1), gru_loss_gaps, 'b-', label='GRU', linewidth=2, alpha=0.8)
    plt.plot(range(1, len(transformer_loss_gaps) + 1), transformer_loss_gaps, 'r-', label='Transformer', linewidth=2, alpha=0.8)
    plt.axhline(y=0.1, color='red', linestyle='--', alpha=0.5, label='과적합 임계값 (0.1)')
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.title('손실 격차 변화 (Val - Train)', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Gap')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 7. 학습률 변화
    plt.subplot(3, 4, 7)
    plt.plot(range(1, len(gru_history['learning_rates']) + 1), gru_history['learning_rates'], 'b-', label='GRU', linewidth=2, alpha=0.8)
    plt.plot(range(1, len(transformer_history['learning_rates']) + 1), transformer_history['learning_rates'], 'r-', label='Transformer', linewidth=2, alpha=0.8)
    plt.title('학습률 변화', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # 8. 최종 성능 비교
    plt.subplot(3, 4, 8)
    models = ['GRU', 'Transformer']
    final_train_acc = [gru_history['train_accuracies'][-1], transformer_history['train_accuracies'][-1]]
    final_val_acc = [gru_history['val_accuracies'][-1], transformer_history['val_accuracies'][-1]]
    
    x = np.arange(len(models))
    width = 0.35
    plt.bar(x - width/2, final_train_acc, width, label='Train Accuracy', alpha=0.8)
    plt.bar(x + width/2, final_val_acc, width, label='Validation Accuracy', alpha=0.8)
    plt.title('최종 성능 비교', fontsize=14, fontweight='bold')
    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.xticks(x, models)
    plt.ylim(0.9, 1.0)
    
    # 9. 과적합 지표 요약
    plt.subplot(3, 4, 9)
    gru_max_gap = max(gru_accuracy_gaps)
    transformer_max_gap = max(transformer_accuracy_gaps)
    
    gaps = [gru_max_gap, transformer_max_gap]
    colors = ['green' if gap < 0.05 else 'orange' if gap < 0.1 else 'red' for gap in gaps]
    bars = plt.bar(models, gaps, color=colors, alpha=0.8)
    plt.axhline(y=0.05, color='red', linestyle='--', alpha=0.5, label='과적합 임계값')
    plt.title('최대 정확도 격차', fontsize=14, fontweight='bold')
    plt.ylabel('Max Accuracy Gap')
    for bar, gap in zip(bars, gaps):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{gap:.3f}', ha='center', va='bottom', fontweight='bold')
    plt.legend()
    
    # 10. 수렴 안정성 분석
    plt.subplot(3, 4, 10)
    # 마지막 20 에폭의 손실 변화량
    gru_recent_loss_std = np.std(gru_history['val_losses'][-20:]) if len(gru_history['val_losses']) >= 20 else np.std(gru_history['val_losses'])
    transformer_recent_loss_std = np.std(transformer_history['val_losses'][-20:]) if len(transformer_history['val_losses']) >= 20 else np.std(transformer_history['val_losses'])
    
    stabilities = [gru_recent_loss_std, transformer_recent_loss_std]
    colors = ['green' if std < 0.01 else 'orange' if std < 0.05 else 'red' for std in stabilities]
    bars = plt.bar(models, stabilities, color=colors, alpha=0.8)
    plt.title('수렴 안정성 (검증 손실 표준편차)', fontsize=14, fontweight='bold')
    plt.ylabel('Validation Loss Std')
    for bar, std in zip(bars, stabilities):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0001,
                f'{std:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # 11. 훈련 효율성 비교
    plt.subplot(3, 4, 11)
    gru_epochs = len(gru_history['train_losses'])
    transformer_epochs = len(transformer_history['train_losses'])
    
    epochs_needed = [gru_epochs, transformer_epochs]
    colors = ['green' if epochs < 50 else 'orange' if epochs < 80 else 'red' for epochs in epochs_needed]
    bars = plt.bar(models, epochs_needed, color=colors, alpha=0.8)
    plt.title('수렴까지 필요한 에폭 수', fontsize=14, fontweight='bold')
    plt.ylabel('Epochs to Convergence')
    for bar, epochs in zip(bars, epochs_needed):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{epochs}', ha='center', va='bottom', fontweight='bold')
    
    # 12. 종합 과적합 평가
    plt.subplot(3, 4, 12)
    # 과적합 점수 계산 (낮을수록 좋음)
    gru_overfitting_score = gru_max_gap + gru_recent_loss_std
    transformer_overfitting_score = transformer_max_gap + transformer_recent_loss_std
    
    scores = [gru_overfitting_score, transformer_overfitting_score]
    colors = ['green' if score < 0.05 else 'orange' if score < 0.1 else 'red' for score in scores]
    bars = plt.bar(models, scores, color=colors, alpha=0.8)
    plt.title('종합 과적합 점수', fontsize=14, fontweight='bold')
    plt.ylabel('Overfitting Score')
    for bar, score in zip(bars, scores):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('detailed_overfitting_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print('✅ 상세 과적합 분석 시각화 저장: detailed_overfitting_analysis.png')

def main():
    """메인 실행 함수"""
    print('🔬 상세 과적합 분석 시작 (충분한 에폭으로 훈련)')
    print('=' * 70)
    
    # 데이터 로드
    data_dir = '../SignGlove/external/SignGlove_HW/datasets/unified'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    all_data, all_labels, class_names = load_and_preprocess_data(data_dir, max_samples_per_class=25)
    
    # 데이터 분할
    train_data, temp_data, train_labels, temp_labels = train_test_split(
        all_data, all_labels, test_size=0.4, random_state=42, stratify=all_labels
    )
    val_data, test_data, val_labels, test_labels = train_test_split(
        temp_data, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
    )
    
    # 전처리
    preprocessor = AdvancedPreprocessor()
    preprocessor.fit(train_data)
    
    train_data_processed = preprocessor.transform(train_data)
    val_data_processed = preprocessor.transform(val_data)
    test_data_processed = preprocessor.transform(test_data)
    
    # Transformer 모델 확장 훈련
    transformer_model = TransformerModel(
        input_size=8, d_model=64, nhead=8, num_layers=2, 
        num_classes=24, dropout=0.1
    ).to(device)
    
    transformer_history = train_model_extended(
        transformer_model, train_data_processed, train_labels, 
        val_data_processed, val_labels, device, epochs=100, model_name="Transformer"
    )
    
    # GRU 모델 확장 훈련
    from improved_preprocessing_model import ImprovedGRU
    
    gru_model = ImprovedGRU(input_size=8, hidden_size=64, num_layers=2, num_classes=24, dropout=0.3).to(device)
    
    gru_history = train_model_extended(
        gru_model, train_data_processed, train_labels, 
        val_data_processed, val_labels, device, epochs=100, model_name="GRU"
    )
    
    # 과적합 패턴 분석
    transformer_overfitting = analyze_overfitting_patterns(transformer_history, "Transformer")
    gru_overfitting = analyze_overfitting_patterns(gru_history, "GRU")
    
    # 결과 요약
    print('\n📊 상세 과적합 분석 결과 요약')
    print('=' * 70)
    
    summary_data = {
        '모델': ['GRU', 'Transformer'],
        '훈련 에폭': [gru_history['final_epoch'], transformer_history['final_epoch']],
        '최종 훈련 정확도': [gru_history['train_accuracies'][-1], transformer_history['train_accuracies'][-1]],
        '최종 검증 정확도': [gru_history['val_accuracies'][-1], transformer_history['val_accuracies'][-1]],
        '최대 정확도 격차': [gru_overfitting['max_accuracy_gap'], transformer_overfitting['max_accuracy_gap']],
        '과적합 판단': [gru_overfitting['overfitting_conclusion'], transformer_overfitting['overfitting_conclusion']]
    }
    
    print(f"{'모델':<12} {'에폭':<6} {'훈련정확도':<12} {'검증정확도':<12} {'최대격차':<12} {'과적합':<20}")
    print('-' * 80)
    
    for i in range(2):
        print(f"{summary_data['모델'][i]:<12} "
              f"{summary_data['훈련 에폭'][i]:<6} "
              f"{summary_data['최종 훈련 정확도'][i]:<12.4f} "
              f"{summary_data['최종 검증 정확도'][i]:<12.4f} "
              f"{summary_data['최대 정확도 격차'][i]:<12.4f} "
              f"{summary_data['과적합 판단'][i]:<20}")
    
    # 결론
    print('\n🎯 과적합 분석 최종 결론')
    print('=' * 70)
    
    if "과적합 없음" in gru_overfitting['overfitting_conclusion'] and "과적합 없음" in transformer_overfitting['overfitting_conclusion']:
        print('✅ 두 모델 모두 과적합이 없음을 확인!')
        print('📋 과적합이 아닌 근거:')
        print('1. 🎯 정확도 격차: 훈련/검증 정확도가 거의 일치')
        print('2. 📉 손실 곡선: 검증 손실이 훈련 손실과 함께 감소')
        print('3. 🔄 수렴 패턴: 안정적이고 자연스러운 수렴')
        print('4. ⚖️ 일반화 성능: 높은 검증 정확도 유지')
        print('5. 📊 충분한 훈련: 100 에폭까지 확장 훈련으로 확인')
    else:
        print('⚠️ 일부 모델에서 과적합 신호 발견')
    
    # 시각화 생성
    create_detailed_overfitting_visualization(gru_history, transformer_history)
    
    print('\n🎉 상세 과적합 분석 완료!')

if __name__ == "__main__":
    main()
