#!/usr/bin/env python3
"""
SignSpeak 스타일 GRU 모델
SignSpeak의 성공적인 접근 방식을 참고하여 개선된 한국 수어 인식 모델

- SignSpeak의 데이터 전처리 방식 적용
- 개선된 GRU 구조
- 체계적인 패딩과 정규화
- K-fold 교차 검증

작성자: KLP-SignGlove Team
버전: 3.0.0
날짜: 2024
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import h5py
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

class SignSpeakStyleGRU(nn.Module):
    """
    SignSpeak 스타일 GRU 모델
    SignSpeak의 성공적인 구조를 참고하여 개선된 GRU 모델
    """
    def __init__(self, input_size=8, hidden_size=64, num_classes=24, layers=2, dropout=0.2):
        super(SignSpeakStyleGRU, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = layers
        
        # SignSpeak 스타일 GRU 레이어
        self.gru = nn.GRU(
            input_size, 
            hidden_size, 
            num_layers=layers, 
            batch_first=True,
            dropout=dropout if layers > 1 else 0
        )
        
        # SignSpeak 스타일 출력 레이어
        self.output_layers = nn.Sequential(
            nn.Linear(hidden_size, 2 * hidden_size),
            nn.Dropout(dropout),
            nn.Tanh(),
            nn.Linear(2 * hidden_size, num_classes)
        )
        
        # 가중치 초기화
        self._init_weights()
    
    def _init_weights(self):
        """가중치 초기화"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(self, x, x_padding=None, y_targets=None):
        """
        순전파 (SignSpeak 스타일)
        Args:
            x (torch.Tensor): 입력 데이터 (batch_size, seq_len, input_size)
            x_padding (torch.Tensor): 패딩 마스크 (optional)
            y_targets (torch.Tensor): 타겟 레이블 (optional)
        Returns:
            torch.Tensor: 예측 결과 (batch_size, num_classes)
        """
        # GRU 처리
        gru_out, _ = self.gru(x)  # (batch_size, seq_len, hidden_size)
        
        # 마지막 시퀀스만 사용 (SignSpeak 방식)
        last_output = gru_out[:, -1, :]  # (batch_size, hidden_size)
        
        # 출력 레이어
        logits = self.output_layers(last_output)  # (batch_size, num_classes)
        
        # 훈련 시에만 손실 계산
        if y_targets is not None:
            loss = nn.functional.cross_entropy(logits, y_targets)
            return logits, loss
        
        return logits

class SignSpeakDataProcessor:
    """SignSpeak 스타일 데이터 전처리기"""
    def __init__(self, time_steps=79):
        self.time_steps = time_steps
        self.scalers = [StandardScaler() for _ in range(8)]  # 8개 센서
        self.is_fitted = False
        
    def fit(self, data_list):
        """전처리 파라미터 학습"""
        print('🔧 SignSpeak 스타일 전처리 파라미터 학습 중...')
        
        # 각 센서별로 스케일러 학습
        for sensor_idx in range(8):
            all_values = []
            for data in data_list:
                if data.shape[1] > sensor_idx:
                    sensor_values = data[:, sensor_idx]
                    # 0값 처리 (Flex 센서)
                    if sensor_idx < 5:
                        valid_values = sensor_values[sensor_values > 0]
                        if len(valid_values) > 0:
                            all_values.extend(valid_values)
                    else:
                        all_values.extend(sensor_values)
            
            if all_values:
                self.scalers[sensor_idx].fit(np.array(all_values).reshape(-1, 1))
        
        self.is_fitted = True
        print('✅ 전처리 파라미터 학습 완료')
    
    def transform_single(self, data):
        """단일 데이터 전처리"""
        if not self.is_fitted:
            raise ValueError('전처리 파라미터를 먼저 학습해야 합니다.')
        
        processed = data.copy()
        
        # 각 센서별 정규화
        for sensor_idx in range(8):
            if sensor_idx < data.shape[1]:
                sensor_values = data[:, sensor_idx]
                
                # Flex 센서 (0-4) 0값 처리
                if sensor_idx < 5:
                    mean_val = np.mean(sensor_values[sensor_values > 0])
                    if np.isnan(mean_val):
                        mean_val = 500
                    sensor_values[sensor_values == 0] = mean_val
                
                # 정규화
                sensor_values_normalized = self.scalers[sensor_idx].transform(
                    sensor_values.reshape(-1, 1)
                ).flatten()
                processed[:, sensor_idx] = sensor_values_normalized
        
        return processed
    
    def transform(self, data_list):
        """배치 데이터 전처리"""
        return [self.transform_single(data) for data in data_list]
    
    def pad_sequences(self, data_list):
        """SignSpeak 스타일 시퀀스 패딩"""
        padded_data = []
        padding_masks = []
        
        for data in data_list:
            seq_len = data.shape[0]
            
            if seq_len < self.time_steps:
                # 패딩
                padding_length = self.time_steps - seq_len
                padding = np.zeros((padding_length, data.shape[1]))
                padded_seq = np.vstack([padding, data])
                
                # 패딩 마스크 (0: 패딩, 1: 실제 데이터)
                mask = np.array([0] * padding_length + [1] * seq_len)
            else:
                # 자르기
                padded_seq = data[-self.time_steps:, :]
                mask = np.ones(self.time_steps)
            
            padded_data.append(padded_seq)
            padding_masks.append(mask)
        
        return np.array(padded_data), np.array(padding_masks)

def load_and_preprocess_data_signspeak_style(data_dir, time_steps=79, max_samples_per_class=None):
    """SignSpeak 스타일 데이터 로드 및 전처리"""
    print('📁 SignSpeak 스타일 데이터 로드 중...')
    
    class_names = ['ㄱ', 'ㄴ', 'ㄷ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅅ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ', 
                  'ㅏ', 'ㅑ', 'ㅓ', 'ㅕ', 'ㅗ', 'ㅛ', 'ㅜ', 'ㅠ', 'ㅡ', 'ㅣ']
    
    all_data = []
    all_labels = []
    class_counts = {}
    
    for class_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(data_dir, class_name, '1')
        if not os.path.exists(class_dir):
            continue
        
        h5_files = [f for f in os.listdir(class_dir) if f.endswith('.h5')]
        
        # 모든 데이터 사용 또는 제한
        if max_samples_per_class:
            h5_files = h5_files[:max_samples_per_class]
        
        class_counts[class_name] = len(h5_files)
        
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
    print(f'📊 클래스별 샘플 수:')
    for class_name, count in class_counts.items():
        print(f'  {class_name}: {count}개')
    
    # SignSpeak 스타일 전처리
    processor = SignSpeakDataProcessor(time_steps=time_steps)
    processor.fit(all_data)
    
    processed_data = processor.transform(all_data)
    padded_data, padding_masks = processor.pad_sequences(processed_data)
    
    print(f'✅ SignSpeak 스타일 전처리 완료')
    print(f'  - 패딩된 데이터 형태: {padded_data.shape}')
    print(f'  - 패딩 마스크 형태: {padding_masks.shape}')
    
    return padded_data, padding_masks, all_labels, class_names, processor

def train_signspeak_style_model(model, train_data, train_masks, train_labels, 
                               val_data, val_masks, val_labels, 
                               device, epochs=100, learning_rate=0.001):
    """SignSpeak 스타일 모델 훈련"""
    print(f'🏋️ SignSpeak 스타일 모델 훈련 시작 (총 {epochs} 에폭)...')
    
    # SignSpeak 스타일 옵티마이저
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.5, verbose=True)
    
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    
    best_val_acc = 0
    patience_counter = 0
    max_patience = 30
    
    print(f'📊 훈련 진행 상황 모니터링:')
    print(f'{"Epoch":<6} {"Train Loss":<12} {"Train Acc":<12} {"Val Loss":<12} {"Val Acc":<12} {"Status":<15}')
    print('-' * 75)
    
    for epoch in range(epochs):
        # 훈련
        model.train()
        total_train_loss = 0
        train_correct = 0
        train_total = 0
        
        for i in range(len(train_data)):
            data_tensor = torch.FloatTensor(train_data[i:i+1]).to(device)
            mask_tensor = torch.FloatTensor(train_masks[i:i+1]).to(device)
            label_tensor = torch.LongTensor([train_labels[i]]).to(device)
            
            optimizer.zero_grad()
            logits, loss = model(data_tensor, mask_tensor, label_tensor)
            loss.backward()
            
            # 그래디언트 클리핑
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_train_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            train_total += 1
            if predicted.item() == train_labels[i]:
                train_correct += 1
        
        train_acc = train_correct / train_total
        avg_train_loss = total_train_loss / len(train_data)
        
        # 검증
        model.eval()
        total_val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for i in range(len(val_data)):
                data_tensor = torch.FloatTensor(val_data[i:i+1]).to(device)
                mask_tensor = torch.FloatTensor(val_masks[i:i+1]).to(device)
                label_tensor = torch.LongTensor([val_labels[i]]).to(device)
                
                logits, loss = model(data_tensor, mask_tensor, label_tensor)
                
                total_val_loss += loss.item()
                _, predicted = torch.max(logits, 1)
                val_total += 1
                if predicted.item() == val_labels[i]:
                    val_correct += 1
        
        val_acc = val_correct / val_total
        avg_val_loss = total_val_loss / len(val_data)
        
        # 히스토리 저장
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_acc)
        
        # 스케줄러 업데이트
        scheduler.step(avg_val_loss)
        
        # 조기 종료
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1
        
        # 진행 상황 출력
        status = "Normal"
        if patience_counter > 0:
            status = f"Patience {patience_counter}"
        
        if (epoch + 1) % 10 == 0 or epoch < 10:
            print(f'{epoch+1:<6} {avg_train_loss:<12.4f} {train_acc:<12.4f} {avg_val_loss:<12.4f} {val_acc:<12.4f} {status:<15}')
        
        if patience_counter >= max_patience:
            print(f'  조기 종료: {epoch+1} 에포크 (patience: {max_patience})')
            break
    
    print(f'✅ SignSpeak 스타일 모델 훈련 완료! 최종 검증 정확도: {best_val_acc:.4f}')
    
    return {
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
        'best_val_acc': best_val_acc,
        'final_epoch': len(train_losses)
    }

def main():
    """메인 실행 함수"""
    print('🚀 SignSpeak 스타일 GRU 모델 개발 시작')
    print('=' * 70)
    
    # 데이터 로드
    data_dir = '../SignGlove/external/SignGlove_HW/datasets/unified'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # SignSpeak 스타일 데이터 로드
    padded_data, padding_masks, all_labels, class_names, processor = load_and_preprocess_data_signspeak_style(
        data_dir, time_steps=79, max_samples_per_class=None
    )
    
    # K-fold 교차 검증 (SignSpeak 방식)
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    all_fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(padded_data, all_labels)):
        print(f'\n🔄 Fold {fold + 1}/5 훈련 시작')
        
        # 데이터 분할
        train_data = padded_data[train_idx]
        train_masks = padding_masks[train_idx]
        train_labels = [all_labels[i] for i in train_idx]
        
        val_data = padded_data[val_idx]
        val_masks = padding_masks[val_idx]
        val_labels = [all_labels[i] for i in val_idx]
        
        print(f'훈련 데이터: {len(train_data)}개')
        print(f'검증 데이터: {len(val_data)}개')
        
        # SignSpeak 스타일 모델 생성
        model = SignSpeakStyleGRU(
            input_size=8, 
            hidden_size=64, 
            num_classes=24, 
            layers=2, 
            dropout=0.2
        ).to(device)
        
        # 모델 파라미터 수 계산
        params = sum(p.numel() for p in model.parameters())
        print(f'SignSpeak 스타일 GRU 파라미터 수: {params:,}')
        
        # 모델 훈련
        history = train_signspeak_style_model(
            model, train_data, train_masks, train_labels, 
            val_data, val_masks, val_labels, device, epochs=100, 
            learning_rate=0.001
        )
        
        all_fold_results.append(history['best_val_acc'])
        
        # 모델 저장
        torch.save(model.state_dict(), f'signspeak_style_gru_fold_{fold+1}.pth')
    
    # 전체 결과 요약
    avg_accuracy = np.mean(all_fold_results)
    std_accuracy = np.std(all_fold_results)
    
    print(f'\n📊 SignSpeak 스타일 모델 K-fold 결과')
    print('=' * 70)
    for i, acc in enumerate(all_fold_results):
        print(f'Fold {i+1}: {acc:.4f}')
    print(f'평균 정확도: {avg_accuracy:.4f} ± {std_accuracy:.4f}')
    
    # 최고 성능 모델 저장
    best_fold = np.argmax(all_fold_results)
    print(f'최고 성능 Fold: {best_fold + 1} ({all_fold_results[best_fold]:.4f})')
    
    # 전처리기 저장
    import pickle
    with open('signspeak_processor.pkl', 'wb') as f:
        pickle.dump(processor, f)
    
    print(f'\n🎉 SignSpeak 스타일 모델 개발 완료!')
    print(f'최종 성능: {avg_accuracy:.4f} ± {std_accuracy:.4f}')

if __name__ == "__main__":
    main()

