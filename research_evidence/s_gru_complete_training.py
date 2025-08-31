#!/usr/bin/env python3
"""
S-GRU 완전 데이터 훈련
전체 600개 샘플을 모두 사용하여 훈련하는 S-GRU 모델

- 전체 600개 샘플 사용 (24개 클래스 × 25개씩)
- 과적합 방지를 위한 강한 정규화
- 실제 하드웨어 데이터에 최적화

작성자: KLP-SignGlove Team
버전: 5.0.0
날짜: 2024
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import h5py
import os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

class SGRU(nn.Module):
    """
    S-GRU 모델 (Simplified GRU)
    과적합 방지를 위한 최소한의 복잡도로 설계된 GRU 모델
    """
    def __init__(self, input_size=8, hidden_size=32, num_classes=24, dropout=0.5):
        super(SGRU, self).__init__()
        
        self.hidden_size = hidden_size
        
        # GRU 레이어
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        
        # 드롭아웃
        self.dropout = nn.Dropout(dropout)
        
        # 분류 레이어
        self.fc = nn.Linear(hidden_size, num_classes)
        
        # 가중치 초기화
        self._init_weights()
    
    def _init_weights(self):
        """가중치 초기화"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(self, x):
        """
        순전파
        Args:
            x (torch.Tensor): 입력 데이터 (batch_size, seq_len, input_size)
        Returns:
            torch.Tensor: 예측 결과 (batch_size, num_classes)
        """
        # GRU 처리
        gru_out, _ = self.gru(x)  # (batch_size, seq_len, hidden_size)
        
        # 마지막 시퀀스 사용
        last_output = gru_out[:, -1, :]  # (batch_size, hidden_size)
        
        # 드롭아웃
        last_output = self.dropout(last_output)
        
        # 분류
        output = self.fc(last_output)  # (batch_size, num_classes)
        
        return output

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

def load_and_preprocess_data(data_dir):
    """전체 데이터 로드 및 전처리"""
    print('📁 전체 데이터 로드 중...')
    
    class_names = ['ㄱ', 'ㄴ', 'ㄷ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅅ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ', 
                  'ㅏ', 'ㅑ', 'ㅓ', 'ㅕ', 'ㅗ', 'ㅛ', 'ㅜ', 'ㅠ', 'ㅡ', 'ㅣ']
    
    all_data = []
    all_labels = []
    class_counts = {}
    
    for class_idx, class_name in enumerate(class_names):
        class_counts[class_name] = 0
        
        # 모든 폴더 (1, 2, 3, 4, 5)에서 데이터 로드
        for folder_num in range(1, 6):
            class_dir = os.path.join(data_dir, class_name, str(folder_num))
            if not os.path.exists(class_dir):
                continue
            
            h5_files = [f for f in os.listdir(class_dir) if f.endswith('.h5')]
            class_counts[class_name] += len(h5_files)
            
            print(f'  {class_name}/폴더{folder_num}: {len(h5_files)}개 샘플 로드 중...')
            
            for h5_file in h5_files:
                file_path = os.path.join(class_dir, h5_file)
                try:
                    with h5py.File(file_path, 'r') as f:
                        data = f['sensor_data'][:]  # (300, 8)
                        all_data.append(data)
                        all_labels.append(class_idx)
                except Exception as e:
                    print(f"    ❌ 파일 읽기 실패: {h5_file} - {e}")
                    continue
    
    print(f'✅ 전체 데이터 로드 완료: {len(all_data)}개 샘플, {len(class_names)}개 클래스')
    print(f'📊 클래스별 샘플 수:')
    for class_name, count in class_counts.items():
        print(f'  {class_name}: {count}개')
    
    return all_data, all_labels, class_names

def train_s_gru_complete(model, all_data, all_labels, device, epochs=150, learning_rate=0.001):
    """전체 데이터로 S-GRU 모델 훈련"""
    print(f'🏋️ S-GRU 모델 전체 데이터 훈련 시작 (총 {epochs} 에폭)...')
    print(f'📊 전체 데이터: {len(all_data)}개 샘플')
    
    # 정규화 적용
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.5, verbose=True)
    
    train_losses = []
    train_accuracies = []
    
    best_acc = 0
    patience_counter = 0
    max_patience = 30
    
    print(f'📊 훈련 진행 상황 모니터링:')
    print(f'{"Epoch":<6} {"Train Loss":<12} {"Train Acc":<12} {"Status":<15}')
    print('-' * 50)
    
    for epoch in range(epochs):
        # 훈련
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        # 데이터 순서를 섞어서 훈련
        indices = torch.randperm(len(all_data))
        
        for idx in indices:
            data = all_data[idx]
            label = all_labels[idx]
            
            data_tensor = torch.FloatTensor(data).unsqueeze(0).to(device)
            label_tensor = torch.LongTensor([label]).to(device)
            
            optimizer.zero_grad()
            outputs = model(data_tensor)
            loss = criterion(outputs, label_tensor)
            loss.backward()
            
            # 그래디언트 클리핑
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += 1
            if predicted.item() == label:
                correct += 1
        
        train_acc = correct / total
        avg_loss = total_loss / len(all_data)
        
        # 히스토리 저장
        train_losses.append(avg_loss)
        train_accuracies.append(train_acc)
        
        # 스케줄러 업데이트
        scheduler.step(avg_loss)
        
        # 조기 종료
        if train_acc > best_acc:
            best_acc = train_acc
            patience_counter = 0
        else:
            patience_counter += 1
        
        # 진행 상황 출력
        status = "Normal"
        if patience_counter > 0:
            status = f"Patience {patience_counter}"
        
        if (epoch + 1) % 10 == 0 or epoch < 10:
            print(f'{epoch+1:<6} {avg_loss:<12.4f} {train_acc:<12.4f} {status:<15}')
        
        if patience_counter >= max_patience:
            print(f'  조기 종료: {epoch+1} 에포크 (patience: {max_patience})')
            break
    
    print(f'✅ S-GRU 모델 전체 데이터 훈련 완료! 최종 정확도: {best_acc:.4f}')
    
    return {
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'best_acc': best_acc,
        'final_epoch': len(train_losses)
    }

def save_s_gru_model(model, filepath='s_gru_complete_model.pth'):
    """S-GRU 모델 저장"""
    torch.save(model.state_dict(), filepath)
    print(f'✅ S-GRU 모델 저장 완료: {filepath}')

def create_training_visualization(history, class_names):
    """훈련 과정 시각화"""
    print('\n📊 훈련 과정 시각화 생성 중...')
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('🚀 S-GRU 완전 데이터 훈련 과정', fontsize=16, fontweight='bold')
    
    # 1. 훈련 손실 곡선
    ax1 = axes[0]
    epochs = range(1, len(history['train_losses']) + 1)
    ax1.plot(epochs, history['train_losses'], 'b-', label='Train Loss', linewidth=2, alpha=0.8)
    ax1.set_title('S-GRU: 훈련 손실 곡선')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 훈련 정확도 곡선
    ax2 = axes[1]
    ax2.plot(epochs, history['train_accuracies'], 'g-', label='Train Accuracy', linewidth=2, alpha=0.8)
    ax2.set_title('S-GRU: 훈련 정확도 곡선')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0.5, 1.0)
    
    plt.tight_layout()
    plt.savefig('s_gru_complete_training_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print('✅ 훈련 과정 시각화 저장: s_gru_complete_training_analysis.png')

def main():
    """메인 실행 함수"""
    print('🚀 S-GRU 완전 데이터 훈련 시작')
    print('=' * 70)
    
    # 데이터 로드
    data_dir = '../SignGlove/external/SignGlove_HW/datasets/unified'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    all_data, all_labels, class_names = load_and_preprocess_data(data_dir)
    
    print(f'\n📊 전체 데이터 사용: {len(all_data)}개 샘플')
    
    # 전처리
    preprocessor = AdvancedPreprocessor()
    preprocessor.fit(all_data)
    
    all_data_processed = preprocessor.transform(all_data)
    
    # S-GRU 모델 생성 및 훈련
    print('\n🔧 S-GRU 모델 완전 데이터 훈련 시작')
    s_gru_model = SGRU(input_size=8, hidden_size=32, num_classes=24, dropout=0.5).to(device)
    
    # 모델 파라미터 수 계산
    params = sum(p.numel() for p in s_gru_model.parameters())
    print(f'S-GRU 파라미터 수: {params:,}')
    
    # 모델 훈련 (전체 데이터 사용)
    history = train_s_gru_complete(
        s_gru_model, all_data_processed, all_labels, device, epochs=150, 
        learning_rate=0.001
    )
    
    # 모델 저장
    save_s_gru_model(s_gru_model, 'models/s_gru_complete_model.pth')
    
    # 결과 요약
    print('\n📊 S-GRU 완전 데이터 훈련 결과 요약')
    print('=' * 70)
    print(f'파라미터 수: {params:,}')
    print(f'훈련 데이터: {len(all_data)}개')
    print(f'최종 정확도: {history["best_acc"]:.4f}')
    print(f'훈련 에폭: {history["final_epoch"]}')
    
    # 시각화 생성
    create_training_visualization(history, class_names)
    
    print(f'\n🎉 S-GRU 완전 데이터 훈련 완료!')
    print(f'최종 성능: {history["best_acc"]:.4f} 정확도')
    print(f'모델 저장: models/s_gru_complete_model.pth')

if __name__ == "__main__":
    main()
