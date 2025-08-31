#!/usr/bin/env python3
"""
S-GRU 필터링된 데이터 균형잡힌 훈련
필터링된 데이터를 사용하여 균형잡힌 분할로 모델을 훈련합니다.
"""

import os
import sys
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.unicode_minus'] = False

class AdvancedPreprocessor:
    """고급 전처리기 (필터링된 데이터용)"""
    
    def __init__(self):
        self.flex_scalers = [StandardScaler() for _ in range(5)]
        self.orientation_scalers = [StandardScaler() for _ in range(3)]
        self.is_fitted = False
    
    def fit(self, data_list):
        """전처리 파라미터 학습"""
        print('🔧 전처리 파라미터 학습 중...')
        
        # Flex 센서 (0-4) 학습
        for sensor_idx in range(5):
            all_values = []
            for data in data_list:
                sensor_values = data[:, sensor_idx]
                # 0값 처리
                valid_values = sensor_values[sensor_values > 0]
                if len(valid_values) > 0:
                    all_values.extend(valid_values)
            
            if all_values:
                self.flex_scalers[sensor_idx].fit(np.array(all_values).reshape(-1, 1))
        
        # Orientation 센서 (5-7) 학습
        for sensor_idx in range(3):
            all_values = []
            for data in data_list:
                sensor_values = data[:, sensor_idx + 5]
                all_values.extend(sensor_values)
            
            if all_values:
                self.orientation_scalers[sensor_idx].fit(np.array(all_values).reshape(-1, 1))
        
        self.is_fitted = True
        print('✅ 전처리 파라미터 학습 완료')
    
    def transform(self, data_list):
        """데이터 변환"""
        if not self.is_fitted:
            raise ValueError("전처리기를 먼저 학습해야 합니다.")
        
        processed_data = []
        
        for data in data_list:
            processed = data.copy()
            
            # Flex 센서 (0-4) 처리
            for sensor_idx in range(5):
                sensor_values = data[:, sensor_idx]
                # 0값 처리
                mean_val = np.mean(sensor_values[sensor_values > 0])
                if np.isnan(mean_val):
                    mean_val = 500
                sensor_values[sensor_values == 0] = mean_val
                
                # 정규화
                normalized = self.flex_scalers[sensor_idx].transform(sensor_values.reshape(-1, 1)).flatten()
                processed[:, sensor_idx] = normalized
            
            # Orientation 센서 (5-7) 처리
            for sensor_idx in range(3):
                sensor_values = data[:, sensor_idx + 5]
                normalized = self.orientation_scalers[sensor_idx].transform(sensor_values.reshape(-1, 1)).flatten()
                processed[:, sensor_idx + 5] = normalized
            
            processed_data.append(processed)
        
        return processed_data
    
    def get_parameters(self):
        """전처리 파라미터 추출"""
        if not self.is_fitted:
            return None
        
        params = {
            'flex_scalers': [],
            'orientation_scalers': []
        }
        
        for scaler in self.flex_scalers:
            params['flex_scalers'].append({
                'mean_': scaler.mean_[0] if scaler.mean_ is not None else 0,
                'scale_': scaler.scale_[0] if scaler.scale_ is not None else 1,
                'var_': scaler.var_[0] if scaler.var_ is not None else 0
            })
        
        for scaler in self.orientation_scalers:
            params['orientation_scalers'].append({
                'mean_': scaler.mean_[0] if scaler.mean_ is not None else 0,
                'scale_': scaler.scale_[0] if scaler.scale_ is not None else 1,
                'var_': scaler.var_[0] if scaler.var_ is not None else 0
            })
        
        return params
    
    def load_parameters(self, params):
        """전처리 파라미터 로드"""
        if params is None:
            return
        
        # Flex 센서 파라미터 로드
        for i, param in enumerate(params['flex_scalers']):
            self.flex_scalers[i].mean_ = np.array([param['mean_']])
            self.flex_scalers[i].scale_ = np.array([param['scale_']])
            self.flex_scalers[i].var_ = np.array([param['var_']])
        
        # Orientation 센서 파라미터 로드
        for i, param in enumerate(params['orientation_scalers']):
            self.orientation_scalers[i].mean_ = np.array([param['mean_']])
            self.orientation_scalers[i].scale_ = np.array([param['scale_']])
            self.orientation_scalers[i].var_ = np.array([param['var_']])
        
        self.is_fitted = True
        print('✅ 전처리 파라미터 로드 완료')

class StandardScaler:
    """간단한 표준 스케일러"""
    
    def __init__(self):
        self.mean_ = None
        self.scale_ = None
        self.var_ = None
    
    def fit(self, X):
        self.mean_ = np.mean(X, axis=0)
        self.var_ = np.var(X, axis=0)
        self.scale_ = np.sqrt(self.var_)
        return self
    
    def transform(self, X):
        return (X - self.mean_) / (self.scale_ + 1e-8)

class SGRU(nn.Module):
    """S-GRU 모델 (Simplified GRU)"""
    
    def __init__(self, input_size=8, hidden_size=32, num_classes=24, dropout=0.5):
        super(SGRU, self).__init__()
        self.hidden_size = hidden_size
        
        # GRU 레이어
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True, dropout=dropout)
        
        # 분류 레이어
        self.fc = nn.Linear(hidden_size, num_classes)
        
        # 드롭아웃
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # GRU 처리
        gru_out, _ = self.gru(x)
        
        # 마지막 시퀀스 출력 사용
        last_output = gru_out[:, -1, :]
        
        # 드롭아웃 적용
        dropped = self.dropout(last_output)
        
        # 분류
        output = self.fc(dropped)
        
        return output

def load_and_split_data_balanced(data_dir, min_samples_per_angle=3):
    """균형잡힌 데이터 로드 및 분할 (필터링된 데이터에 맞게 동적 조정)"""
    print('📁 균형잡힌 데이터 로드 및 분할 중...')
    
    class_names = ['ㄱ', 'ㄴ', 'ㄷ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅅ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ',
                   'ㅏ', 'ㅑ', 'ㅓ', 'ㅕ', 'ㅗ', 'ㅛ', 'ㅜ', 'ㅠ', 'ㅡ', 'ㅣ']
    
    train_data, val_data, test_data = [], [], []
    train_labels, val_labels, test_labels = [], [], []
    
    # 각도별 통계
    angle_stats = {i: {'train': 0, 'val': 0, 'test': 0} for i in range(1, 6)}
    class_stats = {name: {'train': 0, 'val': 0, 'test': 0} for name in class_names}
    
    # 분할 비율 설정
    train_ratio, val_ratio, test_ratio = 0.6, 0.2, 0.2
    
    for class_idx, class_name in enumerate(class_names):
        print(f'  {class_name} 클래스 처리 중...')
        class_dir = Path(data_dir) / class_name
        
        if not class_dir.exists():
            print(f'    ⚠️ 클래스 디렉토리 없음: {class_name}')
            continue
        
        for angle in range(1, 6):
            angle_dir = class_dir / str(angle)
            if not angle_dir.exists():
                print(f'    ⚠️ 각도 디렉토리 없음: {class_name}/{angle}')
                continue
            
            # CSV 파일 목록
            csv_files = list(angle_dir.glob("*.csv"))
            if not csv_files:
                print(f'    ⚠️ CSV 파일 없음: {class_name}/{angle}')
                continue
            
            # 최소 샘플 수 확인
            if len(csv_files) < min_samples_per_angle:
                print(f'    ⚠️ 샘플 수 부족: {class_name}/{angle} ({len(csv_files)}개 < {min_samples_per_angle}개)')
                continue
            
            # 파일을 랜덤하게 섞기
            random.shuffle(csv_files)
            
            # 동적 분할 (비율 기반)
            total_samples = len(csv_files)
            train_count = max(1, int(total_samples * train_ratio))
            val_count = max(1, int(total_samples * val_ratio))
            test_count = total_samples - train_count - val_count
            
            # 최소 1개씩은 보장
            if test_count < 1:
                test_count = 1
                val_count = max(1, total_samples - train_count - test_count)
            
            train_files = csv_files[:train_count]
            val_files = csv_files[train_count:train_count + val_count]
            test_files = csv_files[train_count + val_count:]
            
            print(f'    각도 {angle}: {len(csv_files)}개 샘플 -> 훈련:{len(train_files)}개, 검증:{len(val_files)}개, 테스트:{len(test_files)}개')
            
            # 훈련 데이터 로드
            for file_path in train_files:
                try:
                    df = pd.read_csv(file_path)
                    data = df.iloc[:, :8].values.astype(np.float32)
                    train_data.append(data)
                    train_labels.append(class_idx)
                    angle_stats[angle]['train'] += 1
                    class_stats[class_name]['train'] += 1
                except Exception as e:
                    print(f'    ⚠️ 파일 읽기 실패: {file_path} - {e}')
            
            # 검증 데이터 로드
            for file_path in val_files:
                try:
                    df = pd.read_csv(file_path)
                    data = df.iloc[:, :8].values.astype(np.float32)
                    val_data.append(data)
                    val_labels.append(class_idx)
                    angle_stats[angle]['val'] += 1
                    class_stats[class_name]['val'] += 1
                except Exception as e:
                    print(f'    ⚠️ 파일 읽기 실패: {file_path} - {e}')
            
            # 테스트 데이터 로드
            for file_path in test_files:
                try:
                    df = pd.read_csv(file_path)
                    data = df.iloc[:, :8].values.astype(np.float32)
                    test_data.append(data)
                    test_labels.append(class_idx)
                    angle_stats[angle]['test'] += 1
                    class_stats[class_name]['test'] += 1
                except Exception as e:
                    print(f'    ⚠️ 파일 읽기 실패: {file_path} - {e}')
    
    # 통계 출력
    print('\n📊 각도별 분할 통계:')
    for angle in range(1, 6):
        print(f'  각도 {angle}: 훈련 {angle_stats[angle]["train"]}개, 검증 {angle_stats[angle]["val"]}개, 테스트 {angle_stats[angle]["test"]}개')
    
    print('\n📊 클래스별 분할 통계:')
    for class_name in class_names:
        print(f'  {class_name}: 훈련 {class_stats[class_name]["train"]}개, 검증 {class_stats[class_name]["val"]}개, 테스트 {class_stats[class_name]["test"]}개')
    
    print(f'\n📊 최종 분할 결과:')
    print(f'  - 훈련 데이터: {len(train_data)}개')
    print(f'  - 검증 데이터: {len(val_data)}개')
    print(f'  - 테스트 데이터: {len(test_data)}개')
    
    return train_data, val_data, test_data, train_labels, val_labels, test_labels, class_names

def train_s_gru_balanced(model, train_data, train_labels, val_data, val_labels, device, epochs=150, learning_rate=0.001):
    """S-GRU 균형잡힌 훈련"""
    print(f'🏋️ S-GRU 모델 균형잡힌 훈련 시작 (총 {epochs} 에폭)...')
    print(f'📊 훈련 데이터: {len(train_data)}개, 검증 데이터: {len(val_data)}개')
    
    # 시퀀스 길이 통일 (패딩)
    max_length = max(len(data) for data in train_data + val_data)
    
    # 훈련 데이터 패딩
    padded_train_data = []
    for data in train_data:
        if len(data) < max_length:
            padding = np.tile(data[-1:], (max_length - len(data), 1))
            padded_data = np.vstack([data, padding])
        else:
            padded_data = data
        padded_train_data.append(padded_data)
    
    # 검증 데이터 패딩
    padded_val_data = []
    for data in val_data:
        if len(data) < max_length:
            padding = np.tile(data[-1:], (max_length - len(data), 1))
            padded_data = np.vstack([data, padding])
        else:
            padded_data = data
        padded_val_data.append(padded_data)
    
    # 데이터 로더 생성
    train_dataset = TensorDataset(torch.FloatTensor(np.array(padded_train_data)), torch.LongTensor(train_labels))
    val_dataset = TensorDataset(torch.FloatTensor(np.array(padded_val_data)), torch.LongTensor(val_labels))
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # 손실 함수 및 옵티마이저
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10, verbose=True)
    
    # 훈련 기록
    history = {
        'train_losses': [], 'train_accuracies': [],
        'val_losses': [], 'val_accuracies': [],
        'best_val_acc': 0, 'final_train_acc': 0, 'final_val_acc': 0, 'final_epoch': 0
    }
    
    # 조기 종료
    patience = 30
    patience_counter = 0
    
    print('📊 훈련 진행 상황 모니터링:')
    print(f"{'Epoch':<6} {'Train Loss':<12} {'Train Acc':<12} {'Val Loss':<12} {'Val Acc':<12} {'Status':<12}")
    print('-' * 80)
    
    for epoch in range(epochs):
        # 훈련
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch_data, batch_labels in train_loader:
            batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            
            # 그래디언트 클리핑
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += batch_labels.size(0)
            train_correct += (predicted == batch_labels).sum().item()
        
        # 검증
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_data, batch_labels in val_loader:
                batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
                outputs = model(batch_data)
                loss = criterion(outputs, batch_labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += batch_labels.size(0)
                val_correct += (predicted == batch_labels).sum().item()
        
        # 평균 계산
        train_loss_avg = train_loss / len(train_loader)
        train_acc_avg = train_correct / train_total
        val_loss_avg = val_loss / len(val_loader)
        val_acc_avg = val_correct / val_total
        
        # 기록
        history['train_losses'].append(train_loss_avg)
        history['train_accuracies'].append(train_acc_avg)
        history['val_losses'].append(val_loss_avg)
        history['val_accuracies'].append(val_acc_avg)
        
        # 스케줄러 업데이트
        scheduler.step(val_acc_avg)
        
        # 최고 성능 추적
        if val_acc_avg > history['best_val_acc']:
            history['best_val_acc'] = val_acc_avg
            patience_counter = 0
            status = 'Normal'
        else:
            patience_counter += 1
            status = f'Patience {patience_counter}'
        
        # 진행 상황 출력
        if (epoch + 1) % 10 == 1 or epoch < 10:
            print(f"{epoch+1:<6} {train_loss_avg:<12.4f} {train_acc_avg:<12.4f} {val_loss_avg:<12.4f} {val_acc_avg:<12.4f} {status:<12}")
        
        # 조기 종료
        if patience_counter >= patience:
            print(f'  조기 종료: {epoch + 1} 에포크 (patience: {patience})')
            break
    
    # 최종 결과 저장
    history['final_train_acc'] = train_acc_avg
    history['final_val_acc'] = val_acc_avg
    history['final_epoch'] = epoch + 1
    
    print('✅ S-GRU 모델 균형잡힌 훈련 완료!')
    print(f'  - 최고 검증 정확도: {history["best_val_acc"]:.4f}')
    print(f'  - 최종 훈련 정확도: {history["final_train_acc"]:.4f}')
    print(f'  - 최종 검증 정확도: {history["final_val_acc"]:.4f}')
    
    return history

def evaluate_s_gru_model(model, test_data, test_labels, device, class_names):
    """S-GRU 모델 평가"""
    print('📊 S-GRU 모델 테스트 평가 중...')
    
    # 시퀀스 길이 통일 (패딩)
    max_length = max(len(data) for data in test_data)
    padded_test_data = []
    
    for data in test_data:
        if len(data) < max_length:
            # 패딩: 마지막 값으로 채우기
            padding = np.tile(data[-1:], (max_length - len(data), 1))
            padded_data = np.vstack([data, padding])
        else:
            padded_data = data
        padded_test_data.append(padded_data)
    
    # 데이터 로더 생성
    test_dataset = TensorDataset(torch.FloatTensor(np.array(padded_test_data)), torch.LongTensor(test_labels))
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    model.eval()
    test_correct = 0
    test_total = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch_data, batch_labels in test_loader:
            batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
            outputs = model(batch_data)
            _, predicted = torch.max(outputs.data, 1)
            
            test_total += batch_labels.size(0)
            test_correct += (predicted == batch_labels).sum().item()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())
    
    accuracy = test_correct / test_total
    print(f'테스트 정확도: {accuracy:.4f}')
    
    # 클래스별 F1-score 계산
    from sklearn.metrics import classification_report
    report = classification_report(all_labels, all_predictions, target_names=class_names, output_dict=True)
    
    print('클래스별 F1-score:')
    for class_name in class_names:
        if class_name in report:
            f1_score = report[class_name]['f1-score']
            print(f'  {class_name}: {f1_score:.4f}')
    
    return accuracy, report, None

def save_s_gru_model(model, filepath):
    """S-GRU 모델 저장"""
    torch.save(model.state_dict(), filepath)
    print(f'✅ S-GRU 모델 저장 완료: {filepath}')

def create_training_visualization(history, class_names):
    """훈련 과정 시각화"""
    print('\n📊 Creating training visualization...')
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('🚀 S-GRU Filtered Data Balanced Training Process', fontsize=16, fontweight='bold')
    
    epochs = range(1, len(history['train_losses']) + 1)
    
    # 1. Training/Validation Loss Curves
    ax1 = axes[0, 0]
    ax1.plot(epochs, history['train_losses'], 'b-', label='Training Loss', linewidth=2, alpha=0.8)
    ax1.plot(epochs, history['val_losses'], 'r-', label='Validation Loss', linewidth=2, alpha=0.8)
    ax1.set_title('S-GRU: Loss Curves', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # 2. Training/Validation Accuracy Curves
    ax2 = axes[0, 1]
    ax2.plot(epochs, history['train_accuracies'], 'b-', label='Training Accuracy', linewidth=2, alpha=0.8)
    ax2.plot(epochs, history['val_accuracies'], 'r-', label='Validation Accuracy', linewidth=2, alpha=0.8)
    ax2.set_title('S-GRU: Accuracy Curves', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0.5, 1.0)
    
    # 3. Overfitting Analysis
    ax3 = axes[1, 0]
    train_val_diff = [t - v for t, v in zip(history['train_accuracies'], history['val_accuracies'])]
    ax3.plot(epochs, train_val_diff, 'g-', label='Train-Val Difference', linewidth=2, alpha=0.8)
    ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax3.set_title('Overfitting Analysis (Train-Val Accuracy Difference)', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('Accuracy Difference', fontsize=12)
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3)
    
    # 4. Loss Difference Analysis
    ax4 = axes[1, 1]
    loss_diff = [v - t for t, v in zip(history['train_losses'], history['val_losses'])]
    ax4.plot(epochs, loss_diff, 'm-', label='Val-Train Loss Difference', linewidth=2, alpha=0.8)
    ax4.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax4.set_title('Loss Difference Analysis (Val-Train Loss Difference)', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Epoch', fontsize=12)
    ax4.set_ylabel('Loss Difference', fontsize=12)
    ax4.legend(fontsize=11)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('s_gru_filtered_training_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print('✅ Training visualization saved: s_gru_filtered_training_analysis.png')

def main():
    """메인 실행 함수"""
    print('🚀 S-GRU 필터링된 데이터 균형잡힌 훈련 시작')
    print('=' * 70)
    
    # 시드 설정 (재현 가능성)
    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 데이터 로드 및 균형잡힌 분할
    data_dir = 'real_data_filtered'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    train_data, val_data, test_data, train_labels, val_labels, test_labels, class_names = load_and_split_data_balanced(
        data_dir, min_samples_per_angle=3
    )
    
    # 전처리 (훈련 데이터로만 학습)
    preprocessor = AdvancedPreprocessor()
    preprocessor.fit(train_data)
    
    # 전처리 파라미터 저장
    preprocessor_params = preprocessor.get_parameters()
    torch.save(preprocessor_params, 'models/preprocessor_params_filtered.pth')
    print('✅ 전처리 파라미터 저장 완료: models/preprocessor_params_filtered.pth')
    
    train_data_processed = preprocessor.transform(train_data)
    val_data_processed = preprocessor.transform(val_data)
    test_data_processed = preprocessor.transform(test_data)
    
    # S-GRU 모델 생성 및 훈련
    print('\n🔧 S-GRU 모델 균형잡힌 훈련 시작')
    s_gru_model = SGRU(input_size=8, hidden_size=32, num_classes=24, dropout=0.5).to(device)
    
    # 모델 파라미터 수 계산
    params = sum(p.numel() for p in s_gru_model.parameters())
    print(f'S-GRU 파라미터 수: {params:,}')
    
    # 모델 훈련 (균형잡힌 분할 사용)
    history = train_s_gru_balanced(
        s_gru_model, train_data_processed, train_labels, 
        val_data_processed, val_labels, device, epochs=150, 
        learning_rate=0.001
    )
    
    # 테스트 평가
    test_accuracy, test_report, test_cm = evaluate_s_gru_model(
        s_gru_model, test_data_processed, test_labels, device, class_names
    )
    
    # 모델 저장
    save_s_gru_model(s_gru_model, 'models/s_gru_filtered_model.pth')
    
    # 결과 요약
    print('\n📊 S-GRU 필터링된 데이터 균형잡힌 훈련 결과 요약')
    print('=' * 70)
    print(f'파라미터 수: {params:,}')
    print(f'훈련 데이터: {len(train_data)}개')
    print(f'검증 데이터: {len(val_data)}개')
    print(f'테스트 데이터: {len(test_data)}개')
    print(f'최고 검증 정확도: {history["best_val_acc"]:.4f}')
    print(f'최종 훈련 정확도: {history["final_train_acc"]:.4f}')
    print(f'최종 검증 정확도: {history["final_val_acc"]:.4f}')
    print(f'테스트 정확도: {test_accuracy:.4f}')
    print(f'훈련 에폭: {history["final_epoch"]}')
    
    # 과적합 분석
    overfitting_score = history["final_train_acc"] - history["final_val_acc"]
    print(f'과적합 점수: {overfitting_score:.4f} (0에 가까울수록 좋음)')
    
    # 시각화 생성
    create_training_visualization(history, class_names)
    
    print(f'\n🎉 S-GRU 필터링된 데이터 균형잡힌 훈련 완료!')
    print(f'테스트 성능: {test_accuracy:.4f} 정확도')
    print(f'모델 저장: models/s_gru_filtered_model.pth')

if __name__ == "__main__":
    main()
