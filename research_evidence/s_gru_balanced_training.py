#!/usr/bin/env python3
"""
S-GRU 균형잡힌 훈련 시스템
각 폴더에서 일부씩 가져와서 균형잡힌 분할을 하는 S-GRU 모델

- 각 클래스의 각 폴더에서:
  - 훈련: 3개씩 (총 15개/클래스)
  - 검증: 1개씩 (총 5개/클래스)
  - 테스트: 1개씩 (총 5개/클래스)
- 모든 각도가 훈련/검증/테스트에 포함됨

작성자: KLP-SignGlove Team
버전: 8.0.0
날짜: 2024
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import h5py
import os
import random
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Font settings for English
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 12
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

def load_and_split_data_balanced(data_dir, train_per_folder=3, val_per_folder=1, test_per_folder=1):
    """각 폴더에서 균형잡힌 분할로 데이터 로드"""
    print('📁 균형잡힌 데이터 로드 및 분할 중...')
    
    class_names = ['ㄱ', 'ㄴ', 'ㄷ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅅ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ', 
                  'ㅏ', 'ㅑ', 'ㅓ', 'ㅕ', 'ㅗ', 'ㅛ', 'ㅜ', 'ㅠ', 'ㅡ', 'ㅣ']
    
    train_data = []
    train_labels = []
    val_data = []
    val_labels = []
    test_data = []
    test_labels = []
    
    # 각도별 통계
    angle_stats = {1: {'train': 0, 'val': 0, 'test': 0}, 
                  2: {'train': 0, 'val': 0, 'test': 0},
                  3: {'train': 0, 'val': 0, 'test': 0},
                  4: {'train': 0, 'val': 0, 'test': 0},
                  5: {'train': 0, 'val': 0, 'test': 0}}
    
    for class_idx, class_name in enumerate(class_names):
        print(f'  {class_name} 클래스 처리 중...')
        
        # 각도별로 데이터 로드 및 분할
        for angle in range(1, 6):
            class_dir = os.path.join(data_dir, class_name, str(angle))
            if not os.path.exists(class_dir):
                continue
            
            h5_files = [f for f in os.listdir(class_dir) if f.endswith('.h5')]
            random.shuffle(h5_files)  # 랜덤 섞기
            
            print(f'    각도 {angle}: {len(h5_files)}개 샘플 -> 훈련:{train_per_folder}개, 검증:{val_per_folder}개, 테스트:{test_per_folder}개')
            
            # 훈련 데이터 (앞에서부터)
            for i in range(min(train_per_folder, len(h5_files))):
                file_path = os.path.join(class_dir, h5_files[i])
                try:
                    with h5py.File(file_path, 'r') as f:
                        data = f['sensor_data'][:]
                        train_data.append(data)
                        train_labels.append(class_idx)
                        angle_stats[angle]['train'] += 1
                except Exception as e:
                    print(f"      ❌ 훈련 파일 읽기 실패: {h5_files[i]} - {e}")
            
            # 검증 데이터
            for i in range(train_per_folder, min(train_per_folder + val_per_folder, len(h5_files))):
                file_path = os.path.join(class_dir, h5_files[i])
                try:
                    with h5py.File(file_path, 'r') as f:
                        data = f['sensor_data'][:]
                        val_data.append(data)
                        val_labels.append(class_idx)
                        angle_stats[angle]['val'] += 1
                except Exception as e:
                    print(f"      ❌ 검증 파일 읽기 실패: {h5_files[i]} - {e}")
            
            # 테스트 데이터
            for i in range(train_per_folder + val_per_folder, min(train_per_folder + val_per_folder + test_per_folder, len(h5_files))):
                file_path = os.path.join(class_dir, h5_files[i])
                try:
                    with h5py.File(file_path, 'r') as f:
                        data = f['sensor_data'][:]
                        test_data.append(data)
                        test_labels.append(class_idx)
                        angle_stats[angle]['test'] += 1
                except Exception as e:
                    print(f"      ❌ 테스트 파일 읽기 실패: {h5_files[i]} - {e}")
    
    # 각도별 통계 출력
    print(f'\n📊 각도별 분할 통계:')
    for angle in range(1, 6):
        print(f'  각도 {angle}: 훈련 {angle_stats[angle]["train"]}개, 검증 {angle_stats[angle]["val"]}개, 테스트 {angle_stats[angle]["test"]}개')
    
    # 클래스별 통계
    print(f'\n📊 클래스별 분할 통계:')
    class_stats = {}
    for label in train_labels:
        class_name = class_names[label]
        class_stats[class_name] = class_stats.get(class_name, 0) + 1
    
    for class_name in class_names:
        train_count = class_stats.get(class_name, 0)
        val_count = val_labels.count(class_names.index(class_name))
        test_count = test_labels.count(class_names.index(class_name))
        print(f'  {class_name}: 훈련 {train_count}개, 검증 {val_count}개, 테스트 {test_count}개')
    
    print(f'\n📊 최종 분할 결과:')
    print(f'  - 훈련 데이터: {len(train_data)}개')
    print(f'  - 검증 데이터: {len(val_data)}개')
    print(f'  - 테스트 데이터: {len(test_data)}개')
    
    return train_data, val_data, test_data, train_labels, val_labels, test_labels, class_names

def train_s_gru_balanced(model, train_data, train_labels, val_data, val_labels, device, epochs=150, learning_rate=0.001):
    """균형잡힌 S-GRU 모델 훈련"""
    print(f'🏋️ S-GRU 모델 균형잡힌 훈련 시작 (총 {epochs} 에폭)...')
    print(f'📊 훈련 데이터: {len(train_data)}개, 검증 데이터: {len(val_data)}개')
    
    # 정규화 적용
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
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
    print('-' * 80)
    
    for epoch in range(epochs):
        # 훈련
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        # 훈련 데이터 순서를 섞어서 훈련
        indices = torch.randperm(len(train_data))
        
        for idx in indices:
            data = train_data[idx]
            label = train_labels[idx]
            
            data_tensor = torch.FloatTensor(data).unsqueeze(0).to(device)
            label_tensor = torch.LongTensor([label]).to(device)
            
            optimizer.zero_grad()
            outputs = model(data_tensor)
            loss = criterion(outputs, label_tensor)
            loss.backward()
            
            # 그래디언트 클리핑
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += 1
            if predicted.item() == label:
                train_correct += 1
        
        train_acc = train_correct / train_total
        avg_train_loss = train_loss / len(train_data)
        
        # 검증
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, label in zip(val_data, val_labels):
                data_tensor = torch.FloatTensor(data).unsqueeze(0).to(device)
                label_tensor = torch.LongTensor([label]).to(device)
                
                outputs = model(data_tensor)
                loss = criterion(outputs, label_tensor)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += 1
                if predicted.item() == label:
                    val_correct += 1
        
        val_acc = val_correct / val_total
        avg_val_loss = val_loss / len(val_data)
        
        # 히스토리 저장
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_acc)
        
        # 스케줄러 업데이트 (검증 손실 기준)
        scheduler.step(avg_val_loss)
        
        # 조기 종료 (검증 정확도 기준)
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
    
    print(f'✅ S-GRU 모델 균형잡힌 훈련 완료!')
    print(f'  - 최고 검증 정확도: {best_val_acc:.4f}')
    print(f'  - 최종 훈련 정확도: {train_acc:.4f}')
    print(f'  - 최종 검증 정확도: {val_acc:.4f}')
    
    return {
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
        'best_val_acc': best_val_acc,
        'final_train_acc': train_acc,
        'final_val_acc': val_acc,
        'final_epoch': len(train_losses)
    }

def evaluate_s_gru_model(model, test_data, test_labels, device, class_names):
    """S-GRU 모델 테스트 평가"""
    print('\n📊 S-GRU 모델 테스트 평가 중...')
    
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for data, label in zip(test_data, test_labels):
            data_tensor = torch.FloatTensor(data).unsqueeze(0).to(device)
            outputs = model(data_tensor)
            _, predicted = torch.max(outputs, 1)
            
            all_predictions.append(predicted.item())
            all_labels.append(label)
    
    # 정확도 계산
    accuracy = accuracy_score(all_labels, all_predictions)
    
    # 분류 보고서
    report = classification_report(all_labels, all_predictions, 
                                 target_names=class_names, output_dict=True)
    
    # 혼동 행렬
    cm = confusion_matrix(all_labels, all_predictions)
    
    print(f'테스트 정확도: {accuracy:.4f}')
    print(f'클래스별 F1-score:')
    for i, class_name in enumerate(class_names):
        if class_name in report:
            f1 = report[class_name]['f1-score']
            print(f'  {class_name}: {f1:.4f}')
    
    return accuracy, report, cm

def save_s_gru_model(model, filepath='s_gru_balanced_model.pth'):
    """S-GRU 모델 저장"""
    torch.save(model.state_dict(), filepath)
    print(f'✅ S-GRU 모델 저장 완료: {filepath}')

def create_training_visualization(history, class_names):
    """Training process visualization"""
    print('\n📊 Creating training visualization...')
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('🚀 S-GRU Balanced Training Process', fontsize=16, fontweight='bold')
    
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
    plt.savefig('s_gru_balanced_training_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print('✅ Training visualization saved: s_gru_balanced_training_analysis.png')

def main():
    """메인 실행 함수"""
    print('🚀 S-GRU 균형잡힌 훈련 시작')
    print('=' * 70)
    
    # 시드 설정 (재현 가능성)
    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 데이터 로드 및 균형잡힌 분할
    data_dir = '../SignGlove/external/SignGlove_HW/datasets/unified'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    train_data, val_data, test_data, train_labels, val_labels, test_labels, class_names = load_and_split_data_balanced(
        data_dir, train_per_folder=3, val_per_folder=1, test_per_folder=1
    )
    
    # 전처리 (훈련 데이터로만 학습)
    preprocessor = AdvancedPreprocessor()
    preprocessor.fit(train_data)
    
    # 전처리 파라미터 저장
    preprocessor_params = preprocessor.get_parameters()
    torch.save(preprocessor_params, 'models/preprocessor_params.pth')
    print('✅ 전처리 파라미터 저장 완료: models/preprocessor_params.pth')
    
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
    save_s_gru_model(s_gru_model, 'models/s_gru_balanced_model.pth')
    
    # 결과 요약
    print('\n📊 S-GRU 균형잡힌 훈련 결과 요약')
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
    
    print(f'\n🎉 S-GRU 균형잡힌 훈련 완료!')
    print(f'테스트 성능: {test_accuracy:.4f} 정확도')
    print(f'모델 저장: models/s_gru_balanced_model.pth')

if __name__ == "__main__":
    main()
