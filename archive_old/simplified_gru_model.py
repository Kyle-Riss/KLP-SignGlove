#!/usr/bin/env python3
"""
KLP-SignGlove: 단순화된 GRU 모델
과적합 문제 해결을 위한 간단한 모델 구조

- 모델 복잡도 대폭 감소
- 강한 정규화 적용
- 과적합 방지 기법 적용

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
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

class SimpleGRU(nn.Module):
    """
    단순화된 GRU 모델
    과적합 방지를 위한 간단한 구조
    """
    def __init__(self, input_size=8, hidden_size=32, num_classes=24, dropout=0.5):
        super(SimpleGRU, self).__init__()
        
        self.hidden_size = hidden_size
        
        # 단일 GRU 레이어 (복잡도 감소)
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True, dropout=0.0)
        
        # 강한 드롭아웃
        self.dropout = nn.Dropout(dropout)
        
        # 단순한 분류 레이어
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
        
        # 글로벌 평균 풀링
        pooled = torch.mean(gru_out, dim=1)  # (batch_size, hidden_size)
        
        # 드롭아웃 적용
        pooled = self.dropout(pooled)
        
        # 분류
        output = self.fc(pooled)  # (batch_size, num_classes)
        
        return output

class UltraSimpleGRU(nn.Module):
    """
    초간단 GRU 모델
    최소한의 복잡도로 과적합 완전 방지
    """
    def __init__(self, input_size=8, hidden_size=16, num_classes=24, dropout=0.6):
        super(UltraSimpleGRU, self).__init__()
        
        self.hidden_size = hidden_size
        
        # 매우 작은 GRU
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        
        # 매우 강한 드롭아웃
        self.dropout = nn.Dropout(dropout)
        
        # 단일 분류 레이어
        self.fc = nn.Linear(hidden_size, num_classes)
        
        # 가중치 초기화
        self._init_weights()
    
    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(self, x):
        # GRU 처리
        gru_out, _ = self.gru(x)
        
        # 마지막 시퀀스만 사용 (더 단순)
        last_output = gru_out[:, -1, :]  # (batch_size, hidden_size)
        
        # 강한 드롭아웃
        last_output = self.dropout(last_output)
        
        # 분류
        output = self.fc(last_output)
        
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

def load_and_preprocess_data(data_dir, max_samples_per_class=None):
    """데이터 로드 및 전처리"""
    print('📁 데이터 로드 중...')
    
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
    
    return all_data, all_labels, class_names

def train_simplified_model(model, train_data, train_labels, val_data, val_labels, 
                          device, epochs=100, learning_rate=0.001, model_name="Model"):
    """단순화된 모델 훈련"""
    print(f'🏋️ {model_name} 훈련 시작 (총 {epochs} 에폭)...')
    
    # 강한 정규화 적용
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-3)  # 강한 L2 정규화
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5, verbose=True)
    
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    
    best_val_acc = 0
    patience_counter = 0
    max_patience = 20  # 더 짧은 patience
    
    print(f'📊 훈련 진행 상황 모니터링:')
    print(f'{"Epoch":<6} {"Train Loss":<12} {"Train Acc":<12} {"Val Loss":<12} {"Val Acc":<12} {"Status":<15}')
    print('-' * 75)
    
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
            
            # 그래디언트 클리핑
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
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
    
    print(f'✅ {model_name} 훈련 완료! 최종 검증 정확도: {best_val_acc:.4f}')
    
    return {
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
        'best_val_acc': best_val_acc,
        'final_epoch': len(train_losses)
    }

def evaluate_model(model, test_data, test_labels, device, class_names):
    """모델 평가"""
    print('\n📊 모델 평가 중...')
    
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
    
    print(f'전체 정확도: {accuracy:.4f}')
    print(f'클래스별 F1-score:')
    for i, class_name in enumerate(class_names):
        if class_name in report:
            f1 = report[class_name]['f1-score']
            print(f'  {class_name}: {f1:.4f}')
    
    return accuracy, report, cm

def create_simplified_model_comparison(simple_history, ultra_history, simple_acc, ultra_acc):
    """단순화된 모델 비교 시각화"""
    print('\n📊 단순화된 모델 비교 시각화 생성 중...')
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('🔧 단순화된 GRU 모델 비교', fontsize=16, fontweight='bold')
    
    # 1. 훈련/검증 손실 곡선 (Simple GRU)
    ax1 = axes[0, 0]
    epochs = range(1, len(simple_history['train_losses']) + 1)
    ax1.plot(epochs, simple_history['train_losses'], 'b-', label='Train Loss', linewidth=2, alpha=0.8)
    ax1.plot(epochs, simple_history['val_losses'], 'b--', label='Validation Loss', linewidth=2, alpha=0.8)
    ax1.set_title('Simple GRU: 훈련/검증 손실 곡선')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 훈련/검증 정확도 곡선 (Simple GRU)
    ax2 = axes[0, 1]
    ax2.plot(epochs, simple_history['train_accuracies'], 'g-', label='Train Accuracy', linewidth=2, alpha=0.8)
    ax2.plot(epochs, simple_history['val_accuracies'], 'g--', label='Validation Accuracy', linewidth=2, alpha=0.8)
    ax2.set_title('Simple GRU: 훈련/검증 정확도 곡선')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0.5, 1.0)
    
    # 3. 훈련/검증 손실 곡선 (Ultra Simple GRU)
    ax3 = axes[1, 0]
    epochs = range(1, len(ultra_history['train_losses']) + 1)
    ax3.plot(epochs, ultra_history['train_losses'], 'r-', label='Train Loss', linewidth=2, alpha=0.8)
    ax3.plot(epochs, ultra_history['val_losses'], 'r--', label='Validation Loss', linewidth=2, alpha=0.8)
    ax3.set_title('Ultra Simple GRU: 훈련/검증 손실 곡선')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Loss')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 훈련/검증 정확도 곡선 (Ultra Simple GRU)
    ax4 = axes[1, 1]
    ax4.plot(epochs, ultra_history['train_accuracies'], 'orange', label='Train Accuracy', linewidth=2, alpha=0.8)
    ax4.plot(epochs, ultra_history['val_accuracies'], 'orange', linestyle='--', label='Validation Accuracy', linewidth=2, alpha=0.8)
    ax4.set_title('Ultra Simple GRU: 훈련/검증 정확도 곡선')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Accuracy')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0.5, 1.0)
    
    plt.tight_layout()
    plt.savefig('simplified_model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 성능 비교 차트
    fig2, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    models = ['Simple GRU', 'Ultra Simple GRU']
    accuracies = [simple_acc, ultra_acc]
    
    bars = ax.bar(models, accuracies, color=['skyblue', 'lightcoral'], alpha=0.8)
    ax.set_title('단순화된 모델 성능 비교', fontsize=14, fontweight='bold')
    ax.set_ylabel('Test Accuracy')
    ax.set_ylim(0.5, 1.0)
    
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
               f'{acc:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # 과적합 판단 기준선
    ax.axhline(y=0.95, color='red', linestyle='--', alpha=0.7, label='과적합 의심 임계값 (95%)')
    ax.axhline(y=0.90, color='orange', linestyle='--', alpha=0.7, label='적절한 성능 임계값 (90%)')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('simplified_model_performance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print('✅ 단순화된 모델 비교 시각화 저장:')
    print('  - simplified_model_comparison.png: 훈련 곡선 비교')
    print('  - simplified_model_performance.png: 성능 비교')

def main():
    """메인 실행 함수"""
    print('🔧 단순화된 GRU 모델 개발 시작 (전체 데이터 사용)')
    print('=' * 70)
    
    # 데이터 로드 (전체 데이터 사용)
    data_dir = '../SignGlove/external/SignGlove_HW/datasets/unified'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    all_data, all_labels, class_names = load_and_preprocess_data(data_dir, max_samples_per_class=None)
    
    # 데이터 분할 (더 엄격한 분할)
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
    
    print(f'\n📊 데이터 분할 결과:')
    print(f'훈련 데이터: {len(train_data_processed)}개')
    print(f'검증 데이터: {len(val_data_processed)}개')
    print(f'테스트 데이터: {len(test_data_processed)}개')
    
    # 1. Simple GRU 모델 훈련
    print('\n🔧 Simple GRU 모델 훈련 시작')
    simple_model = SimpleGRU(input_size=8, hidden_size=32, num_classes=24, dropout=0.5).to(device)
    
    # 모델 파라미터 수 계산
    simple_params = sum(p.numel() for p in simple_model.parameters())
    print(f'Simple GRU 파라미터 수: {simple_params:,}')
    
    simple_history = train_simplified_model(
        simple_model, train_data_processed, train_labels, 
        val_data_processed, val_labels, device, epochs=100, 
        learning_rate=0.001, model_name="Simple GRU"
    )
    
    # Simple GRU 평가
    simple_acc, simple_report, simple_cm = evaluate_model(
        simple_model, test_data_processed, test_labels, device, class_names
    )
    
    # 2. Ultra Simple GRU 모델 훈련
    print('\n🔧 Ultra Simple GRU 모델 훈련 시작')
    ultra_model = UltraSimpleGRU(input_size=8, hidden_size=16, num_classes=24, dropout=0.6).to(device)
    
    # 모델 파라미터 수 계산
    ultra_params = sum(p.numel() for p in ultra_model.parameters())
    print(f'Ultra Simple GRU 파라미터 수: {ultra_params:,}')
    
    ultra_history = train_simplified_model(
        ultra_model, train_data_processed, train_labels, 
        val_data_processed, val_labels, device, epochs=100, 
        learning_rate=0.001, model_name="Ultra Simple GRU"
    )
    
    # Ultra Simple GRU 평가
    ultra_acc, ultra_report, ultra_cm = evaluate_model(
        ultra_model, test_data_processed, test_labels, device, class_names
    )
    
    # 결과 요약
    print('\n📊 단순화된 모델 결과 요약 (전체 데이터)')
    print('=' * 70)
    
    print(f'Simple GRU:')
    print(f'  파라미터 수: {simple_params:,}')
    print(f'  테스트 정확도: {simple_acc:.4f}')
    print(f'  최종 검증 정확도: {simple_history["best_val_acc"]:.4f}')
    print(f'  훈련 에폭: {simple_history["final_epoch"]}')
    
    print(f'\nUltra Simple GRU:')
    print(f'  파라미터 수: {ultra_params:,}')
    print(f'  테스트 정확도: {ultra_acc:.4f}')
    print(f'  최종 검증 정확도: {ultra_history["best_val_acc"]:.4f}')
    print(f'  훈련 에폭: {ultra_history["final_epoch"]}')
    
    # 과적합 분석
    print('\n🔍 과적합 분석')
    print('=' * 70)
    
    simple_acc_gap = simple_history['train_accuracies'][-1] - simple_history['val_accuracies'][-1]
    ultra_acc_gap = ultra_history['train_accuracies'][-1] - ultra_history['val_accuracies'][-1]
    
    print(f'Simple GRU 정확도 격차: {simple_acc_gap:.4f}')
    print(f'Ultra Simple GRU 정확도 격차: {ultra_acc_gap:.4f}')
    
    if simple_acc_gap < 0.05 and ultra_acc_gap < 0.05:
        print('✅ 두 모델 모두 과적합이 크게 개선됨!')
    elif simple_acc_gap < 0.1 and ultra_acc_gap < 0.1:
        print('⚠️ 과적합이 일부 개선됨')
    else:
        print('❌ 여전히 과적합 문제 존재')
    
    # 권장 모델 선택
    print('\n🎯 권장 모델')
    print('=' * 70)
    
    if ultra_acc > 0.85 and ultra_acc_gap < 0.05:
        print('✅ Ultra Simple GRU 권장 (낮은 과적합, 적절한 성능)')
        recommended_model = ultra_model
        recommended_name = "Ultra Simple GRU"
    elif simple_acc > 0.85 and simple_acc_gap < 0.05:
        print('✅ Simple GRU 권장 (낮은 과적합, 적절한 성능)')
        recommended_model = simple_model
        recommended_name = "Simple GRU"
    else:
        print('⚠️ 두 모델 모두 추가 개선 필요')
        recommended_model = ultra_model if ultra_acc > simple_acc else simple_model
        recommended_name = "Ultra Simple GRU" if ultra_acc > simple_acc else "Simple GRU"
    
    # 시각화 생성
    create_simplified_model_comparison(simple_history, ultra_history, simple_acc, ultra_acc)
    
    print(f'\n🎉 단순화된 모델 개발 완료! (전체 데이터 사용)')
    print(f'권장 모델: {recommended_name}')

if __name__ == "__main__":
    main()
