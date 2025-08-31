#!/usr/bin/env python3
"""
KLP-SignGlove: 심층 과적합 조사 스크립트
100% 정확도와 빠른 수렴에 대한 의심 해결

- 교차 검증 (Cross-Validation)
- 더 엄격한 과적합 탐지
- 데이터 복잡도 분석
- 다양한 데이터 분할 테스트

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
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

class PositionalEncoding(nn.Module):
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
    def __init__(self):
        self.flex_scalers = [StandardScaler() for _ in range(5)]
        self.orientation_scalers = [StandardScaler() for _ in range(3)]
        self.is_fitted = False
        
    def fit(self, data_list):
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

def analyze_data_complexity(data_list, labels, class_names):
    print('\n🔍 데이터 복잡도 분석')
    
    # 클래스별 샘플 수
    class_counts = {}
    for label in labels:
        class_counts[label] = class_counts.get(label, 0) + 1
    
    print(f"클래스별 샘플 수:")
    for i, class_name in enumerate(class_names):
        count = class_counts.get(i, 0)
        print(f"  {class_name}: {count}개")
    
    # 데이터 변동성 분석
    all_variances = []
    for data in data_list:
        variance = np.var(data, axis=0)
        all_variances.append(variance)
    
    avg_variance = np.mean(all_variances, axis=0)
    print(f"\n센서별 평균 분산:")
    for i, var in enumerate(avg_variance):
        sensor_type = "Flex" if i < 5 else "Orientation"
        print(f"  {sensor_type} {i%5+1}: {var:.4f}")
    
    # 클래스 간 구분 가능성
    class_means = {}
    for class_idx in range(len(class_names)):
        class_data = [data for i, data in enumerate(data_list) if labels[i] == class_idx]
        if class_data:
            class_means[class_idx] = np.mean(class_data, axis=0)
    
    distances = []
    for i in range(len(class_names)):
        for j in range(i+1, len(class_names)):
            if i in class_means and j in class_means:
                dist = np.linalg.norm(class_means[i] - class_means[j])
                distances.append(dist)
    
    avg_distance = np.mean(distances)
    print(f"\n클래스 간 평균 거리: {avg_distance:.4f}")
    
    complexity_score = avg_distance / np.mean(avg_variance)
    print(f"데이터 복잡도 점수: {complexity_score:.4f}")
    
    return {
        'class_counts': class_counts,
        'avg_variance': avg_variance,
        'avg_distance': avg_distance,
        'complexity_score': complexity_score
    }

def train_model_with_rigorous_monitoring(model, train_data, train_labels, val_data, val_labels, 
                                       device, epochs=200, learning_rate=0.001, model_name="Model"):
    print(f'🏋️ {model_name} 엄격한 모니터링 훈련 시작 (총 {epochs} 에폭)...')
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=15, factor=0.5, verbose=True)
    
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    
    best_val_acc = 0
    patience_counter = 0
    max_patience = 50
    
    overfitting_detected = False
    overfitting_epoch = -1
    
    print(f'📊 엄격한 훈련 모니터링:')
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
        
        # 스케줄러 업데이트
        scheduler.step(avg_val_loss)
        
        # 과적합 탐지 (더 엄격한 기준)
        if epoch > 10:
            accuracy_gap = train_acc - val_acc
            loss_gap = avg_val_loss - avg_train_loss
            
            if (accuracy_gap > 0.02 and avg_val_loss > avg_train_loss * 1.5) or \
               (accuracy_gap > 0.05) or \
               (train_acc > 0.99 and val_acc < 0.95):
                if not overfitting_detected:
                    overfitting_detected = True
                    overfitting_epoch = epoch
                    print(f'⚠️ 과적합 신호 탐지! (에폭 {epoch+1})')
        
        # 조기 종료
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1
        
        # 진행 상황 출력
        status = "Normal"
        if overfitting_detected and epoch >= overfitting_epoch:
            status = "Overfitting"
        elif patience_counter > 0:
            status = f"Patience {patience_counter}"
        
        if (epoch + 1) % 20 == 0 or epoch < 10 or overfitting_detected:
            print(f'{epoch+1:<6} {avg_train_loss:<12.4f} {train_acc:<12.4f} {avg_val_loss:<12.4f} {val_acc:<12.4f} {status:<15}')
        
        if patience_counter >= max_patience:
            print(f'  조기 종료: {epoch+1} 에포크 (patience: {max_patience})')
            break
    
    print(f'✅ {model_name} 훈련 완료! 최종 검증 정확도: {best_val_acc:.4f}')
    if overfitting_detected:
        print(f'⚠️ 과적합 신호 탐지됨 (에폭 {overfitting_epoch+1})')
    
    return {
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
        'best_val_acc': best_val_acc,
        'final_epoch': len(train_losses),
        'overfitting_detected': overfitting_detected,
        'overfitting_epoch': overfitting_epoch
    }

def cross_validation_test(model_class, data_list, labels, device, k_folds=5, epochs=100):
    print(f'\n🔄 {k_folds}-Fold 교차 검증 시작')
    
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    fold_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(data_list, labels)):
        print(f'\n📊 Fold {fold+1}/{k_folds}')
        
        # 데이터 분할
        train_data = [data_list[i] for i in train_idx]
        train_labels = [labels[i] for i in train_idx]
        val_data = [data_list[i] for i in val_idx]
        val_labels = [labels[i] for i in val_idx]
        
        # 전처리
        preprocessor = AdvancedPreprocessor()
        preprocessor.fit(train_data)
        
        train_data_processed = preprocessor.transform(train_data)
        val_data_processed = preprocessor.transform(val_data)
        
        # 모델 생성 및 훈련
        model = model_class().to(device)
        history = train_model_with_rigorous_monitoring(
            model, train_data_processed, train_labels, 
            val_data_processed, val_labels, device, epochs=epochs, 
            model_name=f"Fold {fold+1}"
        )
        
        fold_scores.append(history['best_val_acc'])
        print(f'Fold {fold+1} 검증 정확도: {history["best_val_acc"]:.4f}')
    
    mean_score = np.mean(fold_scores)
    std_score = np.std(fold_scores)
    
    print(f'\n📊 교차 검증 결과:')
    print(f'평균 정확도: {mean_score:.4f} ± {std_score:.4f}')
    print(f'최고 정확도: {max(fold_scores):.4f}')
    print(f'최저 정확도: {min(fold_scores):.4f}')
    
    return {
        'fold_scores': fold_scores,
        'mean_score': mean_score,
        'std_score': std_score
    }

def create_investigation_visualization(data_complexity, gru_cv, transformer_cv, gru_history, transformer_history):
    print('\n📊 심층 조사 시각화 생성 중...')
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('🔬 심층 과적합 조사 결과', fontsize=16, fontweight='bold')
    
    # 1. 데이터 복잡도 분석
    ax1 = axes[0, 0]
    class_names = ['ㄱ', 'ㄴ', 'ㄷ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅅ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ', 
                  'ㅏ', 'ㅑ', 'ㅓ', 'ㅕ', 'ㅗ', 'ㅛ', 'ㅜ', 'ㅠ', 'ㅡ', 'ㅣ']
    class_counts = [data_complexity['class_counts'].get(i, 0) for i in range(len(class_names))]
    
    ax1.bar(range(len(class_names)), class_counts, alpha=0.7)
    ax1.set_title('클래스별 샘플 수 분포')
    ax1.set_xlabel('Class')
    ax1.set_ylabel('Sample Count')
    ax1.set_xticks(range(len(class_names)))
    ax1.set_xticklabels(class_names, rotation=45)
    
    # 2. 교차 검증 결과
    ax2 = axes[0, 1]
    models = ['GRU', 'Transformer']
    mean_scores = [gru_cv['mean_score'], transformer_cv['mean_score']]
    std_scores = [gru_cv['std_score'], transformer_cv['std_score']]
    
    bars = ax2.bar(models, mean_scores, yerr=std_scores, capsize=5, alpha=0.7)
    ax2.set_title('교차 검증 평균 정확도')
    ax2.set_ylabel('Accuracy')
    ax2.set_ylim(0.8, 1.0)
    
    for bar, score in zip(bars, mean_scores):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 3. Fold별 정확도 분포
    ax3 = axes[0, 2]
    ax3.plot(range(1, len(gru_cv['fold_scores'])+1), gru_cv['fold_scores'], 'b-o', label='GRU', alpha=0.7)
    ax3.plot(range(1, len(transformer_cv['fold_scores'])+1), transformer_cv['fold_scores'], 'r-s', label='Transformer', alpha=0.7)
    ax3.set_title('Fold별 정확도 분포')
    ax3.set_xlabel('Fold')
    ax3.set_ylabel('Accuracy')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 과적합 탐지 결과
    ax4 = axes[1, 0]
    overfitting_detected = [gru_history['overfitting_detected'], transformer_history['overfitting_detected']]
    colors = ['red' if detected else 'green' for detected in overfitting_detected]
    
    bars = ax4.bar(models, [1 if detected else 0 for detected in overfitting_detected], color=colors, alpha=0.7)
    ax4.set_title('과적합 탐지 결과')
    ax4.set_ylabel('Overfitting Detected')
    ax4.set_ylim(0, 1.2)
    
    for bar, detected in zip(bars, overfitting_detected):
        status = "Detected" if detected else "Not Detected"
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                status, ha='center', va='bottom', fontweight='bold')
    
    # 5. 데이터 복잡도 vs 모델 성능
    ax5 = axes[1, 1]
    complexity = data_complexity['complexity_score']
    performances = [gru_cv['mean_score'], transformer_cv['mean_score']]
    
    ax5.scatter([complexity, complexity], performances, c=['blue', 'red'], s=100, alpha=0.7)
    ax5.annotate('GRU', (complexity, gru_cv['mean_score']), xytext=(5, 5), textcoords='offset points')
    ax5.annotate('Transformer', (complexity, transformer_cv['mean_score']), xytext=(5, 5), textcoords='offset points')
    ax5.set_title('데이터 복잡도 vs 모델 성능')
    ax5.set_xlabel('Data Complexity Score')
    ax5.set_ylabel('Model Performance')
    ax5.grid(True, alpha=0.3)
    
    # 6. 종합 평가
    ax6 = axes[1, 2]
    
    # 과적합 위험도 점수 계산
    def calculate_overfitting_risk(history, cv_result):
        risk_factors = []
        
        # 정확도 격차
        final_acc_gap = history['train_accuracies'][-1] - history['val_accuracies'][-1]
        risk_factors.append(min(final_acc_gap * 10, 1.0))
        
        # 수렴 속도
        convergence_speed = len(history['train_accuracies'])
        risk_factors.append(min(1.0 / convergence_speed * 50, 1.0))
        
        # 교차 검증 일관성
        cv_consistency = 1.0 - cv_result['std_score']
        risk_factors.append(1.0 - cv_consistency)
        
        # 과적합 탐지 여부
        detection_risk = 1.0 if history['overfitting_detected'] else 0.0
        risk_factors.append(detection_risk)
        
        return np.mean(risk_factors)
    
    gru_risk = calculate_overfitting_risk(gru_history, gru_cv)
    transformer_risk = calculate_overfitting_risk(transformer_history, transformer_cv)
    
    risks = [gru_risk, transformer_risk]
    colors = ['green' if risk < 0.3 else 'orange' if risk < 0.7 else 'red' for risk in risks]
    bars = ax6.bar(models, risks, color=colors, alpha=0.7)
    ax6.set_title('과적합 위험도 평가')
    ax6.set_ylabel('Overfitting Risk Score')
    ax6.set_ylim(0, 1.0)
    
    for bar, risk in zip(bars, risks):
        risk_level = "Low" if risk < 0.3 else "Medium" if risk < 0.7 else "High"
        ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{risk:.2f}\n({risk_level})', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('deep_overfitting_investigation.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print('✅ 심층 조사 시각화 저장: deep_overfitting_investigation.png')

def main():
    print('🔬 심층 과적합 조사 시작')
    print('=' * 70)
    
    # 데이터 로드
    data_dir = '../SignGlove/external/SignGlove_HW/datasets/unified'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    all_data, all_labels, class_names = load_and_preprocess_data(data_dir, max_samples_per_class=25)
    
    # 데이터 복잡도 분석
    data_complexity = analyze_data_complexity(all_data, all_labels, class_names)
    
    # 교차 검증 테스트
    from improved_preprocessing_model import ImprovedGRU
    
    print('\n🔄 GRU 교차 검증 시작')
    gru_cv = cross_validation_test(ImprovedGRU, all_data, all_labels, device, k_folds=5, epochs=100)
    
    print('\n🔄 Transformer 교차 검증 시작')
    transformer_cv = cross_validation_test(TransformerModel, all_data, all_labels, device, k_folds=5, epochs=100)
    
    # 엄격한 모니터링 훈련 (더 엄격한 분할)
    train_data, temp_data, train_labels, temp_labels = train_test_split(
        all_data, all_labels, test_size=0.5, random_state=42, stratify=all_labels
    )
    val_data, test_data, val_labels, test_labels = train_test_split(
        temp_data, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
    )
    
    # 전처리
    preprocessor = AdvancedPreprocessor()
    preprocessor.fit(train_data)
    
    train_data_processed = preprocessor.transform(train_data)
    val_data_processed = preprocessor.transform(val_data)
    
    # GRU 엄격한 훈련
    gru_model = ImprovedGRU(input_size=8, hidden_size=64, num_layers=2, num_classes=24, dropout=0.3).to(device)
    gru_history = train_model_with_rigorous_monitoring(
        gru_model, train_data_processed, train_labels, 
        val_data_processed, val_labels, device, epochs=200, model_name="GRU (엄격)"
    )
    
    # Transformer 엄격한 훈련
    transformer_model = TransformerModel(
        input_size=8, d_model=64, nhead=8, num_layers=2, 
        num_classes=24, dropout=0.1
    ).to(device)
    transformer_history = train_model_with_rigorous_monitoring(
        transformer_model, train_data_processed, train_labels, 
        val_data_processed, val_labels, device, epochs=200, model_name="Transformer (엄격)"
    )
    
    # 결과 요약
    print('\n📊 심층 조사 결과 요약')
    print('=' * 70)
    
    print(f'데이터 복잡도 점수: {data_complexity["complexity_score"]:.4f}')
    print(f'클래스 간 평균 거리: {data_complexity["avg_distance"]:.4f}')
    print()
    
    print('교차 검증 결과:')
    print(f'  GRU: {gru_cv["mean_score"]:.4f} ± {gru_cv["std_score"]:.4f}')
    print(f'  Transformer: {transformer_cv["mean_score"]:.4f} ± {transformer_cv["std_score"]:.4f}')
    print()
    
    print('과적합 탐지 결과:')
    print(f'  GRU: {"탐지됨" if gru_history["overfitting_detected"] else "탐지되지 않음"}')
    print(f'  Transformer: {"탐지됨" if transformer_history["overfitting_detected"] else "탐지되지 않음"}')
    print()
    
    # 결론
    print('🎯 심층 조사 결론')
    print('=' * 70)
    
    if data_complexity["complexity_score"] < 1.0:
        print('⚠️ 데이터 복잡도가 낮음 - 과적합 위험 증가')
    else:
        print('✅ 데이터 복잡도가 적절함')
    
    if gru_cv["std_score"] > 0.1 or transformer_cv["std_score"] > 0.1:
        print('⚠️ 교차 검증 결과 불안정 - 과적합 의심')
    else:
        print('✅ 교차 검증 결과 안정적')
    
    if gru_history["overfitting_detected"] or transformer_history["overfitting_detected"]:
        print('⚠️ 과적합 신호 탐지됨 - 모델 선택 재검토 필요')
    else:
        print('✅ 과적합 신호 탐지되지 않음')
    
    # 시각화 생성
    create_investigation_visualization(data_complexity, gru_cv, transformer_cv, gru_history, transformer_history)
    
    print('\n🎉 심층 과적합 조사 완료!')

if __name__ == "__main__":
    main()
