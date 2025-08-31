#!/usr/bin/env python3
"""
KLP-SignGlove: 과적합 분석 비교 스크립트
GRU vs Transformer 모델의 과적합 패턴 분석

과적합 여부를 정확히 판단하기 위한 종합적인 분석 도구
- 훈련/검증 곡선 분석
- 클래스별 성능 분석
- 신뢰도 분포 분석
- t-SNE 시각화
- 이상치 탐지

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
from sklearn.manifold import TSNE
from sklearn.neighbors import LocalOutlierFactor
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

def train_model_with_history(model, train_data, train_labels, val_data, val_labels, 
                           device, epochs=50, learning_rate=0.001, model_name="Model"):
    """모델 훈련 및 히스토리 기록"""
    print(f'🏋️ {model_name} 훈련 시작...')
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    best_val_acc = 0
    patience_counter = 0
    max_patience = 15
    
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
        
        # 조기 종료
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= max_patience:
            print(f'  조기 종료: {epoch+1} 에포크')
            break
        
        if (epoch + 1) % 10 == 0:
            print(f'  Epoch {epoch+1}/{epochs}: Train Loss={avg_train_loss:.4f}, Train Acc={train_acc:.4f}, Val Loss={avg_val_loss:.4f}, Val Acc={val_acc:.4f}')
    
    return {
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
        'best_val_acc': best_val_acc
    }

def evaluate_model_detailed(model, test_data, test_labels, class_names, device):
    """상세한 모델 평가"""
    print('🧪 상세 모델 평가 중...')
    
    model.eval()
    all_predictions = []
    all_true_labels = []
    all_confidences = []
    all_probabilities = []
    inference_times = []
    
    with torch.no_grad():
        for data, label in zip(test_data, test_labels):
            data_tensor = torch.FloatTensor(data).unsqueeze(0).to(device)
            
            start_time = time.time()
            outputs = model(data_tensor)
            inference_time = time.time() - start_time
            inference_times.append(inference_time)
            
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            all_predictions.append(predicted.item())
            all_true_labels.append(label)
            all_confidences.append(confidence.item())
            all_probabilities.append(probabilities.cpu().numpy().flatten())
    
    accuracy = accuracy_score(all_true_labels, all_predictions)
    report = classification_report(all_true_labels, all_predictions, 
                                 target_names=class_names, output_dict=True)
    conf_matrix = confusion_matrix(all_true_labels, all_predictions)
    
    avg_inference_time = np.mean(inference_times)
    
    print(f'✅ 테스트 정확도: {accuracy:.4f} ({accuracy*100:.2f}%)')
    print(f'⏱️ 평균 추론 시간: {avg_inference_time*1000:.2f}ms')
    
    return {
        'accuracy': accuracy,
        'report': report,
        'confusion_matrix': conf_matrix,
        'confidences': all_confidences,
        'probabilities': np.array(all_probabilities),
        'predictions': all_predictions,
        'true_labels': all_true_labels,
        'inference_time': avg_inference_time
    }

def analyze_overfitting(train_history, val_history, model_name):
    """과적합 분석"""
    print(f'\n🔍 {model_name} 과적합 분석')
    
    # 1. 훈련/검증 곡선 분석
    train_losses = train_history['train_losses']
    val_losses = train_history['val_losses']
    train_accuracies = train_history['train_accuracies']
    val_accuracies = train_history['val_accuracies']
    
    # 과적합 지표 계산
    final_train_acc = train_accuracies[-1]
    final_val_acc = val_accuracies[-1]
    accuracy_gap = final_train_acc - final_val_acc
    
    final_train_loss = train_losses[-1]
    final_val_loss = val_losses[-1]
    loss_gap = final_val_loss - final_train_loss
    
    # 과적합 판단
    overfitting_score = 0
    overfitting_signals = []
    
    if accuracy_gap > 0.05:  # 5% 이상 차이
        overfitting_score += 1
        overfitting_signals.append(f"정확도 격차: {accuracy_gap:.3f} (높음)")
    
    if loss_gap > 0.1:  # 손실 격차가 큼
        overfitting_score += 1
        overfitting_signals.append(f"손실 격차: {loss_gap:.3f} (높음)")
    
    if final_train_acc > 0.98 and final_val_acc < 0.95:  # 훈련 정확도가 너무 높음
        overfitting_score += 1
        overfitting_signals.append("훈련 정확도가 너무 높음")
    
    # 과적합 정도 판단
    if overfitting_score == 0:
        overfitting_level = "과적합 없음"
    elif overfitting_score == 1:
        overfitting_level = "약간의 과적합"
    elif overfitting_score == 2:
        overfitting_level = "중간 정도 과적합"
    else:
        overfitting_level = "심각한 과적합"
    
    print(f"  최종 훈련 정확도: {final_train_acc:.4f}")
    print(f"  최종 검증 정확도: {final_val_acc:.4f}")
    print(f"  정확도 격차: {accuracy_gap:.4f}")
    print(f"  손실 격차: {loss_gap:.4f}")
    print(f"  과적합 수준: {overfitting_level}")
    
    if overfitting_signals:
        print(f"  과적합 신호: {', '.join(overfitting_signals)}")
    
    return {
        'overfitting_score': overfitting_score,
        'overfitting_level': overfitting_level,
        'accuracy_gap': accuracy_gap,
        'loss_gap': loss_gap,
        'final_train_acc': final_train_acc,
        'final_val_acc': final_val_acc
    }

def create_overfitting_visualization(gru_history, transformer_history, gru_eval, transformer_eval):
    """과적합 분석 시각화"""
    print('\n📊 과적합 분석 시각화 생성 중...')
    
    fig = plt.figure(figsize=(20, 15))
    
    # 1. 훈련/검증 곡선 비교
    plt.subplot(3, 4, 1)
    gru_epochs = range(1, len(gru_history['train_losses']) + 1)
    transformer_epochs = range(1, len(transformer_history['train_losses']) + 1)
    
    plt.plot(gru_epochs, gru_history['train_losses'], 'b-', label='GRU Train Loss', linewidth=2)
    plt.plot(gru_epochs, gru_history['val_losses'], 'b--', label='GRU Val Loss', linewidth=2)
    plt.plot(transformer_epochs, transformer_history['train_losses'], 'r-', label='Transformer Train Loss', linewidth=2)
    plt.plot(transformer_epochs, transformer_history['val_losses'], 'r--', label='Transformer Val Loss', linewidth=2)
    plt.title('훈련/검증 손실 곡선 비교')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 4, 2)
    plt.plot(gru_epochs, gru_history['train_accuracies'], 'b-', label='GRU Train Acc', linewidth=2)
    plt.plot(gru_epochs, gru_history['val_accuracies'], 'b--', label='GRU Val Acc', linewidth=2)
    plt.plot(transformer_epochs, transformer_history['train_accuracies'], 'r-', label='Transformer Train Acc', linewidth=2)
    plt.plot(transformer_epochs, transformer_history['val_accuracies'], 'r--', label='Transformer Val Acc', linewidth=2)
    plt.title('훈련/검증 정확도 곡선 비교')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. 과적합 지표 비교
    plt.subplot(3, 4, 3)
    models = ['GRU', 'Transformer']
    accuracy_gaps = [gru_history['train_accuracies'][-1] - gru_history['val_accuracies'][-1], 
                    transformer_history['train_accuracies'][-1] - transformer_history['val_accuracies'][-1]]
    colors = ['green' if gap < 0.05 else 'orange' if gap < 0.1 else 'red' for gap in accuracy_gaps]
    bars = plt.bar(models, accuracy_gaps, color=colors, alpha=0.8)
    plt.title('정확도 격차 비교 (과적합 지표)')
    plt.ylabel('Train Acc - Val Acc')
    for bar, gap in zip(bars, accuracy_gaps):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{gap:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 3. 신뢰도 분포 비교
    plt.subplot(3, 4, 4)
    plt.hist(gru_eval['confidences'], bins=20, alpha=0.7, label='GRU', color='blue', density=True)
    plt.hist(transformer_eval['confidences'], bins=20, alpha=0.7, label='Transformer', color='red', density=True)
    plt.title('예측 신뢰도 분포')
    plt.xlabel('Confidence')
    plt.ylabel('Density')
    plt.legend()
    
    # 4. 클래스별 성능 비교
    plt.subplot(3, 4, 5)
    gru_f1_scores = [gru_eval['report'][name]['f1-score'] for name in gru_eval['report'].keys() if name not in ['accuracy', 'macro avg', 'weighted avg']]
    transformer_f1_scores = [transformer_eval['report'][name]['f1-score'] for name in transformer_eval['report'].keys() if name not in ['accuracy', 'macro avg', 'weighted avg']]
    
    x = np.arange(len(gru_f1_scores))
    width = 0.35
    plt.bar(x - width/2, gru_f1_scores, width, label='GRU', alpha=0.8)
    plt.bar(x + width/2, transformer_f1_scores, width, label='Transformer', alpha=0.8)
    plt.title('클래스별 F1-Score 비교')
    plt.xlabel('Class Index')
    plt.ylabel('F1-Score')
    plt.legend()
    plt.xticks(x[::3], x[::3])
    
    # 5. 혼동 행렬 (GRU)
    plt.subplot(3, 4, 6)
    sns.heatmap(gru_eval['confusion_matrix'], annot=True, fmt='d', cmap='Blues', 
                xticklabels=range(24), yticklabels=range(24))
    plt.title('GRU 혼동 행렬')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    # 6. 혼동 행렬 (Transformer)
    plt.subplot(3, 4, 7)
    sns.heatmap(transformer_eval['confusion_matrix'], annot=True, fmt='d', cmap='Reds', 
                xticklabels=range(24), yticklabels=range(24))
    plt.title('Transformer 혼동 행렬')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    # 7. t-SNE 시각화 (GRU)
    plt.subplot(3, 4, 8)
    try:
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(gru_eval['probabilities'])-1))
        gru_embeddings = tsne.fit_transform(gru_eval['probabilities'])
        scatter = plt.scatter(gru_embeddings[:, 0], gru_embeddings[:, 1], 
                            c=gru_eval['true_labels'], cmap='tab20', alpha=0.7)
        plt.title('GRU t-SNE 시각화')
        plt.colorbar(scatter)
    except:
        plt.text(0.5, 0.5, 't-SNE 계산 실패', ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('GRU t-SNE 시각화')
    
    # 8. t-SNE 시각화 (Transformer)
    plt.subplot(3, 4, 9)
    try:
        transformer_embeddings = tsne.fit_transform(transformer_eval['probabilities'])
        scatter = plt.scatter(transformer_embeddings[:, 0], transformer_embeddings[:, 1], 
                            c=transformer_eval['true_labels'], cmap='tab20', alpha=0.7)
        plt.title('Transformer t-SNE 시각화')
        plt.colorbar(scatter)
    except:
        plt.text(0.5, 0.5, 't-SNE 계산 실패', ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Transformer t-SNE 시각화')
    
    # 9. 이상치 탐지 (LOF)
    plt.subplot(3, 4, 10)
    try:
        lof = LocalOutlierFactor(n_neighbors=min(5, len(gru_eval['probabilities'])-1))
        gru_outliers = lof.fit_predict(gru_eval['probabilities'])
        gru_outlier_scores = lof.negative_outlier_factor_
        plt.hist(gru_outlier_scores, bins=20, alpha=0.7, color='blue')
        plt.title('GRU 이상치 점수 분포')
        plt.xlabel('Outlier Score')
        plt.ylabel('Count')
    except:
        plt.text(0.5, 0.5, 'LOF 계산 실패', ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('GRU 이상치 점수 분포')
    
    # 10. 이상치 탐지 (Transformer)
    plt.subplot(3, 4, 11)
    try:
        lof = LocalOutlierFactor(n_neighbors=min(5, len(transformer_eval['probabilities'])-1))
        transformer_outliers = lof.fit_predict(transformer_eval['probabilities'])
        transformer_outlier_scores = lof.negative_outlier_factor_
        plt.hist(transformer_outlier_scores, bins=20, alpha=0.7, color='red')
        plt.title('Transformer 이상치 점수 분포')
        plt.xlabel('Outlier Score')
        plt.ylabel('Count')
    except:
        plt.text(0.5, 0.5, 'LOF 계산 실패', ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Transformer 이상치 점수 분포')
    
    # 11. 종합 성능 비교
    plt.subplot(3, 4, 12)
    metrics = ['Accuracy', 'Avg Confidence', 'Inference Time (ms)']
    gru_metrics = [gru_eval['accuracy'], np.mean(gru_eval['confidences']), gru_eval['inference_time']*1000]
    transformer_metrics = [transformer_eval['accuracy'], np.mean(transformer_eval['confidences']), transformer_eval['inference_time']*1000]
    
    x = np.arange(len(metrics))
    width = 0.35
    plt.bar(x - width/2, gru_metrics, width, label='GRU', alpha=0.8)
    plt.bar(x + width/2, transformer_metrics, width, label='Transformer', alpha=0.8)
    plt.title('종합 성능 비교')
    plt.xlabel('Metrics')
    plt.ylabel('Value')
    plt.legend()
    plt.xticks(x, metrics, rotation=45)
    
    plt.tight_layout()
    plt.savefig('overfitting_analysis_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print('✅ 과적합 분석 시각화 저장: overfitting_analysis_comparison.png')

def main():
    """메인 실행 함수"""
    print('🔬 GRU vs Transformer 과적합 분석 시작')
    print('=' * 60)
    
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
    
    # Transformer 모델 훈련
    transformer_model = TransformerModel(
        input_size=8, d_model=64, nhead=8, num_layers=2, 
        num_classes=24, dropout=0.1
    ).to(device)
    
    transformer_history = train_model_with_history(
        transformer_model, train_data_processed, train_labels, 
        val_data_processed, val_labels, device, epochs=50, model_name="Transformer"
    )
    
    # GRU 모델 훈련
    from improved_preprocessing_model import ImprovedGRU
    
    gru_model = ImprovedGRU(input_size=8, hidden_size=64, num_layers=2, num_classes=24, dropout=0.3).to(device)
    
    gru_history = train_model_with_history(
        gru_model, train_data_processed, train_labels, 
        val_data_processed, val_labels, device, epochs=50, model_name="GRU"
    )
    
    # 모델 평가
    transformer_eval = evaluate_model_detailed(
        transformer_model, test_data_processed, test_labels, class_names, device
    )
    
    gru_eval = evaluate_model_detailed(
        gru_model, test_data_processed, test_labels, class_names, device
    )
    
    # 과적합 분석
    transformer_overfitting = analyze_overfitting(transformer_history, transformer_history, "Transformer")
    gru_overfitting = analyze_overfitting(gru_history, gru_history, "GRU")
    
    # 결과 요약
    print('\n📊 과적합 분석 결과 요약')
    print('=' * 60)
    
    summary_data = {
        '모델': ['GRU', 'Transformer'],
        '정확도': [gru_eval['accuracy'], transformer_eval['accuracy']],
        '과적합 수준': [gru_overfitting['overfitting_level'], transformer_overfitting['overfitting_level']],
        '정확도 격차': [gru_overfitting['accuracy_gap'], transformer_overfitting['accuracy_gap']],
        '평균 신뢰도': [np.mean(gru_eval['confidences']), np.mean(transformer_eval['confidences'])],
        '추론 시간 (ms)': [gru_eval['inference_time']*1000, transformer_eval['inference_time']*1000]
    }
    
    print(f"{'모델':<12} {'정확도':<10} {'과적합':<15} {'정확도격차':<12} {'신뢰도':<10} {'추론시간':<10}")
    print('-' * 75)
    
    for i in range(2):
        print(f"{summary_data['모델'][i]:<12} "
              f"{summary_data['정확도'][i]:<10.4f} "
              f"{summary_data['과적합 수준'][i]:<15} "
              f"{summary_data['정확도 격차'][i]:<12.4f} "
              f"{summary_data['평균 신뢰도'][i]:<10.4f} "
              f"{summary_data['추론 시간 (ms)'][i]:<10.2f}")
    
    # 결론
    print('\n🎯 과적합 분석 결론')
    print('=' * 60)
    
    if gru_overfitting['overfitting_score'] < transformer_overfitting['overfitting_score']:
        print('✅ GRU가 더 적은 과적합 신호를 보임')
    else:
        print('⚠️ Transformer가 더 적은 과적합 신호를 보임')
    
    if gru_overfitting['accuracy_gap'] < transformer_overfitting['accuracy_gap']:
        print('✅ GRU가 더 작은 정확도 격차를 보임')
    else:
        print('⚠️ Transformer가 더 작은 정확도 격차를 보임')
    
    print('\n📋 과적합 관점에서의 모델 선택 근거:')
    print('1. 🎯 일반화 성능: 검증 정확도가 높고 격차가 작은 모델')
    print('2. ⚖️ 안정성: 일관된 성능을 보이는 모델')
    print('3. 🔍 신뢰도: 적절한 신뢰도 분포를 가진 모델')
    print('4. 📊 클래스별 성능: 균등한 성능 분포')
    
    # 시각화 생성
    create_overfitting_visualization(gru_history, transformer_history, gru_eval, transformer_eval)
    
    print('\n🎉 과적합 분석 완료!')

if __name__ == "__main__":
    main()
