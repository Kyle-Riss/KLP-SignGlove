import torch
import torch.nn as nn
import numpy as np
import h5py
import os
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

print('📊 클래스별 상세 성능 분석 시작')

# 한글 폰트 설정
try:
    plt.rcParams['font.family'] = 'NanumGothic'
except:
    plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

class ImprovedGRU(nn.Module):
    """개선된 GRU 모델"""
    def __init__(self, input_size=8, hidden_size=64, num_layers=2, num_classes=24, dropout=0.3):
        super(ImprovedGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # GRU 레이어
        self.gru = nn.GRU(input_size, hidden_size, num_layers, 
                          batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        # 드롭아웃
        self.dropout = nn.Dropout(dropout)
        
        # 출력 레이어
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
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        
        out, _ = self.gru(x, h0)
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out

class AdvancedPreprocessor:
    """고급 전처리 클래스"""
    def __init__(self):
        self.flex_scalers = [StandardScaler() for _ in range(5)]
        self.orientation_scalers = [StandardScaler() for _ in range(3)]
        self.is_fitted = False
        
    def fit(self, data_list):
        """전처리 파라미터 학습"""
        print('🔧 전처리 파라미터 학습 중...')
        
        # Flex 센서별로 스케일러 학습
        for sensor_idx in range(5):
            sensor_data = []
            for data in data_list:
                sensor_values = data[:, sensor_idx]
                # 0값 제외하고 학습
                non_zero_values = sensor_values[sensor_values > 0]
                if len(non_zero_values) > 0:
                    sensor_data.extend(non_zero_values)
            
            if len(sensor_data) > 0:
                sensor_data = np.array(sensor_data).reshape(-1, 1)
                self.flex_scalers[sensor_idx].fit(sensor_data)
        
        # Orientation 센서별로 스케일러 학습
        for sensor_idx in range(3):
            sensor_data = []
            for data in data_list:
                sensor_values = data[:, sensor_idx + 5]
                sensor_data.extend(sensor_values)
            
            sensor_data = np.array(sensor_data).reshape(-1, 1)
            self.orientation_scalers[sensor_idx].fit(sensor_data)
        
        self.is_fitted = True
        print('✅ 전처리 파라미터 학습 완료')
    
    def transform(self, data_list):
        """데이터 변환"""
        if not self.is_fitted:
            raise ValueError("전처리 파라미터를 먼저 학습해야 합니다.")
        
        print('🔄 데이터 변환 중...')
        processed_data = []
        
        for data in data_list:
            processed = data.copy()
            
            # Flex 센서 처리 (0값 처리 + 정규화)
            for sensor_idx in range(5):
                sensor_values = data[:, sensor_idx]
                
                # 0값을 해당 센서의 평균값으로 대체
                mean_val = np.mean(sensor_values[sensor_values > 0])
                if np.isnan(mean_val):
                    mean_val = 500  # 기본값
                
                sensor_values[sensor_values == 0] = mean_val
                
                # 정규화
                sensor_values_normalized = self.flex_scalers[sensor_idx].transform(
                    sensor_values.reshape(-1, 1)
                ).flatten()
                
                processed[:, sensor_idx] = sensor_values_normalized
            
            # Orientation 센서 처리 (정규화)
            for sensor_idx in range(3):
                sensor_values = data[:, sensor_idx + 5]
                sensor_values_normalized = self.orientation_scalers[sensor_idx].transform(
                    sensor_values.reshape(-1, 1)
                ).flatten()
                processed[:, sensor_idx + 5] = sensor_values_normalized
            
            processed_data.append(processed)
        
        print(f'✅ {len(processed_data)}개 샘플 변환 완료')
        return processed_data

def load_and_preprocess_data(data_dir, max_samples_per_class=25):
    """데이터 로드 및 전처리"""
    print(f'📂 데이터 로드 중: {data_dir}')
    
    all_data = []
    all_labels = []
    class_names = []
    
    for class_idx, class_name in enumerate(sorted(os.listdir(data_dir))):
        class_path = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_path):
            continue
            
        class_names.append(class_name)
        print(f'  - 클래스 {class_name} 로딩 중...')
        
        sample_count = 0
        for session in sorted(os.listdir(class_path)):
            if sample_count >= max_samples_per_class:
                break
            session_path = os.path.join(class_path, session)
            if not os.path.isdir(session_path):
                continue
                
            for file_name in sorted(os.listdir(session_path)):
                if sample_count >= max_samples_per_class:
                    break
                if file_name.endswith('.h5'):
                    file_path = os.path.join(session_path, file_name)
                    try:
                        with h5py.File(file_path, 'r') as f:
                            sensor_data = f['sensor_data'][:]
                            if sensor_data.shape[0] >= 20:
                                all_data.append(sensor_data)
                                all_labels.append(class_idx)
                                sample_count += 1
                    except Exception as e:
                        print(f"    경고: {file_path} 로드 실패 - {e}")
    
    print(f'✅ 총 {len(all_data)}개 샘플, {len(class_names)}개 클래스 로드 완료')
    return all_data, all_labels, class_names

def analyze_class_performance(model, test_data, test_labels, class_names, device):
    """클래스별 성능 분석"""
    print('📊 클래스별 성능 분석 중...')
    
    model.eval()
    all_predictions = []
    all_true_labels = []
    all_confidences = []
    all_probabilities = []
    
    with torch.no_grad():
        for data, label in zip(test_data, test_labels):
            data_tensor = torch.FloatTensor(data).unsqueeze(0).to(device)
            outputs = model(data_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            all_predictions.append(predicted.item())
            all_true_labels.append(label)
            all_confidences.append(confidence.item())
            all_probabilities.append(probabilities.cpu().numpy())
    
    # 기본 성능 지표
    accuracy = accuracy_score(all_true_labels, all_predictions)
    report = classification_report(all_true_labels, all_predictions, 
                                 target_names=class_names, output_dict=True)
    cm = confusion_matrix(all_true_labels, all_predictions)
    
    # 클래스별 상세 분석
    class_performance = []
    for i, class_name in enumerate(class_names):
        if class_name in report:
            class_report = report[class_name]
            class_performance.append({
                'class': class_name,
                'precision': class_report['precision'],
                'recall': class_report['recall'],
                'f1_score': class_report['f1-score'],
                'support': class_report['support']
            })
    
    # F1 점수 기준으로 정렬
    class_performance.sort(key=lambda x: x['f1_score'], reverse=True)
    
    return accuracy, report, cm, class_performance, all_confidences, all_probabilities

def analyze_overfitting_signals(train_data, train_labels, val_data, val_labels, model, device):
    """과적합 신호 분석"""
    print('🔍 과적합 신호 분석 중...')
    
    model.eval()
    
    # 훈련 데이터 성능
    train_correct = 0
    train_total = 0
    train_confidences = []
    
    with torch.no_grad():
        for data, label in zip(train_data, train_labels):
            data_tensor = torch.FloatTensor(data).unsqueeze(0).to(device)
            outputs = model(data_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            train_total += 1
            if predicted.item() == label:
                train_correct += 1
            train_confidences.append(confidence.item())
    
    train_accuracy = train_correct / train_total
    
    # 검증 데이터 성능
    val_correct = 0
    val_total = 0
    val_confidences = []
    
    with torch.no_grad():
        for data, label in zip(val_data, val_labels):
            data_tensor = torch.FloatTensor(data).unsqueeze(0).to(device)
            outputs = model(data_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            val_total += 1
            if predicted.item() == label:
                val_correct += 1
            val_confidences.append(confidence.item())
    
    val_accuracy = val_correct / val_total
    
    # 과적합 지표 계산
    accuracy_gap = train_accuracy - val_accuracy
    confidence_gap = np.mean(train_confidences) - np.mean(val_confidences)
    
    overfitting_signals = {
        'train_accuracy': train_accuracy,
        'val_accuracy': val_accuracy,
        'accuracy_gap': accuracy_gap,
        'train_confidence': np.mean(train_confidences),
        'val_confidence': np.mean(val_confidences),
        'confidence_gap': confidence_gap,
        'is_overfitting': accuracy_gap > 0.05 or confidence_gap > 0.05
    }
    
    return overfitting_signals

def visualize_detailed_analysis(accuracy, cm, class_performance, overfitting_signals, class_names):
    """상세 분석 결과 시각화"""
    print('📈 상세 분석 결과 시각화 생성 중...')
    
    fig = plt.figure(figsize=(20, 16))
    
    # 1. 혼동 행렬
    ax1 = plt.subplot(3, 4, 1)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax1)
    ax1.set_title(f'Confusion Matrix (Accuracy: {accuracy:.4f})')
    ax1.set_xlabel('Predicted Label')
    ax1.set_ylabel('True Label')
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
    plt.setp(ax1.get_yticklabels(), rotation=0)
    
    # 2. 클래스별 F1 점수
    ax2 = plt.subplot(3, 4, 2)
    classes = [p['class'] for p in class_performance]
    f1_scores = [p['f1_score'] for p in class_performance]
    colors = ['green' if score >= 0.9 else 'orange' if score >= 0.7 else 'red' for score in f1_scores]
    
    bars = ax2.bar(range(len(classes)), f1_scores, color=colors, alpha=0.7)
    ax2.set_title('Class-wise F1 Scores')
    ax2.set_xlabel('Class')
    ax2.set_ylabel('F1 Score')
    ax2.set_xticks(range(len(classes)))
    ax2.set_xticklabels(classes, rotation=45, ha='right')
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)
    
    for i, (bar, score) in enumerate(zip(bars, f1_scores)):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{score:.3f}', ha='center', va='bottom', fontsize=8)
    
    # 3. 클래스별 정밀도
    ax3 = plt.subplot(3, 4, 3)
    precisions = [p['precision'] for p in class_performance]
    bars = ax3.bar(range(len(classes)), precisions, alpha=0.7, color='skyblue')
    ax3.set_title('Class-wise Precision')
    ax3.set_xlabel('Class')
    ax3.set_ylabel('Precision')
    ax3.set_xticks(range(len(classes)))
    ax3.set_xticklabels(classes, rotation=45, ha='right')
    ax3.set_ylim(0, 1)
    ax3.grid(True, alpha=0.3)
    
    # 4. 클래스별 재현율
    ax4 = plt.subplot(3, 4, 4)
    recalls = [p['recall'] for p in class_performance]
    bars = ax4.bar(range(len(classes)), recalls, alpha=0.7, color='lightcoral')
    ax4.set_title('Class-wise Recall')
    ax4.set_xlabel('Class')
    ax4.set_ylabel('Recall')
    ax4.set_xticks(range(len(classes)))
    ax4.set_xticklabels(classes, rotation=45, ha='right')
    ax4.set_ylim(0, 1)
    ax4.grid(True, alpha=0.3)
    
    # 5. 과적합 신호 분석
    ax5 = plt.subplot(3, 4, 5)
    metrics = ['Train Acc', 'Val Acc', 'Train Conf', 'Val Conf']
    values = [
        overfitting_signals['train_accuracy'],
        overfitting_signals['val_accuracy'],
        overfitting_signals['train_confidence'],
        overfitting_signals['val_confidence']
    ]
    colors = ['blue', 'red', 'green', 'orange']
    
    bars = ax5.bar(metrics, values, color=colors, alpha=0.7)
    ax5.set_title('Overfitting Analysis')
    ax5.set_ylabel('Score')
    ax5.set_ylim(0, 1)
    
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontsize=10)
    
    # 6. 성능 차이
    ax6 = plt.subplot(3, 4, 6)
    gaps = ['Accuracy Gap', 'Confidence Gap']
    gap_values = [overfitting_signals['accuracy_gap'], overfitting_signals['confidence_gap']]
    colors = ['red' if gap > 0.05 else 'green' for gap in gap_values]
    
    bars = ax6.bar(gaps, gap_values, color=colors, alpha=0.7)
    ax6.set_title('Performance Gaps')
    ax6.set_ylabel('Gap')
    ax6.axhline(y=0.05, color='red', linestyle='--', alpha=0.7, label='Overfitting Threshold')
    ax6.legend()
    
    for bar, value in zip(bars, gap_values):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{value:.3f}', ha='center', va='bottom', fontsize=10)
    
    # 7. 클래스별 샘플 수
    ax7 = plt.subplot(3, 4, 7)
    supports = [p['support'] for p in class_performance]
    bars = ax7.bar(range(len(classes)), supports, alpha=0.7, color='purple')
    ax7.set_title('Class-wise Sample Counts')
    ax7.set_xlabel('Class')
    ax7.set_ylabel('Count')
    ax7.set_xticks(range(len(classes)))
    ax7.set_xticklabels(classes, rotation=45, ha='right')
    ax7.grid(True, alpha=0.3)
    
    # 8. 성능 등급 분포
    ax8 = plt.subplot(3, 4, 8)
    performance_grades = {
        'Excellent (≥0.9)': len([p for p in class_performance if p['f1_score'] >= 0.9]),
        'Good (0.7-0.9)': len([p for p in class_performance if 0.7 <= p['f1_score'] < 0.9]),
        'Poor (<0.7)': len([p for p in class_performance if p['f1_score'] < 0.7])
    }
    
    colors = ['green', 'orange', 'red']
    bars = ax8.bar(performance_grades.keys(), performance_grades.values(), color=colors, alpha=0.7)
    ax8.set_title('Performance Grade Distribution')
    ax8.set_ylabel('Number of Classes')
    
    for bar, value in zip(bars, performance_grades.values()):
        height = bar.get_height()
        ax8.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                str(value), ha='center', va='bottom', fontsize=12)
    
    # 9. 상위/하위 성능 클래스
    ax9 = plt.subplot(3, 4, 9)
    top_5_classes = [p['class'] for p in class_performance[:5]]
    top_5_f1 = [p['f1_score'] for p in class_performance[:5]]
    
    bars = ax9.bar(top_5_classes, top_5_f1, alpha=0.7, color='green')
    ax9.set_title('Top 5 Performing Classes')
    ax9.set_ylabel('F1 Score')
    ax9.set_xticklabels(top_5_classes, rotation=45, ha='right')
    ax9.set_ylim(0, 1)
    
    # 10. 하위 5개 성능 클래스
    ax10 = plt.subplot(3, 4, 10)
    bottom_5_classes = [p['class'] for p in class_performance[-5:]]
    bottom_5_f1 = [p['f1_score'] for p in class_performance[-5:]]
    
    bars = ax10.bar(bottom_5_classes, bottom_5_f1, alpha=0.7, color='red')
    ax10.set_title('Bottom 5 Performing Classes')
    ax10.set_ylabel('F1 Score')
    ax10.set_xticklabels(bottom_5_classes, rotation=45, ha='right')
    ax10.set_ylim(0, 1)
    
    # 11. 혼동 행렬 오분류 패턴
    ax11 = plt.subplot(3, 4, 11)
    # 대각선을 제외한 오분류만 표시
    cm_errors = cm.copy()
    np.fill_diagonal(cm_errors, 0)
    
    if np.max(cm_errors) > 0:
        sns.heatmap(cm_errors, annot=True, fmt='d', cmap='Reds',
                    xticklabels=class_names, yticklabels=class_names, ax=ax11)
        ax11.set_title('Confusion Matrix (Errors Only)')
        ax11.set_xlabel('Predicted Label')
        ax11.set_ylabel('True Label')
        plt.setp(ax11.get_xticklabels(), rotation=45, ha='right')
        plt.setp(ax11.get_yticklabels(), rotation=0)
    
    # 12. 과적합 경고
    ax12 = plt.subplot(3, 4, 12)
    if overfitting_signals['is_overfitting']:
        ax12.text(0.5, 0.7, '⚠️ OVERFITTING DETECTED', ha='center', va='center', 
                 fontsize=16, fontweight='bold', color='red', transform=ax12.transAxes)
        ax12.text(0.5, 0.5, f'Accuracy Gap: {overfitting_signals["accuracy_gap"]:.3f}', 
                 ha='center', va='center', fontsize=12, transform=ax12.transAxes)
        ax12.text(0.5, 0.3, f'Confidence Gap: {overfitting_signals["confidence_gap"]:.3f}', 
                 ha='center', va='center', fontsize=12, transform=ax12.transAxes)
    else:
        ax12.text(0.5, 0.5, '✅ No Overfitting Detected', ha='center', va='center', 
                 fontsize=16, fontweight='bold', color='green', transform=ax12.transAxes)
    
    ax12.set_xlim(0, 1)
    ax12.set_ylim(0, 1)
    ax12.axis('off')
    
    plt.tight_layout()
    plt.savefig('detailed_class_performance_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print('✅ 시각화 완료: detailed_class_performance_analysis.png')

def print_detailed_report(accuracy, class_performance, overfitting_signals):
    """상세 리포트 출력"""
    print('\n' + '='*80)
    print('📋 클래스별 상세 성능 분석 리포트')
    print('='*80)
    
    print(f'\n🎯 전체 성능:')
    print(f'  - 테스트 정확도: {accuracy:.4f} ({accuracy*100:.2f}%)')
    
    print(f'\n🔍 과적합 분석:')
    print(f'  - 훈련 정확도: {overfitting_signals["train_accuracy"]:.4f}')
    print(f'  - 검증 정확도: {overfitting_signals["val_accuracy"]:.4f}')
    print(f'  - 정확도 차이: {overfitting_signals["accuracy_gap"]:.4f}')
    print(f'  - 훈련 신뢰도: {overfitting_signals["train_confidence"]:.4f}')
    print(f'  - 검증 신뢰도: {overfitting_signals["val_confidence"]:.4f}')
    print(f'  - 신뢰도 차이: {overfitting_signals["confidence_gap"]:.4f}')
    
    if overfitting_signals['is_overfitting']:
        print(f'  ⚠️  과적합 감지됨!')
    else:
        print(f'  ✅ 과적합 없음')
    
    print(f'\n📊 클래스별 성능 (F1 점수 기준 정렬):')
    print('-' * 60)
    print(f"{'클래스':<4} {'F1':<6} {'Precision':<10} {'Recall':<8} {'Support':<8}")
    print('-' * 60)
    
    for perf in class_performance:
        print(f"{perf['class']:<4} {perf['f1_score']:<6.3f} {perf['precision']:<10.3f} "
              f"{perf['recall']:<8.3f} {perf['support']:<8}")
    
    # 성능 등급 분석
    excellent = [p for p in class_performance if p['f1_score'] >= 0.9]
    good = [p for p in class_performance if 0.7 <= p['f1_score'] < 0.9]
    poor = [p for p in class_performance if p['f1_score'] < 0.7]
    
    print(f'\n🏆 성능 등급 분석:')
    print(f'  - 우수 (F1 ≥ 0.9): {len(excellent)}개 클래스')
    print(f'  - 양호 (0.7 ≤ F1 < 0.9): {len(good)}개 클래스')
    print(f'  - 미흡 (F1 < 0.7): {len(poor)}개 클래스')
    
    if excellent:
        print(f'\n🥇 우수 성능 클래스:')
        for perf in excellent:
            print(f'  - {perf["class"]}: F1={perf["f1_score"]:.3f}')
    
    if poor:
        print(f'\n⚠️  개선 필요 클래스:')
        for perf in poor:
            print(f'  - {perf["class"]}: F1={perf["f1_score"]:.3f}')
    
    print('\n' + '='*80)

def main():
    """메인 함수"""
    # 데이터 경로
    data_dir = '../SignGlove/external/SignGlove_HW/datasets/unified'
    
    # GPU 사용 가능 여부 확인
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'🖥️  사용 디바이스: {device}')
    
    # 1. 데이터 로드
    all_data, all_labels, class_names = load_and_preprocess_data(data_dir, max_samples_per_class=25)
    
    # 2. 라벨 범위 수정
    if min(all_labels) != 0 or max(all_labels) >= len(class_names):
        print('⚠️  라벨 범위 문제 발견! 라벨 재매핑 중...')
        unique_labels = sorted(set(all_labels))
        label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
        all_labels = [label_mapping[label] for label in all_labels]
        print(f'재매핑 후 라벨 범위: {min(all_labels)} ~ {max(all_labels)}')
    
    # 3. 데이터 분할
    train_data, temp_data, train_labels, temp_labels = train_test_split(
        all_data, all_labels, test_size=0.4, random_state=42, stratify=all_labels
    )
    val_data, test_data, val_labels, test_labels = train_test_split(
        temp_data, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
    )
    
    print(f'📊 데이터 분할: 훈련={len(train_data)}, 검증={len(val_data)}, 테스트={len(test_data)}')
    
    # 4. 전처리 적용
    preprocessor = AdvancedPreprocessor()
    preprocessor.fit(train_data)
    
    train_data_processed = preprocessor.transform(train_data)
    val_data_processed = preprocessor.transform(val_data)
    test_data_processed = preprocessor.transform(test_data)
    
    # 5. 모델 로드
    print('🤖 모델 로드 중...')
    model = ImprovedGRU(input_size=8, hidden_size=64, num_layers=2, num_classes=24, dropout=0.3)
    
    checkpoint = torch.load('improved_preprocessing_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    print('✅ 모델 로드 완료')
    
    # 6. 클래스별 성능 분석
    accuracy, report, cm, class_performance, confidences, probabilities = analyze_class_performance(
        model, test_data_processed, test_labels, class_names, device
    )
    
    # 7. 과적합 신호 분석
    overfitting_signals = analyze_overfitting_signals(
        train_data_processed, train_labels, val_data_processed, val_labels, model, device
    )
    
    # 8. 상세 리포트 출력
    print_detailed_report(accuracy, class_performance, overfitting_signals)
    
    # 9. 시각화
    visualize_detailed_analysis(accuracy, cm, class_performance, overfitting_signals, class_names)
    
    print('\n🎉 클래스별 상세 성능 분석 완료!')

if __name__ == "__main__":
    main()
