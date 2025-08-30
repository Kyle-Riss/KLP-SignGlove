import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import h5py
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# 더 나은 MLP 모델 (이전과 동일)
class BetterMLPModel(nn.Module):
    def __init__(self, input_size=160, hidden_sizes=[128, 64, 32], num_classes=24, dropout=0.5):
        super(BetterMLPModel, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, num_classes))
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        return self.network(x)

# 데이터셋 클래스 (이전과 동일)
class EnhancedH5Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, sequence_length=20, use_augmentation=True):
        self.data_dir = data_dir
        self.sequence_length = sequence_length
        self.use_augmentation = use_augmentation
        self.data = []
        self.labels = []
        self.class_names = []
        
        for class_name in sorted(os.listdir(data_dir)):
            class_dir = os.path.join(data_dir, class_name)
            if os.path.isdir(class_dir):
                self.class_names.append(class_name)
                class_idx = len(self.class_names) - 1
                
                for session in ['1', '2', '3', '4', '5']:
                    session_dir = os.path.join(class_dir, session)
                    if os.path.exists(session_dir):
                        for file_name in os.listdir(session_dir):
                            if file_name.endswith('.h5'):
                                file_path = os.path.join(session_dir, file_name)
                                try:
                                    with h5py.File(file_path, 'r') as f:
                                        sensor_data = f['sensor_data'][:]
                                        
                                        if len(sensor_data) >= sequence_length:
                                            data = sensor_data[-sequence_length:]
                                            self.data.append(data)
                                            self.labels.append(class_idx)
                                            
                                            if self.use_augmentation:
                                                # 노이즈 추가
                                                for noise_level in [0.005, 0.01, 0.02]:
                                                    noise_data = data + np.random.normal(0, noise_level, data.shape)
                                                    self.data.append(noise_data)
                                                    self.labels.append(class_idx)
                                                
                                                # 시간 이동
                                                for shift in [2, 4, 6]:
                                                    if len(sensor_data) >= sequence_length + shift:
                                                        shifted_data = sensor_data[-(sequence_length+shift):-shift]
                                                        self.data.append(shifted_data)
                                                        self.labels.append(class_idx)
                                                
                                                # 스케일링
                                                for scale in [0.98, 1.02]:
                                                    scaled_data = data * scale
                                                    self.data.append(scaled_data)
                                                    self.labels.append(class_idx)
                                                
                                                # 마스킹
                                                masked_data = data.copy()
                                                mask_indices = np.random.choice(8, 2, replace=False)
                                                masked_data[:, mask_indices] = 0
                                                self.data.append(masked_data)
                                                self.labels.append(class_idx)
                                                
                                except Exception as e:
                                    print(f"Error loading {file_path}: {e}")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.data[idx]), torch.LongTensor([self.labels[idx]])

print('🔍 클래스별 성능 분석 시작')

# 데이터셋 로드
dataset = EnhancedH5Dataset(
    data_dir='../SignGlove/external/SignGlove_HW/datasets/unified',
    sequence_length=20,
    use_augmentation=True
)

# 데이터 분할
train_indices, test_indices = train_test_split(
    range(len(dataset)), test_size=0.3, random_state=42, 
    stratify=dataset.labels
)
train_indices, val_indices = train_test_split(
    train_indices, test_size=0.2, random_state=42, 
    stratify=[dataset.labels[i] for i in train_indices]
)

# 테스트 데이터로더
test_loader = DataLoader(torch.utils.data.Subset(dataset, test_indices), 
                        batch_size=32, shuffle=False)

# 모델 로드
model = BetterMLPModel(input_size=160, hidden_sizes=[128, 64, 32], num_classes=24, dropout=0.5)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.load_state_dict(torch.load('better_mlp_model.pth'))
model = model.to(device)
model.eval()

# 예측 수행
all_predictions = []
all_targets = []
all_probabilities = []

with torch.no_grad():
    for data, targets in test_loader:
        data, targets = data.to(device), targets.to(device)
        outputs = model(data)
        probabilities = torch.softmax(outputs, dim=1)
        _, predicted = outputs.max(1)
        
        all_predictions.extend(predicted.cpu().numpy())
        all_targets.extend(targets.squeeze().cpu().numpy())
        all_probabilities.extend(probabilities.cpu().numpy())

# 분류 리포트
print('\n📊 전체 분류 리포트:')
report = classification_report(all_targets, all_predictions, 
                             target_names=dataset.class_names, 
                             zero_division=0, output_dict=True)

# 결과를 DataFrame으로 변환
results_df = pd.DataFrame(report).transpose()
results_df = results_df.iloc[:-3]  # 마지막 3행(accuracy, macro avg, weighted avg) 제외

print(f'\n🎯 전체 정확도: {report["accuracy"]:.2%}')

# 클래스별 성능 시각화
fig, axes = plt.subplots(2, 2, figsize=(20, 16))
fig.suptitle('KLP-SignGlove MLP 모델 클래스별 성능 분석', fontsize=16, fontweight='bold')

# 1. 정확도, 정밀도, 재현율 비교
metrics = ['precision', 'recall', 'f1-score']
x = np.arange(len(dataset.class_names))
width = 0.25

for i, metric in enumerate(metrics):
    values = [results_df.loc[class_name, metric] for class_name in dataset.class_names]
    axes[0, 0].bar(x + i*width, values, width, label=metric.capitalize(), alpha=0.8)

axes[0, 0].set_xlabel('클래스')
axes[0, 0].set_ylabel('점수')
axes[0, 0].set_title('클래스별 정확도, 정밀도, 재현율 비교')
axes[0, 0].set_xticks(x + width)
axes[0, 0].set_xticklabels(dataset.class_names, rotation=45)
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. F1-score 순위
f1_scores = [results_df.loc[class_name, 'f1-score'] for class_name in dataset.class_names]
sorted_indices = np.argsort(f1_scores)[::-1]
sorted_classes = [dataset.class_names[i] for i in sorted_indices]
sorted_f1 = [f1_scores[i] for i in sorted_indices]

colors = ['green' if score >= 0.9 else 'orange' if score >= 0.7 else 'red' for score in sorted_f1]
axes[0, 1].barh(range(len(sorted_classes)), sorted_f1, color=colors, alpha=0.8)
axes[0, 1].set_yticks(range(len(sorted_classes)))
axes[0, 1].set_yticklabels(sorted_classes)
axes[0, 1].set_xlabel('F1-Score')
axes[0, 1].set_title('클래스별 F1-Score 순위')
axes[0, 1].grid(True, alpha=0.3)

# 성능 등급 표시
for i, score in enumerate(sorted_f1):
    if score >= 0.9:
        grade = 'A+'
    elif score >= 0.8:
        grade = 'A'
    elif score >= 0.7:
        grade = 'B'
    elif score >= 0.6:
        grade = 'C'
    else:
        grade = 'D'
    axes[0, 1].text(score + 0.01, i, grade, va='center', fontweight='bold')

# 3. 혼동 행렬
cm = confusion_matrix(all_targets, all_predictions)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=dataset.class_names, yticklabels=dataset.class_names,
            ax=axes[1, 0])
axes[1, 0].set_title('혼동 행렬')
axes[1, 0].set_xlabel('예측 클래스')
axes[1, 0].set_ylabel('실제 클래스')

# 4. 클래스별 샘플 수와 성능
sample_counts = [results_df.loc[class_name, 'support'] for class_name in dataset.class_names]
f1_scores = [results_df.loc[class_name, 'f1-score'] for class_name in dataset.class_names]

scatter = axes[1, 1].scatter(sample_counts, f1_scores, c=range(len(dataset.class_names)), 
                           cmap='viridis', s=100, alpha=0.7)
axes[1, 1].set_xlabel('테스트 샘플 수')
axes[1, 1].set_ylabel('F1-Score')
axes[1, 1].set_title('샘플 수 vs 성능')
axes[1, 1].grid(True, alpha=0.3)

# 클래스 이름 표시
for i, class_name in enumerate(dataset.class_names):
    axes[1, 1].annotate(class_name, (sample_counts[i], f1_scores[i]), 
                       xytext=(5, 5), textcoords='offset points', fontsize=8)

plt.tight_layout()
plt.savefig('class_performance_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# 상세 성능 테이블 출력
print('\n📋 클래스별 상세 성능:')
print('=' * 80)
print(f"{'클래스':<4} {'정밀도':<8} {'재현율':<8} {'F1-Score':<10} {'샘플수':<8} {'등급':<4}")
print('=' * 80)

for class_name in dataset.class_names:
    precision = results_df.loc[class_name, 'precision']
    recall = results_df.loc[class_name, 'recall']
    f1 = results_df.loc[class_name, 'f1-score']
    support = results_df.loc[class_name, 'support']
    
    if f1 >= 0.9:
        grade = 'A+'
    elif f1 >= 0.8:
        grade = 'A'
    elif f1 >= 0.7:
        grade = 'B'
    elif f1 >= 0.6:
        grade = 'C'
    else:
        grade = 'D'
    
    print(f"{class_name:<4} {precision:<8.3f} {recall:<8.3f} {f1:<10.3f} {support:<8.0f} {grade:<4}")

print('=' * 80)

# 성능 요약
excellent = sum(1 for f1 in f1_scores if f1 >= 0.9)
good = sum(1 for f1 in f1_scores if 0.8 <= f1 < 0.9)
fair = sum(1 for f1 in f1_scores if 0.7 <= f1 < 0.8)
poor = sum(1 for f1 in f1_scores if f1 < 0.7)

print(f'\n📈 성능 요약:')
print(f'🏆 우수 (A+): {excellent}개 클래스')
print(f'👍 좋음 (A): {good}개 클래스')
print(f'😐 보통 (B): {fair}개 클래스')
print(f'⚠️ 개선필요 (C/D): {poor}개 클래스')

# 가장 문제가 되는 클래스들
worst_classes = []
for i, f1 in enumerate(f1_scores):
    if f1 < 0.7:
        worst_classes.append((dataset.class_names[i], f1))

if worst_classes:
    print(f'\n🔴 개선이 필요한 클래스들:')
    for class_name, f1 in sorted(worst_classes, key=lambda x: x[1]):
        print(f'  {class_name}: {f1:.3f}')

print(f'\n💾 시각화 결과가 "class_performance_analysis.png"에 저장되었습니다.')
