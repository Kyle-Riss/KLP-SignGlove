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
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# 데이터셋 클래스
class DetailedH5Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, sequence_length=20, use_augmentation=False):
        self.data_dir = data_dir
        self.sequence_length = sequence_length
        self.use_augmentation = use_augmentation
        self.data = []
        self.labels = []
        self.class_names = []
        self.file_paths = []  # 파일 경로 저장
        self.session_info = []  # 세션 정보 저장
        
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
                                            self.file_paths.append(file_path)
                                            self.session_info.append(session)
                                            
                                            if self.use_augmentation:
                                                # 원본 데이터만 증강
                                                for noise_level in [0.005, 0.01, 0.02]:
                                                    noise_data = data + np.random.normal(0, noise_level, data.shape)
                                                    self.data.append(noise_data)
                                                    self.labels.append(class_idx)
                                                    self.file_paths.append(file_path + f"_noise_{noise_level}")
                                                    self.session_info.append(session)
                                                
                                except Exception as e:
                                    print(f"Error loading {file_path}: {e}")
        
        print(f"📊 상세 데이터셋 로드 완료: {len(self.data)} 샘플, {len(self.class_names)} 클래스")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.data[idx]), torch.LongTensor([self.labels[idx]])

print('🔍 문제 클래스 상세 분석 시작')

# 문제 클래스들
problem_classes = ['ㅕ', 'ㅈ', 'ㄹ']
good_classes = ['ㅎ', 'ㄷ', 'ㅏ']  # 비교용 우수 클래스

# 데이터셋 로드 (증강 없이)
dataset = DetailedH5Dataset(
    data_dir='../SignGlove/external/SignGlove_HW/datasets/unified',
    sequence_length=20,
    use_augmentation=False
)

# 문제 클래스와 우수 클래스의 인덱스 찾기
problem_indices = [dataset.class_names.index(cls) for cls in problem_classes]
good_indices = [dataset.class_names.index(cls) for cls in good_classes]

print(f'\n📋 분석 대상 클래스:')
print(f'문제 클래스: {problem_classes} (인덱스: {problem_indices})')
print(f'우수 클래스: {good_classes} (인덱스: {good_indices})')

# 클래스별 데이터 분석
class_analysis = {}

for class_name in problem_classes + good_classes:
    class_idx = dataset.class_names.index(class_name)
    class_data_indices = [i for i, label in enumerate(dataset.labels) if label == class_idx]
    
    print(f'\n🔍 {class_name} 클래스 분석:')
    print(f'  총 샘플 수: {len(class_data_indices)}')
    
    # 세션별 분포
    session_counts = defaultdict(int)
    for idx in class_data_indices:
        session_counts[dataset.session_info[idx]] += 1
    
    print(f'  세션별 분포: {dict(session_counts)}')
    
    # 센서 데이터 통계
    class_data = [dataset.data[idx] for idx in class_data_indices]
    class_data_array = np.array(class_data)
    
    print(f'  데이터 형태: {class_data_array.shape}')
    print(f'  센서별 평균: {np.mean(class_data_array, axis=(0,1))}')
    print(f'  센서별 표준편차: {np.std(class_data_array, axis=(0,1))}')
    print(f'  전체 범위: {np.min(class_data_array):.2f} ~ {np.max(class_data_array):.2f}')
    
    class_analysis[class_name] = {
        'indices': class_data_indices,
        'data': class_data_array,
        'session_counts': dict(session_counts),
        'mean': np.mean(class_data_array, axis=(0,1)),
        'std': np.std(class_data_array, axis=(0,1)),
        'min': np.min(class_data_array),
        'max': np.max(class_data_array)
    }

# 시각화
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle('문제 클래스 vs 우수 클래스 데이터 분석', fontsize=16, fontweight='bold')

# 1. 센서별 평균값 비교
sensor_names = ['Flex1', 'Flex2', 'Flex3', 'Flex4', 'Flex5', 'Pitch', 'Roll', 'Yaw']
x = np.arange(len(sensor_names))
width = 0.15

for i, class_name in enumerate(problem_classes + good_classes):
    means = class_analysis[class_name]['mean']
    offset = (i - len(problem_classes)) * width
    color = 'red' if class_name in problem_classes else 'green'
    axes[0, 0].bar(x + offset, means, width, label=class_name, color=color, alpha=0.7)

axes[0, 0].set_xlabel('센서')
axes[0, 0].set_ylabel('평균값')
axes[0, 0].set_title('센서별 평균값 비교')
axes[0, 0].set_xticks(x + width)
axes[0, 0].set_xticklabels(sensor_names, rotation=45)
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. 센서별 표준편차 비교
for i, class_name in enumerate(problem_classes + good_classes):
    stds = class_analysis[class_name]['std']
    offset = (i - len(problem_classes)) * width
    color = 'red' if class_name in problem_classes else 'green'
    axes[0, 1].bar(x + offset, stds, width, label=class_name, color=color, alpha=0.7)

axes[0, 1].set_xlabel('센서')
axes[0, 1].set_ylabel('표준편차')
axes[0, 1].set_title('센서별 표준편차 비교')
axes[0, 1].set_xticks(x + width)
axes[0, 1].set_xticklabels(sensor_names, rotation=45)
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 3. 데이터 범위 비교
ranges = []
class_names = problem_classes + good_classes
colors = ['red'] * len(problem_classes) + ['green'] * len(good_classes)

for class_name in class_names:
    data_range = class_analysis[class_name]['max'] - class_analysis[class_name]['min']
    ranges.append(data_range)

bars = axes[0, 2].bar(class_names, ranges, color=colors, alpha=0.7)
axes[0, 2].set_ylabel('데이터 범위')
axes[0, 2].set_title('전체 데이터 범위 비교')
axes[0, 2].grid(True, alpha=0.3)

# 4. 세션별 분포
session_data = {}
for class_name in class_names:
    session_data[class_name] = class_analysis[class_name]['session_counts']

df_sessions = pd.DataFrame(session_data).fillna(0)
df_sessions.plot(kind='bar', ax=axes[1, 0], color=['red', 'red', 'red', 'green', 'green', 'green'], alpha=0.7)
axes[1, 0].set_xlabel('세션')
axes[1, 0].set_ylabel('샘플 수')
axes[1, 0].set_title('세션별 샘플 분포')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 5. 시계열 데이터 시각화 (첫 번째 샘플)
for i, class_name in enumerate(class_names):
    first_sample = class_analysis[class_name]['data'][0]  # 첫 번째 샘플
    color = 'red' if class_name in problem_classes else 'green'
    axes[1, 1].plot(first_sample, label=class_name, color=color, alpha=0.7)

axes[1, 1].set_xlabel('시간 프레임')
axes[1, 1].set_ylabel('센서 값')
axes[1, 1].set_title('시계열 데이터 비교 (첫 번째 샘플)')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

# 6. 센서별 분포 (Box plot)
all_data = []
all_labels = []
for class_name in class_names:
    data = class_analysis[class_name]['data']
    # Flex 센서만 선택 (0-4)
    flex_data = data[:, :, :5].reshape(-1, 5)
    all_data.extend(flex_data)
    all_labels.extend([class_name] * len(flex_data))

df_box = pd.DataFrame(all_data, columns=['Flex1', 'Flex2', 'Flex3', 'Flex4', 'Flex5'])
df_box['Class'] = all_labels

# Flex1에 대해서만 box plot
sns.boxplot(data=df_box, x='Class', y='Flex1', ax=axes[1, 2])
axes[1, 2].set_title('Flex1 센서 분포 비교')
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('problem_class_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# 상세 분석 결과 출력
print('\n📊 상세 분석 결과:')
print('=' * 100)

for class_name in problem_classes:
    print(f'\n🔴 문제 클래스: {class_name}')
    analysis = class_analysis[class_name]
    
    print(f'  📈 데이터 통계:')
    print(f'    - 평균: {analysis["mean"]}')
    print(f'    - 표준편차: {analysis["std"]}')
    print(f'    - 범위: {analysis["min"]:.2f} ~ {analysis["max"]:.2f}')
    print(f'    - 세션 분포: {analysis["session_counts"]}')
    
    # 데이터 품질 분석
    data_quality = []
    if np.std(analysis["std"]) < 10:
        data_quality.append("낮은 변동성")
    if analysis["max"] - analysis["min"] < 100:
        data_quality.append("좁은 데이터 범위")
    if len(set(analysis["session_counts"].values())) == 1:
        data_quality.append("균등한 세션 분포")
    else:
        data_quality.append("불균등한 세션 분포")
    
    print(f'  🔍 데이터 품질 특징: {", ".join(data_quality)}')

print(f'\n💾 분석 결과가 "problem_class_analysis.png"에 저장되었습니다.')

# 문제점 요약
print(f'\n🎯 문제점 요약:')
print(f'1. ㅕ (여): 매우 낮은 성능 (2.6%) - 데이터 품질 문제 가능성')
print(f'2. ㅈ (지읒): 낮은 성능 (48.4%) - 센서 패턴 구분 어려움')
print(f'3. ㄹ (리을): 보통 성능 (63.9%) - 개선 여지 있음')

print(f'\n💡 개선 방안:')
print(f'1. 문제 클래스에 대한 추가 데이터 수집')
print(f'2. 클래스별 특화된 전처리')
print(f'3. 앙상블 모델 적용')
print(f'4. 데이터 증강 기법 강화')
