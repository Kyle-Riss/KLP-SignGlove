import torch
import numpy as np
import h5py
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

print('🔍 센서 패턴 상세 분석 시작')

# 문제 클래스들
problem_classes = ['ㅕ', 'ㅈ', 'ㄹ']
good_classes = ['ㅎ', 'ㄷ', 'ㅏ']

# 센서 이름
sensor_names = ['Flex1', 'Flex2', 'Flex3', 'Flex4', 'Flex5', 'Pitch', 'Roll', 'Yaw']

def load_class_data(class_name, data_dir):
    """특정 클래스의 모든 데이터를 로드"""
    class_data = []
    class_dir = os.path.join(data_dir, class_name)
    
    for session in ['1', '2', '3', '4', '5']:
        session_dir = os.path.join(class_dir, session)
        if os.path.exists(session_dir):
            for file_name in os.listdir(session_dir):
                if file_name.endswith('.h5'):
                    file_path = os.path.join(session_dir, file_name)
                    try:
                        with h5py.File(file_path, 'r') as f:
                            sensor_data = f['sensor_data'][:]  # (300, 8)
                            # 마지막 20 프레임 사용
                            data = sensor_data[-20:]
                            class_data.append(data)
                    except Exception as e:
                        print(f"Error loading {file_path}: {e}")
    
    return np.array(class_data) if class_data else None

# 데이터 로드
data_dir = '../SignGlove/external/SignGlove_HW/datasets/unified'
class_data = {}

for class_name in problem_classes + good_classes:
    data = load_class_data(class_name, data_dir)
    if data is not None:
        class_data[class_name] = data
        print(f'📊 {class_name} 클래스: {data.shape}')

# 상세 분석
print('\n📈 센서별 상세 분석:')
print('=' * 120)

for class_name in problem_classes + good_classes:
    if class_name not in class_data:
        continue
        
    data = class_data[class_name]
    print(f'\n🔍 {class_name} 클래스 센서 분석:')
    
    # 센서별 통계
    for i, sensor_name in enumerate(sensor_names):
        sensor_values = data[:, :, i].flatten()
        mean_val = np.mean(sensor_values)
        std_val = np.std(sensor_values)
        min_val = np.min(sensor_values)
        max_val = np.max(sensor_values)
        
        print(f'  {sensor_name:>8}: 평균={mean_val:8.2f}, 표준편차={std_val:8.2f}, 범위=[{min_val:6.2f}, {max_val:6.2f}]')

# 시각화
fig, axes = plt.subplots(3, 3, figsize=(20, 15))
fig.suptitle('문제 클래스 vs 우수 클래스 센서 패턴 비교', fontsize=16, fontweight='bold')

# 1. Flex 센서 비교 (Flex1)
for i, class_name in enumerate(problem_classes + good_classes):
    if class_name not in class_data:
        continue
    data = class_data[class_name]
    flex1_data = data[:, :, 0].flatten()  # Flex1
    
    color = 'red' if class_name in problem_classes else 'green'
    axes[0, 0].hist(flex1_data, bins=20, alpha=0.6, label=class_name, color=color)

axes[0, 0].set_xlabel('Flex1 값')
axes[0, 0].set_ylabel('빈도')
axes[0, 0].set_title('Flex1 센서 분포 비교')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. Flex 센서 비교 (Flex2)
for i, class_name in enumerate(problem_classes + good_classes):
    if class_name not in class_data:
        continue
    data = class_data[class_name]
    flex2_data = data[:, :, 1].flatten()  # Flex2
    
    color = 'red' if class_name in problem_classes else 'green'
    axes[0, 1].hist(flex2_data, bins=20, alpha=0.6, label=class_name, color=color)

axes[0, 1].set_xlabel('Flex2 값')
axes[0, 1].set_ylabel('빈도')
axes[0, 1].set_title('Flex2 센서 분포 비교')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 3. Flex 센서 비교 (Flex3)
for i, class_name in enumerate(problem_classes + good_classes):
    if class_name not in class_data:
        continue
    data = class_data[class_name]
    flex3_data = data[:, :, 2].flatten()  # Flex3
    
    color = 'red' if class_name in problem_classes else 'green'
    axes[0, 2].hist(flex3_data, bins=20, alpha=0.6, label=class_name, color=color)

axes[0, 2].set_xlabel('Flex3 값')
axes[0, 2].set_ylabel('빈도')
axes[0, 2].set_title('Flex3 센서 분포 비교')
axes[0, 2].legend()
axes[0, 2].grid(True, alpha=0.3)

# 4. Orientation 센서 비교 (Pitch)
for i, class_name in enumerate(problem_classes + good_classes):
    if class_name not in class_data:
        continue
    data = class_data[class_name]
    pitch_data = data[:, :, 5].flatten()  # Pitch
    
    color = 'red' if class_name in problem_classes else 'green'
    axes[1, 0].hist(pitch_data, bins=20, alpha=0.6, label=class_name, color=color)

axes[1, 0].set_xlabel('Pitch 값')
axes[1, 0].set_ylabel('빈도')
axes[1, 0].set_title('Pitch 센서 분포 비교')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 5. Orientation 센서 비교 (Roll)
for i, class_name in enumerate(problem_classes + good_classes):
    if class_name not in class_data:
        continue
    data = class_data[class_name]
    roll_data = data[:, :, 6].flatten()  # Roll
    
    color = 'red' if class_name in problem_classes else 'green'
    axes[1, 1].hist(roll_data, bins=20, alpha=0.6, label=class_name, color=color)

axes[1, 1].set_xlabel('Roll 값')
axes[1, 1].set_ylabel('빈도')
axes[1, 1].set_title('Roll 센서 분포 비교')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

# 6. Orientation 센서 비교 (Yaw)
for i, class_name in enumerate(problem_classes + good_classes):
    if class_name not in class_data:
        continue
    data = class_data[class_name]
    yaw_data = data[:, :, 7].flatten()  # Yaw
    
    color = 'red' if class_name in problem_classes else 'green'
    axes[1, 2].hist(yaw_data, bins=20, alpha=0.6, label=class_name, color=color)

axes[1, 2].set_xlabel('Yaw 값')
axes[1, 2].set_ylabel('빈도')
axes[1, 2].set_title('Yaw 센서 분포 비교')
axes[1, 2].legend()
axes[1, 2].grid(True, alpha=0.3)

# 7. 시계열 패턴 비교 (Flex1)
for i, class_name in enumerate(problem_classes + good_classes):
    if class_name not in class_data:
        continue
    data = class_data[class_name]
    # 첫 번째 샘플의 Flex1 시계열
    flex1_series = data[0, :, 0]
    
    color = 'red' if class_name in problem_classes else 'green'
    axes[2, 0].plot(flex1_series, label=class_name, color=color, alpha=0.7)

axes[2, 0].set_xlabel('시간 프레임')
axes[2, 0].set_ylabel('Flex1 값')
axes[2, 0].set_title('Flex1 시계열 패턴 (첫 번째 샘플)')
axes[2, 0].legend()
axes[2, 0].grid(True, alpha=0.3)

# 8. 시계열 패턴 비교 (Pitch)
for i, class_name in enumerate(problem_classes + good_classes):
    if class_name not in class_data:
        continue
    data = class_data[class_name]
    # 첫 번째 샘플의 Pitch 시계열
    pitch_series = data[0, :, 5]
    
    color = 'red' if class_name in problem_classes else 'green'
    axes[2, 1].plot(pitch_series, label=class_name, color=color, alpha=0.7)

axes[2, 1].set_xlabel('시간 프레임')
axes[2, 1].set_ylabel('Pitch 값')
axes[2, 1].set_title('Pitch 시계열 패턴 (첫 번째 샘플)')
axes[2, 1].legend()
axes[2, 1].grid(True, alpha=0.3)

# 9. 센서 간 상관관계 히트맵 (ㅕ 클래스)
if 'ㅕ' in class_data:
    data = class_data['ㅕ']
    # 모든 샘플의 평균 계산
    mean_data = np.mean(data, axis=0)  # (20, 8)
    
    # 센서 간 상관관계 계산
    corr_matrix = np.corrcoef(mean_data.T)
    
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                xticklabels=sensor_names, yticklabels=sensor_names,
                ax=axes[2, 2])
    axes[2, 2].set_title('ㅕ 클래스 센서 상관관계')

plt.tight_layout()
plt.savefig('detailed_sensor_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# 문제점 분석
print('\n🎯 문제점 상세 분석:')
print('=' * 80)

# 1. ㅕ 클래스 분석
if 'ㅕ' in class_data:
    data = class_data['ㅕ']
    print(f'\n🔴 ㅕ (여) 클래스 문제점:')
    
    # Flex2 센서의 높은 표준편차 확인
    flex2_data = data[:, :, 1].flatten()
    flex2_std = np.std(flex2_data)
    print(f'  - Flex2 표준편차: {flex2_std:.2f} (매우 높음)')
    
    # Pitch 값의 특이성 확인
    pitch_data = data[:, :, 5].flatten()
    pitch_mean = np.mean(pitch_data)
    print(f'  - Pitch 평균: {pitch_mean:.2f} (다른 클래스와 매우 다름)')
    
    # 데이터 일관성 확인
    sample_std = np.std(data, axis=(1, 2))  # 각 샘플의 표준편차
    print(f'  - 샘플별 변동성: {np.mean(sample_std):.2f}')

# 2. ㅈ 클래스 분석
if 'ㅈ' in class_data:
    data = class_data['ㅈ']
    print(f'\n🔴 ㅈ (지읒) 클래스 문제점:')
    
    # 낮은 표준편차 확인
    overall_std = np.std(data)
    print(f'  - 전체 표준편차: {overall_std:.2f} (매우 낮음)')
    
    # Flex4 센서의 높은 표준편차
    flex4_data = data[:, :, 3].flatten()
    flex4_std = np.std(flex4_data)
    print(f'  - Flex4 표준편차: {flex4_std:.2f} (높음)')
    
    # Pitch 값의 특이성
    pitch_data = data[:, :, 5].flatten()
    pitch_mean = np.mean(pitch_data)
    print(f'  - Pitch 평균: {pitch_mean:.2f} (다른 클래스와 다름)')

# 3. ㄹ 클래스 분석
if 'ㄹ' in class_data:
    data = class_data['ㄹ']
    print(f'\n🔴 ㄹ (리을) 클래스 문제점:')
    
    # 전반적으로 낮은 표준편차
    overall_std = np.std(data)
    print(f'  - 전체 표준편차: {overall_std:.2f} (낮음)')
    
    # Yaw 센서의 낮은 표준편차
    yaw_data = data[:, :, 7].flatten()
    yaw_std = np.std(yaw_data)
    print(f'  - Yaw 표준편차: {yaw_std:.2f} (매우 낮음)')

# 우수 클래스와의 비교
print(f'\n✅ 우수 클래스 특징:')
for class_name in good_classes:
    if class_name not in class_data:
        continue
    data = class_data[class_name]
    overall_std = np.std(data)
    print(f'  - {class_name}: 전체 표준편차 {overall_std:.2f}')

print(f'\n💡 개선 방안:')
print(f'1. ㅕ 클래스: Flex2 센서의 높은 변동성 문제 해결 필요')
print(f'2. ㅈ 클래스: 데이터의 낮은 변동성으로 인한 구분 어려움')
print(f'3. ㄹ 클래스: 전반적으로 낮은 변동성, 더 명확한 패턴 필요')
print(f'4. 추가 데이터 수집 시 더 일관된 제스처 수행 필요')

print(f'\n💾 상세 분석 결과가 "detailed_sensor_analysis.png"에 저장되었습니다.')
