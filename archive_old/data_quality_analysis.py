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

print('🔍 전체 데이터셋 품질 분석 시작')

# 센서 이름
sensor_names = ['Flex1', 'Flex2', 'Flex3', 'Flex4', 'Flex5', 'Pitch', 'Roll', 'Yaw']

def load_all_data(data_dir):
    """모든 클래스의 데이터를 로드하고 품질 분석"""
    all_data = {}
    quality_issues = defaultdict(list)
    
    for class_name in sorted(os.listdir(data_dir)):
        class_dir = os.path.join(data_dir, class_name)
        if os.path.isdir(class_dir):
            print(f'📊 {class_name} 클래스 분석 중...')
            class_data = []
            file_paths = []
            
            for session in ['1', '2', '3', '4', '5']:
                session_dir = os.path.join(class_dir, session)
                if os.path.exists(session_dir):
                    for file_name in os.listdir(session_dir):
                        if file_name.endswith('.h5'):
                            file_path = os.path.join(session_dir, file_name)
                            try:
                                with h5py.File(file_path, 'r') as f:
                                    sensor_data = f['sensor_data'][:]  # (300, 8)
                                    
                                    # 데이터 품질 검사
                                    quality_check = check_data_quality(sensor_data, file_path)
                                    if quality_check['issues']:
                                        quality_issues[class_name].extend(quality_check['issues'])
                                    
                                    # 마지막 20 프레임 사용
                                    data = sensor_data[-20:]
                                    class_data.append(data)
                                    file_paths.append(file_path)
                                    
                            except Exception as e:
                                print(f"Error loading {file_path}: {e}")
                                quality_issues[class_name].append(f"파일 로드 오류: {e}")
            
            if class_data:
                all_data[class_name] = {
                    'data': np.array(class_data),
                    'file_paths': file_paths,
                    'quality_issues': quality_issues[class_name]
                }
    
    return all_data

def check_data_quality(data, file_path):
    """데이터 품질 검사"""
    issues = []
    
    # 1. NaN 값 검사
    if np.isnan(data).any():
        issues.append("NaN 값 존재")
    
    # 2. 무한값 검사
    if np.isinf(data).any():
        issues.append("무한값 존재")
    
    # 3. 센서별 범위 검사
    for i, sensor_name in enumerate(sensor_names):
        sensor_data = data[:, i]
        
        # Flex 센서 범위 검사 (0-1000)
        if i < 5:  # Flex 센서
            if np.any(sensor_data < 0) or np.any(sensor_data > 1000):
                issues.append(f"{sensor_name} 범위 오류: {np.min(sensor_data):.2f}~{np.max(sensor_data):.2f}")
        
        # Orientation 센서 범위 검사
        else:  # Pitch, Roll, Yaw
            if np.any(sensor_data < -180) or np.any(sensor_data > 180):
                issues.append(f"{sensor_name} 범위 오류: {np.min(sensor_data):.2f}~{np.max(sensor_data):.2f}")
    
    # 4. 데이터 길이 검사
    if len(data) < 20:
        issues.append(f"데이터 길이 부족: {len(data)}")
    
    # 5. 변동성 검사
    for i, sensor_name in enumerate(sensor_names):
        sensor_data = data[:, i]
        std_val = np.std(sensor_data)
        
        # 너무 낮은 변동성
        if std_val < 0.1:
            issues.append(f"{sensor_name} 변동성 부족: {std_val:.4f}")
        
        # 너무 높은 변동성 (Flex 센서)
        if i < 5 and std_val > 200:
            issues.append(f"{sensor_name} 변동성 과다: {std_val:.2f}")
    
    return {'issues': issues}

# 데이터 로드
data_dir = '../SignGlove/external/SignGlove_HW/datasets/unified'
all_data = load_all_data(data_dir)

print(f'\n📊 데이터 로드 완료: {len(all_data)} 클래스')

# 품질 이슈 요약
print('\n🔍 데이터 품질 이슈 요약:')
print('=' * 80)

total_issues = 0
for class_name, data_info in all_data.items():
    issues = data_info['quality_issues']
    if issues:
        print(f'\n🔴 {class_name} 클래스 이슈 ({len(issues)}개):')
        for issue in issues:
            print(f'  - {issue}')
        total_issues += len(issues)
    else:
        print(f'✅ {class_name} 클래스: 문제없음')

print(f'\n📈 전체 이슈 수: {total_issues}')

# 시각화
fig, axes = plt.subplots(3, 3, figsize=(20, 15))
fig.suptitle('전체 데이터셋 품질 분석', fontsize=16, fontweight='bold')

# 1. 클래스별 샘플 수
sample_counts = [len(data_info['data']) for data_info in all_data.values()]
class_names = list(all_data.keys())

axes[0, 0].bar(class_names, sample_counts, color='skyblue', alpha=0.7)
axes[0, 0].set_xlabel('클래스')
axes[0, 0].set_ylabel('샘플 수')
axes[0, 0].set_title('클래스별 샘플 수')
axes[0, 0].tick_params(axis='x', rotation=45)
axes[0, 0].grid(True, alpha=0.3)

# 2. 클래스별 이슈 수
issue_counts = [len(data_info['quality_issues']) for data_info in all_data.values()]

colors = ['red' if count > 0 else 'green' for count in issue_counts]
axes[0, 1].bar(class_names, issue_counts, color=colors, alpha=0.7)
axes[0, 1].set_xlabel('클래스')
axes[0, 1].set_ylabel('이슈 수')
axes[0, 1].set_title('클래스별 품질 이슈 수')
axes[0, 1].tick_params(axis='x', rotation=45)
axes[0, 1].grid(True, alpha=0.3)

# 3. 센서별 평균값 분포
all_means = []
all_labels = []
for class_name, data_info in all_data.items():
    data = data_info['data']
    means = np.mean(data, axis=(0, 1))  # (8,)
    all_means.extend(means)
    all_labels.extend([f'{class_name}_{sensor}' for sensor in sensor_names])

# Flex 센서만 선택
flex_means = []
flex_labels = []
for i, label in enumerate(all_labels):
    if 'Flex' in label:
        flex_means.append(all_means[i])
        flex_labels.append(label)

axes[0, 2].hist(flex_means, bins=20, color='orange', alpha=0.7)
axes[0, 2].set_xlabel('Flex 센서 평균값')
axes[0, 2].set_ylabel('빈도')
axes[0, 2].set_title('Flex 센서 평균값 분포')
axes[0, 2].grid(True, alpha=0.3)

# 4. 센서별 표준편차 분포
all_stds = []
for class_name, data_info in all_data.items():
    data = data_info['data']
    stds = np.std(data, axis=(0, 1))  # (8,)
    all_stds.extend(stds)

# Flex 센서만 선택
flex_stds = []
for i, label in enumerate(all_labels):
    if 'Flex' in label:
        flex_stds.append(all_stds[i])

axes[1, 0].hist(flex_stds, bins=20, color='green', alpha=0.7)
axes[1, 0].set_xlabel('Flex 센서 표준편차')
axes[1, 0].set_ylabel('빈도')
axes[1, 0].set_title('Flex 센서 표준편차 분포')
axes[1, 0].grid(True, alpha=0.3)

# 5. 클래스별 데이터 범위
ranges = []
for class_name, data_info in all_data.items():
    data = data_info['data']
    data_range = np.max(data) - np.min(data)
    ranges.append(data_range)

axes[1, 1].bar(class_names, ranges, color='purple', alpha=0.7)
axes[1, 1].set_xlabel('클래스')
axes[1, 1].set_ylabel('데이터 범위')
axes[1, 1].set_title('클래스별 데이터 범위')
axes[1, 1].tick_params(axis='x', rotation=45)
axes[1, 1].grid(True, alpha=0.3)

# 6. 센서별 상관관계 히트맵 (전체 평균)
all_sensor_data = []
for class_name, data_info in all_data.items():
    data = data_info['data']
    # 모든 샘플의 평균 계산
    mean_data = np.mean(data, axis=0)  # (20, 8)
    all_sensor_data.append(mean_data)

if all_sensor_data:
    combined_data = np.vstack(all_sensor_data)  # (N*20, 8)
    corr_matrix = np.corrcoef(combined_data.T)
    
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                xticklabels=sensor_names, yticklabels=sensor_names,
                ax=axes[1, 2])
    axes[1, 2].set_title('전체 센서 상관관계')

# 7. 문제가 있는 클래스들의 센서 패턴
problem_classes = ['ㅕ', 'ㅈ', 'ㄹ']
good_classes = ['ㅎ', 'ㄷ', 'ㅏ']

for class_name in problem_classes + good_classes:
    if class_name in all_data:
        data = all_data[class_name]['data']
        # Flex1 센서의 분포
        flex1_data = data[:, :, 0].flatten()
        
        color = 'red' if class_name in problem_classes else 'green'
        axes[2, 0].hist(flex1_data, bins=20, alpha=0.6, label=class_name, color=color)

axes[2, 0].set_xlabel('Flex1 값')
axes[2, 0].set_ylabel('빈도')
axes[2, 0].set_title('문제 클래스 vs 우수 클래스 Flex1 분포')
axes[2, 0].legend()
axes[2, 0].grid(True, alpha=0.3)

# 8. 클래스별 데이터 일관성 (표준편차의 표준편차)
consistency_scores = []
for class_name, data_info in all_data.items():
    data = data_info['data']
    # 각 샘플의 표준편차 계산
    sample_stds = np.std(data, axis=2)  # (N, 20)
    # 표준편차의 표준편차 (일관성 지표)
    consistency = np.std(sample_stds)
    consistency_scores.append(consistency)

axes[2, 1].bar(class_names, consistency_scores, color='brown', alpha=0.7)
axes[2, 1].set_xlabel('클래스')
axes[2, 1].set_ylabel('일관성 점수 (낮을수록 일관)')
axes[2, 1].set_title('클래스별 데이터 일관성')
axes[2, 1].tick_params(axis='x', rotation=45)
axes[2, 1].grid(True, alpha=0.3)

# 9. 품질 점수 계산
quality_scores = []
for class_name, data_info in all_data.items():
    issues = data_info['quality_issues']
    # 이슈가 적을수록 높은 점수
    score = max(0, 100 - len(issues) * 10)
    quality_scores.append(score)

axes[2, 2].bar(class_names, quality_scores, color=['red' if score < 80 else 'orange' if score < 90 else 'green' for score in quality_scores], alpha=0.7)
axes[2, 2].set_xlabel('클래스')
axes[2, 2].set_ylabel('품질 점수')
axes[2, 2].set_title('클래스별 데이터 품질 점수')
axes[2, 2].tick_params(axis='x', rotation=45)
axes[2, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('data_quality_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# 상세 분석 결과 출력
print('\n📊 상세 분석 결과:')
print('=' * 100)

# 품질 점수별 분류
excellent_classes = []
good_classes = []
poor_classes = []

for class_name, data_info in all_data.items():
    issues = data_info['quality_issues']
    score = max(0, 100 - len(issues) * 10)
    
    if score >= 90:
        excellent_classes.append((class_name, score))
    elif score >= 70:
        good_classes.append((class_name, score))
    else:
        poor_classes.append((class_name, score))

print(f'\n🏆 우수 품질 (90점 이상): {len(excellent_classes)}개 클래스')
for class_name, score in excellent_classes:
    print(f'  - {class_name}: {score}점')

print(f'\n👍 양호 품질 (70-89점): {len(good_classes)}개 클래스')
for class_name, score in good_classes:
    print(f'  - {class_name}: {score}점')

print(f'\n⚠️ 개선필요 (70점 미만): {len(poor_classes)}개 클래스')
for class_name, score in poor_classes:
    print(f'  - {class_name}: {score}점')
    issues = all_data[class_name]['quality_issues']
    for issue in issues:
        print(f'    * {issue}')

# 데이터 정제 권장사항
print(f'\n💡 데이터 정제 권장사항:')
print(f'1. 범위 오류 수정: 센서값이 정상 범위를 벗어나는 데이터 제거')
print(f'2. 변동성 조정: 너무 낮거나 높은 변동성을 가진 데이터 정규화')
print(f'3. 일관성 개선: 클래스 내 데이터 일관성 향상을 위한 전처리')
print(f'4. 추가 수집: 품질이 낮은 클래스에 대한 추가 데이터 수집')

print(f'\n💾 분석 결과가 "data_quality_analysis.png"에 저장되었습니다.')
