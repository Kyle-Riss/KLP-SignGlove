import numpy as np
import h5py
import os
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
import pandas as pd
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

print('🔍 상세 데이터 품질 분석 시작')

# 한글 폰트 설정
try:
    plt.rcParams['font.family'] = 'NanumGothic'
except:
    plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def load_all_data_detailed(data_dir):
    """모든 데이터를 상세히 로드"""
    print(f'📂 데이터 로드 중: {data_dir}')
    
    all_data = []
    all_labels = []
    class_names = []
    file_paths = []
    
    # 클래스별로 데이터 로드
    for class_idx, class_name in enumerate(sorted(os.listdir(data_dir))):
        class_path = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_path):
            continue
            
        class_names.append(class_name)
        print(f'  - 클래스 {class_name} 로딩 중...')
        
        # 각 세션의 데이터 로드
        for session in sorted(os.listdir(class_path)):
            session_path = os.path.join(class_path, session)
            if not os.path.isdir(session_path):
                continue
                
            # H5 파일들 로드
            for file_name in sorted(os.listdir(session_path)):
                if file_name.endswith('.h5'):
                    file_path = os.path.join(session_path, file_name)
                    try:
                        with h5py.File(file_path, 'r') as f:
                            sensor_data = f['sensor_data'][:]
                            all_data.append(sensor_data)
                            all_labels.append(len(class_names) - 1)  # 0부터 시작하는 인덱스
                            file_paths.append(file_path)
                    except Exception as e:
                        print(f"    경고: {file_path} 로드 실패 - {e}")
    
    print(f'✅ 총 {len(all_data)}개 샘플, {len(class_names)}개 클래스 로드 완료')
    return all_data, all_labels, class_names, file_paths

def analyze_sensor_statistics(data_list, class_names, labels):
    """센서 통계 분석"""
    print('📊 센서 통계 분석 중...')
    
    # 클래스별 통계
    class_stats = defaultdict(list)
    
    for i, (data, label) in enumerate(zip(data_list, labels)):
        class_name = class_names[label]
        
        # 기본 통계
        mean_vals = np.mean(data, axis=0)
        std_vals = np.std(data, axis=0)
        min_vals = np.min(data, axis=0)
        max_vals = np.max(data, axis=0)
        range_vals = max_vals - min_vals
        
        # 각 센서별 통계
        for sensor_idx in range(8):
            sensor_name = f'Flex_{sensor_idx+1}' if sensor_idx < 5 else f'Orientation_{sensor_idx-4}'
            
            class_stats[f'{class_name}_{sensor_name}_mean'].append(mean_vals[sensor_idx])
            class_stats[f'{class_name}_{sensor_name}_std'].append(std_vals[sensor_idx])
            class_stats[f'{class_name}_{sensor_name}_range'].append(range_vals[sensor_idx])
            class_stats[f'{class_name}_{sensor_name}_min'].append(min_vals[sensor_idx])
            class_stats[f'{class_name}_{sensor_name}_max'].append(max_vals[sensor_idx])
    
    return class_stats

def check_data_quality_issues(data_list, file_paths):
    """데이터 품질 문제 체크"""
    print('🔍 데이터 품질 문제 체크 중...')
    
    issues = {
        'nan_values': [],
        'inf_values': [],
        'zero_variance': [],
        'extreme_values': [],
        'short_sequences': [],
        'sensor_range_violations': []
    }
    
    for i, (data, file_path) in enumerate(zip(data_list, file_paths)):
        # NaN 값 체크
        if np.isnan(data).any():
            issues['nan_values'].append((i, file_path, np.isnan(data).sum()))
        
        # Inf 값 체크
        if np.isinf(data).any():
            issues['inf_values'].append((i, file_path, np.isinf(data).sum()))
        
        # 제로 분산 체크
        for sensor_idx in range(8):
            if np.std(data[:, sensor_idx]) == 0:
                issues['zero_variance'].append((i, file_path, sensor_idx))
        
        # 극단적 값 체크
        if np.any(np.abs(data) > 10000):
            issues['extreme_values'].append((i, file_path, np.max(np.abs(data))))
        
        # 짧은 시퀀스 체크
        if data.shape[0] < 20:
            issues['short_sequences'].append((i, file_path, data.shape[0]))
        
        # 센서 범위 위반 체크
        flex_sensors = data[:, :5]
        orientation_sensors = data[:, 5:]
        
        if np.any(flex_sensors < 0) or np.any(flex_sensors > 1000):
            issues['sensor_range_violations'].append((i, file_path, 'flex'))
        
        if np.any(orientation_sensors < -180) or np.any(orientation_sensors > 180):
            issues['sensor_range_violations'].append((i, file_path, 'orientation'))
    
    return issues

def analyze_class_separability(data_list, labels, class_names):
    """클래스 분리 가능성 분석"""
    print('🎯 클래스 분리 가능성 분석 중...')
    
    # 각 클래스의 대표 특성 추출
    class_features = defaultdict(list)
    
    for data, label in zip(data_list, labels):
        class_name = class_names[label]
        
        # 특성 추출
        features = []
        
        # 1. 각 센서의 평균, 표준편차
        for sensor_idx in range(8):
            sensor_data = data[:, sensor_idx]
            features.extend([np.mean(sensor_data), np.std(sensor_data)])
        
        # 2. 센서 간 상관관계 (상위 5개)
        corr_matrix = np.corrcoef(data.T)
        upper_tri = corr_matrix[np.triu_indices(8, k=1)]
        top_corrs = np.sort(upper_tri)[-5:]
        features.extend(top_corrs)
        
        # 3. 전체 데이터의 통계
        features.extend([
            np.mean(data), np.std(data), np.max(data), np.min(data),
            np.percentile(data, 25), np.percentile(data, 75)
        ])
        
        class_features[class_name].append(features)
    
    # 클래스별 평균 특성 계산
    class_avg_features = {}
    for class_name, feature_list in class_features.items():
        class_avg_features[class_name] = np.mean(feature_list, axis=0)
    
    return class_avg_features

def visualize_data_analysis(data_list, labels, class_names, class_stats, issues, class_features):
    """데이터 분석 결과 시각화"""
    print('📈 데이터 분석 결과 시각화 생성 중...')
    
    fig = plt.figure(figsize=(20, 16))
    
    # 1. 클래스별 샘플 수
    ax1 = plt.subplot(3, 4, 1)
    class_counts = Counter([class_names[label] for label in labels])
    bars = ax1.bar(class_counts.keys(), class_counts.values(), alpha=0.7)
    ax1.set_title('Class-wise Sample Counts')
    ax1.set_xlabel('Class')
    ax1.set_ylabel('Count')
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
    
    # 2. 센서별 값 분포 (Flex 센서)
    ax2 = plt.subplot(3, 4, 2)
    flex_data = []
    for data in data_list:
        flex_data.extend(data[:, :5].flatten())
    ax2.hist(flex_data, bins=50, alpha=0.7, color='blue')
    ax2.set_title('Flex Sensor Values Distribution')
    ax2.set_xlabel('Value')
    ax2.set_ylabel('Frequency')
    
    # 3. 센서별 값 분포 (Orientation 센서)
    ax3 = plt.subplot(3, 4, 3)
    orientation_data = []
    for data in data_list:
        orientation_data.extend(data[:, 5:].flatten())
    ax3.hist(orientation_data, bins=50, alpha=0.7, color='red')
    ax3.set_title('Orientation Sensor Values Distribution')
    ax3.set_xlabel('Value')
    ax3.set_ylabel('Frequency')
    
    # 4. 시퀀스 길이 분포
    ax4 = plt.subplot(3, 4, 4)
    sequence_lengths = [data.shape[0] for data in data_list]
    ax4.hist(sequence_lengths, bins=30, alpha=0.7, color='green')
    ax4.set_title('Sequence Length Distribution')
    ax4.set_xlabel('Length')
    ax4.set_ylabel('Frequency')
    
    # 5. 클래스별 Flex 센서 평균값
    ax5 = plt.subplot(3, 4, 5)
    flex_means = defaultdict(list)
    for data, label in zip(data_list, labels):
        class_name = class_names[label]
        flex_means[class_name].append(np.mean(data[:, :5]))
    
    class_names_sorted = sorted(flex_means.keys())
    flex_avg_means = [np.mean(flex_means[cls]) for cls in class_names_sorted]
    bars = ax5.bar(class_names_sorted, flex_avg_means, alpha=0.7)
    ax5.set_title('Class-wise Average Flex Sensor Values')
    ax5.set_xlabel('Class')
    ax5.set_ylabel('Average Value')
    plt.setp(ax5.get_xticklabels(), rotation=45, ha='right')
    
    # 6. 클래스별 Orientation 센서 평균값
    ax6 = plt.subplot(3, 4, 6)
    orientation_means = defaultdict(list)
    for data, label in zip(data_list, labels):
        class_name = class_names[label]
        orientation_means[class_name].append(np.mean(data[:, 5:]))
    
    orientation_avg_means = [np.mean(orientation_means[cls]) for cls in class_names_sorted]
    bars = ax6.bar(class_names_sorted, orientation_avg_means, alpha=0.7, color='orange')
    ax6.set_title('Class-wise Average Orientation Sensor Values')
    ax6.set_xlabel('Class')
    ax6.set_ylabel('Average Value')
    plt.setp(ax6.get_xticklabels(), rotation=45, ha='right')
    
    # 7. 데이터 품질 문제 요약
    ax7 = plt.subplot(3, 4, 7)
    issue_counts = {k: len(v) for k, v in issues.items()}
    if issue_counts:
        bars = ax7.bar(issue_counts.keys(), issue_counts.values(), alpha=0.7, color='red')
        ax7.set_title('Data Quality Issues')
        ax7.set_xlabel('Issue Type')
        ax7.set_ylabel('Count')
        plt.setp(ax7.get_xticklabels(), rotation=45, ha='right')
    
    # 8. 센서별 표준편차 분포
    ax8 = plt.subplot(3, 4, 8)
    sensor_stds = []
    for data in data_list:
        sensor_stds.extend(np.std(data, axis=0))
    ax8.hist(sensor_stds, bins=50, alpha=0.7, color='purple')
    ax8.set_title('Sensor Standard Deviation Distribution')
    ax8.set_xlabel('Standard Deviation')
    ax8.set_ylabel('Frequency')
    
    # 9. 클래스별 특성 히트맵 (상위 10개 특성)
    ax9 = plt.subplot(3, 4, 9)
    if class_features:
        feature_names = [f'Feature_{i}' for i in range(10)]
        class_names_list = list(class_features.keys())
        feature_matrix = np.array([class_features[cls][:10] for cls in class_names_list])
        
        im = ax9.imshow(feature_matrix, cmap='viridis', aspect='auto')
        ax9.set_title('Class Feature Heatmap (Top 10)')
        ax9.set_xlabel('Feature')
        ax9.set_ylabel('Class')
        ax9.set_xticks(range(10))
        ax9.set_xticklabels(feature_names, rotation=45)
        ax9.set_yticks(range(len(class_names_list)))
        ax9.set_yticklabels(class_names_list)
        plt.colorbar(im, ax=ax9)
    
    # 10. 센서 간 상관관계
    ax10 = plt.subplot(3, 4, 10)
    # 모든 데이터를 합쳐서 상관관계 계산
    all_data_combined = np.vstack(data_list)
    corr_matrix = np.corrcoef(all_data_combined.T)
    
    sensor_names = [f'Flex_{i+1}' for i in range(5)] + [f'Ori_{i+1}' for i in range(3)]
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                xticklabels=sensor_names, yticklabels=sensor_names, ax=ax10)
    ax10.set_title('Sensor Correlation Matrix')
    
    # 11. 클래스별 시퀀스 길이
    ax11 = plt.subplot(3, 4, 11)
    class_lengths = defaultdict(list)
    for data, label in zip(data_list, labels):
        class_name = class_names[label]
        class_lengths[class_name].append(data.shape[0])
    
    class_avg_lengths = [np.mean(class_lengths[cls]) for cls in class_names_sorted]
    bars = ax11.bar(class_names_sorted, class_avg_lengths, alpha=0.7, color='cyan')
    ax11.set_title('Class-wise Average Sequence Length')
    ax11.set_xlabel('Class')
    ax11.set_ylabel('Average Length')
    plt.setp(ax11.get_xticklabels(), rotation=45, ha='right')
    
    # 12. 데이터 품질 점수
    ax12 = plt.subplot(3, 4, 12)
    total_samples = len(data_list)
    quality_scores = {}
    
    # 각 클래스별 품질 점수 계산
    for class_name in class_names:
        class_indices = [i for i, label in enumerate(labels) if class_names[label] == class_name]
        class_issues = 0
        
        for issue_type, issue_list in issues.items():
            class_issues += len([issue for issue in issue_list if issue[0] in class_indices])
        
        quality_score = 1 - (class_issues / len(class_indices)) if class_indices else 0
        quality_scores[class_name] = quality_score
    
    quality_values = [quality_scores[cls] for cls in class_names_sorted]
    bars = ax12.bar(class_names_sorted, quality_values, alpha=0.7, color='gold')
    ax12.set_title('Class-wise Data Quality Score')
    ax12.set_xlabel('Class')
    ax12.set_ylabel('Quality Score')
    ax12.set_ylim(0, 1)
    plt.setp(ax12.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig('detailed_data_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print('✅ 시각화 완료: detailed_data_analysis.png')

def print_detailed_report(data_list, labels, class_names, issues, class_features):
    """상세 리포트 출력"""
    print('\n' + '='*80)
    print('📋 상세 데이터 품질 분석 리포트')
    print('='*80)
    
    print(f'\n📊 기본 정보:')
    print(f'  - 총 샘플 수: {len(data_list)}')
    print(f'  - 클래스 수: {len(class_names)}')
    print(f'  - 센서 수: 8 (Flex: 5, Orientation: 3)')
    
    # 클래스별 샘플 수
    class_counts = Counter([class_names[label] for label in labels])
    print(f'\n📈 클래스별 샘플 분포:')
    for class_name in sorted(class_counts.keys()):
        print(f'  - {class_name}: {class_counts[class_name]}개')
    
    # 데이터 품질 문제 요약
    print(f'\n⚠️  데이터 품질 문제:')
    total_issues = sum(len(issue_list) for issue_list in issues.values())
    print(f'  - 총 문제 수: {total_issues}')
    
    for issue_type, issue_list in issues.items():
        if issue_list:
            print(f'  - {issue_type}: {len(issue_list)}개')
            if len(issue_list) <= 5:  # 상위 5개만 표시
                for issue in issue_list[:5]:
                    print(f'    * {issue}')
    
    # 센서 통계
    print(f'\n📊 센서 통계:')
    all_data_combined = np.vstack(data_list)
    
    for i in range(8):
        sensor_name = f'Flex_{i+1}' if i < 5 else f'Orientation_{i-4}'
        sensor_data = all_data_combined[:, i]
        print(f'  - {sensor_name}:')
        print(f'    * 평균: {np.mean(sensor_data):.2f}')
        print(f'    * 표준편차: {np.std(sensor_data):.2f}')
        print(f'    * 최소값: {np.min(sensor_data):.2f}')
        print(f'    * 최대값: {np.max(sensor_data):.2f}')
    
    # 시퀀스 길이 통계
    sequence_lengths = [data.shape[0] for data in data_list]
    print(f'\n📏 시퀀스 길이 통계:')
    print(f'  - 평균 길이: {np.mean(sequence_lengths):.2f}')
    print(f'  - 표준편차: {np.std(sequence_lengths):.2f}')
    print(f'  - 최소 길이: {np.min(sequence_lengths)}')
    print(f'  - 최대 길이: {np.max(sequence_lengths)}')
    
    # 클래스 분리 가능성 분석
    print(f'\n🎯 클래스 분리 가능성:')
    if class_features:
        print(f'  - 특성 수: {len(list(class_features.values())[0])}')
        print(f'  - 클래스별 특성 차이 분석 완료')
    
    print('\n' + '='*80)

def main():
    """메인 함수"""
    # 데이터 경로
    data_dir = '../SignGlove/external/SignGlove_HW/datasets/unified'
    
    # 1. 모든 데이터 로드
    all_data, all_labels, class_names, file_paths = load_all_data_detailed(data_dir)
    
    # 2. 센서 통계 분석
    class_stats = analyze_sensor_statistics(all_data, class_names, all_labels)
    
    # 3. 데이터 품질 문제 체크
    issues = check_data_quality_issues(all_data, file_paths)
    
    # 4. 클래스 분리 가능성 분석
    class_features = analyze_class_separability(all_data, all_labels, class_names)
    
    # 5. 상세 리포트 출력
    print_detailed_report(all_data, all_labels, class_names, issues, class_features)
    
    # 6. 시각화
    visualize_data_analysis(all_data, all_labels, class_names, class_stats, issues, class_features)
    
    print('\n🎉 상세 데이터 품질 분석 완료!')

if __name__ == "__main__":
    main()
