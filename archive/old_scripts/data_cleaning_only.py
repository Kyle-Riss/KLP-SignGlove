import numpy as np
import h5py
import os
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

print('🧹 1순위 데이터 정제 시작')

# 센서 이름
sensor_names = ['Flex1', 'Flex2', 'Flex3', 'Flex4', 'Flex5', 'Pitch', 'Roll', 'Yaw']

class DataCleaner:
    """1순위 데이터 정제 클래스"""
    
    def __init__(self):
        self.cleaned_data = {}
        self.removed_samples = defaultdict(int)
        self.normalized_samples = defaultdict(int)
        self.original_stats = {}
        self.cleaned_stats = {}
        
    def clean_data(self, data_dir):
        """1순위 정제: 범위 오류 제거, 극단적 변동성 정규화"""
        print('🔧 1순위 데이터 정제 시작...')
        
        for class_name in sorted(os.listdir(data_dir)):
            class_dir = os.path.join(data_dir, class_name)
            if os.path.isdir(class_dir):
                print(f'  📊 {class_name} 클래스 정제 중...')
                class_data = []
                class_paths = []
                original_class_data = []
                
                for session in ['1', '2', '3', '4', '5']:
                    session_dir = os.path.join(class_dir, session)
                    if os.path.exists(session_dir):
                        for file_name in os.listdir(session_dir):
                            if file_name.endswith('.h5'):
                                file_path = os.path.join(session_dir, file_name)
                                try:
                                    with h5py.File(file_path, 'r') as f:
                                        sensor_data = f['sensor_data'][:]  # (300, 8)
                                        
                                        # 원본 데이터 저장
                                        original_class_data.append(sensor_data[-20:])
                                        
                                        # 데이터 정제
                                        cleaned_data = self._clean_single_sample(sensor_data, class_name)
                                        if cleaned_data is not None:
                                            # 마지막 20 프레임 사용
                                            data = cleaned_data[-20:]
                                            class_data.append(data)
                                            class_paths.append(file_path)
                                            
                                except Exception as e:
                                    print(f"    Error loading {file_path}: {e}")
                
                if class_data:
                    self.cleaned_data[class_name] = {
                        'data': np.array(class_data),
                        'paths': class_paths
                    }
                    
                    # 통계 저장
                    self.original_stats[class_name] = np.array(original_class_data)
                    self.cleaned_stats[class_name] = np.array(class_data)
        
        print(f'✅ 1순위 정제 완료!')
        self._print_cleaning_summary()
        return self.cleaned_data
    
    def _clean_single_sample(self, sensor_data, class_name):
        """단일 샘플 정제"""
        # 1. 범위 오류 검사 및 제거
        for i, sensor_name in enumerate(sensor_names):
            sensor_values = sensor_data[:, i]
            
            # Flex 센서 범위 검사 (0-1000)
            if i < 5:  # Flex 센서
                if np.any(sensor_values < 0) or np.any(sensor_values > 1000):
                    self.removed_samples[class_name] += 1
                    return None  # 샘플 제거
            
            # Orientation 센서 범위 검사
            else:  # Pitch, Roll, Yaw
                if np.any(sensor_values < -180) or np.any(sensor_values > 180):
                    self.removed_samples[class_name] += 1
                    return None  # 샘플 제거
        
        # 2. 극단적 변동성 정규화
        cleaned_data = sensor_data.copy()
        for i, sensor_name in enumerate(sensor_names):
            sensor_values = sensor_data[:, i]
            std_val = np.std(sensor_values)
            
            # Flex 센서 극단적 변동성 정규화 (std > 200)
            if i < 5 and std_val > 200:
                # Z-score 정규화 후 스케일링
                mean_val = np.mean(sensor_values)
                normalized = (sensor_values - mean_val) / std_val
                # 적절한 범위로 스케일링 (표준편차 100으로)
                cleaned_data[:, i] = mean_val + normalized * 100
                self.normalized_samples[class_name] += 1
        
        return cleaned_data
    
    def _print_cleaning_summary(self):
        """정제 요약 출력"""
        print('\n📊 정제 요약:')
        print('=' * 80)
        
        total_removed = sum(self.removed_samples.values())
        total_normalized = sum(self.normalized_samples.values())
        
        print(f'🗑️ 제거된 샘플: {total_removed}개')
        print(f'🔧 정규화된 샘플: {total_normalized}개')
        
        if total_removed > 0:
            print(f'\n📋 클래스별 제거된 샘플:')
            for class_name, count in sorted(self.removed_samples.items()):
                if count > 0:
                    print(f'  {class_name}: {count}개')
        
        if total_normalized > 0:
            print(f'\n🔧 클래스별 정규화된 샘플:')
            for class_name, count in sorted(self.normalized_samples.items()):
                if count > 0:
                    print(f'  {class_name}: {count}개')
    
    def analyze_cleaning_effects(self):
        """정제 효과 분석"""
        print('\n📈 정제 효과 분석:')
        print('=' * 80)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('1순위 데이터 정제 효과 분석', fontsize=16, fontweight='bold')
        
        # 1. 클래스별 샘플 수 변화
        original_counts = []
        cleaned_counts = []
        class_names = []
        
        for class_name in sorted(self.original_stats.keys()):
            original_counts.append(len(self.original_stats[class_name]))
            cleaned_counts.append(len(self.cleaned_stats[class_name]))
            class_names.append(class_name)
        
        x = np.arange(len(class_names))
        width = 0.35
        
        axes[0, 0].bar(x - width/2, original_counts, width, label='정제 전', color='lightcoral', alpha=0.7)
        axes[0, 0].bar(x + width/2, cleaned_counts, width, label='정제 후', color='lightblue', alpha=0.7)
        axes[0, 0].set_xlabel('클래스')
        axes[0, 0].set_ylabel('샘플 수')
        axes[0, 0].set_title('클래스별 샘플 수 변화')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(class_names, rotation=45)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 센서별 표준편차 분포 (정제 전후)
        all_original_stds = []
        all_cleaned_stds = []
        
        for class_name in class_names:
            if class_name in self.original_stats and class_name in self.cleaned_stats:
                original_data = self.original_stats[class_name]
                cleaned_data = self.cleaned_stats[class_name]
                
                # Flex 센서만
                for i in range(5):  # Flex1-5
                    original_stds = np.std(original_data[:, :, i], axis=1)
                    cleaned_stds = np.std(cleaned_data[:, :, i], axis=1)
                    
                    all_original_stds.extend(original_stds)
                    all_cleaned_stds.extend(cleaned_stds)
        
        axes[0, 1].hist(all_original_stds, bins=30, alpha=0.7, label='정제 전', color='red')
        axes[0, 1].hist(all_cleaned_stds, bins=30, alpha=0.7, label='정제 후', color='blue')
        axes[0, 1].set_xlabel('Flex 센서 표준편차')
        axes[0, 1].set_ylabel('빈도')
        axes[0, 1].set_title('Flex 센서 변동성 분포')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Orientation 센서 표준편차 분포
        all_original_orient_stds = []
        all_cleaned_orient_stds = []
        
        for class_name in class_names:
            if class_name in self.original_stats and class_name in self.cleaned_stats:
                original_data = self.original_stats[class_name]
                cleaned_data = self.cleaned_stats[class_name]
                
                # Orientation 센서만
                for i in range(5, 8):  # Pitch, Roll, Yaw
                    original_stds = np.std(original_data[:, :, i], axis=1)
                    cleaned_stds = np.std(cleaned_data[:, :, i], axis=1)
                    
                    all_original_orient_stds.extend(original_stds)
                    all_cleaned_orient_stds.extend(cleaned_stds)
        
        axes[0, 2].hist(all_original_orient_stds, bins=30, alpha=0.7, label='정제 전', color='red')
        axes[0, 2].hist(all_cleaned_orient_stds, bins=30, alpha=0.7, label='정제 후', color='blue')
        axes[0, 2].set_xlabel('Orientation 센서 표준편차')
        axes[0, 2].set_ylabel('빈도')
        axes[0, 2].set_title('Orientation 센서 변동성 분포')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. 클래스별 평균 변동성 변화
        class_avg_std_original = []
        class_avg_std_cleaned = []
        
        for class_name in class_names:
            if class_name in self.original_stats and class_name in self.cleaned_stats:
                original_data = self.original_stats[class_name]
                cleaned_data = self.cleaned_stats[class_name]
                
                # 전체 센서의 평균 표준편차
                original_std = np.mean([np.std(original_data[:, :, i], axis=1) for i in range(8)])
                cleaned_std = np.mean([np.std(cleaned_data[:, :, i], axis=1) for i in range(8)])
                
                class_avg_std_original.append(np.mean(original_std))
                class_avg_std_cleaned.append(np.mean(cleaned_std))
        
        axes[1, 0].bar(x - width/2, class_avg_std_original, width, label='정제 전', color='lightcoral', alpha=0.7)
        axes[1, 0].bar(x + width/2, class_avg_std_cleaned, width, label='정제 후', color='lightblue', alpha=0.7)
        axes[1, 0].set_xlabel('클래스')
        axes[1, 0].set_ylabel('평균 변동성')
        axes[1, 0].set_title('클래스별 평균 변동성 변화')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(class_names, rotation=45)
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. 센서별 범위 분포
        all_original_ranges = []
        all_cleaned_ranges = []
        
        for class_name in class_names:
            if class_name in self.original_stats and class_name in self.cleaned_stats:
                original_data = self.original_stats[class_name]
                cleaned_data = self.cleaned_stats[class_name]
                
                for i in range(8):
                    original_range = np.max(original_data[:, :, i]) - np.min(original_data[:, :, i])
                    cleaned_range = np.max(cleaned_data[:, :, i]) - np.min(cleaned_data[:, :, i])
                    
                    all_original_ranges.append(original_range)
                    all_cleaned_ranges.append(cleaned_range)
        
        axes[1, 1].hist(all_original_ranges, bins=30, alpha=0.7, label='정제 전', color='red')
        axes[1, 1].hist(all_cleaned_ranges, bins=30, alpha=0.7, label='정제 후', color='blue')
        axes[1, 1].set_xlabel('센서 값 범위')
        axes[1, 1].set_ylabel('빈도')
        axes[1, 1].set_title('센서 값 범위 분포')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. 정제 효과 요약
        total_original = sum(original_counts)
        total_cleaned = sum(cleaned_counts)
        retention_rate = (total_cleaned / total_original) * 100
        
        summary_text = f"""
정제 효과 요약:
• 총 샘플: {total_original} → {total_cleaned} ({retention_rate:.1f}% 유지)
• 제거된 샘플: {sum(self.removed_samples.values())}개
• 정규화된 샘플: {sum(self.normalized_samples.values())}개
• 클래스 수: {len(class_names)}개
        """
        
        axes[1, 2].text(0.1, 0.5, summary_text, transform=axes[1, 2].transAxes, 
                       fontsize=12, verticalalignment='center',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
        axes[1, 2].set_title('정제 효과 요약')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig('data_cleaning_effects.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 상세 통계 출력
        print(f'\n📊 정제 후 데이터 통계:')
        print('=' * 80)
        
        total_samples = 0
        for class_name in sorted(self.cleaned_data.keys()):
            samples = len(self.cleaned_data[class_name]['data'])
            total_samples += samples
            removed = self.removed_samples.get(class_name, 0)
            normalized = self.normalized_samples.get(class_name, 0)
            
            print(f'  {class_name}: {samples}개 샘플', end='')
            if removed > 0:
                print(f' (제거: {removed}개)', end='')
            if normalized > 0:
                print(f' (정규화: {normalized}개)', end='')
            print()
        
        print(f'\n  총 샘플 수: {total_samples}')
        print(f'  유지율: {(total_samples / (total_samples + sum(self.removed_samples.values()))) * 100:.1f}%')

# 메인 실행
if __name__ == "__main__":
    # 데이터 정제
    data_dir = '../SignGlove/external/SignGlove_HW/datasets/unified'
    cleaner = DataCleaner()
    cleaned_data = cleaner.clean_data(data_dir)
    
    # 정제 효과 분석
    cleaner.analyze_cleaning_effects()
    
    print(f'\n🎉 1순위 데이터 정제 완료!')
    print(f'💾 저장된 파일:')
    print(f'  - data_cleaning_effects.png: 정제 효과 분석')
    print(f'\n📝 다음 단계:')
    print(f'  1. 정제된 데이터로 모델 훈련')
    print(f'  2. 필요시 2순위 정제 (증강) 적용')
    print(f'  3. 성능 비교 분석')
