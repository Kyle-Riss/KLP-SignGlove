#!/usr/bin/env python3
"""
개선된 전처리 파이프라인
데이터 품질 분석 결과를 바탕으로 한 종합적인 전처리 시스템
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from sklearn.preprocessing import StandardScaler, RobustScaler
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from training.label_mapping import KSLLabelMapper

class ImprovedPreprocessingPipeline:
    """개선된 전처리 파이프라인"""
    
    def __init__(self, config=None):
        self.config = config or self._get_default_config()
        self.label_mapper = KSLLabelMapper()
        
        # 클래스별 일관성 점수 (분석 결과 기반)
        self.consistency_scores = {
            'ㄱ': 0.179, 'ㄴ': 0.660, 'ㄷ': 0.661, 'ㄹ': 0.855, 'ㅁ': 0.558,
            'ㅂ': 0.777, 'ㅅ': 0.805, 'ㅇ': 0.824, 'ㅈ': 0.796, 'ㅊ': 0.772,
            'ㅋ': 0.730, 'ㅌ': 0.863, 'ㅍ': 0.734, 'ㅎ': 0.743, 'ㅏ': 0.680,
            'ㅑ': 0.760, 'ㅓ': 1.503, 'ㅕ': 0.652, 'ㅗ': 0.682, 'ㅛ': 0.787,
            'ㅜ': 0.685, 'ㅠ': 0.616, 'ㅡ': 0.635, 'ㅣ': 0.705
        }
        
        # 클래스 그룹 분류
        self.low_consistency_classes = ['ㄱ', 'ㄴ', 'ㄷ', 'ㅁ', 'ㅕ']
        self.high_consistency_classes = ['ㄹ', 'ㅌ', 'ㅅ', 'ㅇ']
        self.noisy_yaw_classes = ['ㄱ', 'ㅁ', 'ㅍ', 'ㅓ']
        
        print(f"🔧 개선된 전처리 파이프라인 초기화 완료")
        print(f"📊 설정: {self.config}")
    
    def _get_default_config(self):
        """기본 설정"""
        return {
            'target_length': 300,
            'complementary_filter_alpha': 0.96,
            'noise_reduction_window': 5,
            'outlier_threshold': 3.0,
            'robust_scaling': True,
            'class_specific_weights': True,
            'data_augmentation': True,
            'flex_normalization': True,
            'imu_enhancement': True
        }
    
    def apply_complementary_filter(self, data, alpha=0.96):
        """상보 필터 적용 (IMU 센서 안정화)"""
        filtered_data = data.copy()
        
        # IMU 센서에만 적용
        imu_sensors = ['pitch', 'roll', 'yaw']
        
        for sensor in imu_sensors:
            if sensor in filtered_data.columns:
                values = filtered_data[sensor].values
                filtered_values = np.zeros_like(values)
                filtered_values[0] = values[0]
                
                for i in range(1, len(values)):
                    filtered_values[i] = alpha * filtered_values[i-1] + (1 - alpha) * values[i]
                
                filtered_data[sensor] = filtered_values
        
        return filtered_data
    
    def reduce_noise(self, data, window_size=5):
        """노이즈 감소 (이동 평균 필터)"""
        denoised_data = data.copy()
        
        for column in denoised_data.columns:
            if column in ['pitch', 'roll', 'yaw', 'flex1', 'flex2', 'flex3', 'flex4', 'flex5']:
                values = denoised_data[column].values
                denoised_values = np.convolve(values, np.ones(window_size)/window_size, mode='same')
                denoised_data[column] = denoised_values
        
        return denoised_data
    
    def remove_outliers(self, data, threshold=3.0):
        """이상치 제거"""
        cleaned_data = data.copy()
        
        for column in cleaned_data.columns:
            if column in ['pitch', 'roll', 'yaw', 'flex1', 'flex2', 'flex3', 'flex4', 'flex5']:
                values = cleaned_data[column].values
                mean_val = np.mean(values)
                std_val = np.std(values)
                
                # 이상치 마스크
                outlier_mask = np.abs(values - mean_val) > threshold * std_val
                
                # 이상치를 중간값으로 대체
                median_val = np.median(values)
                cleaned_data.loc[outlier_mask, column] = median_val
        
        return cleaned_data
    
    def normalize_flex_sensors(self, data):
        """Flex 센서 정규화 및 강화"""
        normalized_data = data.copy()
        
        flex_sensors = ['flex1', 'flex2', 'flex3', 'flex4', 'flex5']
        
        for sensor in flex_sensors:
            if sensor in normalized_data.columns:
                values = normalized_data[sensor].values
                
                # 0-1 정규화
                min_val = np.min(values)
                max_val = np.max(values)
                if max_val > min_val:
                    normalized_values = (values - min_val) / (max_val - min_val)
                else:
                    normalized_values = values
                
                # 스케일링 강화 (1.2배)
                normalized_values = normalized_values * 1.2
                
                normalized_data[sensor] = normalized_values
        
        return normalized_data
    
    def apply_class_specific_weights(self, data, class_name):
        """클래스별 센서 가중치 적용"""
        weighted_data = data.copy()
        
        # 기본 가중치
        weights = {
            'pitch': 1.0, 'roll': 1.0, 'yaw': 1.0,
            'flex1': 1.0, 'flex2': 1.0, 'flex3': 1.0, 'flex4': 1.0, 'flex5': 1.0
        }
        
        # 클래스별 특화 가중치
        if class_name in self.low_consistency_classes:
            # 낮은 일관성 클래스: 더 강한 정규화
            weights.update({
                'pitch': 1.3, 'roll': 1.3, 'yaw': 1.5,
                'flex1': 1.2, 'flex2': 1.2, 'flex3': 1.2, 'flex4': 1.2, 'flex5': 1.2
            })
        
        elif class_name in self.high_consistency_classes:
            # 높은 일관성 클래스: 표준 가중치
            weights.update({
                'pitch': 1.1, 'roll': 1.1, 'yaw': 1.1,
                'flex1': 1.0, 'flex2': 1.0, 'flex3': 1.0, 'flex4': 1.0, 'flex5': 1.0
            })
        
        # yaw 노이즈가 높은 클래스 특별 처리
        if class_name in self.noisy_yaw_classes:
            weights['yaw'] = 0.8  # yaw 센서 가중치 감소
        
        # 가중치 적용
        for sensor, weight in weights.items():
            if sensor in weighted_data.columns:
                weighted_data[sensor] = weighted_data[sensor] * weight
        
        return weighted_data
    
    def normalize_data_length(self, data, target_length=300):
        """데이터 길이 정규화"""
        current_length = len(data)
        
        if current_length == target_length:
            return data
        
        elif current_length > target_length:
            # 자르기 (중앙 부분 유지)
            start_idx = (current_length - target_length) // 2
            return data.iloc[start_idx:start_idx + target_length].reset_index(drop=True)
        
        else:
            # 패딩 (마지막 값으로 반복)
            padding_length = target_length - current_length
            padding_data = pd.DataFrame([data.iloc[-1].values] * padding_length, columns=data.columns)
            return pd.concat([data, padding_data], ignore_index=True)
    
    def apply_data_augmentation(self, data, class_name, probability=0.3):
        """데이터 증강"""
        if not self.config['data_augmentation']:
            return data
        
        augmented_data = data.copy()
        
        # 클래스별 증강 강도 조절
        if class_name in self.low_consistency_classes:
            aug_prob = probability * 1.5  # 더 강한 증강
        elif class_name in self.high_consistency_classes:
            aug_prob = probability * 0.5  # 약한 증강
        else:
            aug_prob = probability
        
        # 가우시안 노이즈 추가
        if np.random.random() < aug_prob:
            noise_std = 0.01
            for column in augmented_data.columns:
                if column in ['pitch', 'roll', 'yaw', 'flex1', 'flex2', 'flex3', 'flex4', 'flex5']:
                    noise = np.random.normal(0, noise_std, len(augmented_data))
                    augmented_data[column] += noise
        
        # 시간 이동
        if np.random.random() < aug_prob * 0.5:
            shift = np.random.randint(-5, 6)
            if shift != 0:
                augmented_data = augmented_data.shift(shift).fillna(method='bfill').fillna(method='ffill')
        
        return augmented_data
    
    def preprocess_single_file(self, file_path, class_name):
        """단일 파일 전처리"""
        try:
            # 데이터 로드
            data = pd.read_csv(file_path)
            
            if data.empty:
                return None
            
            # 1. 기본 정리
            data = self.remove_outliers(data, self.config['outlier_threshold'])
            
            # 2. 상보 필터 적용
            data = self.apply_complementary_filter(data, self.config['complementary_filter_alpha'])
            
            # 3. 노이즈 감소
            data = self.reduce_noise(data, self.config['noise_reduction_window'])
            
            # 4. Flex 센서 정규화
            if self.config['flex_normalization']:
                data = self.normalize_flex_sensors(data)
            
            # 5. 클래스별 가중치 적용
            if self.config['class_specific_weights']:
                data = self.apply_class_specific_weights(data, class_name)
            
            # 6. 데이터 길이 정규화
            data = self.normalize_data_length(data, self.config['target_length'])
            
            # 7. 데이터 증강
            data = self.apply_data_augmentation(data, class_name)
            
            return data
            
        except Exception as e:
            print(f"⚠️ 파일 전처리 실패: {file_path}, 오류: {e}")
            return None
    
    def preprocess_dataset(self, data_dir, output_dir=None):
        """전체 데이터셋 전처리"""
        print(f"🔄 전체 데이터셋 전처리 시작")
        
        if output_dir is None:
            output_dir = os.path.join(data_dir, 'preprocessed')
        
        os.makedirs(output_dir, exist_ok=True)
        
        processed_files = 0
        total_files = 0
        
        for class_name in self.label_mapper.get_all_classes():
            print(f"  📁 {class_name} 클래스 처리 중...")
            
            # 클래스별 출력 디렉토리 생성
            class_output_dir = os.path.join(output_dir, class_name)
            os.makedirs(class_output_dir, exist_ok=True)
            
            # 파일 패턴 매칭
            pattern = os.path.join(data_dir, f"**/{class_name}/**/episode_*.csv")
            files = glob.glob(pattern, recursive=True)
            
            for file_path in files:
                total_files += 1
                
                # 파일명 추출
                file_name = os.path.basename(file_path)
                
                # 전처리 적용
                processed_data = self.preprocess_single_file(file_path, class_name)
                
                if processed_data is not None:
                    # 저장
                    output_path = os.path.join(class_output_dir, f"preprocessed_{file_name}")
                    processed_data.to_csv(output_path, index=False)
                    processed_files += 1
        
        print(f"✅ 전처리 완료: {processed_files}/{total_files} 파일 처리됨")
        print(f"📁 출력 디렉토리: {output_dir}")
        
        return output_dir

class ImprovedDataset(Dataset):
    """개선된 전처리 파이프라인을 사용하는 데이터셋"""
    
    def __init__(self, data_dir, preprocessor=None, transform=None):
        self.data_dir = data_dir
        self.preprocessor = preprocessor or ImprovedPreprocessingPipeline()
        self.transform = transform
        self.label_mapper = KSLLabelMapper()
        
        self.data_files = []
        self.labels = []
        
        self._load_data()
        
        print(f"📊 데이터셋 로드 완료: {len(self.data_files)} 파일")
    
    def _load_data(self):
        """데이터 파일 목록 로드"""
        for class_name in self.label_mapper.get_all_classes():
            class_id = self.label_mapper.get_label_id(class_name)
            
            pattern = os.path.join(self.data_dir, f"**/{class_name}/**/episode_*.csv")
            files = glob.glob(pattern, recursive=True)
            
            for file_path in files:
                self.data_files.append((file_path, class_name))
                self.labels.append(class_id)
    
    def __len__(self):
        return len(self.data_files)
    
    def __getitem__(self, idx):
        file_path, class_name = self.data_files[idx]
        label = self.labels[idx]
        
        # 전처리 적용
        processed_data = self.preprocessor.preprocess_single_file(file_path, class_name)
        
        if processed_data is None:
            # 오류 시 더미 데이터 반환
            dummy_data = np.zeros((self.preprocessor.config['target_length'], 8))
            return torch.FloatTensor(dummy_data), torch.LongTensor([label])
        
        # 텐서 변환
        sensor_data = processed_data[['pitch', 'roll', 'yaw', 'flex1', 'flex2', 'flex3', 'flex4', 'flex5']].values
        sensor_tensor = torch.FloatTensor(sensor_data)
        label_tensor = torch.LongTensor([label])
        
        if self.transform:
            sensor_tensor = self.transform(sensor_tensor)
        
        return sensor_tensor, label_tensor

def create_preprocessing_visualization(preprocessor, sample_file, class_name):
    """전처리 과정 시각화"""
    print(f"📊 전처리 과정 시각화 생성 중...")
    
    # 원본 데이터 로드
    original_data = pd.read_csv(sample_file)
    
    # 각 단계별 전처리 결과
    steps = {
        'Original': original_data,
        'Outlier_Removed': preprocessor.remove_outliers(original_data),
        'Complementary_Filter': preprocessor.apply_complementary_filter(original_data),
        'Noise_Reduced': preprocessor.reduce_noise(original_data),
        'Flex_Normalized': preprocessor.normalize_flex_sensors(original_data),
        'Class_Weighted': preprocessor.apply_class_specific_weights(original_data, class_name),
        'Length_Normalized': preprocessor.normalize_data_length(original_data)
    }
    
    # 시각화
    fig, axes = plt.subplots(4, 2, figsize=(15, 12))
    fig.suptitle(f'전처리 과정 시각화 - {class_name}', fontsize=16)
    
    sensors = ['pitch', 'roll', 'yaw', 'flex1', 'flex2', 'flex3', 'flex4', 'flex5']
    
    for i, (step_name, data) in enumerate(steps.items()):
        row = i // 2
        col = i % 2
        
        if i < len(axes.flat):
            ax = axes[row, col]
            
            # 센서별 플롯
            for sensor in sensors[:3]:  # IMU 센서만 표시
                if sensor in data.columns:
                    ax.plot(data[sensor].values, label=sensor, alpha=0.7)
            
            ax.set_title(step_name)
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('preprocessing_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("    ✅ 시각화 파일 저장 완료")

def main():
    """메인 함수"""
    # 전처리 파이프라인 초기화
    preprocessor = ImprovedPreprocessingPipeline()
    
    # 데이터 디렉토리
    data_dir = 'integrations/SignGlove_HW/github_unified_data'
    
    # 샘플 파일로 시각화 생성
    sample_pattern = os.path.join(data_dir, "**/ㄱ/**/episode_*.csv")
    sample_files = glob.glob(sample_pattern, recursive=True)
    
    if sample_files:
        create_preprocessing_visualization(preprocessor, sample_files[0], 'ㄱ')
    
    # 데이터셋 생성 테스트
    dataset = ImprovedDataset(data_dir, preprocessor)
    
    # 데이터로더 생성
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # 샘플 배치 확인
    for batch_idx, (data, labels) in enumerate(dataloader):
        print(f"배치 {batch_idx + 1}:")
        print(f"  데이터 형태: {data.shape}")
        print(f"  라벨 형태: {labels.shape}")
        print(f"  라벨 값: {labels.flatten().tolist()}")
        break
    
    print(f"\n🎉 전처리 파이프라인 테스트 완료!")

if __name__ == "__main__":
    main()
