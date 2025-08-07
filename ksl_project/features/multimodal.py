# 멀티모달 입력 처리 클래스
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional, Union
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from preprocessing.filters import ButterworthFilter
from preprocessing.normalization import SensorNormalization

class MultiModalInput:
    def __init__(self, apply_filtering: bool = True, apply_normalization: bool = True):
        """
        멀티모달 센서 입력 처리 클래스
        
        Args:
            apply_filtering: Butterworth 필터 적용 여부
            apply_normalization: 센서 정규화 적용 여부
        """
        self.apply_filtering = apply_filtering
        self.apply_normalization = apply_normalization
        
        # 전처리 모듈 초기화
        if apply_filtering:
            self.filter = ButterworthFilter(cutoff_freq=5.0, sampling_rate=100.0)
        
        if apply_normalization:
            self.normalizer = SensorNormalization(method='minmax')
        
        # 센서별 특성 저장
        self.flex_features = None      # Shape: (batch, time, 5)
        self.imu_features = None       # Shape: (batch, time, 6) [accel_xyz + gyro_xyz]
        self.orientation_features = None # Shape: (batch, time, 3) [roll, pitch, yaw]
        
        # 융합된 특징
        self.fused_features = None
        
    def process_raw_data(self, raw_data: np.ndarray) -> Dict[str, np.ndarray]:
        """
        원본 센서 데이터를 센서별로 분리 및 전처리
        
        Args:
            raw_data: 원본 데이터 (time, 8) [flex1-5, pitch, roll, yaw]
            
        Returns:
            센서별 분리된 데이터
        """
        # 센서별 데이터 분리
        flex_data = raw_data[:, :5]           # flex1~5
        orientation_data = raw_data[:, 5:]    # pitch, roll, yaw
        
        # IMU 데이터는 현재 없으므로 가상 생성 (실제 프로젝트에서는 실제 센서 데이터 사용)
        # 실제로는 가속도(3) + 자이로(3) = 6차원
        imu_data = np.zeros((raw_data.shape[0], 6))  # placeholder
        
        processed_data = {}
        
        # 1. 필터링 적용
        if self.apply_filtering:
            processed_data['flex'] = self.filter.apply_filter(flex_data, axis=0)
            processed_data['imu'] = self.filter.apply_filter(imu_data, axis=0)
            processed_data['orientation'] = self.filter.apply_filter(orientation_data, axis=0)
        else:
            processed_data['flex'] = flex_data
            processed_data['imu'] = imu_data
            processed_data['orientation'] = orientation_data
        
        # 2. 정규화 적용
        if self.apply_normalization:
            norm_flex, norm_imu, norm_ori = self.normalizer.normalize_ksl_sensors(
                processed_data['flex'], 
                processed_data['imu'], 
                processed_data['orientation']
            )
            processed_data['flex'] = norm_flex
            processed_data['imu'] = norm_imu
            processed_data['orientation'] = norm_ori
        
        return processed_data
    
    def create_multimodal_features(self, processed_data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        전처리된 센서 데이터로부터 멀티모달 특징 생성
        
        Args:
            processed_data: 전처리된 센서별 데이터
            
        Returns:
            다양한 형태의 특징 벡터
        """
        features = {}
        
        # 1. 연결형 특징 (Concatenated Features) - Classical ML용
        concat_features = np.concatenate([
            processed_data['flex'],
            processed_data['imu'],
            processed_data['orientation']
        ], axis=1)  # Shape: (time, 14)
        
        features['concatenated'] = concat_features
        
        # 2. 시계열 보존형 (Sequence Features) - Deep Learning용
        # 각 센서별로 별도 보관하여 모델에서 각각 처리 가능
        features['flex_sequence'] = processed_data['flex']
        features['imu_sequence'] = processed_data['imu']
        features['orientation_sequence'] = processed_data['orientation']
        
        # 3. 통계적 특징 (Statistical Features)
        features['statistical'] = self._extract_statistical_features(processed_data)
        
        # 4. 파생 특징 (Derived Features)
        features['derived'] = self._extract_derived_features(processed_data)
        
        return features
    
    def _extract_statistical_features(self, data: Dict[str, np.ndarray]) -> np.ndarray:
        """각 센서별 통계적 특징 추출"""
        stat_features = []
        
        for sensor_type, sensor_data in data.items():
            # 각 센서 채널별로 통계 특징 계산
            for channel in range(sensor_data.shape[1]):
                channel_data = sensor_data[:, channel]
                
                # 기본 통계량
                stats = [
                    np.mean(channel_data),      # 평균
                    np.std(channel_data),       # 표준편차
                    np.max(channel_data),       # 최대값
                    np.min(channel_data),       # 최소값
                    np.median(channel_data),    # 중앙값
                    np.var(channel_data),       # 분산
                    np.ptp(channel_data)        # 범위 (max - min)
                ]
                
                stat_features.extend(stats)
        
        return np.array(stat_features)
    
    def _extract_derived_features(self, data: Dict[str, np.ndarray]) -> np.ndarray:
        """센서간 상관관계 및 파생 특징 추출"""
        derived_features = []
        
        # 1. Flex 센서간 상관관계
        flex_data = data['flex']
        flex_corr = np.corrcoef(flex_data.T)
        
        # 상관계수 매트릭스의 상삼각 부분만 사용 (중복 제거)
        upper_tri_indices = np.triu_indices_from(flex_corr, k=1)
        flex_correlations = flex_corr[upper_tri_indices]
        derived_features.extend(flex_correlations)
        
        # 2. 움직임 강도 (Movement Intensity)
        orientation_data = data['orientation']
        movement_intensity = np.sqrt(np.sum(np.diff(orientation_data, axis=0)**2, axis=1))
        derived_features.extend([
            np.mean(movement_intensity),
            np.std(movement_intensity),
            np.max(movement_intensity)
        ])
        
        # 3. 제스처 복잡도 (Gesture Complexity)
        # 각 센서의 변화율 기반
        complexity_scores = []
        for sensor_data in data.values():
            gradient = np.gradient(sensor_data, axis=0)
            complexity = np.mean(np.linalg.norm(gradient, axis=1))
            complexity_scores.append(complexity)
        
        derived_features.extend(complexity_scores)
        
        return np.array(derived_features)
    
    def prepare_for_classical_ml(self, window_data: np.ndarray) -> np.ndarray:
        """
        Classical ML 모델을 위한 특징 벡터 준비
        
        Args:
            window_data: 윈도우된 센서 데이터 (time, features)
            
        Returns:
            평탄화된 특징 벡터
        """
        # 원본 데이터 전처리
        processed_data = self.process_raw_data(window_data)
        
        # 멀티모달 특징 생성
        features = self.create_multimodal_features(processed_data)
        
        # 통계적 특징 + 파생 특징 결합
        ml_features = np.concatenate([
            features['statistical'],
            features['derived']
        ])
        
        return ml_features
    
    def prepare_for_deep_learning(self, window_data: np.ndarray) -> Dict[str, torch.Tensor]:
        """
        Deep Learning 모델을 위한 텐서 준비
        
        Args:
            window_data: 윈도우된 센서 데이터 (time, features)
            
        Returns:
            센서별 분리된 텐서 딕셔너리
        """
        # 원본 데이터 전처리
        processed_data = self.process_raw_data(window_data)
        
        # 멀티모달 특징 생성
        features = self.create_multimodal_features(processed_data)
        
        # PyTorch 텐서로 변환
        tensor_features = {
            'flex': torch.FloatTensor(features['flex_sequence']).unsqueeze(0),  # (1, time, 5)
            'imu': torch.FloatTensor(features['imu_sequence']).unsqueeze(0),    # (1, time, 6)
            'orientation': torch.FloatTensor(features['orientation_sequence']).unsqueeze(0),  # (1, time, 3)
            'concatenated': torch.FloatTensor(features['concatenated']).unsqueeze(0)  # (1, time, 14)
        }
        
        return tensor_features
    
    def get_feature_dimensions(self) -> Dict[str, int]:
        """각 특징 타입별 차원 정보 반환"""
        return {
            'flex_channels': 5,
            'imu_channels': 6,
            'orientation_channels': 3,
            'concatenated_channels': 14,
            'statistical_features': 98,  # (5+6+3) * 7 통계량
            'derived_features': 16       # 상관관계 + 움직임 + 복잡도
        }
    
    def get_processing_info(self) -> Dict:
        """전처리 설정 정보 반환"""
        return {
            'filtering_enabled': self.apply_filtering,
            'normalization_enabled': self.apply_normalization,
            'filter_info': self.filter.get_filter_info() if self.apply_filtering else None,
            'normalizer_info': self.normalizer.get_normalization_info() if self.apply_normalization else None,
            'feature_dimensions': self.get_feature_dimensions()
        }

# 사용 예시
if __name__ == "__main__":
    # 테스트용 데이터 생성
    np.random.seed(42)
    test_window = np.random.randn(20, 8)  # 20 타임스텝, 8개 센서
    
    # 멀티모달 처리기 초기화
    multimodal = MultiModalInput(apply_filtering=True, apply_normalization=True)
    
    print("=== 멀티모달 특징 처리 테스트 ===")
    
    # Classical ML용 특징
    ml_features = multimodal.prepare_for_classical_ml(test_window)
    print(f"Classical ML 특징 차원: {ml_features.shape}")
    
    # Deep Learning용 특징
    dl_features = multimodal.prepare_for_deep_learning(test_window)
    print(f"Deep Learning 특징:")
    for name, tensor in dl_features.items():
        print(f"  {name}: {tensor.shape}")
    
    # 처리 정보 출력
    print(f"\n처리 정보: {multimodal.get_processing_info()}")
    print("멀티모달 처리 테스트 완료!")
