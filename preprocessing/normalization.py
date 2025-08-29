# 센서 정규화 구현
import numpy as np
import pandas as pd
from typing import Union, Dict, Tuple, Optional

class SensorNormalization:
    def __init__(self, method: str = 'minmax', sensor_ranges: Optional[Dict] = None):
        """
        센서 정규화 클래스
        
        Args:
            method: 정규화 방법 ('minmax', 'zscore', 'robust')
            sensor_ranges: 센서별 예상 범위 (선택적)
        """
        self.method = method
        self.sensor_ranges = sensor_ranges or {}
        self.fitted_params = {}
        self.is_fitted = False
        
        # 기본 센서 범위 정의 (KSL 시스템 기준)
        self.default_ranges = {
            'flex': (700, 900),      # Flex 센서 일반적 범위
            'accel': (-2, 2),        # 가속도 (g 단위)
            'gyro': (-250, 250),     # 자이로스코프 (deg/s)
            'orientation': (-180, 180)  # 오일러 각도 (degrees)
        }
    
    def fit(self, data: np.ndarray, sensor_types: list = None) -> 'SensorNormalization':
        """
        데이터에서 정규화 파라미터 학습
        
        Args:
            data: 학습 데이터 (N x Features)
            sensor_types: 각 컬럼의 센서 타입 리스트
            
        Returns:
            self (method chaining 지원)
        """
        if sensor_types is None:
            sensor_types = ['unknown'] * data.shape[1]
            
        self.fitted_params = {}
        
        for i, sensor_type in enumerate(sensor_types):
            col_data = data[:, i]
            
            if self.method == 'minmax':
                # Min-Max 정규화 (0-1 범위)
                if sensor_type in self.sensor_ranges:
                    min_val, max_val = self.sensor_ranges[sensor_type]
                else:
                    min_val, max_val = np.min(col_data), np.max(col_data)
                    
                self.fitted_params[i] = {
                    'min': min_val,
                    'max': max_val,
                    'range': max_val - min_val
                }
                
            elif self.method == 'zscore':
                # Z-score 정규화 (평균 0, 표준편차 1)
                self.fitted_params[i] = {
                    'mean': np.mean(col_data),
                    'std': np.std(col_data)
                }
                
            elif self.method == 'robust':
                # Robust 정규화 (중앙값, IQR 기반)
                self.fitted_params[i] = {
                    'median': np.median(col_data),
                    'q25': np.percentile(col_data, 25),
                    'q75': np.percentile(col_data, 75)
                }
        
        self.is_fitted = True
        return self
    
    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        데이터 정규화 적용
        
        Args:
            data: 정규화할 데이터
            
        Returns:
            정규화된 데이터
        """
        if not self.is_fitted:
            raise ValueError("정규화기가 학습되지 않았습니다. fit() 메서드를 먼저 호출하세요.")
        
        normalized_data = data.copy()
        
        for i, params in self.fitted_params.items():
            if i >= data.shape[1]:
                continue
                
            col_data = data[:, i]
            
            if self.method == 'minmax':
                # Min-Max 정규화
                normalized_data[:, i] = (col_data - params['min']) / params['range']
                # 0-1 범위로 클리핑
                normalized_data[:, i] = np.clip(normalized_data[:, i], 0, 1)
                
            elif self.method == 'zscore':
                # Z-score 정규화
                if params['std'] > 0:
                    normalized_data[:, i] = (col_data - params['mean']) / params['std']
                else:
                    normalized_data[:, i] = col_data - params['mean']
                    
            elif self.method == 'robust':
                # Robust 정규화
                iqr = params['q75'] - params['q25']
                if iqr > 0:
                    normalized_data[:, i] = (col_data - params['median']) / iqr
                else:
                    normalized_data[:, i] = col_data - params['median']
        
        return normalized_data
    
    def fit_transform(self, data: np.ndarray, sensor_types: list = None) -> np.ndarray:
        """학습과 변환을 한 번에 수행"""
        return self.fit(data, sensor_types).transform(data)
    
    def inverse_transform(self, normalized_data: np.ndarray) -> np.ndarray:
        """정규화를 역변환하여 원본 스케일로 복원"""
        if not self.is_fitted:
            raise ValueError("정규화기가 학습되지 않았습니다.")
        
        original_data = normalized_data.copy()
        
        for i, params in self.fitted_params.items():
            if i >= normalized_data.shape[1]:
                continue
                
            col_data = normalized_data[:, i]
            
            if self.method == 'minmax':
                original_data[:, i] = col_data * params['range'] + params['min']
                
            elif self.method == 'zscore':
                original_data[:, i] = col_data * params['std'] + params['mean']
                
            elif self.method == 'robust':
                iqr = params['q75'] - params['q25']
                original_data[:, i] = col_data * iqr + params['median']
        
        return original_data
    
    def normalize_ksl_sensors(self, flex_data: np.ndarray, imu_data: np.ndarray, 
                             orientation_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        KSL 센서 데이터 전용 정규화
        
        Args:
            flex_data: Flex 센서 데이터 (N x 5)
            imu_data: IMU 데이터 (N x 6) [accel_xyz, gyro_xyz]
            orientation_data: 방향 데이터 (N x 3) [roll, pitch, yaw]
            
        Returns:
            정규화된 (flex, imu, orientation) 데이터
        """
        # Flex 센서 정규화 (0-1 범위)
        flex_normalizer = SensorNormalization('minmax', {'flex': self.default_ranges['flex']})
        sensor_types_flex = ['flex'] * flex_data.shape[1]
        normalized_flex = flex_normalizer.fit_transform(flex_data, sensor_types_flex)
        
        # IMU 데이터 정규화
        imu_normalizer = SensorNormalization('minmax')
        # 앞의 3개는 가속도, 뒤의 3개는 자이로스코프
        sensor_types_imu = ['accel'] * 3 + ['gyro'] * 3
        normalized_imu = imu_normalizer.fit_transform(imu_data, sensor_types_imu)
        
        # 방향 데이터 정규화 (-1 ~ 1 범위로 변환)
        orientation_normalizer = SensorNormalization('minmax', 
                                                    {'orientation': self.default_ranges['orientation']})
        sensor_types_ori = ['orientation'] * orientation_data.shape[1]
        normalized_orientation = orientation_normalizer.fit_transform(orientation_data, sensor_types_ori)
        # -1 ~ 1 범위로 조정
        normalized_orientation = normalized_orientation * 2 - 1
        
        return normalized_flex, normalized_imu, normalized_orientation
    
    def get_normalization_info(self) -> dict:
        """정규화 정보 반환"""
        return {
            'method': self.method,
            'is_fitted': self.is_fitted,
            'fitted_params': self.fitted_params,
            'sensor_ranges': self.sensor_ranges
        }

# 편의 함수들
def normalize_sensor_data(data: np.ndarray, method: str = 'minmax') -> np.ndarray:
    """센서 데이터 간단 정규화 함수"""
    normalizer = SensorNormalization(method)
    return normalizer.fit_transform(data)

def normalize_ksl_data(flex_data: np.ndarray, imu_data: np.ndarray) -> np.ndarray:
    """KSL 데이터 통합 정규화 함수"""
    # Flex 센서 정규화 (0-1)
    flex_norm = (flex_data - 700) / (900 - 700)
    flex_norm = np.clip(flex_norm, 0, 1)
    
    # IMU 정규화 (-1 to 1)
    imu_norm = imu_data / 180.0
    imu_norm = np.clip(imu_norm, -1, 1)
    
    return np.hstack([flex_norm, imu_norm])

# 사용 예시
if __name__ == "__main__":
    # 테스트 데이터 생성
    np.random.seed(42)
    
    # 가상의 센서 데이터
    flex_test = np.random.uniform(750, 850, (100, 5))  # Flex 센서
    imu_test = np.random.uniform(-1, 1, (100, 6))      # IMU 센서
    orientation_test = np.random.uniform(-90, 90, (100, 3))  # 방향
    
    # 정규화 테스트
    normalizer = SensorNormalization()
    
    print("=== 센서 정규화 테스트 ===")
    print(f"원본 Flex 범위: {flex_test.min():.2f} ~ {flex_test.max():.2f}")
    
    norm_flex, norm_imu, norm_ori = normalizer.normalize_ksl_sensors(
        flex_test, imu_test, orientation_test)
    
    print(f"정규화 Flex 범위: {norm_flex.min():.2f} ~ {norm_flex.max():.2f}")
    print(f"정규화 IMU 범위: {norm_imu.min():.2f} ~ {norm_imu.max():.2f}")
    print(f"정규화 방향 범위: {norm_ori.min():.2f} ~ {norm_ori.max():.2f}")
    print("정규화 테스트 완료!")
