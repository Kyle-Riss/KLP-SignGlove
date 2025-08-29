# Butterworth 필터 구현
import numpy as np
from scipy.signal import butter, filtfilt
from typing import Union, Tuple

class ButterworthFilter:
    def __init__(self, cutoff_freq: float = 5.0, sampling_rate: float = 100.0, 
                 filter_order: int = 4, filter_type: str = 'low'):
        """
        Butterworth 필터 초기화
        
        Args:
            cutoff_freq: 차단 주파수 (Hz)
            sampling_rate: 샘플링 주파수 (Hz)  
            filter_order: 필터 차수
            filter_type: 필터 타입 ('low', 'high', 'band')
        """
        self.cutoff_freq = cutoff_freq
        self.sampling_rate = sampling_rate
        self.filter_order = filter_order
        self.filter_type = filter_type
        
        # 정규화된 차단 주파수 계산 (Nyquist 주파수 기준)
        nyquist_freq = sampling_rate / 2.0
        self.normalized_cutoff = cutoff_freq / nyquist_freq
        
        # 필터 계수 계산
        self.b, self.a = butter(filter_order, self.normalized_cutoff, 
                               btype=filter_type, analog=False)
    
    def apply_filter(self, data: np.ndarray, axis: int = 0) -> np.ndarray:
        """
        데이터에 Butterworth 필터 적용
        
        Args:
            data: 입력 데이터 (1D 또는 2D 배열)
            axis: 필터 적용 축 (시간 축)
            
        Returns:
            필터링된 데이터
        """
        if data.size == 0:
            return data
            
        # filtfilt를 사용하여 zero-phase 필터링 (위상 지연 없음)
        filtered_data = filtfilt(self.b, self.a, data, axis=axis)
        return filtered_data
    
    def apply_to_dataframe(self, df, sensor_columns: list) -> np.ndarray:
        """
        DataFrame의 센서 컬럼들에 필터 적용
        
        Args:
            df: pandas DataFrame
            sensor_columns: 필터링할 컬럼명 리스트
            
        Returns:
            필터링된 데이터 배열
        """
        import pandas as pd
        
        filtered_data = []
        for col in sensor_columns:
            if col in df.columns:
                filtered_col = self.apply_filter(df[col].values)
                filtered_data.append(filtered_col)
            else:
                print(f"Warning: 컬럼 '{col}'을 찾을 수 없습니다.")
                
        return np.column_stack(filtered_data) if filtered_data else np.array([])
    
    def get_filter_info(self) -> dict:
        """필터 설정 정보 반환"""
        return {
            'cutoff_frequency': self.cutoff_freq,
            'sampling_rate': self.sampling_rate,
            'filter_order': self.filter_order,
            'filter_type': self.filter_type,
            'normalized_cutoff': self.normalized_cutoff
        }

# 편의 함수들
def apply_low_pass_filter(data: np.ndarray, cutoff: float = 10.0, fs: float = 50.0, order: int = 4) -> np.ndarray:
    """저역 통과 필터 적용"""
    filter_obj = ButterworthFilter(cutoff, fs, order, 'low')
    return filter_obj.apply_filter(data)

def apply_moving_average(data: np.ndarray, window_size: int = 5) -> np.ndarray:
    """이동 평균 필터 적용"""
    if len(data) < window_size:
        return data
    
    filtered = np.zeros_like(data)
    for i in range(len(data)):
        start_idx = max(0, i - window_size // 2)
        end_idx = min(len(data), i + window_size // 2 + 1)
        if data.ndim == 1:
            filtered[i] = np.mean(data[start_idx:end_idx])
        else:
            filtered[i] = np.mean(data[start_idx:end_idx], axis=0)
    
    return filtered

# 사용 예시
if __name__ == "__main__":
    # 테스트 데이터 생성
    import matplotlib.pyplot as plt
    
    # 노이즈가 포함된 신호 생성
    t = np.linspace(0, 1, 100)  # 1초, 100Hz
    clean_signal = np.sin(2 * np.pi * 2 * t)  # 2Hz 신호
    noise = 0.5 * np.sin(2 * np.pi * 20 * t)  # 20Hz 노이즈
    noisy_signal = clean_signal + noise
    
    # 필터 적용
    butter_filter = ButterworthFilter(cutoff_freq=5.0, sampling_rate=100.0)
    filtered_signal = butter_filter.apply_filter(noisy_signal)
    
    print("Butterworth 필터 테스트 완료")
    print(f"필터 정보: {butter_filter.get_filter_info()}")
    print(f"원본 신호 RMS: {np.sqrt(np.mean(noisy_signal**2)):.3f}")
    print(f"필터링 신호 RMS: {np.sqrt(np.mean(filtered_signal**2)):.3f}")
