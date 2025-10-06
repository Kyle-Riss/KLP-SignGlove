"""
추론 전처리 유틸리티

원시 센서 데이터를 모델 입력 형식으로 변환
"""

import numpy as np
import torch
from typing import Union, List
from sklearn.preprocessing import StandardScaler


class InferencePreprocessor:
    """
    추론용 데이터 전처리기
    
    원시 센서 데이터를 모델 입력 텐서로 변환
    - 타임스텝 정규화 (패딩 또는 트렁케이션)
    - 특징 정규화 (StandardScaler)
    - 텐서 변환
    """
    
    def __init__(
        self,
        target_timesteps: int = 87,
        n_channels: int = 8,
        scaler: StandardScaler = None,
        enable_dtw: bool = False
    ):
        """
        Args:
            target_timesteps: 목표 타임스텝 길이 (기본: 87)
            n_channels: 입력 채널 수 (기본: 8)
            scaler: 사전 학습된 StandardScaler (None이면 새로 생성)
        """
        self.target_timesteps = target_timesteps
        self.n_channels = n_channels
        self.scaler = scaler if scaler is not None else StandardScaler()
        self.enable_dtw = enable_dtw
    
    def normalize_timesteps(self, data: np.ndarray) -> np.ndarray:
        """
        타임스텝 길이 정규화
        
        Args:
            data: 입력 데이터 (timesteps, channels)
        
        Returns:
            normalized: 정규화된 데이터 (target_timesteps, channels)
        """
        if len(data) == 0:
            return np.zeros((self.target_timesteps, self.n_channels))
        
        current_length = len(data)
        
        if current_length == self.target_timesteps:
            return data
        elif current_length > self.target_timesteps:
            # 트렁케이션 대신 옵션에 따라 균등 샘플링(DTW 대체 접근)
            if self.enable_dtw:
                indices = np.linspace(0, current_length - 1, self.target_timesteps, dtype=int)
                return data[indices]
            else:
                # 기본: 단순 트렁케이션
                return data[:self.target_timesteps]
        else:
            # 패딩
            padding_length = self.target_timesteps - current_length
            padded_data = np.pad(
                data,
                ((0, padding_length), (0, 0)),
                mode='constant',
                constant_values=0.0
            )
            return padded_data
    
    def normalize_features(self, data: np.ndarray, fit: bool = False) -> np.ndarray:
        """
        특징 정규화 (StandardScaler)
        
        Args:
            data: 입력 데이터 (timesteps, channels) 또는 (samples, timesteps, channels)
            fit: True이면 scaler를 학습, False이면 기존 scaler 사용
        
        Returns:
            normalized: 정규화된 데이터
        """
        original_shape = data.shape
        
        # 2D로 reshape
        if len(original_shape) == 2:
            data_reshaped = data
        else:
            data_reshaped = data.reshape(-1, original_shape[-1])
        
        # 정규화
        if fit:
            normalized = self.scaler.fit_transform(data_reshaped)
        else:
            normalized = self.scaler.transform(data_reshaped)
        
        # 원래 shape으로 복원
        normalized = normalized.reshape(original_shape)
        
        return normalized
    
    def preprocess_single(
        self,
        raw_data: Union[np.ndarray, List[List[float]]],
        normalize: bool = True
    ) -> torch.Tensor:
        """
        단일 샘플 전처리
        
        Args:
            raw_data: 원시 센서 데이터 (timesteps, channels) 또는 List
            normalize: 특징 정규화 여부
        
        Returns:
            tensor: 전처리된 텐서 (1, target_timesteps, channels)
        """
        # numpy 배열로 변환
        if isinstance(raw_data, list):
            raw_data = np.array(raw_data)
        
        # 타임스텝 정규화
        normalized_data = self.normalize_timesteps(raw_data)
        
        # 특징 정규화
        if normalize:
            normalized_data = self.normalize_features(normalized_data)
        
        # 텐서 변환 및 배치 차원 추가
        tensor = torch.from_numpy(normalized_data).float().unsqueeze(0)
        
        return tensor
    
    def preprocess_batch(
        self,
        raw_data_list: List[Union[np.ndarray, List[List[float]]]],
        normalize: bool = True
    ) -> torch.Tensor:
        """
        배치 전처리
        
        Args:
            raw_data_list: 원시 센서 데이터 리스트
            normalize: 특징 정규화 여부
        
        Returns:
            tensor: 전처리된 텐서 (batch_size, target_timesteps, channels)
        """
        batch_data = []
        
        for raw_data in raw_data_list:
            # numpy 배열로 변환
            if isinstance(raw_data, list):
                raw_data = np.array(raw_data)
            
            # 타임스텝 정규화
            normalized_data = self.normalize_timesteps(raw_data)
            batch_data.append(normalized_data)
        
        # numpy 배열로 변환
        batch_array = np.array(batch_data)
        
        # 특징 정규화
        if normalize:
            batch_array = self.normalize_features(batch_array)
        
        # 텐서 변환
        tensor = torch.from_numpy(batch_array).float()
        
        return tensor
    
    def save_scaler(self, filepath: str):
        """
        Scaler 저장
        
        Args:
            filepath: 저장 경로 (.pkl)
        """
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(self.scaler, f)
    
    @classmethod
    def load_scaler(cls, filepath: str, **kwargs):
        """
        Scaler 로드
        
        Args:
            filepath: 로드 경로 (.pkl)
            **kwargs: InferencePreprocessor 초기화 인자
        
        Returns:
            preprocessor: Scaler가 로드된 전처리기
        """
        import pickle
        with open(filepath, 'rb') as f:
            scaler = pickle.load(f)
        
        return cls(scaler=scaler, **kwargs)


# 테스트 코드
if __name__ == "__main__":
    print("🧪 InferencePreprocessor 테스트...")
    
    # 전처리기 생성
    preprocessor = InferencePreprocessor(
        target_timesteps=87,
        n_channels=8
    )
    
    # 테스트 데이터 (가변 길이)
    raw_data_short = np.random.randn(50, 8)  # 짧은 시퀀스
    raw_data_long = np.random.randn(100, 8)  # 긴 시퀀스
    
    # 단일 샘플 전처리
    print("\n📊 단일 샘플 전처리:")
    tensor_short = preprocessor.preprocess_single(raw_data_short, normalize=False)
    print(f"  짧은 시퀀스 (50) → {tensor_short.shape}")
    
    tensor_long = preprocessor.preprocess_single(raw_data_long, normalize=False)
    print(f"  긴 시퀀스 (100) → {tensor_long.shape}")
    
    # 배치 전처리
    print("\n📊 배치 전처리:")
    raw_data_list = [raw_data_short, raw_data_long]
    batch_tensor = preprocessor.preprocess_batch(raw_data_list, normalize=False)
    print(f"  배치 shape: {batch_tensor.shape}")
    
    print("\n✅ 테스트 완료!")

