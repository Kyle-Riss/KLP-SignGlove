"""
추론 전처리 유틸리티

센서 데이터 정규화, 패딩, 트렁케이션 등
"""

import numpy as np
import torch
from typing import Union, List
import pickle
from pathlib import Path


class InferencePreprocessor:
    """
    추론 전처리기
    
    훈련 시 사용한 정규화 파라미터를 로딩하여
    일관된 전처리 수행
    """
    
    def __init__(
        self,
        target_timesteps: int = 87,
        n_channels: int = 8,
        scaler=None
    ):
        """
        Args:
            target_timesteps: 타겟 타임스텝 길이
            n_channels: 센서 채널 수
            scaler: sklearn StandardScaler 객체 (None이면 정규화 안함)
        """
        self.target_timesteps = target_timesteps
        self.n_channels = n_channels
        self.scaler = scaler
    
    def pad_or_truncate(
        self,
        data: np.ndarray
    ) -> np.ndarray:
        """
        데이터를 타겟 길이로 패딩 또는 트렁케이션
        
        Args:
            data: (timesteps, channels) 센서 데이터
        
        Returns:
            processed: (target_timesteps, channels) 처리된 데이터
        """
        current_timesteps = data.shape[0]
        
        if current_timesteps == self.target_timesteps:
            return data
        elif current_timesteps < self.target_timesteps:
            # 패딩
            padding = np.zeros((self.target_timesteps - current_timesteps, self.n_channels))
            return np.vstack([data, padding])
        else:
            # 트렁케이션 (앞부분 사용)
            return data[:self.target_timesteps]
    
    def normalize(
        self,
        data: np.ndarray
    ) -> np.ndarray:
        """
        데이터 정규화
        
        Args:
            data: (timesteps, channels) 센서 데이터
        
        Returns:
            normalized: 정규화된 데이터
        """
        if self.scaler is None:
            return data
        
        # StandardScaler 적용
        original_shape = data.shape
        data_flat = data.reshape(-1, self.n_channels)
        normalized_flat = self.scaler.transform(data_flat)
        normalized = normalized_flat.reshape(original_shape)
        
        return normalized
    
    def preprocess_single(
        self,
        raw_data: Union[np.ndarray, List[List[float]]],
        normalize: bool = True
    ) -> torch.Tensor:
        """
        단일 샘플 전처리
        
        Args:
            raw_data: 원시 센서 데이터
            normalize: 정규화 여부
        
        Returns:
            tensor: (1, target_timesteps, channels) 텐서
        """
        # numpy 변환
        if isinstance(raw_data, list):
            data = np.array(raw_data, dtype=np.float32)
        else:
            data = raw_data.astype(np.float32)
        
        # 패딩/트렁케이션
        data = self.pad_or_truncate(data)
        
        # 정규화
        if normalize:
            data = self.normalize(data)
        
        # 텐서 변환
        tensor = torch.from_numpy(data).unsqueeze(0)  # (1, timesteps, channels)
        
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
            normalize: 정규화 여부
        
        Returns:
            tensor: (batch_size, target_timesteps, channels) 텐서
        """
        processed_list = []
        
        for raw_data in raw_data_list:
            # numpy 변환
            if isinstance(raw_data, list):
                data = np.array(raw_data, dtype=np.float32)
            else:
                data = raw_data.astype(np.float32)
            
            # 패딩/트렁케이션
            data = self.pad_or_truncate(data)
            
            # 정규화
            if normalize:
                data = self.normalize(data)
            
            processed_list.append(data)
        
        # 배치 텐서로 변환
        batch_array = np.stack(processed_list, axis=0)
        tensor = torch.from_numpy(batch_array)
        
        return tensor
    
    @classmethod
    def load_scaler(
        cls,
        scaler_path: str,
        target_timesteps: int = 87,
        n_channels: int = 8
    ) -> 'InferencePreprocessor':
        """
        저장된 scaler를 로딩하여 전처리기 생성
        
        Args:
            scaler_path: scaler 파일 경로
            target_timesteps: 타겟 타임스텝 길이
            n_channels: 센서 채널 수
        
        Returns:
            preprocessor: 전처리기 인스턴스
        """
        scaler_file = Path(scaler_path)
        
        if scaler_file.exists():
            with open(scaler_file, 'rb') as f:
                scaler = pickle.load(f)
        else:
            raise FileNotFoundError(f"Scaler 파일을 찾을 수 없습니다: {scaler_path}")
        
        return cls(
            target_timesteps=target_timesteps,
            n_channels=n_channels,
            scaler=scaler
        )
    
    def save_scaler(self, scaler_path: str):
        """
        Scaler 저장
        
        Args:
            scaler_path: 저장할 파일 경로
        """
        if self.scaler is None:
            raise ValueError("저장할 scaler가 없습니다")
        
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)


# 테스트 코드
if __name__ == "__main__":
    print("🧪 InferencePreprocessor 테스트...")
    
    # 전처리기 초기화 (scaler 없이)
    preprocessor = InferencePreprocessor(target_timesteps=87, n_channels=8)
    
    # 테스트 데이터
    print("\n1️⃣ 단일 샘플 전처리:")
    raw_data = np.random.randn(50, 8)  # 짧은 시퀀스
    tensor = preprocessor.preprocess_single(raw_data, normalize=False)
    print(f"  입력 shape: {raw_data.shape}")
    print(f"  출력 shape: {tensor.shape}")
    print(f"  ✅ 패딩 성공!")
    
    # 배치 전처리
    print("\n2️⃣ 배치 전처리:")
    raw_data_list = [
        np.random.randn(50, 8),
        np.random.randn(100, 8),
        np.random.randn(87, 8)
    ]
    batch_tensor = preprocessor.preprocess_batch(raw_data_list, normalize=False)
    print(f"  배치 크기: {len(raw_data_list)}")
    print(f"  출력 shape: {batch_tensor.shape}")
    print(f"  ✅ 배치 전처리 성공!")
