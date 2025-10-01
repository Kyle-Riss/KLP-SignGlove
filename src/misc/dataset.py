"""
PyTorch Dataset 클래스
"""
import torch
from torch.utils.data import Dataset
from typing import Dict
from torch import Tensor


class SignGloveDataset(Dataset):
    """SignGlove 데이터를 위한 PyTorch Dataset"""

    def __init__(self, x: torch.Tensor, y: torch.Tensor, x_padding: torch.Tensor = None):
        """텐서를 저장합니다."""
        self.x = x.float()
        self.y = y.long()
        self.x_padding = x_padding.float() if x_padding is not None else None

    def __len__(self) -> int:
        """데이터셋의 샘플 수를 반환합니다."""
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        """주어진 인덱스의 샘플과 라벨을 반환합니다."""
        sample = {
            "measurement": self.x[idx],
            "label": self.y[idx],
        }
        # 패딩 정보가 있으면 추가
        if self.x_padding is not None:
            sample["measurement_padding"] = self.x_padding[idx]
        
        return sample
