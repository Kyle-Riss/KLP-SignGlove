"""
PyTorch Dataset 클래스
"""
import torch
from torch.utils.data import Dataset
from typing import Dict
from torch import Tensor


class SignGloveDataset(Dataset):
    """SignGlove 데이터를 위한 PyTorch Dataset"""

    def __init__(self, x: torch.Tensor, y: torch.Tensor):
        """텐서를 저장합니다."""
        self.x = x.float()
        self.y = y.long()

    def __len__(self) -> int:
        """데이터셋의 샘플 수를 반환합니다."""
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        """주어진 인덱스의 샘플과 라벨을 반환합니다."""
        sample = {
            "measurement": self.x[idx],
            "label": self.y[idx],
        }
        return sample
