"""
간결한 DynamicDataModule - Lightning 모듈만 포함
"""
import torch
import pytorch_lightning as L
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold, train_test_split
from typing import Tuple

from .data_loader import find_signglove_files
from .data_preprocessor import preprocess_data, print_split_statistics
from .dataset import SignGloveDataset


class DynamicDataModule(L.LightningDataModule):
    """SignGlove 데이터를 위한 Lightning DataModule (24 classes: 14 consonants + 10 vowels)"""

    def __init__(
        self,
        data_dir: str = "/home/billy/25-1kp/SignGlove_HW/datasets/unified",
        time_steps: int = 87,
        n_channels: int = 8,
        batch_size: int = 32,
        kfold: int = 0,
        splits: int = 5,
        seed: int = 42,
        shuffle: bool = True,
        test_size: float = 0.2,
        val_size: float = 0.2,
        use_test_split: bool = True,
        resampling_method: str = "padding",
    ):
        super().__init__()
        
        self.data_dir = data_dir
        self.time_steps = time_steps
        self.n_channels = n_channels
        self.batch_size = batch_size
        self.kfold = kfold
        self.splits = splits
        self.seed = seed
        self.test_size = test_size
        self.val_size = val_size
        self.use_test_split = use_test_split
        self.resampling_method = resampling_method
        
        # Initialize generator for reproducible splits
        self.generator = torch.Generator(device="cpu").manual_seed(self.seed)
        
        # Params for DataLoader
        self.params = {
            "batch_size": batch_size,
            "shuffle": shuffle,
            "generator": self.generator,
        }
        
        # Store class names
        self.class_names = []
        
        # Cache for data splits (to ensure consistency across setup calls)
        self._data_split_cache = None

    def split_data(self, X: np.ndarray, y: np.ndarray, X_padding: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """데이터를 train/val/test 세트로 분할합니다."""
        if self.use_test_split:
            # 3-way split: train/val/test
            X_temp, X_test, y_temp, y_test, X_padding_temp, X_padding_test = train_test_split(
                X, y, X_padding, test_size=self.test_size, random_state=self.seed, stratify=y
            )
            
            val_size_adjusted = self.val_size / (1 - self.test_size)
            X_train, X_val, y_train, y_val, X_padding_train, X_padding_val = train_test_split(
                X_temp, y_temp, X_padding_temp, test_size=val_size_adjusted, random_state=self.seed, stratify=y_temp
            )
            
            return X_train, X_val, X_test, y_train, y_val, y_test, X_padding_train, X_padding_val, X_padding_test
        else:
            # Use K-fold for train/val split
            kfold = StratifiedKFold(n_splits=self.splits, shuffle=True, random_state=self.seed)
            
            for i, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
                if i == self.kfold:
                    X_train, X_val = X[train_idx], X[val_idx]
                    y_train, y_val = y[train_idx], y[val_idx]
                    X_padding_train, X_padding_val = X_padding[train_idx], X_padding[val_idx]
                    
                    # Create independent test set (20% of remaining data)
                    test_size = 0.2
                    val_test_size = len(val_idx)
                    test_samples = int(val_test_size * test_size)
                    test_idx = val_idx[:test_samples]
                    val_idx = val_idx[test_samples:]
                    
                    X_test = X[test_idx]
                    y_test = y[test_idx]
                    X_padding_test = X_padding[test_idx]
                    
                    X_val = X[val_idx]
                    y_val = y[val_idx]
                    X_padding_val = X_padding[val_idx]
                    
                    return X_train, X_val, X_test, y_train, y_val, y_test, X_padding_train, X_padding_val, X_padding_test

    def setup(self, stage: str):
        """데이터를 준비합니다."""
        # Check if data is already loaded and split
        if self._data_split_cache is not None:
            print("✅ 캐시된 데이터 분할 사용 (일관된 train/val/test set)")
            X_train, X_val, X_test, y_train, y_val, y_test, X_padding_train, X_padding_val, X_padding_test = self._data_split_cache
        else:
            print("🚀 DynamicDataModule 설정 시작...")
            
            # Find data files
            files = find_signglove_files(self.data_dir)
            
            # Load and preprocess data
            X, y, X_padding, class_names, scaler = preprocess_data(
                files, self.time_steps, self.n_channels, self.resampling_method
            )
            
            # Store class names
            self.class_names = class_names
            
            # Split data (only once!)
            X_train, X_val, X_test, y_train, y_val, y_test, X_padding_train, X_padding_val, X_padding_test = self.split_data(X, y, X_padding)
            
            # Cache the split for future setup() calls
            self._data_split_cache = (X_train, X_val, X_test, y_train, y_val, y_test, X_padding_train, X_padding_val, X_padding_test)
            
            # Print statistics
            print_split_statistics(y_train, y_val, y_test, self.class_names)
        
        # Create datasets with padding information
        self.train_dataset = SignGloveDataset(
            torch.from_numpy(X_train), 
            torch.from_numpy(y_train),
            torch.from_numpy(X_padding_train)
        )
        self.val_dataset = SignGloveDataset(
            torch.from_numpy(X_val), 
            torch.from_numpy(y_val),
            torch.from_numpy(X_padding_val)
        )
        self.test_dataset = SignGloveDataset(
            torch.from_numpy(X_test), 
            torch.from_numpy(y_test),
            torch.from_numpy(X_padding_test)
        )
        
        print("✅ DynamicDataModule 설정 완료!")

    def train_dataloader(self) -> DataLoader:
        """훈련 데이터로더를 반환합니다."""
        return DataLoader(self.train_dataset, **self.params)

    def val_dataloader(self) -> DataLoader:
        """검증 데이터로더를 반환합니다."""
        return DataLoader(self.val_dataset, batch_size=len(self.val_dataset))

    def test_dataloader(self) -> DataLoader:
        """테스트 데이터로더를 반환합니다."""
        return DataLoader(self.test_dataset, batch_size=len(self.test_dataset))


# Test the DynamicDataModule
if __name__ == "__main__":
    print("🧪 DynamicDataModule 테스트 시작...")
    
    # Test with SignGlove dataset
    dm = DynamicDataModule(
        data_dir="/home/billy/25-1kp/SignGlove_HW/datasets/unified",
        time_steps=87,
        n_channels=8,
        batch_size=16,
        use_test_split=True
    )
    
    dm.setup("test")
    
    # Test dataloaders
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    test_loader = dm.test_dataloader()
    
    print(f"\n📊 데이터로더 테스트:")
    print(f"훈련 배치 수: {len(train_loader)}")
    print(f"검증 배치 수: {len(val_loader)}")
    print(f"테스트 배치 수: {len(test_loader)}")
    
    # Test a single batch
    for batch in train_loader:
        print(f"\n🔍 배치 정보:")
        print(f"  입력 shape: {batch['measurement'].shape}")
        print(f"  라벨 shape: {batch['label'].shape}")
        print(f"  라벨 값: {batch['label'][:5].tolist()}")
        break
    
    print("\n✅ DynamicDataModule 테스트 완료!")