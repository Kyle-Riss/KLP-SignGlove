import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pytorch_lightning as L
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import os
import glob
from typing import Dict, List, Tuple, Optional
from torch import Tensor
import json


class getDataset(Dataset):
    """pytorch Dataset object generation for SignGlove data"""

    def __init__(self, x: np.ndarray, y: np.ndarray):
        """Assign tensors to self states"""
        self.x = torch.from_numpy(x).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self) -> int:
        """Length is the number of examples in a dataset"""
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        """get item and it's label with given index idx"""
        sample = {
            "measurement": self.x[idx],
            "label": self.y[idx],
        }
        return sample


class DynamicDataModule(L.LightningDataModule):
    """Lightning Module for dynamic loading of SignGlove-DataAnalysis data"""

    def __init__(
        self,
        data_dir: str = "/home/billy/25-1kp/SignGlove_HW/datasets/unified",
        time_steps: int = 87,  # 새로운 SignGlove 데이터셋은 87 타임스텝
        n_channels: int = 8,
        batch_size: int = 32,
        kfold: int = 0,
        splits: int = 5,
        seed: int = 42,
        shuffle: bool = True,
        test_size: float = 0.2,
        val_size: float = 0.2,
        use_test_split: bool = True,
        resampling_method: str = "padding",  # "interpolation" or "padding"
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
        
        # Store class names and scaler
        self.class_names = []
        self.scaler = StandardScaler()

    def find_data_files(self) -> List[str]:
        """Find all episode CSV files in the new SignGlove dataset directory"""
        # 새로운 SignGlove 데이터셋: 34개 클래스 (자음 14개 + 모음 10개 + 숫자 10개)
        consonants = ['ㄱ', 'ㄴ', 'ㄷ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅅ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
        vowels = ['ㅏ', 'ㅑ', 'ㅓ', 'ㅕ', 'ㅗ', 'ㅛ', 'ㅜ', 'ㅠ', 'ㅡ', 'ㅣ']
        numbers = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        
        consonant_files = []
        vowel_files = []
        number_files = []
        
        # 새로운 SignGlove 데이터셋 구조: datasets/{class}/{session}/episode_*.csv
        for consonant in consonants:
            consonant_pattern = os.path.join(self.data_dir, consonant, "*", "episode_*.csv")
            files = glob.glob(consonant_pattern)
            consonant_files.extend(files)
        
        for vowel in vowels:
            vowel_pattern = os.path.join(self.data_dir, vowel, "*", "episode_*.csv")
            files = glob.glob(vowel_pattern)
            vowel_files.extend(files)
        
        for number in numbers:
            number_pattern = os.path.join(self.data_dir, number, "*", "episode_*.csv")
            files = glob.glob(number_pattern)
            number_files.extend(files)
        
        files = consonant_files + vowel_files + number_files
        print(f"Found {len(consonant_files)} consonant files, {len(vowel_files)} vowel files, {len(number_files)} number files")
        print(f"Total: {len(files)} episode files from new SignGlove dataset (34 classes)")
        return files

    def extract_class_from_filename(self, filepath: str) -> str:
        """Extract class name from filepath (new SignGlove dataset structure)"""
        # 새로운 SignGlove 데이터셋 구조: datasets/{class}/{session}/episode_*.csv
        path_parts = filepath.split('/')
        for part in path_parts:
            # 한글 자모 (ㄱ-ㅎ, ㅏ-ㅣ)
            if len(part) == 1 and ord(part) >= 0x3131 and ord(part) <= 0x318E:
                return part
            # 숫자 (0-9)
            elif len(part) == 1 and part.isdigit():
                return part
        return "unknown"

    def load_csv_file(self, filepath: str) -> np.ndarray:
        """Load a single CSV file and extract 8 channels"""
        try:
            df = pd.read_csv(filepath)
            
            # Extract 8 channels: flex1-5, pitch, roll, yaw
            channels = ['flex1', 'flex2', 'flex3', 'flex4', 'flex5', 'pitch', 'roll', 'yaw']
            data = df[channels].values
            
            return data
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return np.array([])

    def normalize_timesteps(self, data: np.ndarray) -> np.ndarray:
        """Normalize timestep length to self.time_steps"""
        if len(data) == 0:
            return np.array([])
            
        current_length = len(data)
        
        if current_length == self.time_steps:
            return data
        elif current_length > self.time_steps:
            # Truncate to target length
            return data[:self.time_steps]
        else:
            # Pad or interpolate to target length
            if self.resampling_method == "interpolation":
                # Linear interpolation (requires scipy)
                try:
                    from scipy.interpolate import interp1d
                    x_old = np.linspace(0, 1, current_length)
                    x_new = np.linspace(0, 1, self.time_steps)
                    
                    interpolated_data = np.zeros((self.time_steps, self.n_channels))
                    for i in range(self.n_channels):
                        f = interp1d(x_old, data[:, i], kind='linear', fill_value='extrapolate')
                        interpolated_data[:, i] = f(x_new)
                    
                    return interpolated_data
                except ImportError:
                    print("Warning: scipy not available, falling back to padding")
                    # Fallback to padding if scipy is not available
                    pass
            
            # Default to ASL-style padding
            if True:  # Always use padding now (default behavior)
                # ASL-style Padding (constant_values=1.0)
                padding_length = self.time_steps - current_length
                padded_data = np.pad(data, ((0, padding_length), (0, 0)), mode='constant', constant_values=1.0)
                return padded_data

    def preprocess_data(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Load and preprocess all data files"""
        files = self.find_data_files()
        
        all_data = []
        all_labels = []
        class_to_idx = {}
        
        # 클래스 이름 초기화
        self.class_names = []
        
        print("Loading and preprocessing data...")
        
        for filepath in files:
            class_name = self.extract_class_from_filename(filepath)
            
            # 중복 클래스 방지
            if class_name not in class_to_idx:
                class_to_idx[class_name] = len(class_to_idx)
                self.class_names.append(class_name)
            
            data = self.load_csv_file(filepath)
            if len(data) > 0:
                # Normalize timesteps
                normalized_data = self.normalize_timesteps(data)
                if len(normalized_data) > 0:
                    all_data.append(normalized_data)
                    all_labels.append(class_to_idx[class_name])
        
        if not all_data:
            raise ValueError("No valid data found!")
        
        # Convert to numpy arrays
        X = np.array(all_data)  # Shape: (samples, time_steps, channels)
        y = np.array(all_labels)
        
        print(f"Data shape: {X.shape}")
        print(f"Number of classes: {len(self.class_names)}")
        print(f"Classes: {self.class_names}")
        
        # Normalize features
        original_shape = X.shape
        X_reshaped = X.reshape(-1, X.shape[-1])  # Reshape for scaling
        X_scaled = self.scaler.fit_transform(X_reshaped)
        X = X_scaled.reshape(original_shape)  # Reshape back
        
        return X, y, self.class_names

    def split_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split data into train/val/test sets"""
        if self.use_test_split:
            # 3-way split: train/val/test
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y, test_size=self.test_size, random_state=self.seed, stratify=y
            )
            
            val_size_adjusted = self.val_size / (1 - self.test_size)
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=val_size_adjusted, random_state=self.seed, stratify=y_temp
            )
            
            return X_train, X_val, X_test, y_train, y_val, y_test
        else:
            # Use K-fold for train/val split
            kfold = StratifiedKFold(n_splits=self.splits, shuffle=True, random_state=self.seed)
            
            for i, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
                if i == self.kfold:
                    X_train, X_val = X[train_idx], X[val_idx]
                    y_train, y_val = y[train_idx], y[val_idx]
                    
                    # Create dummy test set (same as val for K-fold)
                    X_test, y_test = X_val.copy(), y_val.copy()
                    
                    return X_train, X_val, X_test, y_train, y_val, y_test

    def print_split_statistics(self, y_train: np.ndarray, y_val: np.ndarray, y_test: np.ndarray):
        """Print detailed statistics of data splits"""
        print("\n" + "="*60)
        print("📊 데이터 분할 통계")
        print("="*60)
        
        total_samples = len(y_train) + len(y_val) + len(y_test)
        print(f"총 샘플 수: {total_samples}")
        print(f"훈련 세트: {len(y_train)} ({len(y_train)/total_samples*100:.1f}%)")
        print(f"검증 세트: {len(y_val)} ({len(y_val)/total_samples*100:.1f}%)")
        print(f"테스트 세트: {len(y_test)} ({len(y_test)/total_samples*100:.1f}%)")
        
        print(f"\n클래스별 분포:")
        for i, class_name in enumerate(self.class_names):
            train_count = np.sum(y_train == i)
            val_count = np.sum(y_val == i)
            test_count = np.sum(y_test == i)
            total_count = train_count + val_count + test_count
            
            print(f"  {class_name}: 총 {total_count}개 (훈련: {train_count}, 검증: {val_count}, 테스트: {test_count})")

    def setup(self, stage: str):
        """Prepare data as tensors"""
        print("🚀 DynamicDataModule 설정 시작...")
        
        # Load and preprocess data
        X, y, class_names = self.preprocess_data()
        
        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(X, y)
        
        # Print statistics
        self.print_split_statistics(y_train, y_val, y_test)
        
        # Create datasets
        self.train_dataset = getDataset(X_train, y_train)
        self.val_dataset = getDataset(X_val, y_val)
        self.test_dataset = getDataset(X_test, y_test)
        
        print("✅ DynamicDataModule 설정 완료!")

    def train_dataloader(self) -> DataLoader:
        """Called when trainer.fit() is used"""
        return DataLoader(self.train_dataset, **self.params)

    def val_dataloader(self) -> DataLoader:
        """Called when trainer.val() is used in training cycle"""
        return DataLoader(self.val_dataset, batch_size=len(self.val_dataset))

    def test_dataloader(self) -> DataLoader:
        """Called when trainer.test() is used"""
        return DataLoader(self.test_dataset, batch_size=len(self.test_dataset))


# Test the DynamicDataModule
if __name__ == "__main__":
    print("🧪 DynamicDataModule 테스트 시작...")
    
    # Test with new SignGlove dataset
    dm = DynamicDataModule(
        data_dir="/home/billy/25-1kp/SignGlove_HW/datasets/unified",  # 새로운 데이터셋 경로로 변경 필요
        time_steps=87,  # 새로운 SignGlove 데이터셋은 87 타임스텝
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