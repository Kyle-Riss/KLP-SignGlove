"""
데이터 전처리 관련 유틸리티 함수들
"""
import numpy as np
from typing import Tuple, List
from sklearn.preprocessing import StandardScaler


def normalize_timesteps(data: np.ndarray, target_length: int, method: str = "padding") -> np.ndarray:
    """타임스텝 길이를 정규화합니다."""
    if len(data) == 0:
        return np.array([])
        
    current_length = len(data)
    
    if current_length == target_length:
        return data
    elif current_length > target_length:
        # Truncate to target length
        return data[:target_length]
    else:
        # Pad or interpolate to target length
        if method == "interpolation":
            # Linear interpolation (requires scipy)
            try:
                from scipy.interpolate import interp1d
                x_old = np.linspace(0, 1, current_length)
                x_new = np.linspace(0, 1, target_length)
                
                interpolated_data = np.zeros((target_length, data.shape[1]))
                for i in range(data.shape[1]):
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
            padding_length = target_length - current_length
            padded_data = np.pad(data, ((0, padding_length), (0, 0)), mode='constant', constant_values=1.0)
            return padded_data


def preprocess_data(
    files: List[str], 
    target_timesteps: int, 
    n_channels: int = 8,
    resampling_method: str = "padding"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """모든 데이터 파일을 로드하고 전처리합니다."""
    from .data_loader import load_csv_file, extract_class_from_filename
    
    all_data = []
    all_labels = []
    all_padding_info = []  # 패딩 정보 저장
    class_to_idx = {}
    class_names = []
    
    print("Loading and preprocessing data...")
    
    for filepath in files:
        class_name = extract_class_from_filename(filepath)
        
        # 중복 클래스 방지
        if class_name not in class_to_idx:
            class_to_idx[class_name] = len(class_to_idx)
            class_names.append(class_name)
        
        data = load_csv_file(filepath)
        if len(data) > 0:
            original_length = len(data)
            # Normalize timesteps
            normalized_data = normalize_timesteps(data, target_timesteps, resampling_method)
            if len(normalized_data) > 0:
                all_data.append(normalized_data)
                all_labels.append(class_to_idx[class_name])
                
                # 패딩 정보 생성 (ASL 방식)
                if original_length < target_timesteps:
                    # 패딩된 경우: 패딩 부분을 1.0, 실제 데이터 부분을 0.0
                    padding_info = np.zeros(target_timesteps)
                    padding_info[original_length:] = 1.0  # 패딩된 부분
                else:
                    # 트렁케이션된 경우: 모두 0.0 (실제 데이터)
                    padding_info = np.zeros(target_timesteps)
                
                all_padding_info.append(padding_info)
    
    if not all_data:
        raise ValueError("No valid data found!")
    
    # Convert to numpy arrays
    X = np.array(all_data)  # Shape: (samples, time_steps, channels)
    y = np.array(all_labels)
    X_padding = np.array(all_padding_info)  # Shape: (samples, time_steps)
    
    print(f"Data shape: {X.shape}")
    print(f"Padding shape: {X_padding.shape}")
    print(f"Number of classes: {len(class_names)} (24 classes: 14 consonants + 10 vowels)")
    print(f"Classes: {class_names}")
    
    # Normalize features
    original_shape = X.shape
    X_reshaped = X.reshape(-1, X.shape[-1])  # Reshape for scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_reshaped)
    X = X_scaled.reshape(original_shape)  # Reshape back
    
    return X, y, X_padding, class_names, scaler


def print_split_statistics(y_train: np.ndarray, y_val: np.ndarray, y_test: np.ndarray, class_names: List[str]):
    """데이터 분할 통계를 출력합니다."""
    print("\n" + "="*60)
    print("📊 데이터 분할 통계")
    print("="*60)
    
    total_samples = len(y_train) + len(y_val) + len(y_test)
    print(f"총 샘플 수: {total_samples}")
    print(f"훈련 세트: {len(y_train)} ({len(y_train)/total_samples*100:.1f}%)")
    print(f"검증 세트: {len(y_val)} ({len(y_val)/total_samples*100:.1f}%)")
    print(f"테스트 세트: {len(y_test)} ({len(y_test)/total_samples*100:.1f}%)")
    
    print(f"\n클래스별 분포:")
    for i, class_name in enumerate(class_names):
        train_count = np.sum(y_train == i)
        val_count = np.sum(y_val == i)
        test_count = np.sum(y_test == i)
        total_count = train_count + val_count + test_count
        
        print(f"  {class_name}: 총 {total_count}개 (훈련: {train_count}, 검증: {val_count}, 테스트: {test_count})")
