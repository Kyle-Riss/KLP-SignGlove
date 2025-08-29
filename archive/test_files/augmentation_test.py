#!/usr/bin/env python3
"""
데이터 증강 기법 효과 검증 스크립트
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import seaborn as sns
from pathlib import Path
import json

def load_sample_data():
    """샘플 데이터 로드"""
    data_dir = Path("../SignGlove_HW/datasets/unified")
    
    # 첫 번째 클래스의 첫 번째 사용자 데이터 사용
    first_class_dir = data_dir / "ㄱ" / "1"
    if not first_class_dir.exists():
        print(f"❌ 데이터 디렉토리를 찾을 수 없습니다: {first_class_dir}")
        return []
    
    sample_files = list(first_class_dir.glob("*.h5"))[:3]  # 3개 파일만
    
    all_data = []
    for file_path in sample_files:
        try:
            # H5 파일 읽기
            import h5py
            with h5py.File(file_path, 'r') as f:
                # 데이터셋 키 확인
                keys = list(f.keys())
                print(f"📁 {file_path.name} 키: {keys}")
                
                # 첫 번째 키의 데이터 사용
                if keys:
                    data_key = keys[0]
                    sensor_data = f[data_key][:50]  # 처음 50개 샘플만
                    
                    # 첫 번째 센서 채널만 사용
                    if len(sensor_data.shape) > 1:
                        sensor_data = sensor_data[:, 0]  # 첫 번째 채널
                    
                    all_data.append(sensor_data.tolist())
                    print(f"✅ {file_path.name} 로드 완료: {len(sensor_data)}개 샘플")
                else:
                    print(f"⚠️ 데이터를 찾을 수 없습니다: {file_path.name}")
                
        except Exception as e:
            print(f"❌ 파일 로드 실패 {file_path.name}: {e}")
            continue
    
    return all_data

def apply_augmentation(data, method='noise', intensity=0.01):
    """데이터 증강 적용"""
    augmented = []
    
    for sequence in data:
        if method == 'noise':
            # 가우시안 노이즈 추가
            noise = np.random.normal(0, intensity, len(sequence))
            augmented_seq = np.array(sequence) + noise
            
        elif method == 'time_shift':
            # 시간 이동
            shift = np.random.randint(-2, 3)
            if shift > 0:
                # 오른쪽으로 이동
                augmented_seq = np.array(sequence[shift:])
                # 부족한 부분을 마지막 값으로 채움
                if len(augmented_seq) < len(sequence):
                    padding = [sequence[-1]] * (len(sequence) - len(augmented_seq))
                    augmented_seq = np.concatenate([augmented_seq, padding])
            else:
                # 왼쪽으로 이동
                augmented_seq = np.array(sequence[:shift])
                # 부족한 부분을 첫 번째 값으로 채움
                if len(augmented_seq) < len(sequence):
                    padding = [sequence[0]] * (len(sequence) - len(augmented_seq))
                    augmented_seq = np.concatenate([padding, augmented_seq])
                
        elif method == 'scaling':
            # 스케일링
            scale = np.random.uniform(0.9, 1.1)
            augmented_seq = np.array(sequence) * scale
            
        else:
            augmented_seq = np.array(sequence)
            
        augmented.append(augmented_seq)
    
    return augmented

def analyze_augmentation_effects():
    """증강 기법 효과 분석"""
    print("🔬 데이터 증강 효과 분석 시작")
    print("=" * 50)
    
    # 샘플 데이터 로드
    original_data = load_sample_data()
    if not original_data:
        print("❌ 데이터를 찾을 수 없습니다.")
        return
    
    print(f"📊 원본 데이터: {len(original_data)}개 시퀀스")
    
    # 다양한 증강 기법 테스트
    augmentation_methods = {
        'noise_small': ('noise', 0.005),
        'noise_medium': ('noise', 0.01),
        'noise_large': ('noise', 0.05),
        'time_shift': ('time_shift', 0),
        'scaling': ('scaling', 0)
    }
    
    results = {}
    
    for method_name, (method, intensity) in augmentation_methods.items():
        print(f"\n🔧 {method_name} 증강 테스트...")
        
        augmented_data = apply_augmentation(original_data, method, intensity)
        
        # 원본과 증강 데이터 비교
        mse_values = []
        correlation_values = []
        
        for orig, aug in zip(original_data, augmented_data):
            # 길이 맞추기
            min_len = min(len(orig), len(aug))
            orig_trim = orig[:min_len]
            aug_trim = aug[:min_len]
            
            # MSE 계산
            mse = mean_squared_error(orig_trim, aug_trim)
            mse_values.append(mse)
            
            # 상관계수 계산
            corr = np.corrcoef(orig_trim, aug_trim)[0, 1]
            correlation_values.append(corr)
        
        results[method_name] = {
            'mse_mean': np.mean(mse_values),
            'mse_std': np.std(mse_values),
            'correlation_mean': np.mean(correlation_values),
            'correlation_std': np.std(correlation_values),
            'data': augmented_data
        }
        
        print(f"  - MSE: {results[method_name]['mse_mean']:.6f} ± {results[method_name]['mse_std']:.6f}")
        print(f"  - 상관계수: {results[method_name]['correlation_mean']:.4f} ± {results[method_name]['correlation_std']:.4f}")
    
    # 결과 시각화
    visualize_augmentation_results(original_data, results)
    
    # 권장사항 제시
    print_recommendations(results)

def visualize_augmentation_results(original_data, results):
    """증강 결과 시각화"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('데이터 증강 기법 효과 비교', fontsize=16, fontweight='bold')
    
    # 원본 데이터 플롯
    axes[0, 0].plot(original_data[0], 'b-', linewidth=2, label='원본')
    axes[0, 0].set_title('원본 데이터')
    axes[0, 0].set_xlabel('시간')
    axes[0, 0].set_ylabel('센서 값')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 각 증강 기법별 플롯
    positions = [(0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
    method_names = list(results.keys())
    
    for i, (method_name, pos) in enumerate(zip(method_names, positions)):
        row, col = pos
        original = original_data[0]
        augmented = results[method_name]['data'][0]
        
        # 길이 맞추기
        min_len = min(len(original), len(augmented))
        orig_trim = original[:min_len]
        aug_trim = augmented[:min_len]
        
        axes[row, col].plot(orig_trim, 'b-', linewidth=2, label='원본', alpha=0.7)
        axes[row, col].plot(aug_trim, 'r--', linewidth=2, label='증강', alpha=0.7)
        axes[row, col].set_title(f'{method_name}')
        axes[row, col].set_xlabel('시간')
        axes[row, col].set_ylabel('센서 값')
        axes[row, col].legend()
        axes[row, col].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('augmentation_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\n📈 시각화 결과가 'augmentation_analysis.png'에 저장되었습니다.")

def print_recommendations(results):
    """증강 기법 권장사항 출력"""
    print("\n🎯 증강 기법 권장사항:")
    print("=" * 50)
    
    # 안전한 기법들
    safe_methods = []
    for method_name, result in results.items():
        if result['correlation_mean'] > 0.95 and result['mse_mean'] < 0.01:
            safe_methods.append(method_name)
    
    print("✅ 안전한 증강 기법:")
    for method in safe_methods:
        result = results[method]
        print(f"  - {method}: 상관계수 {result['correlation_mean']:.4f}, MSE {result['mse_mean']:.6f}")
    
    # 위험한 기법들
    dangerous_methods = []
    for method_name, result in results.items():
        if result['correlation_mean'] < 0.8 or result['mse_mean'] > 0.1:
            dangerous_methods.append(method_name)
    
    if dangerous_methods:
        print("\n⚠️ 위험한 증강 기법 (사용 금지):")
        for method in dangerous_methods:
            result = results[method]
            print(f"  - {method}: 상관계수 {result['correlation_mean']:.4f}, MSE {result['mse_mean']:.6f}")
    
    print("\n💡 권장 증강 전략:")
    print("  1. 노이즈: 0.005-0.01 표준편차 (매우 작은 노이즈)")
    print("  2. 시간 이동: ±1-2 샘플 (미세한 시작점 조정)")
    print("  3. 스케일링: 0.95-1.05 배율 (전체적인 크기 조정)")
    print("  4. 조합 사용: 여러 기법을 순차적으로 적용")
    print("  5. 검증 필수: 증강 후 데이터 품질 확인")

if __name__ == "__main__":
    analyze_augmentation_effects()
