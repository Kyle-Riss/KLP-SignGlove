"""
A-GRU 클래스별 중요도 패턴 분석
자음 vs 모음에서 A-Net의 작동 차이 분석
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm

from src.models.AGRUModels import AGRUModel
from src.misc.DynamicDataModule import DynamicDataModule

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# 출력 디렉토리
OUTPUT_DIR = Path("visualizations/agru_class_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_trained_model(checkpoint_path):
    """학습된 A-GRU 모델 로드"""
    model = AGRUModel.load_from_checkpoint(
        checkpoint_path,
        learning_rate=0.001,
        input_size=8,
        hidden_size=64,
        classes=24,
        layers=2,
        dropout=0.3,
        gamma=1.0
    )
    model.eval()
    return model


def analyze_class_patterns(model, datamodule):
    """
    클래스별 A-Net 중요도 패턴 분석
    
    목표:
    1. 자음 vs 모음의 중요도 패턴 차이
    2. 각 클래스의 중요도 시간적 분포
    3. 채널별 중요도 차이 (Flex vs IMU)
    """
    datamodule.setup('test')
    test_loader = datamodule.test_dataloader()
    
    # 모델을 CPU로 이동
    device = torch.device('cpu')
    model = model.to(device)
    
    # 클래스 정의
    consonants = ['ㄱ', 'ㄴ', 'ㄷ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅅ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
    vowels = ['ㅏ', 'ㅑ', 'ㅓ', 'ㅕ', 'ㅗ', 'ㅛ', 'ㅜ', 'ㅠ', 'ㅡ', 'ㅣ']
    all_classes = consonants + vowels
    
    # 결과 저장용
    class_results = {
        'consonants': {
            'importances': [],      # A-Net 중요도
            'flex_signals': [],     # Flex 센서
            'imu_signals': [],      # IMU 센서
            'labels': []
        },
        'vowels': {
            'importances': [],
            'flex_signals': [],
            'imu_signals': [],
            'labels': []
        }
    }
    
    print("📊 클래스별 중요도 추출 중...")
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Processing batches"):
            x = batch['measurement'].to(device)  # (batch, time, 8)
            y = batch['label'].cpu().numpy()
            
            # A-GRU forward pass
            outputs, h_n, all_importances = model.agru(x)
            
            # Layer 1의 importance (더 raw한 패턴)
            importance = all_importances[0].cpu().numpy()  # (batch, time, 8)
            x_np = x.cpu().numpy()
            
            # 배치 내 각 샘플 처리
            for i in range(x.size(0)):
                class_idx = y[i]
                class_name = all_classes[class_idx]
                
                # 자음 vs 모음 구분
                if class_name in consonants:
                    category = 'consonants'
                else:
                    category = 'vowels'
                
                class_results[category]['importances'].append(importance[i])
                class_results[category]['flex_signals'].append(x_np[i, :, :5])  # Flex 5개
                class_results[category]['imu_signals'].append(x_np[i, :, 5:])   # IMU 3개
                class_results[category]['labels'].append(class_name)
    
    # 리스트를 numpy 배열로 변환
    for category in ['consonants', 'vowels']:
        class_results[category]['importances'] = np.array(class_results[category]['importances'])
        class_results[category]['flex_signals'] = np.array(class_results[category]['flex_signals'])
        class_results[category]['imu_signals'] = np.array(class_results[category]['imu_signals'])
    
    print(f"✅ 자음 샘플: {len(class_results['consonants']['labels'])}개")
    print(f"✅ 모음 샘플: {len(class_results['vowels']['labels'])}개")
    
    return class_results, all_classes


def plot_category_comparison(class_results):
    """자음 vs 모음 비교 플롯"""
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    
    categories = ['consonants', 'vowels']
    titles = ['Consonants', 'Vowels']
    
    for col, (category, title) in enumerate(zip(categories, titles)):
        importance = class_results[category]['importances']  # (N, time, 8)
        flex = class_results[category]['flex_signals']       # (N, time, 5)
        imu = class_results[category]['imu_signals']         # (N, time, 3)
        
        # 평균 계산
        mean_importance = importance.mean(axis=0)  # (time, 8)
        mean_flex = flex.mean(axis=0)              # (time, 5)
        mean_imu = imu.mean(axis=0)                # (time, 3)
        
        # 1. Flex 센서 신호
        ax = axes[0, col]
        for i in range(5):
            ax.plot(mean_flex[:, i], alpha=0.7, label=f'Flex {i+1}')
        ax.set_title(f'{title} - Flex Sensors', fontsize=12, fontweight='bold')
        ax.set_ylabel('Signal Value')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # 2. A-Net 중요도 (Flex 채널)
        ax = axes[1, col]
        for i in range(5):
            ax.plot(mean_importance[:, i], alpha=0.7, label=f'Importance Ch{i+1}')
        ax.set_title(f'{title} - A-Net Importance (Flex Channels)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Importance')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # 3. 평균 비교 (Flex vs Importance)
        ax = axes[2, col]
        avg_flex = mean_flex.mean(axis=1)
        avg_importance = mean_importance[:, :5].mean(axis=1)
        
        ax.plot(avg_flex, label='Avg Flex Signal', linewidth=2, color='blue', alpha=0.7)
        ax.plot(avg_importance, label='Avg A-Net Importance', linewidth=2, color='red', alpha=0.7)
        ax.set_title(f'{title} - Average Patterns', fontsize=12, fontweight='bold')
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / "category_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ 자음/모음 비교 플롯 저장: {output_path}")
    plt.close()


def plot_channel_importance_heatmap(class_results):
    """채널별 중요도 히트맵 (Flex vs IMU)"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    categories = ['consonants', 'vowels']
    titles = ['Consonants', 'Vowels']
    
    for ax, category, title in zip(axes, categories, titles):
        importance = class_results[category]['importances']  # (N, time, 8)
        
        # 시간 축 평균 → (N, 8)
        mean_over_time = importance.mean(axis=1)
        
        # 전체 평균 → (8,)
        channel_importance = mean_over_time.mean(axis=0)
        
        # 히트맵 데이터 준비 (8개 채널을 세로로 표시)
        data = channel_importance.reshape(-1, 1)  # (8, 1)
        
        # 히트맵 그리기
        channel_labels = [f'Flex {i+1}' for i in range(5)] + [f'IMU {i+1}' for i in range(3)]
        sns.heatmap(data.T, annot=True, fmt='.3f', cmap='YlOrRd', 
                   xticklabels=channel_labels,
                   yticklabels=['Importance'],
                   cbar_kws={'label': 'Importance'},
                   ax=ax)
        
        ax.set_title(f'{title} - Channel Importance', fontsize=12, fontweight='bold')
        ax.set_xlabel('Channels')
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / "channel_importance_heatmap.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ 채널 중요도 히트맵 저장: {output_path}")
    plt.close()


def plot_temporal_importance_distribution(class_results):
    """시간적 중요도 분포 비교"""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    categories = ['consonants', 'vowels']
    titles = ['Consonants', 'Vowels']
    colors = ['blue', 'red']
    
    for ax, category, title, color in zip(axes, categories, titles, colors):
        importance = class_results[category]['importances']  # (N, time, 8)
        
        # 채널 평균 → (N, time)
        importance_over_channels = importance.mean(axis=2)
        
        # 전체 샘플 평균과 표준편차
        mean_importance = importance_over_channels.mean(axis=0)
        std_importance = importance_over_channels.std(axis=0)
        
        # 플롯
        time_steps = np.arange(len(mean_importance))
        ax.plot(time_steps, mean_importance, color=color, linewidth=2, label='Mean Importance')
        ax.fill_between(time_steps, 
                        mean_importance - std_importance,
                        mean_importance + std_importance,
                        color=color, alpha=0.2, label='±1 std')
        
        ax.set_title(f'{title} - Temporal Importance Distribution', fontsize=12, fontweight='bold')
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('A-Net Importance')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / "temporal_importance_distribution.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ 시간적 중요도 분포 저장: {output_path}")
    plt.close()


def calculate_statistics(class_results):
    """정량적 통계 계산"""
    stats = {}
    
    for category in ['consonants', 'vowels']:
        importance = class_results[category]['importances']  # (N, time, 8)
        flex = class_results[category]['flex_signals']       # (N, time, 5)
        
        # Flex 채널만 사용 (첫 5개 채널)
        importance_flex = importance[:, :, :5]  # (N, time, 5)
        
        # 상관계수 계산
        correlations = []
        peak_diffs = []
        
        for i in range(importance.shape[0]):
            # 평균 Flex vs 평균 Importance
            avg_flex = flex[i].mean(axis=1)
            avg_importance = importance_flex[i].mean(axis=1)
            
            # Pearson 상관계수
            corr = np.corrcoef(avg_flex, avg_importance)[0, 1]
            correlations.append(corr)
            
            # 피크 위치 차이
            flex_peak = np.argmax(avg_flex)
            importance_peak = np.argmax(avg_importance)
            peak_diffs.append(importance_peak - flex_peak)
        
        stats[category] = {
            'correlation': {
                'mean': np.mean(correlations),
                'std': np.std(correlations)
            },
            'peak_diff': {
                'mean': np.mean(peak_diffs),
                'std': np.std(peak_diffs)
            },
            'importance_mean': importance.mean(),
            'importance_std': importance.std(),
            'flex_mean': flex.mean(),
            'flex_std': flex.std()
        }
    
    return stats


def print_statistics(stats):
    """통계 출력"""
    print("\n" + "="*60)
    print("📊 클래스별 정량적 분석 결과")
    print("="*60)
    
    for category in ['consonants', 'vowels']:
        category_name = "Consonants" if category == 'consonants' else "Vowels"
        print(f"\n[{category_name}]")
        print(f"  상관계수: {stats[category]['correlation']['mean']:.3f} ± {stats[category]['correlation']['std']:.3f}")
        print(f"  시간 차이: {stats[category]['peak_diff']['mean']:.1f} ± {stats[category]['peak_diff']['std']:.1f} steps")
        print(f"  평균 중요도: {stats[category]['importance_mean']:.3f} ± {stats[category]['importance_std']:.3f}")
        print(f"  평균 Flex 신호: {stats[category]['flex_mean']:.3f} ± {stats[category]['flex_std']:.3f}")
    
    # 자음 vs 모음 비교
    print("\n" + "-"*60)
    print("[Consonants vs Vowels Comparison]")
    
    corr_diff = stats['consonants']['correlation']['mean'] - stats['vowels']['correlation']['mean']
    print(f"  상관계수 차이: {corr_diff:.3f}")
    
    peak_diff = stats['consonants']['peak_diff']['mean'] - stats['vowels']['peak_diff']['mean']
    print(f"  시간 차이 차이: {peak_diff:.1f} steps")
    
    importance_diff = stats['consonants']['importance_mean'] - stats['vowels']['importance_mean']
    print(f"  평균 중요도 차이: {importance_diff:.3f}")
    
    print("="*60)


def main():
    print("🔬 A-GRU 클래스별 중요도 패턴 분석 시작...\n")
    
    # 1. 모델 로드
    checkpoint_path = "checkpoints/best_model_epoch=46_val/loss=0.00.ckpt"
    print(f"📂 체크포인트 로드: {checkpoint_path}")
    model = load_trained_model(checkpoint_path)
    print("✅ 모델 로드 완료\n")
    
    # 2. 데이터 로드
    print("📂 데이터 로드 중...")
    datamodule = DynamicDataModule(
        data_dir='/home/billy/25-1kp/SignGlove_HW/datasets/unified',
        batch_size=32,
        test_size=0.2,
        val_size=0.2,
        use_test_split=True
    )
    print("✅ 데이터 로드 완료\n")
    
    # 3. 클래스별 분석
    class_results, all_classes = analyze_class_patterns(model, datamodule)
    
    # 4. 시각화
    print("\n🎨 시각화 생성 중...")
    plot_category_comparison(class_results)
    plot_channel_importance_heatmap(class_results)
    plot_temporal_importance_distribution(class_results)
    
    # 5. 통계 계산 및 출력
    print("\n📈 통계 계산 중...")
    stats = calculate_statistics(class_results)
    print_statistics(stats)
    
    # 6. 결과 저장
    print(f"\n💾 결과 저장 중...")
    np.savez(OUTPUT_DIR / "class_analysis_results.npz",
             **class_results,
             stats=stats)
    
    print(f"\n✅ 분석 완료!")
    print(f"📁 결과 저장 위치: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()

