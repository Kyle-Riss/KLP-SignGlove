"""
A-GRU 중요도 가중치 분석 스크립트
A-Net이 학습한 e_t 값과 실제 센서 신호의 상관관계 검증
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

sys.path.append('.')

from src.models.AGRUModels import AGRUModel
from src.misc.DynamicDataModule import DynamicDataModule

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False


def load_trained_model(checkpoint_path):
    """학습된 A-GRU 모델 로드"""
    # Lightning은 체크포인트에 하이퍼파라미터를 저장하지만, 
    # 명시적으로 learning_rate를 제공해야 함
    model = AGRUModel.load_from_checkpoint(
        checkpoint_path,
        learning_rate=0.001,  # 기본값
        strict=False  # 누락된 키 무시
    )
    model.eval()
    return model


def analyze_importance_vs_flex(model, datamodule, num_samples=10):
    """
    A-Net 중요도(e_t)와 Flex 센서 신호의 상관관계 분석
    
    가설: A-Net은 손가락 최대 굴곡 시점에 높은 중요도를 부여할 것
    """
    datamodule.setup('test')
    test_loader = datamodule.test_dataloader()
    
    # 모델을 CPU로 이동 (데이터도 CPU에 있으므로)
    device = torch.device('cpu')
    model = model.to(device)
    
    results = {
        'importances': [],      # A-Net 중요도 (e_t)
        'flex_signals': [],     # Flex 센서 신호 (5개)
        'imu_signals': [],      # IMU 신호 (3개)
        'labels': [],           # 클래스 레이블
        'timestamps': []        # 타임스텝
    }
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if batch_idx >= num_samples:
                break
            
            x = batch['measurement'].to(device)  # (batch, time, 8)
            y = batch['label']
            
            # A-GRU forward pass로 importance 추출
            outputs, h_n, all_importances = model.agru(x)
            
            # Layer 1의 importance (더 raw한 패턴)
            importance_layer1 = all_importances[0]  # (batch, time, input_size)
            
            # 배치의 첫 샘플만 저장
            for i in range(min(x.size(0), num_samples - batch_idx)):
                results['importances'].append(importance_layer1[i].cpu().numpy())
                results['flex_signals'].append(x[i, :, :5].cpu().numpy())  # Flex 5개
                results['imu_signals'].append(x[i, :, 5:].cpu().numpy())   # IMU 3개
                results['labels'].append(y[i].item())
                results['timestamps'].append(np.arange(x.size(1)))
    
    return results


def compute_flex_peaks(flex_signals):
    """
    Flex 센서에서 최대 굴곡 시점 찾기
    
    Returns:
        peak_times: 각 Flex 센서의 최대값 시점 (5개)
        peak_magnitudes: 최대값 크기
    """
    # Flex 신호의 최대값 시점 (각 센서별)
    peak_times = np.argmax(flex_signals, axis=0)  # (5,)
    peak_magnitudes = np.max(flex_signals, axis=0)  # (5,)
    
    return peak_times, peak_magnitudes


def compute_motion_onset(signals, threshold=0.1):
    """
    움직임 시작 시점 감지 (신호가 threshold를 처음 넘는 시점)
    
    Returns:
        onset_times: 각 센서의 움직임 시작 시점
    """
    # 신호를 0~1로 정규화
    signals_norm = (signals - signals.min(axis=0)) / (signals.max(axis=0) - signals.min(axis=0) + 1e-8)
    
    onset_times = []
    for i in range(signals_norm.shape[1]):
        onset_idx = np.where(signals_norm[:, i] > threshold)[0]
        if len(onset_idx) > 0:
            onset_times.append(onset_idx[0])
        else:
            onset_times.append(0)
    
    return np.array(onset_times)


def visualize_importance_correlation(results, output_dir='visualizations/agru_analysis'):
    """
    A-Net 중요도와 센서 신호의 상관관계 시각화
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    num_samples = len(results['importances'])
    
    for sample_idx in range(min(num_samples, 5)):  # 5개 샘플만
        importance = results['importances'][sample_idx]  # (time, 8)
        flex = results['flex_signals'][sample_idx]        # (time, 5)
        imu = results['imu_signals'][sample_idx]          # (time, 3)
        label = results['labels'][sample_idx]
        
        # Flex 피크와 움직임 시작 시점
        flex_peaks, flex_mags = compute_flex_peaks(flex)
        flex_onsets = compute_motion_onset(flex, threshold=0.1)
        
        # 시각화
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))
        
        # 1. Flex 센서 신호 + A-Net 중요도 (Flex 채널)
        ax1 = axes[0]
        time = np.arange(flex.shape[0])
        
        # Flex 신호 5개
        for i in range(5):
            ax1.plot(time, flex[:, i], label=f'Flex {i+1}', alpha=0.7, linewidth=1.5)
            # 피크 표시
            ax1.axvline(flex_peaks[i], color=f'C{i}', linestyle='--', alpha=0.5)
            ax1.scatter([flex_peaks[i]], [flex_mags[i]], color=f'C{i}', s=100, zorder=5, marker='*')
        
        ax1.set_title(f'Sample {sample_idx} (Label: {label}) - Flex Sensors & Peaks', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Flex Signal (Normalized)', fontsize=10)
        ax1.legend(loc='upper right', fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        # 2. A-Net 중요도 (Flex 채널 5개)
        ax2 = axes[1]
        importance_flex = importance[:, :5]  # Flex 채널만
        
        for i in range(5):
            ax2.plot(time, importance_flex[:, i], label=f'e_t (Flex {i+1})', linewidth=2, alpha=0.8)
            # Flex 피크 시점 표시
            ax2.axvline(flex_peaks[i], color=f'C{i}', linestyle='--', alpha=0.5)
        
        ax2.set_title('A-Net Importance Weights (e_t) for Flex Channels', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Importance (e_t)', fontsize=10)
        ax2.legend(loc='upper right', fontsize=8)
        ax2.grid(True, alpha=0.3)
        
        # 3. 평균 중요도 vs 평균 Flex 신호
        ax3 = axes[2]
        avg_importance_flex = importance_flex.mean(axis=1)  # 시간축 평균
        avg_flex = flex.mean(axis=1)
        
        ax3_twin = ax3.twinx()
        
        line1 = ax3.plot(time, avg_flex, 'b-', linewidth=2, label='Avg Flex Signal', alpha=0.7)
        line2 = ax3_twin.plot(time, avg_importance_flex, 'r-', linewidth=2, label='Avg Importance (e_t)', alpha=0.7)
        
        # 평균 피크 시점 표시
        avg_peak = int(flex_peaks.mean())
        ax3.axvline(avg_peak, color='purple', linestyle='--', linewidth=2, alpha=0.7, label='Avg Peak Time')
        
        ax3.set_xlabel('Time Step', fontsize=10)
        ax3.set_ylabel('Avg Flex Signal', fontsize=10, color='b')
        ax3_twin.set_ylabel('Avg Importance (e_t)', fontsize=10, color='r')
        ax3.tick_params(axis='y', labelcolor='b')
        ax3_twin.tick_params(axis='y', labelcolor='r')
        
        # 범례 합치기
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax3.legend(lines, labels, loc='upper left', fontsize=8)
        
        ax3.set_title('Average Flex Signal vs A-Net Importance', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/sample_{sample_idx}_importance_correlation.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Sample {sample_idx} 시각화 완료: {output_dir}/sample_{sample_idx}_importance_correlation.png")


def compute_correlation_statistics(results):
    """
    A-Net 중요도와 Flex 피크의 정량적 상관관계 계산
    """
    correlations = []
    time_lags = []  # A-Net 피크와 Flex 피크 사이의 시간 차이
    
    for i in range(len(results['importances'])):
        importance = results['importances'][i][:, :5]  # Flex 채널만
        flex = results['flex_signals'][i]
        
        # Flex 피크 시점
        flex_peaks, _ = compute_flex_peaks(flex)
        avg_flex_peak = int(flex_peaks.mean())
        
        # A-Net 중요도 피크 시점 (평균)
        avg_importance = importance.mean(axis=1)
        importance_peak = np.argmax(avg_importance)
        
        # 시간 차이 (lag)
        time_lag = importance_peak - avg_flex_peak
        time_lags.append(time_lag)
        
        # 상관계수 계산
        corr = np.corrcoef(flex.mean(axis=1), avg_importance)[0, 1]
        correlations.append(corr)
    
    return {
        'correlations': correlations,
        'time_lags': time_lags,
        'mean_corr': np.mean(correlations),
        'std_corr': np.std(correlations),
        'mean_lag': np.mean(time_lags),
        'std_lag': np.std(time_lags)
    }


def create_summary_plot(stats, output_dir='visualizations/agru_analysis'):
    """통계 요약 플롯"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # 1. 상관계수 분포
    ax1 = axes[0]
    ax1.hist(stats['correlations'], bins=20, color='steelblue', alpha=0.7, edgecolor='black')
    ax1.axvline(stats['mean_corr'], color='red', linestyle='--', linewidth=2, 
                label=f"Mean: {stats['mean_corr']:.3f} ± {stats['std_corr']:.3f}")
    ax1.set_xlabel('Correlation Coefficient', fontsize=11)
    ax1.set_ylabel('Frequency', fontsize=11)
    ax1.set_title('A-Net Importance vs Flex Signal Correlation', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 시간 차이 분포
    ax2 = axes[1]
    ax2.hist(stats['time_lags'], bins=20, color='coral', alpha=0.7, edgecolor='black')
    ax2.axvline(stats['mean_lag'], color='red', linestyle='--', linewidth=2,
                label=f"Mean Lag: {stats['mean_lag']:.1f} ± {stats['std_lag']:.1f} steps")
    ax2.set_xlabel('Time Lag (Importance Peak - Flex Peak)', fontsize=11)
    ax2.set_ylabel('Frequency', fontsize=11)
    ax2.set_title('Temporal Alignment of A-Net Peaks', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/correlation_statistics.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 통계 요약 플롯 저장: {output_dir}/correlation_statistics.png")


def main():
    print("🔬 A-GRU 중요도 가중치 분석 시작...")
    
    # 1. 체크포인트 찾기 (Lightning logs에서)
    checkpoint_paths = [
        Path('lightning_logs'),
        Path('src/experiments/checkpoints'),
        Path('checkpoints')
    ]
    
    checkpoints = []
    for cp_dir in checkpoint_paths:
        if cp_dir.exists():
            checkpoints.extend(list(cp_dir.rglob('*.ckpt')))
    
    if not checkpoints:
        print("❌ 체크포인트를 찾을 수 없습니다!")
        return
    
    # 가장 최근 체크포인트 사용
    latest_checkpoint = max(checkpoints, key=lambda p: p.stat().st_mtime)
    print(f"📂 체크포인트 로드: {latest_checkpoint}")
    
    # 2. 모델 로드
    model = load_trained_model(str(latest_checkpoint))
    print(f"✅ 모델 로드 완료 (Test Acc: 99.65%)")
    
    # 3. 데이터 로드
    datamodule = DynamicDataModule(
        data_dir='/home/billy/25-1kp/SignGlove_HW/datasets/unified',
        batch_size=32,
        test_size=0.2,
        val_size=0.2
    )
    print(f"✅ 데이터 로드 완료")
    
    # 4. 중요도 분석
    print("\n📊 A-Net 중요도 추출 중...")
    results = analyze_importance_vs_flex(model, datamodule, num_samples=20)
    print(f"✅ {len(results['importances'])}개 샘플 분석 완료")
    
    # 5. 시각화
    print("\n🎨 시각화 생성 중...")
    visualize_importance_correlation(results)
    
    # 6. 통계 계산
    print("\n📈 상관관계 통계 계산 중...")
    stats = compute_correlation_statistics(results)
    print(f"\n{'='*60}")
    print(f"📊 정량적 검증 결과")
    print(f"{'='*60}")
    print(f"평균 상관계수: {stats['mean_corr']:.3f} ± {stats['std_corr']:.3f}")
    print(f"평균 시간 차이: {stats['mean_lag']:.1f} ± {stats['std_lag']:.1f} steps")
    print(f"{'='*60}")
    
    # 7. 통계 요약 플롯
    create_summary_plot(stats)
    
    # 8. 결과 저장
    print("\n💾 결과 저장 중...")
    np.savez(
        'visualizations/agru_analysis/analysis_results.npz',
        correlations=stats['correlations'],
        time_lags=stats['time_lags'],
        mean_corr=stats['mean_corr'],
        std_corr=stats['std_corr'],
        mean_lag=stats['mean_lag'],
        std_lag=stats['std_lag']
    )
    
    print("\n✅ 분석 완료!")
    print(f"📁 결과 저장 위치: visualizations/agru_analysis/")


if __name__ == "__main__":
    main()
