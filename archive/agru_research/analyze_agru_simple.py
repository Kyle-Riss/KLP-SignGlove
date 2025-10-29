"""
간단한 A-GRU 중요도 시각화
더미 데이터로 A-Net의 작동 원리 검증
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.append('.')

from src.models.AmygdalaGRU import StackedAGRU

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False


def generate_flex_signal(seq_len=87, peak_position=40, noise_level=0.1):
    """
    손가락 굴곡 패턴을 시뮬레이션
    
    Args:
        seq_len: 시퀀스 길이
        peak_position: 최대 굴곡 시점
        noise_level: 노이즈 레벨
    
    Returns:
        flex_signal: (seq_len, 5) Flex 센서 신호
        imu_signal: (seq_len, 3) IMU 신호
    """
    time = np.arange(seq_len)
    
    # Flex 신호: 가우시안 형태 (손가락 굴곡)
    flex_signals = []
    for i in range(5):
        # 각 손가락의 피크 시점을 약간 다르게
        peak_offset = np.random.randint(-5, 5)
        peak = peak_position + peak_offset
        
        # 가우시안 envelope
        signal = np.exp(-((time - peak) ** 2) / (2 * 10 ** 2))
        signal += np.random.normal(0, noise_level, seq_len)
        signal = np.clip(signal, 0, 1)
        
        flex_signals.append(signal)
    
    flex_signal = np.array(flex_signals).T  # (seq_len, 5)
    
    # IMU 신호: 움직임 시작 ~ 종료
    imu_signal = np.random.normal(0, 0.05, (seq_len, 3))
    
    return flex_signal, imu_signal


def visualize_agru_importance():
    """A-GRU의 중요도 가중치 시각화"""
    print("🧪 A-GRU 중요도 가중치 시각화...")
    
    # 1. 모델 생성 (학습되지 않은 모델로 테스트)
    input_size = 8
    hidden_size = 64
    agru = StackedAGRU(input_size, hidden_size, num_layers=2, gamma=1.0, dropout=0.0)
    agru.eval()
    
    # 2. 테스트 신호 생성 (3개 샘플)
    output_dir = Path('visualizations/agru_analysis')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    peak_positions = [30, 45, 60]  # 다양한 피크 위치
    
    for idx, peak_pos in enumerate(peak_positions):
        flex, imu = generate_flex_signal(seq_len=87, peak_position=peak_pos)
        
        # 입력 결합
        x = np.concatenate([flex, imu], axis=1)  # (87, 8)
        x_tensor = torch.FloatTensor(x).unsqueeze(0)  # (1, 87, 8)
        
        # 3. A-GRU forward pass
        with torch.no_grad():
            outputs, h_n, all_importances = agru(x_tensor)
        
        # Layer 1의 importance
        importance = all_importances[0][0].numpy()  # (87, 8)
        
        # 4. 시각화
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))
        time = np.arange(87)
        
        # (1) Flex 신호
        ax1 = axes[0]
        for i in range(5):
            ax1.plot(time, flex[:, i], label=f'Flex {i+1}', alpha=0.7, linewidth=1.5)
        ax1.axvline(peak_pos, color='red', linestyle='--', linewidth=2, alpha=0.7, label='True Peak')
        ax1.set_title(f'Sample {idx+1} - Flex Sensor Signals (Peak at t={peak_pos})', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Flex Signal', fontsize=10)
        ax1.legend(loc='upper right', fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        # (2) A-Net 중요도 (Flex 채널)
        ax2 = axes[1]
        importance_flex = importance[:, :5]
        for i in range(5):
            ax2.plot(time, importance_flex[:, i], label=f'e_t (Flex {i+1})', linewidth=2, alpha=0.8)
        ax2.axvline(peak_pos, color='red', linestyle='--', linewidth=2, alpha=0.7, label='True Peak')
        ax2.set_title('A-Net Importance Weights (e_t) for Flex Channels', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Importance (e_t)', fontsize=10)
        ax2.legend(loc='upper right', fontsize=8)
        ax2.grid(True, alpha=0.3)
        
        # (3) 평균 신호 vs 평균 중요도
        ax3 = axes[2]
        avg_flex = flex.mean(axis=1)
        avg_importance = importance_flex.mean(axis=1)
        
        ax3_twin = ax3.twinx()
        
        line1 = ax3.plot(time, avg_flex, 'b-', linewidth=2.5, label='Avg Flex Signal', alpha=0.7)
        line2 = ax3_twin.plot(time, avg_importance, 'r-', linewidth=2.5, label='Avg Importance (e_t)', alpha=0.7)
        ax3.axvline(peak_pos, color='purple', linestyle='--', linewidth=2, alpha=0.7, label='True Peak')
        
        # A-Net이 예측한 피크
        predicted_peak = np.argmax(avg_importance)
        ax3.axvline(predicted_peak, color='orange', linestyle=':', linewidth=2, alpha=0.7, label=f'Predicted Peak (t={predicted_peak})')
        
        ax3.set_xlabel('Time Step', fontsize=10)
        ax3.set_ylabel('Avg Flex Signal', fontsize=10, color='b')
        ax3_twin.set_ylabel('Avg Importance (e_t)', fontsize=10, color='r')
        ax3.tick_params(axis='y', labelcolor='b')
        ax3_twin.tick_params(axis='y', labelcolor='r')
        
        # 범례
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax3.legend(lines + [ax3.axvline(peak_pos, alpha=0), ax3.axvline(predicted_peak, alpha=0)], 
                  labels + ['True Peak', f'Predicted Peak (t={predicted_peak})'], 
                  loc='upper left', fontsize=8)
        
        ax3.set_title(f'Average Signals (Error: {abs(predicted_peak - peak_pos)} steps)', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/demo_sample_{idx+1}_peak{peak_pos}.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Sample {idx+1} (Peak={peak_pos}) → Predicted={predicted_peak} (Error={abs(predicted_peak-peak_pos)})")
        print(f"   저장: {output_dir}/demo_sample_{idx+1}_peak{peak_pos}.png")
    
    print(f"\n✅ 시각화 완료! 결과: {output_dir}/")


if __name__ == "__main__":
    visualize_agru_importance()
