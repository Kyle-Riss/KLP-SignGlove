#!/usr/bin/env python3
"""
Scale-Aware GRU 스케일 중요도 분석 스크립트
각 CNN 스케일(k=3,5,7)의 중요도를 분석합니다.
"""
import sys
import os.path as op
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Add project root to path
path = op.dirname(op.realpath(__file__))
sys.path.append(path)

from src.models.MSCSGRUModels_ScaleAware import MSCSGRU_ScaleAware, MSCSGRU_ScaleHard
from src.misc.DynamicDataModule import DynamicDataModule

def load_best_model(model_class, checkpoint_dir="lightning_logs"):
    """최고 성능 체크포인트 로드"""
    import glob
    
    # Find latest version directory
    version_dirs = glob.glob(f"{checkpoint_dir}/version_*")
    if not version_dirs:
        print(f"❌ 체크포인트 디렉토리를 찾을 수 없습니다: {checkpoint_dir}")
        return None
    
    latest_version = max(version_dirs, key=lambda x: int(x.split('_')[-1]))
    checkpoint_files = glob.glob(f"{latest_version}/checkpoints/*.ckpt")
    
    if not checkpoint_files:
        print(f"❌ 체크포인트 파일을 찾을 수 없습니다: {latest_version}/checkpoints/")
        return None
    
    # Load best checkpoint
    best_checkpoint = checkpoint_files[0]
    print(f"✅ 체크포인트 로드: {best_checkpoint}")
    
    model = model_class.load_from_checkpoint(
        best_checkpoint,
        learning_rate=0.001,
        input_size=8,
        hidden_size=64,
        classes=24
    )
    model.eval()
    return model

def analyze_scale_weights(model):
    """스케일별 가중치 분석"""
    print("\n" + "="*80)
    print("📊 스케일별 가중치 분석")
    print("="*80)
    
    # GRU1의 첫 번째 셀 분석
    gru1_cell = model.gru1.cells[0]
    
    # Update gate weights
    W_z3_norm = torch.norm(gru1_cell.W_z3.weight).item()
    W_z5_norm = torch.norm(gru1_cell.W_z5.weight).item()
    W_z7_norm = torch.norm(gru1_cell.W_z7.weight).item()
    
    total_z = W_z3_norm + W_z5_norm + W_z7_norm
    
    print("\n1️⃣  Update Gate (z_t) 가중치 크기:")
    print(f"  - W_z3 (k=3): {W_z3_norm:.4f} ({W_z3_norm/total_z*100:.1f}%)")
    print(f"  - W_z5 (k=5): {W_z5_norm:.4f} ({W_z5_norm/total_z*100:.1f}%)")
    print(f"  - W_z7 (k=7): {W_z7_norm:.4f} ({W_z7_norm/total_z*100:.1f}%)")
    
    # Reset gate weights
    W_r3_norm = torch.norm(gru1_cell.W_r3.weight).item()
    W_r5_norm = torch.norm(gru1_cell.W_r5.weight).item()
    W_r7_norm = torch.norm(gru1_cell.W_r7.weight).item()
    
    total_r = W_r3_norm + W_r5_norm + W_r7_norm
    
    print("\n2️⃣  Reset Gate (r_t) 가중치 크기:")
    print(f"  - W_r3 (k=3): {W_r3_norm:.4f} ({W_r3_norm/total_r*100:.1f}%)")
    print(f"  - W_r5 (k=5): {W_r5_norm:.4f} ({W_r5_norm/total_r*100:.1f}%)")
    print(f"  - W_r7 (k=7): {W_r7_norm:.4f} ({W_r7_norm/total_r*100:.1f}%)")
    
    # Hidden gate weights
    W_h3_norm = torch.norm(gru1_cell.W_h3.weight).item()
    W_h5_norm = torch.norm(gru1_cell.W_h5.weight).item()
    W_h7_norm = torch.norm(gru1_cell.W_h7.weight).item()
    
    total_h = W_h3_norm + W_h5_norm + W_h7_norm
    
    print("\n3️⃣  Hidden Gate (h̃_t) 가중치 크기:")
    print(f"  - W_h3 (k=3): {W_h3_norm:.4f} ({W_h3_norm/total_h*100:.1f}%)")
    print(f"  - W_h5 (k=5): {W_h5_norm:.4f} ({W_h5_norm/total_h*100:.1f}%)")
    print(f"  - W_h7 (k=7): {W_h7_norm:.4f} ({W_h7_norm/total_h*100:.1f}%)")
    
    # Overall importance
    avg_3 = (W_z3_norm/total_z + W_r3_norm/total_r + W_h3_norm/total_h) / 3
    avg_5 = (W_z5_norm/total_z + W_r5_norm/total_r + W_h5_norm/total_h) / 3
    avg_7 = (W_z7_norm/total_z + W_r7_norm/total_r + W_h7_norm/total_h) / 3
    
    print("\n🎯 전체 평균 중요도:")
    print(f"  - Scale k=3 (짧은 패턴): {avg_3*100:.1f}%")
    print(f"  - Scale k=5 (중간 패턴): {avg_5*100:.1f}%")
    print(f"  - Scale k=7 (긴 패턴): {avg_7*100:.1f}%")
    
    # Determine most important scale
    scales = {'k=3': avg_3, 'k=5': avg_5, 'k=7': avg_7}
    most_important = max(scales, key=scales.get)
    
    print(f"\n✨ 가장 중요한 스케일: {most_important} ({scales[most_important]*100:.1f}%)")
    
    return {
        'update_gate': [W_z3_norm/total_z, W_z5_norm/total_z, W_z7_norm/total_z],
        'reset_gate': [W_r3_norm/total_r, W_r5_norm/total_r, W_r7_norm/total_r],
        'hidden_gate': [W_h3_norm/total_h, W_h5_norm/total_h, W_h7_norm/total_h],
        'overall': [avg_3, avg_5, avg_7]
    }

def visualize_scale_importance(importance_data):
    """스케일 중요도 시각화"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Scale Importance Analysis in Scale-Aware GRU', fontsize=16, fontweight='bold')
    
    scales = ['k=3\n(Short)', 'k=5\n(Medium)', 'k=7\n(Long)']
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    # 1. Update Gate
    ax = axes[0, 0]
    bars = ax.bar(scales, importance_data['update_gate'], color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax.set_ylabel('Relative Importance', fontsize=12)
    ax.set_title('Update Gate (z_t)', fontsize=13, fontweight='bold')
    ax.set_ylim(0, 0.6)
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars, importance_data['update_gate']):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{val*100:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # 2. Reset Gate
    ax = axes[0, 1]
    bars = ax.bar(scales, importance_data['reset_gate'], color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax.set_ylabel('Relative Importance', fontsize=12)
    ax.set_title('Reset Gate (r_t)', fontsize=13, fontweight='bold')
    ax.set_ylim(0, 0.6)
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars, importance_data['reset_gate']):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{val*100:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # 3. Hidden Gate
    ax = axes[1, 0]
    bars = ax.bar(scales, importance_data['hidden_gate'], color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax.set_ylabel('Relative Importance', fontsize=12)
    ax.set_title('Hidden Gate (h̃_t)', fontsize=13, fontweight='bold')
    ax.set_ylim(0, 0.6)
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars, importance_data['hidden_gate']):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{val*100:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # 4. Overall Average
    ax = axes[1, 1]
    bars = ax.bar(scales, importance_data['overall'], color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax.set_ylabel('Average Importance', fontsize=12)
    ax.set_title('Overall Scale Importance', fontsize=13, fontweight='bold')
    ax.set_ylim(0, 0.6)
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars, importance_data['overall']):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{val*100:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('scale_importance_analysis.png', dpi=300, bbox_inches='tight')
    print("\n✅ 스케일 중요도 플롯 저장: scale_importance_analysis.png")

def test_scale_ablation(model, datamodule):
    """스케일별 Ablation Study"""
    print("\n" + "="*80)
    print("🧪 스케일 Ablation Study")
    print("="*80)
    print("\n각 스케일을 제거했을 때의 성능 변화를 측정합니다...")
    
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    test_loader = datamodule.test_dataloader()
    
    def evaluate_with_ablation(ablate_scale=None):
        """특정 스케일을 제거하고 평가"""
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in test_loader:
                x, y, padding = batch
                x = x.to(device)
                y = y.to(device)
                padding = padding.to(device)
                
                # Forward pass with ablation
                if ablate_scale:
                    # Temporarily zero out specific scale weights
                    gru1_cell = model.gru1.cells[0]
                    
                    if ablate_scale == 3:
                        original_z3 = gru1_cell.W_z3.weight.data.clone()
                        original_r3 = gru1_cell.W_r3.weight.data.clone()
                        original_h3 = gru1_cell.W_h3.weight.data.clone()
                        gru1_cell.W_z3.weight.data.zero_()
                        gru1_cell.W_r3.weight.data.zero_()
                        gru1_cell.W_h3.weight.data.zero_()
                    elif ablate_scale == 5:
                        original_z5 = gru1_cell.W_z5.weight.data.clone()
                        original_r5 = gru1_cell.W_r5.weight.data.clone()
                        original_h5 = gru1_cell.W_h5.weight.data.clone()
                        gru1_cell.W_z5.weight.data.zero_()
                        gru1_cell.W_r5.weight.data.zero_()
                        gru1_cell.W_h5.weight.data.zero_()
                    elif ablate_scale == 7:
                        original_z7 = gru1_cell.W_z7.weight.data.clone()
                        original_r7 = gru1_cell.W_r7.weight.data.clone()
                        original_h7 = gru1_cell.W_h7.weight.data.clone()
                        gru1_cell.W_z7.weight.data.zero_()
                        gru1_cell.W_r7.weight.data.zero_()
                        gru1_cell.W_h7.weight.data.zero_()
                
                logits, _ = model(x, padding, y)
                _, predicted = torch.max(logits, 1)
                
                total += y.size(0)
                correct += (predicted == y).sum().item()
                
                # Restore weights
                if ablate_scale:
                    if ablate_scale == 3:
                        gru1_cell.W_z3.weight.data = original_z3
                        gru1_cell.W_r3.weight.data = original_r3
                        gru1_cell.W_h3.weight.data = original_h3
                    elif ablate_scale == 5:
                        gru1_cell.W_z5.weight.data = original_z5
                        gru1_cell.W_r5.weight.data = original_r5
                        gru1_cell.W_h5.weight.data = original_h5
                    elif ablate_scale == 7:
                        gru1_cell.W_z7.weight.data = original_z7
                        gru1_cell.W_r7.weight.data = original_r7
                        gru1_cell.W_h7.weight.data = original_h7
        
        return correct / total
    
    # Full model
    print("\n📊 평가 중...")
    full_acc = evaluate_with_ablation(ablate_scale=None)
    print(f"  ✅ 전체 모델: {full_acc*100:.2f}%")
    
    # Ablate k=3
    acc_without_3 = evaluate_with_ablation(ablate_scale=3)
    drop_3 = (full_acc - acc_without_3) * 100
    print(f"  ❌ k=3 제거: {acc_without_3*100:.2f}% (성능 하락: {drop_3:.2f}%)")
    
    # Ablate k=5
    acc_without_5 = evaluate_with_ablation(ablate_scale=5)
    drop_5 = (full_acc - acc_without_5) * 100
    print(f"  ❌ k=5 제거: {acc_without_5*100:.2f}% (성능 하락: {drop_5:.2f}%)")
    
    # Ablate k=7
    acc_without_7 = evaluate_with_ablation(ablate_scale=7)
    drop_7 = (full_acc - acc_without_7) * 100
    print(f"  ❌ k=7 제거: {acc_without_7*100:.2f}% (성능 하락: {drop_7:.2f}%)")
    
    print("\n💡 결론:")
    drops = {'k=3': drop_3, 'k=5': drop_5, 'k=7': drop_7}
    most_critical = max(drops, key=drops.get)
    print(f"  가장 중요한 스케일: {most_critical} (제거 시 {drops[most_critical]:.2f}% 하락)")
    
    return {
        'full': full_acc,
        'without_3': acc_without_3,
        'without_5': acc_without_5,
        'without_7': acc_without_7,
        'drops': [drop_3, drop_5, drop_7]
    }

def main():
    print("""
╔════════════════════════════════════════════════════════════════════════════╗
║                  Scale-Aware GRU 스케일 중요도 분석                          ║
╚════════════════════════════════════════════════════════════════════════════╝
    """)
    
    # Load model
    print("\n📦 모델 로딩 중...")
    model = load_best_model(MSCSGRU_ScaleAware)
    
    if model is None:
        print("\n❌ 모델을 로드할 수 없습니다. 먼저 학습을 완료해주세요.")
        return
    
    # Analyze weights
    importance_data = analyze_scale_weights(model)
    
    # Visualize
    print("\n📊 시각화 생성 중...")
    visualize_scale_importance(importance_data)
    
    # Ablation study
    print("\n📦 데이터 로딩 중...")
    datamodule = DynamicDataModule(
        time_steps=87,
        batch_size=32,
        kfold=0,
        splits=5,
        seed=42
    )
    datamodule.setup()
    
    ablation_results = test_scale_ablation(model, datamodule)
    
    print("""
╔════════════════════════════════════════════════════════════════════════════╗
║                         분석 완료!                                          ║
╚════════════════════════════════════════════════════════════════════════════╝

생성된 파일:
  📊 scale_importance_analysis.png - 스케일 중요도 시각화

주요 발견사항:
  - 가중치 분석을 통해 각 스케일의 상대적 중요도 확인
  - Ablation study를 통해 각 스케일의 실제 기여도 측정
  - 이 정보는 모델 최적화 및 해석에 활용 가능
    """)

if __name__ == "__main__":
    main()

