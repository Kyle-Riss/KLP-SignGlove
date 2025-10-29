"""
A-GRU 노이즈 견고성 분석
Accuracy degradation under various noise levels
"""

import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from src.models.AGRUModels import AGRUModel
from src.models.GRUModels import StackedGRU, GRU
from src.models.MSCSGRUModels import MSCSGRU
from src.misc.DynamicDataModule import DynamicDataModule

# 출력 디렉토리
OUTPUT_DIR = Path("visualizations/noise_robustness")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def add_gaussian_noise(x, noise_level):
    """
    가우시안 노이즈 추가
    
    Args:
        x: (batch, time, features) 입력 데이터
        noise_level: 노이즈 강도 (0.0 ~ 1.0)
    
    Returns:
        노이즈가 추가된 데이터
    """
    noise = torch.randn_like(x) * noise_level
    return x + noise


def evaluate_with_noise(model, datamodule, noise_level=0.0):
    """
    특정 노이즈 레벨에서 모델 성능 평가
    
    Args:
        model: 평가할 모델
        datamodule: 데이터 모듈
        noise_level: 노이즈 강도
    
    Returns:
        accuracy: 정확도
    """
    datamodule.setup('test')
    test_loader = datamodule.test_dataloader()
    
    model.eval()
    device = torch.device('cpu')
    model = model.to(device)
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in test_loader:
            x = batch['measurement'].to(device)
            y = batch['label'].to(device)
            
            # padding 키 확인 (LightningModel.py와 동일하게)
            x_padding = batch.get('measurement_padding', None)
            if x_padding is not None:
                x_padding = x_padding.to(device)
            else:
                x_padding = torch.zeros(x.size(0), x.size(1), device=device)
            
            # 노이즈 추가
            if noise_level > 0:
                x = add_gaussian_noise(x, noise_level)
            
            # 예측 (모든 LightningModel은 동일한 forward signature)
            # forward(x, x_padding, y) - label 필요
            logits, _ = model(x, x_padding, y)
            
            pred = torch.argmax(logits, dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    
    accuracy = correct / total if total > 0 else 0
    return accuracy


def analyze_noise_robustness(models, datamodule, noise_levels):
    """
    여러 노이즈 레벨에서 모델 성능 분석
    
    Args:
        models: dict of models {name: model}
        datamodule: 데이터 모듈
        noise_levels: list of noise levels
    
    Returns:
        results: dict of {model_name: [accuracies]}
    """
    results = {}
    
    print("\n" + "="*60)
    print("📊 노이즈 견고성 분석")
    print("="*60)
    
    for name, model in models.items():
        print(f"\n🔍 Analyzing {name}...")
        accuracies = []
        
        for noise_level in tqdm(noise_levels, desc=f"  {name}"):
            acc = evaluate_with_noise(model, datamodule, noise_level)
            accuracies.append(acc * 100)  # Convert to percentage
            
        results[name] = accuracies
        
        # 결과 출력
        print(f"  Clean (0.0): {accuracies[0]:.2f}%")
        print(f"  Noisy (0.1): {accuracies[-1]:.2f}%")
        print(f"  Degradation: {accuracies[0] - accuracies[-1]:.2f}%p")
    
    return results


def plot_noise_robustness(results, noise_levels):
    """노이즈 견고성 플롯"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Color palette
    colors = {'GRU': '#3498db', 'A-GRU': '#e74c3c', 'MS-CSGRU': '#2ecc71'}
    
    # 1. Accuracy vs Noise Level
    ax = axes[0]
    for name, accuracies in results.items():
        ax.plot(noise_levels, accuracies, marker='o', linewidth=2, 
                label=name, color=colors.get(name, 'gray'))
    
    ax.set_xlabel('Noise Level (σ)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Accuracy vs Noise Level', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)
    
    # 2. Accuracy Degradation
    ax = axes[1]
    
    model_names = list(results.keys())
    clean_accs = [results[name][0] for name in model_names]
    noisy_accs = [results[name][-1] for name in model_names]
    degradations = [clean_accs[i] - noisy_accs[i] for i in range(len(model_names))]
    
    x_pos = np.arange(len(model_names))
    bars = ax.bar(x_pos, degradations, color=[colors.get(name, 'gray') for name in model_names],
                  alpha=0.7, edgecolor='black')
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(model_names)
    ax.set_ylabel('Accuracy Degradation (%p)', fontsize=12, fontweight='bold')
    ax.set_title('Accuracy Degradation (Clean → Noise 0.1)', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # 값 표시
    for bar, val in zip(bars, degradations):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}%p',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / "noise_robustness.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ 노이즈 견고성 플롯 저장: {output_path}")
    plt.close()


def plot_noise_sensitivity_heatmap(results, noise_levels):
    """노이즈 민감도 히트맵"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    model_names = list(results.keys())
    data = np.array([results[name] for name in model_names])
    
    # 히트맵 생성
    sns.heatmap(data, annot=True, fmt='.1f', cmap='RdYlGn',
                xticklabels=[f'{level:.2f}' for level in noise_levels],
                yticklabels=model_names,
                cbar_kws={'label': 'Accuracy (%)'},
                vmin=0, vmax=100,
                ax=ax)
    
    ax.set_xlabel('Noise Level (σ)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Model', fontsize=12, fontweight='bold')
    ax.set_title('Accuracy under Different Noise Levels', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / "noise_sensitivity_heatmap.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ 노이즈 민감도 히트맵 저장: {output_path}")
    plt.close()


def plot_relative_robustness(results, noise_levels):
    """상대적 견고성 플롯 (GRU 기준)"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # GRU를 기준으로 상대적 성능 계산
    gru_accs = results['GRU']
    
    colors = {'A-GRU': '#e74c3c', 'MS-CSGRU': '#2ecc71'}
    
    for name, accuracies in results.items():
        if name == 'GRU':
            continue
        
        # 상대적 성능 (GRU 대비 차이)
        relative_perf = [acc - gru_acc for acc, gru_acc in zip(accuracies, gru_accs)]
        
        ax.plot(noise_levels, relative_perf, marker='o', linewidth=2,
                label=f'{name} vs GRU', color=colors.get(name, 'gray'))
    
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlabel('Noise Level (σ)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy Difference vs GRU (%p)', fontsize=12, fontweight='bold')
    ax.set_title('Relative Performance vs Baseline GRU', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / "relative_robustness.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ 상대적 견고성 플롯 저장: {output_path}")
    plt.close()


def print_summary_table(results, noise_levels):
    """결과 요약 테이블 출력"""
    print("\n" + "="*80)
    print("📊 노이즈 견고성 분석 결과 요약")
    print("="*80)
    
    # 헤더
    print(f"{'Model':<15} {'Clean (0.0)':<15} {'Noise (0.05)':<15} {'Noise (0.1)':<15} {'Degradation':<15}")
    print("-"*80)
    
    # 각 모델
    for name, accuracies in results.items():
        clean = accuracies[0]
        noise_05 = accuracies[len(accuracies)//2] if len(accuracies) > 1 else accuracies[0]
        noise_10 = accuracies[-1]
        degradation = clean - noise_10
        
        print(f"{name:<15} {clean:>13.2f}% {noise_05:>13.2f}% {noise_10:>13.2f}% {degradation:>13.2f}%p")
    
    print("="*80)
    
    # 최고 견고성 모델
    print("\n" + "="*80)
    print("🏆 노이즈 견고성 순위")
    print("="*80)
    
    degradations = {name: accuracies[0] - accuracies[-1] 
                   for name, accuracies in results.items()}
    sorted_models = sorted(degradations.items(), key=lambda x: x[1])
    
    for rank, (name, deg) in enumerate(sorted_models, 1):
        symbol = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "  "
        print(f"{symbol} {rank}. {name:<15}: {deg:.2f}%p degradation")
    
    print("="*80)
    
    # A-GRU 분석
    if 'A-GRU' in results:
        print("\n" + "="*80)
        print("🔬 A-GRU 견고성 분석")
        print("="*80)
        
        agru_deg = degradations['A-GRU']
        gru_deg = degradations.get('GRU', 0)
        
        improvement = gru_deg - agru_deg
        improvement_pct = (improvement / gru_deg * 100) if gru_deg > 0 else 0
        
        print(f"A-GRU degradation: {agru_deg:.2f}%p")
        print(f"GRU degradation: {gru_deg:.2f}%p")
        print(f"Improvement: {improvement:.2f}%p ({improvement_pct:+.1f}%)")
        
        if improvement > 0:
            print(f"\n✅ A-GRU는 GRU보다 {improvement:.2f}%p 더 견고합니다!")
        elif improvement < 0:
            print(f"\n⚠️ A-GRU는 GRU보다 {abs(improvement):.2f}%p 덜 견고합니다.")
        else:
            print(f"\n➖ A-GRU와 GRU의 견고성은 비슷합니다.")
        
        print("="*80)


def load_trained_models():
    """학습된 모델들 로드"""
    models = {}
    
    # 1. GRU
    print("📦 Loading GRU...")
    try:
        # GRU 모델은 Linear + GRU 구조로 저장됨
        gru = GRU.load_from_checkpoint(
            "checkpoints/GRU_best.ckpt",
            learning_rate=0.001,
            input_size=8,
            hidden_size=64,
            classes=24,
            layers=2,
            dropout=0.3
        )
        models['GRU'] = gru
        print("   ✅ GRU loaded")
    except Exception as e:
        print(f"   ⚠️ GRU not found, creating new: {e}")
        gru = GRU(
            learning_rate=0.001,
            input_size=8,
            hidden_size=64,
            classes=24,
            layers=2,
            dropout=0.3
        )
        models['GRU'] = gru
    
    # 2. A-GRU
    print("📦 Loading A-GRU...")
    try:
        agru = AGRUModel.load_from_checkpoint(
            "checkpoints/AGRU_best.ckpt",
            learning_rate=0.001,
            input_size=8,
            hidden_size=64,
            classes=24,
            layers=2,
            dropout=0.3,
            gamma=1.0
        )
        models['A-GRU'] = agru
        print("   ✅ A-GRU loaded (Test Acc: 99.65%)")
    except Exception as e:
        print(f"   ❌ A-GRU load failed: {e}")
        return None
    
    # 3. MS-CSGRU
    print("📦 Loading MS-CSGRU...")
    try:
        mscsgru = MSCSGRU.load_from_checkpoint(
            "checkpoints/MSCSGRU_best.ckpt",
            learning_rate=0.001,
            input_size=8,
            hidden_size=64,
            classes=24,
            layers=2,
            dropout=0.3
        )
        models['MS-CSGRU'] = mscsgru
        print("   ✅ MS-CSGRU loaded")
    except Exception as e:
        print(f"   ⚠️ MS-CSGRU not found, creating new: {e}")
        mscsgru = MSCSGRU(
            learning_rate=0.001,
            input_size=8,
            hidden_size=64,
            classes=24,
            layers=2,
            dropout=0.3
        )
        models['MS-CSGRU'] = mscsgru
    
    return models


def main():
    print("🚀 A-GRU 노이즈 견고성 분석 시작...\n")
    
    # 1. 모델 로드
    models = load_trained_models()
    if models is None or 'A-GRU' not in models:
        print("❌ A-GRU 모델을 찾을 수 없습니다!")
        return
    
    # 2. 데이터 로드
    print("\n📂 데이터 로드 중...")
    datamodule = DynamicDataModule(
        data_dir='/home/billy/25-1kp/SignGlove_HW/datasets/unified',
        batch_size=32,
        test_size=0.2,
        val_size=0.2,
        use_test_split=True
    )
    print("✅ 데이터 로드 완료\n")
    
    # 3. 노이즈 레벨 설정
    noise_levels = [0.0, 0.02, 0.04, 0.06, 0.08, 0.1]
    
    # 4. 노이즈 견고성 분석
    results = analyze_noise_robustness(models, datamodule, noise_levels)
    
    # 5. 시각화
    print("\n🎨 시각화 생성 중...")
    plot_noise_robustness(results, noise_levels)
    plot_noise_sensitivity_heatmap(results, noise_levels)
    plot_relative_robustness(results, noise_levels)
    
    # 6. 결과 요약
    print_summary_table(results, noise_levels)
    
    # 7. 결과 저장
    print(f"\n💾 결과 저장 중...")
    np.savez(OUTPUT_DIR / "noise_robustness_results.npz",
             results=results,
             noise_levels=noise_levels)
    
    print(f"\n✅ 분석 완료!")
    print(f"📁 결과 저장 위치: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()

