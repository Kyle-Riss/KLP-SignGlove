"""
A-GRU 계산 효율성 분석 (수정된 버전)
Latency, FLOPs, Parameters 비교: GRU vs A-GRU vs MS-CSGRU
"""

import torch
import time
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from src.models.GRUModels import StackedGRU
from src.models.AGRUModels import AGRUModel
from src.models.MSCSGRUModels import MSCSGRU

# 출력 디렉토리
OUTPUT_DIR = Path("visualizations/efficiency_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def count_parameters(model):
    """모델의 총 파라미터 수 계산"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def estimate_flops(model, input_shape=(1, 87, 8)):
    """
    FLOPs 추정 (간단한 방법)
    """
    # 더미 입력 생성
    x = torch.randn(input_shape)
    x_padding = torch.zeros(input_shape[0], input_shape[1])
    y_targets = torch.randint(0, 24, (input_shape[0],))
    
    # 간단한 FLOPs 추정 (정확한 계산 대신 근사치)
    batch_size, seq_len, input_size = input_shape
    
    # 모델별 FLOPs 추정
    if model.__class__.__name__ == 'AGRUModel':
        # A-GRU: GRU + A-Net
        # GRU: 3 * (input_size + hidden_size + 1) * hidden_size * batch * seq
        # A-Net: (input_size + hidden_size + 1) * batch * seq
        hidden_size = 64  # A-GRU 기본값
        gru_flops = 3 * (input_size + hidden_size + 1) * hidden_size * batch_size * seq_len
        anet_flops = (input_size + hidden_size + 1) * batch_size * seq_len
        total_flops = gru_flops + anet_flops
    elif model.__class__.__name__ == 'StackedGRU':
        # Stacked GRU: 2 layers
        hidden_size = 64
        gru_flops = 2 * 3 * (input_size + hidden_size + 1) * hidden_size * batch_size * seq_len
        # Classifier
        classifier_flops = hidden_size * 24 * batch_size  # 24 classes
        total_flops = gru_flops + classifier_flops
    elif model.__class__.__name__ == 'MSCSGRU':
        # MS-CSGRU: CNN + GRU
        # CNN: 3 towers with different kernel sizes
        cnn_flops = 3 * (3*3*8*32 + 5*5*8*32 + 7*7*8*32) * batch_size * seq_len  # 3 towers
        # GRU
        hidden_size = 64
        gru_flops = 3 * (96 + hidden_size + 1) * hidden_size * batch_size * seq_len  # 96 = 3*32
        # Classifier
        classifier_flops = hidden_size * 24 * batch_size
        total_flops = cnn_flops + gru_flops + classifier_flops
    else:
        # 기본 GRU
        hidden_size = 64
        total_flops = 3 * (input_size + hidden_size + 1) * hidden_size * batch_size * seq_len
    
    return total_flops


def measure_latency(model, input_shape=(1, 87, 8), num_runs=100, warmup=10):
    """
    추론 지연 시간 측정 (CPU) - 수정된 버전
    """
    model.eval()
    device = torch.device('cpu')
    model = model.to(device)
    
    # 더미 데이터
    x = torch.randn(input_shape, device=device)
    x_padding = torch.zeros(input_shape[0], input_shape[1], device=device)
    y_targets = torch.randint(0, 24, (input_shape[0],), device=device)
    
    # 모델 타입에 따른 호출 방식 결정
    model_name = model.__class__.__name__
    
    def forward_pass():
        if model_name == 'AGRUModel':
            return model(x, x_padding, y_targets)
        else:
            return model(x)
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            try:
                _ = forward_pass()
            except Exception as e:
                print(f"Warmup error: {e}")
                pass
    
    # 측정
    latencies = []
    with torch.no_grad():
        for _ in tqdm(range(num_runs), desc="Measuring latency"):
            start_time = time.perf_counter()
            try:
                _ = forward_pass()
            except Exception as e:
                print(f"Forward pass error: {e}")
                pass
            end_time = time.perf_counter()
            latencies.append((end_time - start_time) * 1000)  # ms로 변환
    
    return np.mean(latencies), np.std(latencies)


def create_models():
    """비교할 모델들 생성"""
    models = {}
    
    # 1. Standard GRU (Stacked)
    print("📦 Creating Standard GRU...")
    gru_model = StackedGRU(
        learning_rate=0.001,
        input_size=8,
        hidden_size=64,
        classes=24,
        layers=2,
        dropout=0.1
    )
    models['GRU'] = gru_model
    
    # 2. A-GRU
    print("📦 Creating A-GRU...")
    agru_model = AGRUModel(
        learning_rate=0.001,
        input_size=8,
        hidden_size=64,
        num_layers=2,
        num_classes=24,
        dropout=0.1,
        gamma=1.0
    )
    models['A-GRU'] = agru_model
    
    # 3. MS-CSGRU
    print("📦 Creating MS-CSGRU...")
    mscsgru_model = MSCSGRU(
        learning_rate=0.001,
        input_size=8,
        num_classes=24,
        dropout=0.1
    )
    models['MS-CSGRU'] = mscsgru_model
    
    return models


def analyze_efficiency():
    """계산 효율성 분석"""
    print("🚀 A-GRU 계산 효율성 분석 시작...")
    
    # 모델 생성
    models = create_models()
    
    # 결과 저장
    results = {}
    
    print("\n" + "="*60)
    print("📊 계산 효율성 분석")
    print("="*60)
    
    for name, model in models.items():
        print(f"\n🔍 Analyzing {name}...")
        
        # Parameters
        params = count_parameters(model)
        print(f"   Parameters: {params:,}")
        
        # FLOPs
        flops = estimate_flops(model)
        print(f"   FLOPs: {flops:,}")
        
        # Latency
        latency_mean, latency_std = measure_latency(model)
        print(f"   Latency: {latency_mean:.3f} ± {latency_std:.3f} ms")
        
        results[name] = {
            'parameters': params,
            'flops': flops,
            'latency_mean': latency_mean,
            'latency_std': latency_std
        }
    
    return results


def create_visualizations(results):
    """시각화 생성"""
    print("\n🎨 시각화 생성 중...")
    
    # 데이터 준비
    models = list(results.keys())
    params = [results[m]['parameters'] for m in models]
    flops = [results[m]['flops'] for m in models]
    latencies = [results[m]['latency_mean'] for m in models]
    
    # 1. 효율성 비교 플롯
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('A-GRU vs Baseline Models: Computational Efficiency', fontsize=16, fontweight='bold')
    
    # Parameters 비교
    axes[0, 0].bar(models, params, color=['skyblue', 'lightcoral', 'lightgreen'])
    axes[0, 0].set_title('Parameters Comparison', fontweight='bold')
    axes[0, 0].set_ylabel('Parameters')
    axes[0, 0].tick_params(axis='x', rotation=45)
    for i, v in enumerate(params):
        axes[0, 0].text(i, v + max(params)*0.01, f'{v:,}', ha='center', va='bottom')
    
    # FLOPs 비교
    axes[0, 1].bar(models, flops, color=['skyblue', 'lightcoral', 'lightgreen'])
    axes[0, 1].set_title('FLOPs Comparison', fontweight='bold')
    axes[0, 1].set_ylabel('FLOPs')
    axes[0, 1].tick_params(axis='x', rotation=45)
    for i, v in enumerate(flops):
        axes[0, 1].text(i, v + max(flops)*0.01, f'{v:,}', ha='center', va='bottom')
    
    # Latency 비교
    axes[1, 0].bar(models, latencies, color=['skyblue', 'lightcoral', 'lightgreen'])
    axes[1, 0].set_title('Latency Comparison', fontweight='bold')
    axes[1, 0].set_ylabel('Latency (ms)')
    axes[1, 0].tick_params(axis='x', rotation=45)
    for i, v in enumerate(latencies):
        axes[1, 0].text(i, v + max(latencies)*0.01, f'{v:.2f}ms', ha='center', va='bottom')
    
    # 상대 효율성 (GRU 기준)
    gru_params = results['GRU']['parameters']
    gru_flops = results['GRU']['flops']
    gru_latency = results['GRU']['latency_mean']
    
    param_ratios = [results[m]['parameters'] / gru_params for m in models]
    flop_ratios = [results[m]['flops'] / gru_flops for m in models]
    latency_ratios = [results[m]['latency_mean'] / gru_latency for m in models]
    
    x = np.arange(len(models))
    width = 0.25
    
    axes[1, 1].bar(x - width, param_ratios, width, label='Parameters', color='skyblue')
    axes[1, 1].bar(x, flop_ratios, width, label='FLOPs', color='lightcoral')
    axes[1, 1].bar(x + width, latency_ratios, width, label='Latency', color='lightgreen')
    
    axes[1, 1].set_title('Relative Efficiency (vs GRU)', fontweight='bold')
    axes[1, 1].set_ylabel('Ratio')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(models)
    axes[1, 1].legend()
    axes[1, 1].axhline(y=1.0, color='red', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'efficiency_comparison_fixed.png', dpi=300, bbox_inches='tight')
    print(f"✅ 효율성 비교 플롯 저장: {OUTPUT_DIR / 'efficiency_comparison_fixed.png'}")
    
    # 2. 효율성 히트맵
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 데이터 정규화 (0-1 스케일)
    data = np.array([
        [p/max(params) for p in params],
        [f/max(flops) for f in flops],
        [l/max(latencies) for l in latencies]
    ])
    
    im = ax.imshow(data, cmap='RdYlGn_r', aspect='auto')
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models)
    ax.set_yticks(range(3))
    ax.set_yticklabels(['Parameters', 'FLOPs', 'Latency'])
    
    # 값 표시
    for i in range(3):
        for j in range(len(models)):
            text = ax.text(j, i, f'{data[i, j]:.2f}', ha="center", va="center", color="black", fontweight='bold')
    
    ax.set_title('Normalized Efficiency Heatmap\n(Lower is Better)', fontweight='bold')
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'efficiency_heatmap_fixed.png', dpi=300, bbox_inches='tight')
    print(f"✅ 효율성 히트맵 저장: {OUTPUT_DIR / 'efficiency_heatmap_fixed.png'}")
    
    plt.close()


def print_summary(results):
    """결과 요약 출력"""
    print("\n" + "="*80)
    print("📊 효율성 분석 결과 요약")
    print("="*80)
    print(f"{'Model':<15} {'Parameters':<20} {'FLOPs':<20} {'Latency (ms)':<15}")
    print("-" * 80)
    
    for name, result in results.items():
        params = result['parameters']
        flops = result['flops']
        latency = result['latency_mean']
        print(f"{name:<15} {params:,} ({params/1000:.1f}K) {flops:,} ({flops/1000000:.1f}M) {latency:.3f} ± {result['latency_std']:.3f}")
    
    print("\n" + "="*80)
    print("📈 GRU 대비 상대 효율성")
    print("="*80)
    print(f"{'Model':<15} {'Params Ratio':<15} {'FLOPs Ratio':<15} {'Latency Ratio':<15}")
    print("-" * 80)
    
    gru_params = results['GRU']['parameters']
    gru_flops = results['GRU']['flops']
    gru_latency = results['GRU']['latency_mean']
    
    for name, result in results.items():
        param_ratio = result['parameters'] / gru_params
        flop_ratio = result['flops'] / gru_flops
        latency_ratio = result['latency_mean'] / gru_latency
        print(f"{name:<15} {param_ratio:.3f}x {flop_ratio:.3f}x {latency_ratio:.3f}x")
    
    print("\n" + "="*80)
    print("🏆 A-GRU 효율성 분석")
    print("="*80)
    
    agru_params = results['A-GRU']['parameters']
    agru_latency = results['A-GRU']['latency_mean']
    mscsgru_params = results['MS-CSGRU']['parameters']
    mscsgru_latency = results['MS-CSGRU']['latency_mean']
    
    print(f"A-GRU vs Standard GRU:")
    print(f"  Parameters overhead: +{((agru_params/gru_params-1)*100):.1f}% (+{agru_params-gru_params:,} params)")
    print(f"  Latency overhead: +{((agru_latency/gru_latency-1)*100):.1f}%")
    print()
    print(f"A-GRU vs MS-CSGRU:")
    print(f"  Parameters saving: {((1-agru_params/mscsgru_params)*100):.1f}% ({mscsgru_params-agru_params:,} params)")
    print(f"  Latency overhead: +{((agru_latency/mscsgru_latency-1)*100):.1f}%")


def save_results(results):
    """결과 저장"""
    print("\n💾 결과 저장 중...")
    
    # NumPy 배열로 저장
    np.savez(OUTPUT_DIR / 'efficiency_results_fixed.npz', 
             models=list(results.keys()),
             parameters=[results[m]['parameters'] for m in results.keys()],
             flops=[results[m]['flops'] for m in results.keys()],
             latencies=[results[m]['latency_mean'] for m in results.keys()])
    
    print(f"✅ 분석 완료!")
    print(f"📁 결과 저장 위치: {OUTPUT_DIR}/")


def main():
    """메인 함수"""
    # 효율성 분석
    results = analyze_efficiency()
    
    # 시각화 생성
    create_visualizations(results)
    
    # 결과 요약
    print_summary(results)
    
    # 결과 저장
    save_results(results)


if __name__ == "__main__":
    main()
