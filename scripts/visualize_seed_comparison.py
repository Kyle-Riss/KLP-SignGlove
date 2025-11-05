#!/usr/bin/env python3
"""
5회 seed 실행 결과 시각화
각 seed별 성능 비교 및 분석
"""
import re
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def extract_seed_results():
    """로그 파일에서 seed별 결과 추출"""
    log_dir = Path('lightning_logs')
    if not log_dir.exists():
        return None
    
    # 모델 매핑
    model_mapping = {
        'ms3dgru': 'MS3DGRU',
        'gru': 'GRU',
        'stackedgru': 'StackedGRU',
        'ms3dstackedgru': 'MS3DStackedGRU'
    }
    
    # 데이터셋 매핑
    dataset_mapping = {
        'unified': 'Unified',
        'yubeen': 'Yubeen',
        'jaeyeon': 'Jaeyeon'
    }
    
    results = defaultdict(list)
    
    # multi_seed 로그 파일 처리
    for log_file in log_dir.glob('multi_seed_*.log'):
        try:
            filename = log_file.stem.replace('multi_seed_', '')
            
            # 모델명 추출
            model_key = None
            for key in model_mapping.keys():
                if filename.startswith(key):
                    model_key = key
                    break
            
            if model_key is None:
                continue
            
            model_name = model_mapping[model_key]
            
            # 데이터셋 추출
            dataset_key = None
            for key in dataset_mapping.keys():
                if key in filename:
                    dataset_key = key
                    break
            
            if dataset_key is None:
                continue
            
            dataset_name = dataset_mapping[dataset_key]
            
            # seed 추출
            seed_match = re.search(r'seed(\d+)_run(\d+)', filename)
            if not seed_match:
                continue
            
            seed = int(seed_match.group(1))
            run = int(seed_match.group(2))
            
            with open(log_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
                # test metrics 추출
                test_acc_match = re.search(r'test/accuracy\s*[│|\|]\s*([0-9.]+)', content)
                test_f1_match = re.search(r'test/f1_score\s*[│|\|]\s*([0-9.]+)', content)
                test_loss_match = re.search(r'test/loss\s*[│|\|]\s*([0-9.]+)', content)
                
                if test_acc_match:
                    key = f'{model_name}_{dataset_name}'
                    results[key].append({
                        'seed': seed,
                        'run': run,
                        'test_acc': float(test_acc_match.group(1)) * 100,
                        'test_f1': float(test_f1_match.group(1)) if test_f1_match else None,
                        'test_loss': float(test_loss_match.group(1)) if test_loss_match else None,
                    })
        except Exception as e:
            print(f'Error processing {log_file}: {e}')
            continue
    
    return results

def create_seed_comparison_visualization(results, output_dir='visualizations/seed_comparison'):
    """Seed별 비교 시각화 생성"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if not results:
        print("⚠️  No data found.")
        return
    
    # Seed 순서 정의
    seeds_order = [1337, 42, 1234, 5678, 9999]
    seed_names = [f'Seed {s}' for s in seeds_order]
    
    # 각 모델-데이터셋 조합별로 시각화
    for key, seed_results in sorted(results.items()):
        model, dataset = key.split('_')
        
        # Seed별로 정렬
        sorted_results = sorted(seed_results, key=lambda x: (x['seed'], x['run']))
        
        # 데이터 추출
        seeds = []
        accuracies = []
        f1_scores = []
        losses = []
        
        for result in sorted_results:
            seeds.append(result['seed'])
            accuracies.append(result['test_acc'])
            if result['test_f1']:
                f1_scores.append(result['test_f1'])
            if result['test_loss']:
                losses.append(result['test_loss'])
        
        # 1. Seed별 Accuracy 비교 (Bar Plot)
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'{model} - {dataset} Dataset: Seed Comparison', fontsize=16, fontweight='bold')
        
        # 1-1. Accuracy Bar Plot
        ax1 = axes[0, 0]
        seed_acc_dict = defaultdict(list)
        for seed, acc in zip(seeds, accuracies):
            seed_acc_dict[seed].append(acc)
        
        seed_means = [np.mean(seed_acc_dict[s]) for s in seeds_order if s in seed_acc_dict and len(seed_acc_dict[s]) > 0]
        seed_stds = [np.std(seed_acc_dict[s]) if len(seed_acc_dict[s]) > 1 else 0 for s in seeds_order if s in seed_acc_dict and len(seed_acc_dict[s]) > 0]
        seed_labels = [f'Seed {s}' for s in seeds_order if s in seed_acc_dict and len(seed_acc_dict[s]) > 0]
        
        if len(seed_means) > 0:
            x_pos = np.arange(len(seed_labels))
            bars = ax1.bar(x_pos, seed_means, yerr=seed_stds, capsize=5, alpha=0.7, color='steelblue', edgecolor='black')
            ax1.set_xlabel('Seed', fontsize=12)
            ax1.set_ylabel('Test Accuracy (%)', fontsize=12)
            ax1.set_title('Test Accuracy by Seed', fontsize=14, fontweight='bold')
            ax1.set_xticks(x_pos)
            ax1.set_xticklabels(seed_labels, rotation=45, ha='right')
            ax1.grid(axis='y', alpha=0.3)
            ax1.set_ylim([max(0, min(seed_means) - 10), min(100, max(seed_means) + 5)])
            
            # 값 표시
            for i, (bar, mean, std) in enumerate(zip(bars, seed_means, seed_stds)):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{mean:.2f}%\n±{std:.2f}',
                        ha='center', va='bottom', fontsize=9)
        else:
            ax1.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax1.transAxes, fontsize=12)
            ax1.set_title('Test Accuracy by Seed', fontsize=14, fontweight='bold')
        
        # 1-2. Accuracy Line Plot (Trend)
        ax2 = axes[0, 1]
        for seed in seeds_order:
            if seed in seed_acc_dict:
                seed_accs = seed_acc_dict[seed]
                ax2.plot([seed] * len(seed_accs), seed_accs, 'o-', label=f'Seed {seed}', linewidth=2, markersize=8)
        
        ax2.set_xlabel('Seed', fontsize=12)
        ax2.set_ylabel('Test Accuracy (%)', fontsize=12)
        ax2.set_title('Test Accuracy Trend by Seed', fontsize=14, fontweight='bold')
        ax2.set_xticks(seeds_order)
        ax2.set_xticklabels([f'Seed {s}' for s in seeds_order], rotation=45, ha='right')
        ax2.grid(alpha=0.3)
        ax2.legend(loc='best')
        
        # 1-3. F1-Score 비교 (if available)
        ax3 = axes[1, 0]
        if f1_scores:
            seed_f1_dict = defaultdict(list)
            for seed, f1 in zip(seeds, f1_scores):
                seed_f1_dict[seed].append(f1)
            
            seed_f1_means = [np.mean(seed_f1_dict[s]) for s in seeds_order if s in seed_f1_dict]
            seed_f1_stds = [np.std(seed_f1_dict[s]) if len(seed_f1_dict[s]) > 1 else 0 for s in seeds_order if s in seed_f1_dict]
            seed_f1_labels = [f'Seed {s}' for s in seeds_order if s in seed_f1_dict]
            
            x_pos_f1 = np.arange(len(seed_f1_labels))
            bars_f1 = ax3.bar(x_pos_f1, seed_f1_means, yerr=seed_f1_stds, capsize=5, alpha=0.7, color='coral', edgecolor='black')
            ax3.set_xlabel('Seed', fontsize=12)
            ax3.set_ylabel('Test F1-Score', fontsize=12)
            ax3.set_title('Test F1-Score by Seed', fontsize=14, fontweight='bold')
            ax3.set_xticks(x_pos_f1)
            ax3.set_xticklabels(seed_f1_labels, rotation=45, ha='right')
            ax3.grid(axis='y', alpha=0.3)
            
            # 값 표시
            for i, (bar, mean, std) in enumerate(zip(bars_f1, seed_f1_means, seed_f1_stds)):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height,
                        f'{mean:.4f}\n±{std:.4f}',
                        ha='center', va='bottom', fontsize=9)
        else:
            ax3.text(0.5, 0.5, 'F1-Score data not available', ha='center', va='center', transform=ax3.transAxes, fontsize=12)
            ax3.set_title('Test F1-Score by Seed', fontsize=14, fontweight='bold')
        
        # 1-4. Loss 비교 (if available)
        ax4 = axes[1, 1]
        if losses:
            seed_loss_dict = defaultdict(list)
            for seed, loss in zip(seeds, losses):
                seed_loss_dict[seed].append(loss)
            
            seed_loss_means = [np.mean(seed_loss_dict[s]) for s in seeds_order if s in seed_loss_dict]
            seed_loss_stds = [np.std(seed_loss_dict[s]) if len(seed_loss_dict[s]) > 1 else 0 for s in seeds_order if s in seed_loss_dict]
            seed_loss_labels = [f'Seed {s}' for s in seeds_order if s in seed_loss_dict]
            
            x_pos_loss = np.arange(len(seed_loss_labels))
            bars_loss = ax4.bar(x_pos_loss, seed_loss_means, yerr=seed_loss_stds, capsize=5, alpha=0.7, color='lightgreen', edgecolor='black')
            ax4.set_xlabel('Seed', fontsize=12)
            ax4.set_ylabel('Test Loss', fontsize=12)
            ax4.set_title('Test Loss by Seed', fontsize=14, fontweight='bold')
            ax4.set_xticks(x_pos_loss)
            ax4.set_xticklabels(seed_loss_labels, rotation=45, ha='right')
            ax4.grid(axis='y', alpha=0.3)
            
            # 값 표시
            for i, (bar, mean, std) in enumerate(zip(bars_loss, seed_loss_means, seed_loss_stds)):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height,
                        f'{mean:.4f}\n±{std:.4f}',
                        ha='center', va='bottom', fontsize=9)
        else:
            ax4.text(0.5, 0.5, 'Loss data not available', ha='center', va='center', transform=ax4.transAxes, fontsize=12)
            ax4.set_title('Test Loss by Seed', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        filename = f'seed_comparison_{model}_{dataset}.png'
        plt.savefig(output_path / filename, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {output_path / filename}")
        plt.close()
    
    # 2. 전체 모델 비교 (Heatmap)
    create_seed_heatmap(results, output_path)
    
    # 3. Seed별 성능 순위 비교
    create_seed_ranking(results, output_path)

def create_seed_heatmap(results, output_path):
    """Seed별 성능 Heatmap 생성"""
    seeds_order = [1337, 42, 1234, 5678, 9999]
    
    # 데이터 준비
    heatmap_data = []
    row_labels = []
    
    for key in sorted(results.keys()):
        model, dataset = key.split('_')
        seed_results = results[key]
        
        seed_acc_dict = defaultdict(list)
        for result in seed_results:
            seed_acc_dict[result['seed']].append(result['test_acc'])
        
        row_label = f'{model}\n{dataset}'
        row_data = []
        for seed in seeds_order:
            if seed in seed_acc_dict:
                row_data.append(np.mean(seed_acc_dict[seed]))
            else:
                row_data.append(np.nan)
        
        heatmap_data.append(row_data)
        row_labels.append(row_label)
    
    # Heatmap 생성
    fig, ax = plt.subplots(figsize=(12, 10))
    heatmap_data = np.array(heatmap_data)
    
    im = ax.imshow(heatmap_data, cmap='YlOrRd', aspect='auto')
    
    # 축 설정
    ax.set_xticks(np.arange(len(seeds_order)))
    ax.set_xticklabels([f'Seed {s}' for s in seeds_order])
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    
    # 값 표시
    for i in range(len(row_labels)):
        for j in range(len(seeds_order)):
            if not np.isnan(heatmap_data[i, j]):
                text = ax.text(j, i, f'{heatmap_data[i, j]:.1f}%',
                             ha="center", va="center", color="black", fontsize=9)
    
    ax.set_title('Test Accuracy Heatmap: Model-Dataset by Seed', fontsize=16, fontweight='bold', pad=20)
    plt.colorbar(im, ax=ax, label='Test Accuracy (%)')
    plt.tight_layout()
    plt.savefig(output_path / 'seed_heatmap.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path / 'seed_heatmap.png'}")
    plt.close()

def create_seed_ranking(results, output_path):
    """Seed별 성능 순위 비교"""
    seeds_order = [1337, 42, 1234, 5678, 9999]
    
    # 각 seed별 평균 성능 계산
    seed_performance = defaultdict(list)
    
    for key in results.keys():
        seed_results = results[key]
        seed_acc_dict = defaultdict(list)
        for result in seed_results:
            seed_acc_dict[result['seed']].append(result['test_acc'])
        
        for seed in seeds_order:
            if seed in seed_acc_dict:
                seed_performance[seed].append(np.mean(seed_acc_dict[seed]))
    
    # Seed별 평균 및 표준편차 계산
    seed_stats = {}
    for seed in seeds_order:
        if seed in seed_performance:
            seed_stats[seed] = {
                'mean': np.mean(seed_performance[seed]),
                'std': np.std(seed_performance[seed]),
                'min': np.min(seed_performance[seed]),
                'max': np.max(seed_performance[seed])
            }
    
    # 순위 비교 그래프
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Seed Performance Ranking', fontsize=16, fontweight='bold')
    
    # 1. 평균 성능 비교
    ax1 = axes[0, 0]
    seeds = [f'Seed {s}' for s in seeds_order if s in seed_stats]
    means = [seed_stats[s]['mean'] for s in seeds_order if s in seed_stats]
    stds = [seed_stats[s]['std'] for s in seeds_order if s in seed_stats]
    
    x_pos = np.arange(len(seeds))
    bars = ax1.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7, color='steelblue', edgecolor='black')
    ax1.set_xlabel('Seed', fontsize=12)
    ax1.set_ylabel('Average Test Accuracy (%)', fontsize=12)
    ax1.set_title('Average Performance by Seed (All Models-Datasets)', fontsize=14, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(seeds, rotation=45, ha='right')
    ax1.grid(axis='y', alpha=0.3)
    
    # 값 표시
    for bar, mean, std in zip(bars, means, stds):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{mean:.2f}%\n±{std:.2f}',
                ha='center', va='bottom', fontsize=9)
    
    # 2. 성능 범위 비교
    ax2 = axes[0, 1]
    mins = [seed_stats[s]['min'] for s in seeds_order if s in seed_stats]
    maxs = [seed_stats[s]['max'] for s in seeds_order if s in seed_stats]
    
    for i, (seed_name, mean, min_val, max_val) in enumerate(zip(seeds, means, mins, maxs)):
        ax2.plot([i, i], [min_val, max_val], 'o-', linewidth=3, markersize=8, color='steelblue', alpha=0.7)
        ax2.plot(i, mean, 'o', markersize=10, color='red', label='Mean' if i == 0 else '')
    
    ax2.set_xlabel('Seed', fontsize=12)
    ax2.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax2.set_title('Performance Range by Seed', fontsize=14, fontweight='bold')
    ax2.set_xticks(np.arange(len(seeds)))
    ax2.set_xticklabels(seeds, rotation=45, ha='right')
    ax2.grid(alpha=0.3)
    ax2.legend()
    
    # 3. Box Plot
    ax3 = axes[1, 0]
    box_data = [seed_performance[s] for s in seeds_order if s in seed_performance]
    box_labels = [f'Seed {s}' for s in seeds_order if s in seed_performance]
    
    bp = ax3.boxplot(box_data, labels=box_labels, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
        patch.set_alpha(0.7)
    
    ax3.set_xlabel('Seed', fontsize=12)
    ax3.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax3.set_title('Performance Distribution by Seed', fontsize=14, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    plt.setp(ax3.get_xticklabels(), rotation=45, ha='right')
    
    # 4. Violin Plot
    ax4 = axes[1, 1]
    parts = ax4.violinplot(box_data, positions=range(len(box_labels)), showmeans=True, showmedians=True)
    for pc in parts['bodies']:
        pc.set_facecolor('lightcoral')
        pc.set_alpha(0.7)
    
    ax4.set_xlabel('Seed', fontsize=12)
    ax4.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax4.set_title('Performance Distribution (Violin Plot)', fontsize=14, fontweight='bold')
    ax4.set_xticks(range(len(box_labels)))
    ax4.set_xticklabels(box_labels, rotation=45, ha='right')
    ax4.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / 'seed_ranking.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path / 'seed_ranking.png'}")
    plt.close()

def main():
    print("=" * 80)
    print("Seed Comparison Visualization")
    print("=" * 80)
    print()
    
    print("📊 Extracting seed results from log files...")
    results = extract_seed_results()
    
    if not results:
        print("⚠️  No data found. Please check log files.")
        return
    
    print(f"✅ Found {len(results)} model-dataset combinations")
    print()
    
    print("🎨 Generating visualizations...")
    create_seed_comparison_visualization(results)
    
    print()
    print("=" * 80)
    print("✅ Complete!")
    print("=" * 80)

if __name__ == '__main__':
    main()

