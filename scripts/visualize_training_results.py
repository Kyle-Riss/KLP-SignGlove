#!/usr/bin/env python3
"""
학습 결과 시각화 스크립트
모델별, 데이터셋별 성능 비교
"""
import os
import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def extract_results_from_logs():
    """로그 파일에서 결과 추출"""
    results = {}
    log_dir = Path("lightning_logs")
    
    for log_file in log_dir.glob("seq_*.log"):
        if not log_file.exists():
            continue
            
        # 파일명에서 모델과 데이터셋 추출
        name = log_file.stem.replace("seq_", "")
        
        # 모델과 데이터셋 분리 (순서 중요: 더 긴 이름 먼저 체크)
        if "ms3dstk" in name:
            model = "MS3DStackedGRU"
            dataset = name.replace("ms3dstk_", "")
        elif "stackedgru" in name:
            model = "StackedGRU"
            dataset = name.replace("stackedgru_", "")
        elif "ms3dgru" in name:
            model = "MS3DGRU"
            dataset = name.replace("ms3dgru_", "")
        elif "gru_" in name:
            model = "GRU"
            dataset = name.replace("gru_", "")
        else:
            continue
        
        # 로그 파일에서 정확도 추출
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                content = f.read()
                if "[DONE]" not in content:
                    continue
                
                # test/accuracy 찾기
                acc_match = re.search(r'test/accuracy\s*\│\s*([0-9]+\.[0-9]+)', content)
                if acc_match:
                    accuracy = float(acc_match.group(1))
                    if model not in results:
                        results[model] = {}
                    results[model][dataset] = accuracy
        except Exception as e:
            print(f"Error reading {log_file}: {e}")
            continue
    
    return results

def create_visualizations(results):
    """시각화 생성"""
    output_dir = Path("visualizations/training_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 데이터셋 이름 매핑
    dataset_names = {
        "unified": "Unified",
        "yubeen": "Yubeen",
        "jaeyeon": "Jaeyeon"
    }
    
    models = list(results.keys())
    datasets = ["unified", "yubeen", "jaeyeon"]
    
    # 1. 모델별 성능 비교 (데이터셋별)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Model Performance Comparison by Dataset', fontsize=16, fontweight='bold')
    
    for idx, dataset in enumerate(datasets):
        ax = axes[idx]
        model_accs = []
        model_names = []
        
        for model in models:
            if dataset in results.get(model, {}):
                model_accs.append(results[model][dataset] * 100)
                model_names.append(model)
        
        if model_accs:
            bars = ax.bar(model_names, model_accs, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
            ax.set_title(f'{dataset_names[dataset]} Dataset', fontsize=12, fontweight='bold')
            ax.set_ylabel('Test Accuracy (%)', fontsize=11)
            ax.set_ylim(90, 100)
            ax.grid(axis='y', alpha=0.3)
            
            # 값 표시
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}%',
                       ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'model_comparison_by_dataset.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_dir / 'model_comparison_by_dataset.png'}")
    
    # 2. 데이터셋별 성능 비교 (모델별)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Model Performance Comparison by Model', fontsize=16, fontweight='bold')
    axes = axes.flatten()
    
    for idx, model in enumerate(models):
        if idx >= 4:
            break
        ax = axes[idx]
        dataset_accs = []
        dataset_labels = []
        
        for dataset in datasets:
            if dataset in results.get(model, {}):
                dataset_accs.append(results[model][dataset] * 100)
                dataset_labels.append(dataset_names[dataset])
        
        if dataset_accs:
            bars = ax.bar(dataset_labels, dataset_accs, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
            ax.set_title(f'{model}', fontsize=12, fontweight='bold')
            ax.set_ylabel('Test Accuracy (%)', fontsize=11)
            ax.set_ylim(90, 100)
            ax.grid(axis='y', alpha=0.3)
            
            # 값 표시
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}%',
                       ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'model_comparison_by_model.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_dir / 'model_comparison_by_model.png'}")
    
    # 3. 히트맵
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 데이터 준비
    heatmap_data = []
    for model in models:
        row = []
        for dataset in datasets:
            if dataset in results.get(model, {}):
                row.append(results[model][dataset] * 100)
            else:
                row.append(np.nan)
        heatmap_data.append(row)
    
    heatmap_data = np.array(heatmap_data)
    
    im = ax.imshow(heatmap_data, cmap='YlGnBu', aspect='auto', vmin=90, vmax=100)
    
    # 축 레이블
    ax.set_xticks(np.arange(len(datasets)))
    ax.set_yticks(np.arange(len(models)))
    ax.set_xticklabels([dataset_names[d] for d in datasets])
    ax.set_yticklabels(models)
    
    # 값 표시
    for i in range(len(models)):
        for j in range(len(datasets)):
            if not np.isnan(heatmap_data[i, j]):
                text = ax.text(j, i, f'{heatmap_data[i, j]:.2f}%',
                             ha="center", va="center", color="black", fontweight='bold')
    
    ax.set_title('Model Performance Heatmap (Test Accuracy %)', fontsize=14, fontweight='bold', pad=20)
    plt.colorbar(im, ax=ax, label='Test Accuracy (%)')
    plt.tight_layout()
    plt.savefig(output_dir / 'performance_heatmap.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_dir / 'performance_heatmap.png'}")
    
    # 4. 결과 테이블 저장
    results_table = []
    for model in models:
        for dataset in datasets:
            if dataset in results.get(model, {}):
                results_table.append({
                    'Model': model,
                    'Dataset': dataset_names[dataset],
                    'Accuracy (%)': f"{results[model][dataset] * 100:.2f}"
                })
    
    # JSON으로 저장
    with open(output_dir / 'results.json', 'w', encoding='utf-8') as f:
        json.dump(results_table, f, indent=2, ensure_ascii=False)
    print(f"✅ Saved: {output_dir / 'results.json'}")
    
    # 텍스트 테이블도 저장
    with open(output_dir / 'results_table.txt', 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("Training Results Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"{'Model':<20} {'Dataset':<15} {'Accuracy (%)':<15}\n")
        f.write("-" * 60 + "\n")
        for row in results_table:
            f.write(f"{row['Model']:<20} {row['Dataset']:<15} {row['Accuracy (%)']:<15}\n")
    
    print(f"✅ Saved: {output_dir / 'results_table.txt'}")
    
    plt.close('all')

def main():
    print("=" * 60)
    print("Training Results Visualization")
    print("=" * 60)
    print()
    
    # 결과 추출
    print("📊 Extracting results from log files...")
    results = extract_results_from_logs()
    
    if not results:
        print("❌ No results found in log files!")
        return
    
    print(f"✅ Found results for {len(results)} models")
    for model, datasets in results.items():
        print(f"   - {model}: {len(datasets)} datasets")
    
    print()
    print("📈 Creating visualizations...")
    create_visualizations(results)
    
    print()
    print("=" * 60)
    print("✅ Visualization complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()

