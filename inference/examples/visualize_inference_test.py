#!/usr/bin/env python3
"""
4개 모델 추론 테스트 결과 시각화
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from inference import SignGloveInference

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def test_all_models_and_visualize():
    """4개 모델 추론 테스트 및 시각화"""
    
    print("=" * 80)
    print("🚀 4개 모델 추론 테스트 및 시각화")
    print("=" * 80)
    print()
    
    # 모델 설정
    models_config = {
        'GRU': {
            'path': 'archive/checkpoints_backup/checkpoints_backup/GRU_best.ckpt',
            'type': 'GRU',
            'hidden_size': 64,
            'dropout': 0.2,
            'expected_acc': '98.36%'
        },
        'StackedGRU': {
            'path': 'archive/checkpoints_backup/checkpoints_backup/GRU_best.ckpt',
            'type': 'StackedGRU',
            'hidden_size': 64,
            'dropout': 0.2,
            'expected_acc': '95.43%'
        },
        'MS3DGRU': {
            'path': 'best_model/ms3dgru_best.ckpt',
            'type': 'MS3DGRU',
            'cnn_filters': 32,
            'dropout': 0.1,
            'expected_acc': '98.40%'
        },
        'MS3DStackedGRU': {
            'path': 'best_model/ms3dgru_best.ckpt',
            'type': 'MS3DStackedGRU',
            'cnn_filters': 32,
            'dropout': 0.05,
            'expected_acc': '98.24%'
        }
    }
    
    # 테스트 데이터 생성
    print("📊 테스트 데이터 생성...")
    n_samples = 100
    test_data_list = [np.random.randn(87, 8) for _ in range(n_samples)]
    single_test_data = test_data_list[0]
    
    print(f"  단일 샘플: {single_test_data.shape}")
    print(f"  배치 샘플: {len(test_data_list)}개")
    print()
    
    results = {}
    
    # 각 모델로 추론 테스트
    for model_name, config in models_config.items():
        print(f"🤖 모델: {model_name} 테스트 중...")
        
        try:
            # 추론 엔진 초기화
            init_params = {
                'model_path': config['path'],
                'model_type': config['type'],
                'input_size': 8,
                'hidden_size': config.get('hidden_size', 64),
                'classes': 24,
                'device': 'cpu',
                'dropout': config['dropout']
            }
            
            if 'cnn_filters' in config:
                init_params['cnn_filters'] = config['cnn_filters']
            
            engine = SignGloveInference(**init_params)
            info = engine.get_model_info()
            
            # 단일 샘플 추론 시간 측정
            single_times = []
            for _ in range(50):
                start = time.time()
                _ = engine.predict_single(single_test_data, return_all_info=False)
                single_times.append((time.time() - start) * 1000)
            
            # 배치 추론 시간 측정
            batch_sizes = [1, 4, 8, 16, 32]
            batch_results = {}
            for batch_size in batch_sizes:
                batch_data = test_data_list[:batch_size]
                times = []
                for _ in range(10):
                    start = time.time()
                    _ = engine.predict_batch(batch_data)
                    times.append((time.time() - start) * 1000)
                batch_results[batch_size] = {
                    'mean': np.mean(times),
                    'std': np.std(times),
                    'per_sample': np.mean(times) / batch_size
                }
            
            # 예측 결과 수집
            predictions = []
            confidences = []
            for data in test_data_list[:50]:
                result = engine.predict_single(data)
                predictions.append(result['predicted_class'])
                confidences.append(result['confidence'])
            
            results[model_name] = {
                'success': True,
                'params': info.get('total_parameters', 0),
                'single_times': single_times,
                'batch_results': batch_results,
                'predictions': predictions,
                'confidences': confidences,
                'expected_acc': config['expected_acc']
            }
            
            print(f"  ✅ 완료")
            
        except Exception as e:
            print(f"  ❌ 오류: {e}")
            results[model_name] = {'success': False, 'error': str(e)}
    
    print()
    print("🎨 시각화 생성 중...")
    
    # 시각화 생성
    output_dir = Path('visualizations/inference_test')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    create_inference_visualizations(results, output_dir)
    
    print()
    print("=" * 80)
    print("✅ 추론 테스트 및 시각화 완료!")
    print(f"📁 결과 저장 위치: {output_dir}")
    print("=" * 80)

def create_inference_visualizations(results, output_dir):
    """추론 테스트 결과 시각화"""
    
    # 성공한 모델만 필터링
    successful_models = {k: v for k, v in results.items() if v.get('success', False)}
    
    if not successful_models:
        print("⚠️  시각화할 성공한 모델이 없습니다.")
        return
    
    # 1. 추론 시간 비교 (Bar Plot)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Inference Performance Comparison', fontsize=16, fontweight='bold')
    
    # 1-1. 단일 샘플 추론 시간
    ax1 = axes[0, 0]
    model_names = list(successful_models.keys())
    single_means = [np.mean(results[m]['single_times']) for m in model_names]
    single_stds = [np.std(results[m]['single_times']) for m in model_names]
    
    x_pos = np.arange(len(model_names))
    bars = ax1.bar(x_pos, single_means, yerr=single_stds, capsize=5, alpha=0.7, 
                   color=['steelblue', 'coral', 'lightgreen', 'orange'], edgecolor='black')
    ax1.set_xlabel('Model', fontsize=12)
    ax1.set_ylabel('Inference Time (ms)', fontsize=12)
    ax1.set_title('Single Sample Inference Time', fontsize=14, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(model_names, rotation=45, ha='right')
    ax1.grid(axis='y', alpha=0.3)
    
    for bar, mean, std in zip(bars, single_means, single_stds):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{mean:.2f}ms\n±{std:.2f}',
                ha='center', va='bottom', fontsize=9)
    
    # 1-2. 처리량 비교
    ax2 = axes[0, 1]
    throughputs = [1000 / np.mean(results[m]['single_times']) for m in model_names]
    
    bars2 = ax2.bar(x_pos, throughputs, alpha=0.7, 
                    color=['steelblue', 'coral', 'lightgreen', 'orange'], edgecolor='black')
    ax2.set_xlabel('Model', fontsize=12)
    ax2.set_ylabel('Throughput (samples/sec)', fontsize=12)
    ax2.set_title('Inference Throughput', fontsize=14, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(model_names, rotation=45, ha='right')
    ax2.grid(axis='y', alpha=0.3)
    
    for bar, thr in zip(bars2, throughputs):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{thr:.1f}\nsamples/sec',
                ha='center', va='bottom', fontsize=9)
    
    # 1-3. 배치 크기별 추론 시간
    ax3 = axes[1, 0]
    batch_sizes = [1, 4, 8, 16, 32]
    x_batch = np.arange(len(batch_sizes))
    width = 0.2
    
    for i, model_name in enumerate(model_names):
        batch_times = [results[model_name]['batch_results'][bs]['per_sample'] for bs in batch_sizes]
        ax3.bar(x_batch + i*width, batch_times, width, label=model_name, alpha=0.7)
    
    ax3.set_xlabel('Batch Size', fontsize=12)
    ax3.set_ylabel('Time per Sample (ms)', fontsize=12)
    ax3.set_title('Batch Size vs Inference Time per Sample', fontsize=14, fontweight='bold')
    ax3.set_xticks(x_batch + width * 1.5)
    ax3.set_xticklabels(batch_sizes)
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)
    
    # 1-4. 파라미터 수 vs 추론 시간
    ax4 = axes[1, 1]
    params = [results[m]['params'] for m in model_names]
    times = [np.mean(results[m]['single_times']) for m in model_names]
    
    scatter = ax4.scatter(params, times, s=200, alpha=0.6, 
                         c=['steelblue', 'coral', 'lightgreen', 'orange'], edgecolors='black')
    for i, model_name in enumerate(model_names):
        ax4.annotate(model_name, (params[i], times[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=10)
    
    ax4.set_xlabel('Number of Parameters', fontsize=12)
    ax4.set_ylabel('Inference Time (ms)', fontsize=12)
    ax4.set_title('Parameters vs Inference Time', fontsize=14, fontweight='bold')
    ax4.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'inference_performance_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_dir / 'inference_performance_comparison.png'}")
    plt.close()
    
    # 2. 추론 시간 분포 (Box Plot)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Inference Time Distribution', fontsize=16, fontweight='bold')
    
    # 2-1. 단일 샘플 추론 시간 분포
    ax1 = axes[0]
    box_data = [results[m]['single_times'] for m in model_names]
    bp = ax1.boxplot(box_data, labels=model_names, patch_artist=True)
    colors = ['steelblue', 'coral', 'lightgreen', 'orange']
    for patch, color in zip(bp['boxes'], colors[:len(model_names)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax1.set_xlabel('Model', fontsize=12)
    ax1.set_ylabel('Inference Time (ms)', fontsize=12)
    ax1.set_title('Single Sample Inference Time Distribution', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
    
    # 2-2. 확률 분포
    ax2 = axes[1]
    conf_data = [results[m]['confidences'] for m in model_names]
    bp2 = ax2.boxplot(conf_data, labels=model_names, patch_artist=True)
    for patch, color in zip(bp2['boxes'], colors[:len(model_names)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax2.set_xlabel('Model', fontsize=12)
    ax2.set_ylabel('Confidence', fontsize=12)
    ax2.set_title('Prediction Confidence Distribution', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'inference_time_distribution.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_dir / 'inference_time_distribution.png'}")
    plt.close()
    
    # 3. 배치 성능 비교 (Heatmap)
    fig, ax = plt.subplots(figsize=(12, 8))
    
    batch_sizes = [1, 4, 8, 16, 32]
    heatmap_data = []
    for model_name in model_names:
        row = [results[model_name]['batch_results'][bs]['per_sample'] for bs in batch_sizes]
        heatmap_data.append(row)
    
    heatmap_data = np.array(heatmap_data)
    im = ax.imshow(heatmap_data, cmap='YlOrRd', aspect='auto')
    
    ax.set_xticks(np.arange(len(batch_sizes)))
    ax.set_xticklabels(batch_sizes)
    ax.set_yticks(np.arange(len(model_names)))
    ax.set_yticklabels(model_names)
    
    for i in range(len(model_names)):
        for j in range(len(batch_sizes)):
            text = ax.text(j, i, f'{heatmap_data[i, j]:.2f}ms',
                         ha="center", va="center", color="black", fontsize=10)
    
    ax.set_xlabel('Batch Size', fontsize=12)
    ax.set_ylabel('Model', fontsize=12)
    ax.set_title('Inference Time per Sample by Batch Size', fontsize=14, fontweight='bold', pad=20)
    plt.colorbar(im, ax=ax, label='Time per Sample (ms)')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'inference_batch_heatmap.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_dir / 'inference_batch_heatmap.png'}")
    plt.close()
    
    # 4. 종합 비교 (Radar Chart)
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    categories = ['Speed', 'Throughput', 'Efficiency', 'Params']
    num_vars = len(categories)
    
    # 각 모델의 정규화된 점수 계산
    max_times = max([np.mean(results[m]['single_times']) for m in model_names])
    max_thr = max([1000 / np.mean(results[m]['single_times']) for m in model_names])
    max_params = max([results[m]['params'] for m in model_names])
    
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # 완전한 원을 만들기 위해
    
    colors = ['steelblue', 'coral', 'lightgreen', 'orange']
    
    for i, model_name in enumerate(model_names):
        # 정규화된 점수 (높을수록 좋음)
        speed_score = 1 - (np.mean(results[model_name]['single_times']) / max_times)  # 빠를수록 높음
        thr_score = (1000 / np.mean(results[model_name]['single_times'])) / max_thr  # 높을수록 좋음
        eff_score = (1000 / np.mean(results[model_name]['single_times'])) / results[model_name]['params'] * 10000  # 효율성
        eff_score = eff_score / max([(1000 / np.mean(results[m]['single_times'])) / results[m]['params'] * 10000 for m in model_names])  # 정규화
        params_score = 1 - (results[model_name]['params'] / max_params)  # 적을수록 높음
        
        values = [speed_score, thr_score, eff_score, params_score]
        values += values[:1]  # 완전한 원을 만들기 위해
        
        ax.plot(angles, values, 'o-', linewidth=2, label=model_name, color=colors[i])
        ax.fill(angles, values, alpha=0.25, color=colors[i])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 1)
    ax.set_title('Model Performance Radar Chart', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'inference_radar_chart.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_dir / 'inference_radar_chart.png'}")
    plt.close()
    
    # 5. 결과 요약 테이블 이미지
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('tight')
    ax.axis('off')
    
    table_data = []
    headers = ['Model', 'Params', 'Avg Time (ms)', 'Throughput (/sec)', 'Expected Acc', 'Status']
    
    for model_name in model_names:
        row = [
            model_name,
            f"{results[model_name]['params']:,}",
            f"{np.mean(results[model_name]['single_times']):.2f}",
            f"{1000 / np.mean(results[model_name]['single_times']):.1f}",
            results[model_name]['expected_acc'],
            '✅ Success'
        ]
        table_data.append(row)
    
    table = ax.table(cellText=table_data, colLabels=headers, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2)
    
    # 헤더 스타일
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # 데이터 행 스타일
    colors = ['#E3F2FD', '#FFF3E0', '#F1F8E9', '#FCE4EC']
    for i, model_name in enumerate(model_names):
        for j in range(len(headers)):
            table[(i+1, j)].set_facecolor(colors[i % len(colors)])
    
    plt.title('Inference Test Results Summary', fontsize=16, fontweight='bold', pad=20)
    plt.savefig(output_dir / 'inference_results_summary.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_dir / 'inference_results_summary.png'}")
    plt.close()

if __name__ == "__main__":
    test_all_models_and_visualize()



