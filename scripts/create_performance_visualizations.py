"""
SignGlove 모델 성능 시각화 생성 스크립트
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# 스타일 설정
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# 한글 폰트 설정 (한글이 깨지지 않도록)
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# 데이터 준비
data = {
    'Dataset': ['yubeen', 'yubeen', 'yubeen', 'yubeen',
                'jaeyeon', 'jaeyeon', 'jaeyeon', 'jaeyeon',
                'yubeen & jaeyeon', 'yubeen & jaeyeon', 'yubeen & jaeyeon', 'yubeen & jaeyeon'],
    'Model': ['GRU', 'StackedGRU', 'MS3DGRU', 'MS3DStackedGRU',
              'GRU', 'StackedGRU', 'MS3DGRU', 'MS3DStackedGRU',
              'GRU', 'StackedGRU', 'MS3DGRU', 'MS3DStackedGRU'],
    'Parameters': [74776, 50584, 58840, 167032,
                   74776, 50584, 58840, 167032,
                   74776, 50584, 58840, 167032],
    'Test_Acc': [98.44, 97.06, 98.78, 98.78,
                 98.44, 97.06, 98.78, 95.14,
                 98.44, 97.06, 98.78, 98.44],
    'Test_F1': [0.9844, 0.9698, 0.9877, 0.9878,
                0.9844, 0.9698, 0.9877, 0.9494,
                0.9844, 0.9698, 0.9877, 0.9843],
    'Test_Loss': [0.061, 0.092, 0.052, 0.045,
                  0.061, 0.092, 0.052, 0.124,
                  0.061, 0.092, 0.052, 0.046],
    'Train_Acc': [99.7, 99.5, 99.8, 99.9,
                  99.7, 99.5, 99.8, 99.2,
                  99.7, 99.5, 99.8, 99.6],
    'Val_Acc': [99.3, 98.8, 99.5, 99.6,
                99.3, 98.8, 99.5, 96.5,
                99.3, 98.8, 99.5, 99.2],
    'Val_Loss': [0.024, 0.036, 0.021, 0.018,
                 0.024, 0.036, 0.021, 0.089,
                 0.024, 0.036, 0.021, 0.025],
    'Overfitting': ['경미', '중간', '경미', '경미',
                    '경미', '중간', '경미', '심함',
                    '경미', '중간', '경미', '경미'],
    'Stability': ['매우 안정', '매우 안정', '매우 안정', '매우 안정',
                  '매우 안정', '매우 안정', '매우 안정', '불안정',
                  '매우 안정', '매우 안정', '매우 안정', '안정']
}

df = pd.DataFrame(data)

# 출력 디렉토리 생성
output_dir = Path('inference/performance_visualizations')
output_dir.mkdir(parents=True, exist_ok=True)

print('=' * 80)
print('📊 SignGlove 모델 성능 시각화 생성')
print('=' * 80)
print()

# 1. 데이터셋별 Test Accuracy 비교
print('📌 시각화 1: 데이터셋별 Test Accuracy 비교')
fig, ax = plt.subplots(figsize=(14, 8))
pivot_acc = df.pivot(index='Model', columns='Dataset', values='Test_Acc')
pivot_acc.plot(kind='bar', ax=ax, width=0.8)
ax.set_title('Test Accuracy by Dataset and Model', fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Model', fontsize=12)
ax.set_ylabel('Test Accuracy (%)', fontsize=12)
ax.legend(title='Dataset', fontsize=10)
ax.set_ylim([94, 100])
ax.grid(axis='y', alpha=0.3)
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(output_dir / '1_dataset_model_test_accuracy.png', dpi=300, bbox_inches='tight')
plt.close()
print('   ✅ 저장: 1_dataset_model_test_accuracy.png')

# 2. 모델별 성능 지표 비교 (Heatmap)
print('📌 시각화 2: 모델별 성능 지표 비교 (Heatmap)')
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for idx, metric in enumerate(['Test_Acc', 'Test_F1', 'Test_Loss']):
    pivot = df.pivot(index='Model', columns='Dataset', values=metric)
    if metric == 'Test_Loss':
        sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn_r', ax=axes[idx], cbar_kws={'label': metric})
    else:
        sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdYlGn', ax=axes[idx], cbar_kws={'label': metric})
    axes[idx].set_title(metric.replace('_', ' '), fontsize=11, fontweight='bold')
    axes[idx].set_xlabel('')
    axes[idx].set_ylabel('')

plt.suptitle('Performance Metrics Heatmap', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(output_dir / '2_performance_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()
print('   ✅ 저장: 2_performance_heatmap.png')

# 3. 파라미터 수 vs 성능 (효율성)
print('📌 시각화 3: 파라미터 수 vs 성능 (효율성 분석)')
fig, ax = plt.subplots(figsize=(12, 8))

# 각 모델의 평균 성능 계산
model_avg = df.groupby('Model').agg({
    'Parameters': 'first',
    'Test_Acc': 'mean',
    'Test_F1': 'mean',
    'Test_Loss': 'mean'
}).reset_index()

scatter = ax.scatter(model_avg['Parameters'] / 1000, model_avg['Test_Acc'], 
                     s=200, alpha=0.7, c=model_avg['Test_Acc'], 
                     cmap='viridis', edgecolors='black', linewidth=2)

for i, row in model_avg.iterrows():
    ax.annotate(row['Model'], 
                (row['Parameters'] / 1000, row['Test_Acc']),
                xytext=(5, 5), textcoords='offset points', fontsize=10, fontweight='bold')

ax.set_xlabel('Parameters (K)', fontsize=12)
ax.set_ylabel('Test Accuracy (%)', fontsize=12)
ax.set_title('Model Efficiency: Parameters vs Test Accuracy', fontsize=14, fontweight='bold', pad=20)
ax.grid(True, alpha=0.3)
plt.colorbar(scatter, label='Test Accuracy (%)')
plt.tight_layout()
plt.savefig(output_dir / '3_parameter_efficiency.png', dpi=300, bbox_inches='tight')
plt.close()
print('   ✅ 저장: 3_parameter_efficiency.png')

# 4. Overfitting 분석 (Train vs Val vs Test)
print('📌 시각화 4: Overfitting 분석 (Train vs Val vs Test)')
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

datasets = ['yubeen', 'jaeyeon', 'yubeen & jaeyeon']
for idx, dataset in enumerate(datasets):
    dataset_df = df[df['Dataset'] == dataset]
    
    x = np.arange(len(dataset_df))
    width = 0.25
    
    axes[idx].bar(x - width, dataset_df['Train_Acc'], width, label='Train Acc', alpha=0.8)
    axes[idx].bar(x, dataset_df['Val_Acc'], width, label='Val Acc', alpha=0.8)
    axes[idx].bar(x + width, dataset_df['Test_Acc'], width, label='Test Acc', alpha=0.8)
    
    axes[idx].set_xlabel('Model', fontsize=10)
    axes[idx].set_ylabel('Accuracy (%)', fontsize=10)
    axes[idx].set_title(f'{dataset}', fontsize=12, fontweight='bold')
    axes[idx].set_xticks(x)
    axes[idx].set_xticklabels(dataset_df['Model'], rotation=45, ha='right')
    axes[idx].legend(fontsize=9)
    axes[idx].set_ylim([94, 100])
    axes[idx].grid(axis='y', alpha=0.3)

plt.suptitle('Overfitting Analysis: Train vs Validation vs Test Accuracy', 
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(output_dir / '4_overfitting_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print('   ✅ 저장: 4_overfitting_analysis.png')

# 5. Loss 비교 (Train vs Val vs Test)
print('📌 시각화 5: Loss 비교 (Train vs Val vs Test)')
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for idx, dataset in enumerate(datasets):
    dataset_df = df[df['Dataset'] == dataset]
    
    x = np.arange(len(dataset_df))
    width = 0.35
    
    axes[idx].bar(x - width/2, dataset_df['Val_Loss'], width, label='Val Loss', alpha=0.8, color='orange')
    axes[idx].bar(x + width/2, dataset_df['Test_Loss'], width, label='Test Loss', alpha=0.8, color='red')
    
    axes[idx].set_xlabel('Model', fontsize=10)
    axes[idx].set_ylabel('Loss', fontsize=10)
    axes[idx].set_title(f'{dataset}', fontsize=12, fontweight='bold')
    axes[idx].set_xticks(x)
    axes[idx].set_xticklabels(dataset_df['Model'], rotation=45, ha='right')
    axes[idx].legend(fontsize=9)
    axes[idx].grid(axis='y', alpha=0.3)

plt.suptitle('Loss Comparison: Validation vs Test Loss', 
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(output_dir / '5_loss_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print('   ✅ 저장: 5_loss_comparison.png')

# 6. 데이터셋별 모델 성능 랭킹
print('📌 시각화 6: 데이터셋별 모델 성능 랭킹')
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for idx, dataset in enumerate(datasets):
    dataset_df = df[df['Dataset'] == dataset].sort_values('Test_Acc', ascending=True)
    
    y_pos = np.arange(len(dataset_df))
    bars = axes[idx].barh(y_pos, dataset_df['Test_Acc'], alpha=0.8)
    
    # 색상 코딩 (높은 순위일수록 진한 색)
    colors = plt.cm.viridis(np.linspace(0.3, 1, len(dataset_df)))
    for i, (bar, color) in enumerate(zip(bars, colors)):
        bar.set_color(color)
    
    axes[idx].set_yticks(y_pos)
    axes[idx].set_yticklabels(dataset_df['Model'], fontsize=10)
    axes[idx].set_xlabel('Test Accuracy (%)', fontsize=10)
    axes[idx].set_title(f'{dataset}', fontsize=12, fontweight='bold')
    axes[idx].set_xlim([94, 100])
    axes[idx].grid(axis='x', alpha=0.3)
    
    # 값 표시
    for i, v in enumerate(dataset_df['Test_Acc']):
        axes[idx].text(v + 0.1, i, f'{v:.2f}%', va='center', fontsize=9)

plt.suptitle('Model Performance Ranking by Dataset', 
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(output_dir / '6_model_ranking.png', dpi=300, bbox_inches='tight')
plt.close()
print('   ✅ 저장: 6_model_ranking.png')

# 7. 안정성 및 과적합 수준 분석
print('📌 시각화 7: 안정성 및 과적합 수준 분석')
fig, ax = plt.subplots(figsize=(12, 8))

# 안정성과 과적합을 수치로 변환
stability_map = {'매우 안정': 3, '안정': 2, '불안정': 1}
overfitting_map = {'경미': 3, '중간': 2, '심함': 1}

df['Stability_Score'] = df['Stability'].map(stability_map)
df['Overfitting_Score'] = df['Overfitting'].map(overfitting_map)

scatter = ax.scatter(df['Test_Acc'], df['Stability_Score'] + df['Overfitting_Score'],
                     s=df['Parameters'] / 500, alpha=0.6, c=df['Test_Acc'],
                     cmap='RdYlGn', edgecolors='black', linewidth=1.5)

# 데이터셋별로 다른 마커
for dataset, marker in zip(['yubeen', 'jaeyeon', 'yubeen & jaeyeon'], ['o', 's', '^']):
    dataset_df = df[df['Dataset'] == dataset]
    ax.scatter(dataset_df['Test_Acc'], 
               dataset_df['Stability_Score'] + dataset_df['Overfitting_Score'],
               s=dataset_df['Parameters'] / 500, alpha=0.6,
               marker=marker, label=dataset, edgecolors='black', linewidth=1.5)

ax.set_xlabel('Test Accuracy (%)', fontsize=12)
ax.set_ylabel('Quality Score (Stability + Overfitting)', fontsize=12)
ax.set_title('Model Quality: Performance vs Stability & Overfitting', 
             fontsize=14, fontweight='bold', pad=20)
ax.legend(title='Dataset', fontsize=10)
ax.grid(True, alpha=0.3)
plt.colorbar(scatter, label='Test Accuracy (%)')
plt.tight_layout()
plt.savefig(output_dir / '7_quality_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print('   ✅ 저장: 7_quality_analysis.png')

# 8. 종합 성능 비교 (Radar Chart)
print('📌 시각화 8: 종합 성능 비교 (종합 점수)')
fig, axes = plt.subplots(1, 4, figsize=(20, 5), subplot_kw=dict(projection='polar'))

models = ['GRU', 'StackedGRU', 'MS3DGRU', 'MS3DStackedGRU']
categories = ['Test Acc', 'Test F1', 'Efficiency', 'Stability', 'Overfitting']
N = len(categories)

# 각 모델의 평균 계산 (yubeen & jaeyeon 데이터셋 제외, jaeyeon의 불안정한 경우 제외)
filtered_df = df[(df['Dataset'] != 'jaeyeon') | (df['Model'] != 'MS3DStackedGRU')]

for idx, model in enumerate(models):
    model_df = filtered_df[filtered_df['Model'] == model]
    
    # 정규화된 값 계산
    values = [
        model_df['Test_Acc'].mean() / 100,  # 0-1
        model_df['Test_F1'].mean(),  # 이미 0-1
        1 - (model_df['Parameters'].mean() / 200000),  # 효율성 (역수)
        model_df['Stability_Score'].mean() / 3,  # 0-1
        model_df['Overfitting_Score'].mean() / 3  # 0-1
    ]
    
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    values += values[:1]  # 닫기
    angles += angles[:1]
    
    axes[idx].plot(angles, values, 'o-', linewidth=2, label=model)
    axes[idx].fill(angles, values, alpha=0.25)
    axes[idx].set_xticks(angles[:-1])
    axes[idx].set_xticklabels(categories, fontsize=9)
    axes[idx].set_ylim([0, 1])
    axes[idx].set_title(model, fontsize=11, fontweight='bold', pad=20)
    axes[idx].grid(True)

plt.suptitle('Comprehensive Performance Comparison (Normalized)', 
             fontsize=16, fontweight='bold', y=1.05)
plt.tight_layout()
plt.savefig(output_dir / '8_comprehensive_radar.png', dpi=300, bbox_inches='tight')
plt.close()
print('   ✅ 저장: 8_comprehensive_radar.png')

# 9. 데이터셋별 상세 비교 (Grouped Bar)
print('📌 시각화 9: 데이터셋별 상세 비교')
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

metrics = [('Test_Acc', 'Test Accuracy (%)'), 
           ('Test_F1', 'Test F1-Score'),
           ('Test_Loss', 'Test Loss'),
           ('Parameters', 'Parameters (K)')]

for idx, (metric, label) in enumerate(metrics):
    ax = axes[idx // 2, idx % 2]
    
    x = np.arange(len(datasets))
    width = 0.2
    models = df['Model'].unique()
    
    for i, model in enumerate(models):
        model_values = []
        for dataset in datasets:
            value = df[(df['Dataset'] == dataset) & (df['Model'] == model)][metric].values
            if len(value) > 0:
                if metric == 'Parameters':
                    model_values.append(value[0] / 1000)  # K로 변환
                else:
                    model_values.append(value[0])
            else:
                model_values.append(0)
        
        offset = (i - len(models)/2 + 0.5) * width
        ax.bar(x + offset, model_values, width, label=model, alpha=0.8)
    
    ax.set_xlabel('Dataset', fontsize=10)
    ax.set_ylabel(label, fontsize=10)
    ax.set_title(label, fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=15, ha='right')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

plt.suptitle('Detailed Performance Comparison by Dataset', 
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(output_dir / '9_detailed_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print('   ✅ 저장: 9_detailed_comparison.png')

# 10. CSV 파일로 데이터 저장
print('📌 시각화 10: 원본 데이터 CSV 저장')
df.to_csv(output_dir / 'performance_data.csv', index=False, encoding='utf-8-sig')
print('   ✅ 저장: performance_data.csv')

# 요약 통계도 저장
summary = df.groupby('Model').agg({
    'Test_Acc': ['mean', 'std', 'min', 'max'],
    'Test_F1': ['mean', 'std', 'min', 'max'],
    'Test_Loss': ['mean', 'std', 'min', 'max'],
    'Parameters': 'first'
}).round(2)
summary.to_csv(output_dir / 'performance_summary.csv', encoding='utf-8-sig')
print('   ✅ 저장: performance_summary.csv')

print()
print('=' * 80)
print('✅ 모든 시각화 생성 완료!')
print('=' * 80)
print(f'📁 출력 디렉토리: {output_dir}')
print(f'📊 생성된 파일: 10개 시각화 + 2개 CSV')
print()
print('생성된 파일 목록:')
for file in sorted(output_dir.glob('*')):
    size = file.stat().st_size / 1024
    print(f'  • {file.name} ({size:.1f} KB)')

