"""
Performance Visualizations 종합 분석
1-9번 시각화와 실제 테스트 결과 비교
"""

import pandas as pd
import numpy as np
from pathlib import Path

print('=' * 80)
print('📊 Performance Visualizations 종합 분석')
print('=' * 80)
print()

# 1. 데이터 로드
df = pd.read_csv('inference/performance_visualizations/performance_data.csv')
summary_df = pd.read_csv('inference/performance_visualizations/performance_summary.csv', index_col=0)

print('📌 1. 기본 성능 데이터 (performance_data.csv)')
print('-' * 80)
print(df.to_string(index=False))
print()

print('📌 2. 요약 통계 (performance_summary.csv)')
print('-' * 80)
print(summary_df)
print()

# 3. 실제 테스트 결과 확인
test_report_file = Path('inference/performance_visualizations/real_test_report_ms3dgru_final.txt')
if test_report_file.exists():
    with open(test_report_file, 'r', encoding='utf-8') as f:
        content = f.read()
        
    print('📌 3. 실제 테스트 결과 (real_test_report_ms3dgru_final.txt)')
    print('-' * 80)
    lines = content.split('\n')
    for line in lines[:10]:
        if line.strip():
            print(line)
    print()

# 4. 각 시각화 데이터 분석
print('=' * 80)
print('📊 시각화별 데이터 요약')
print('=' * 80)
print()

# 데이터셋별로 그룹화
datasets = df['Dataset'].unique()
models = df['Model'].unique()

print('1️⃣  데이터셋별 Test Accuracy (1_dataset_model_test_accuracy.png 기준):')
print('-' * 80)
for dataset in datasets:
    dataset_df = df[df['Dataset'] == dataset]
    print(f'\n📊 {dataset}:')
    for _, row in dataset_df.iterrows():
        print(f'   {row["Model"]:20s}: {row["Test_Acc"]:5.2f}%')
print()

print('2️⃣  모델별 성능 지표 (2_performance_heatmap.png 기준):')
print('-' * 80)
model_stats = df.groupby('Model').agg({
    'Test_Acc': ['mean', 'std', 'min', 'max'],
    'Test_F1': ['mean', 'std'],
    'Test_Loss': ['mean', 'std']
}).round(2)
print(model_stats)
print()

print('3️⃣  파라미터 효율성 (3_parameter_efficiency.png 기준):')
print('-' * 80)
for model in models:
    model_df = df[df['Model'] == model].iloc[0]
    params_k = model_df['Parameters'] / 1000
    avg_acc = df[df['Model'] == model]['Test_Acc'].mean()
    efficiency = avg_acc / params_k  # 정확도 / 파라미터(K)
    print(f'   {model:20s}: {avg_acc:5.2f}% / {params_k:6.1f}K params = {efficiency:.4f} 효율')
print()

print('4️⃣  Overfitting 분석 (4_overfitting_analysis.png 기준):')
print('-' * 80)
for dataset in datasets:
    dataset_df = df[df['Dataset'] == dataset]
    print(f'\n📊 {dataset}:')
    for _, row in dataset_df.iterrows():
        train_acc = row['Train_Acc']
        val_acc = row['Val_Acc']
        test_acc = row['Test_Acc']
        gap = train_acc - test_acc
        status = '✅ 경미' if gap < 1.5 else '⚠️  중간' if gap < 3 else '❌ 심함'
        print(f'   {row["Model"]:20s}: Train={train_acc:5.2f}%, Val={val_acc:5.2f}%, Test={test_acc:5.2f}% | Gap={gap:4.2f}% {status}')
print()

print('5️⃣  Loss 비교 (5_loss_comparison.png 기준):')
print('-' * 80)
for dataset in datasets:
    dataset_df = df[df['Dataset'] == dataset]
    print(f'\n📊 {dataset}:')
    for _, row in dataset_df.iterrows():
        val_loss = row['Val_Loss']
        test_loss = row['Test_Loss']
        print(f'   {row["Model"]:20s}: Val Loss={val_loss:.3f}, Test Loss={test_loss:.3f}')
print()

print('6️⃣  모델 성능 랭킹 (6_model_ranking.png 기준):')
print('-' * 80)
for dataset in datasets:
    dataset_df = df[df['Dataset'] == dataset].sort_values('Test_Acc', ascending=False)
    print(f'\n📊 {dataset} (순위순):')
    for rank, (_, row) in enumerate(dataset_df.iterrows(), 1):
        print(f'   {rank}. {row["Model"]:20s}: {row["Test_Acc"]:5.2f}%')
print()

print('7️⃣  품질 분석 (7_quality_analysis.png 기준):')
print('-' * 80)
for dataset in datasets:
    dataset_df = df[df['Dataset'] == dataset]
    print(f'\n📊 {dataset}:')
    for _, row in dataset_df.iterrows():
        stability = row['Stability']
        overfitting = row['Overfitting']
        test_acc = row['Test_Acc']
        quality_score = row['Stability_Score'] + row['Overfitting_Score']
        print(f'   {row["Model"]:20s}: Acc={test_acc:5.2f}%, Stability={stability:10s}, Overfitting={overfitting:6s}, Quality={quality_score}')
print()

print('8️⃣  종합 성능 비교 (8_comprehensive_radar.png 기준):')
print('-' * 80)
for model in models:
    model_df = df[df['Model'] == model]
    avg_acc = model_df['Test_Acc'].mean()
    avg_f1 = model_df['Test_F1'].mean()
    avg_params = model_df['Parameters'].iloc[0]
    avg_stability = model_df['Stability_Score'].mean()
    avg_overfitting = model_df['Overfitting_Score'].mean()
    
    # 정규화된 점수
    norm_acc = avg_acc / 100
    norm_f1 = avg_f1
    norm_efficiency = 1 - (avg_params / 200000)  # 파라미터가 적을수록 높음
    norm_stability = avg_stability / 3
    norm_overfitting = avg_overfitting / 3
    
    print(f'\n   {model}:')
    print(f'      정확도: {norm_acc:.3f} | F1: {norm_f1:.3f} | 효율: {norm_efficiency:.3f} | 안정성: {norm_stability:.3f} | 과적합방지: {norm_overfitting:.3f}')
print()

print('9️⃣  상세 비교 (9_detailed_comparison.png 기준):')
print('-' * 80)
metrics = ['Test_Acc', 'Test_F1', 'Test_Loss', 'Parameters']
for metric in metrics:
    print(f'\n📊 {metric}:')
    pivot = df.pivot(index='Model', columns='Dataset', values=metric)
    print(pivot.to_string())
print()

# 5. 실제 테스트 결과와 비교
print('=' * 80)
print('⚠️  실제 테스트 결과 비교')
print('=' * 80)
print()
print('📌 훈련 시 성능 (performance_data.csv):')
ms3dgru_train = df[df['Model'] == 'MS3DGRU']
print(f'   평균 Test Accuracy: {ms3dgru_train["Test_Acc"].mean():.2f}%')
print(f'   yubeen & jaeyeon: {ms3dgru_train[ms3dgru_train["Dataset"] == "yubeen & jaeyeon"]["Test_Acc"].values[0]:.2f}%')
print()

print('📌 실제 테스트 성능 (real_test_report):')
print(f'   Test Accuracy: 4.16% ⚠️')
print(f'   문제: 모델이 모든 입력을 클래스 13 (ㅎ)로만 예측')
print()

print('=' * 80)
print('🔍 문제 분석')
print('=' * 80)
print()
print('1. 성능 차이:')
print('   - 훈련 로그: 98.78%')
print('   - 체크포인트 정보: 98.78%')
print('   - 직접 테스트: 4.16%')
print('   → 체크포인트 저장/로드 문제 가능성 높음')
print()
print('2. 가능한 원인:')
print('   - 체크포인트가 잘못된 모델 상태 저장')
print('   - 모델 로드 시 파라미터 불일치')
print('   - 전처리 파이프라인 차이')
print('   - Dropout/Training mode 차이')
print()
print('3. 권장 사항:')
print('   - PyTorch Lightning Trainer.test()로 재검증')
print('   - 체크포인트 재저장 (훈련 완료 직후)')
print('   - 전처리 파이프라인 일치 확인')
print()

print('=' * 80)

