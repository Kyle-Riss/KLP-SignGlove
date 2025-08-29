#!/usr/bin/env python3
"""
모음 성능 비교 스크립트
기존 모델과 개선된 모델의 모음 성능을 비교
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# 기존 모델과 개선된 모델 import
from simple_robust_model import SimpleRobustGRU
from vowel_improvement import VowelEnhancedGRU, VowelImprovedDataset

def load_models():
    """모델 로드"""
    print("📊 모델 로딩 중...")
    
    # 기존 모델
    original_model = SimpleRobustGRU(
        input_size=8,
        hidden_size=128,
        num_layers=2,
        num_classes=24,
        dropout=0.3
    )
    
    try:
        original_model.load_state_dict(torch.load('simple_robust_model.pth', map_location='cpu'))
        print("✅ 기존 모델 로드 완료")
    except:
        print("⚠️ 기존 모델 파일을 찾을 수 없습니다.")
        return None, None
    
    # 개선된 모델
    enhanced_model = VowelEnhancedGRU(
        input_size=8,
        hidden_size=128,
        num_layers=2,
        num_classes=24,
        dropout=0.3
    )
    
    try:
        enhanced_model.load_state_dict(torch.load('vowel_enhanced_model.pth', map_location='cpu'))
        print("✅ 개선된 모델 로드 완료")
    except:
        print("⚠️ 개선된 모델 파일을 찾을 수 없습니다.")
        return original_model, None
    
    return original_model, enhanced_model

def evaluate_model_performance(model, test_loader, model_name, class_names):
    """모델 성능 평가"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for data, targets in test_loader:
            data, targets = data.to(device), targets.to(device)
            
            if model_name == "Enhanced":
                outputs, attention_weights = model(data)
            else:
                outputs = model(data)
            
            _, predicted = torch.max(outputs.data, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(targets.squeeze().cpu().numpy())
    
    # 모음 클래스만 추출
    vowels = ['ㅏ', 'ㅑ', 'ㅓ', 'ㅕ', 'ㅗ', 'ㅛ', 'ㅜ', 'ㅠ', 'ㅡ', 'ㅣ']
    vowel_indices = [class_names.index(v) for v in vowels]
    
    vowel_predictions = []
    vowel_targets = []
    
    for pred, target in zip(all_predictions, all_targets):
        if target in vowel_indices:
            vowel_predictions.append(pred)
            vowel_targets.append(target)
    
    # 모음 성능 리포트
    vowel_class_names = [class_names[i] for i in vowel_indices]
    report = classification_report(vowel_targets, vowel_predictions, 
                                 target_names=vowel_class_names, output_dict=True)
    
    return report, vowel_predictions, vowel_targets

def compare_vowel_performance(original_model, enhanced_model, test_loader, class_names):
    """모음 성능 비교"""
    print("\n📊 모음 성능 비교")
    print("=" * 50)
    
    results = {}
    
    # 기존 모델 평가
    if original_model is not None:
        print("🔍 기존 모델 평가 중...")
        original_report, _, _ = evaluate_model_performance(
            original_model, test_loader, "Original", class_names
        )
        results['Original'] = original_report
    
    # 개선된 모델 평가
    if enhanced_model is not None:
        print("🔍 개선된 모델 평가 중...")
        enhanced_report, _, _ = evaluate_model_performance(
            enhanced_model, test_loader, "Enhanced", class_names
        )
        results['Enhanced'] = enhanced_report
    
    return results

def visualize_comparison(results, class_names):
    """성능 비교 시각화"""
    print("\n📈 성능 비교 시각화")
    print("=" * 50)
    
    if len(results) < 2:
        print("⚠️ 비교할 모델이 충분하지 않습니다.")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('기존 모델 vs 개선된 모델 모음 성능 비교', fontsize=16, fontweight='bold')
    
    vowels = ['ㅏ', 'ㅑ', 'ㅓ', 'ㅕ', 'ㅗ', 'ㅛ', 'ㅜ', 'ㅠ', 'ㅡ', 'ㅣ']
    
    # 1. 모음별 정확도 비교
    original_accuracies = []
    enhanced_accuracies = []
    
    for vowel in vowels:
        if vowel in results['Original']:
            original_accuracies.append(results['Original'][vowel]['precision'] * 100)
        else:
            original_accuracies.append(0)
        
        if vowel in results['Enhanced']:
            enhanced_accuracies.append(results['Enhanced'][vowel]['precision'] * 100)
        else:
            enhanced_accuracies.append(0)
    
    x = np.arange(len(vowels))
    width = 0.35
    
    bars1 = axes[0, 0].bar(x - width/2, original_accuracies, width, label='기존 모델', color='lightblue', alpha=0.7)
    bars2 = axes[0, 0].bar(x + width/2, enhanced_accuracies, width, label='개선된 모델', color='lightcoral', alpha=0.7)
    
    axes[0, 0].set_title('모음별 정확도 비교')
    axes[0, 0].set_xlabel('모음')
    axes[0, 0].set_ylabel('정확도 (%)')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(vowels)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 값 표시
    for bar in bars1:
        height = bar.get_height()
        axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 1,
                       f'{height:.1f}%', ha='center', va='bottom', fontsize=8)
    
    for bar in bars2:
        height = bar.get_height()
        axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 1,
                       f'{height:.1f}%', ha='center', va='bottom', fontsize=8)
    
    # 2. 전체 성능 비교
    metrics = ['precision', 'recall', 'f1-score']
    original_overall = []
    enhanced_overall = []
    
    for metric in metrics:
        if 'weighted avg' in results['Original']:
            original_overall.append(results['Original']['weighted avg'][metric] * 100)
        else:
            original_overall.append(0)
        
        if 'weighted avg' in results['Enhanced']:
            enhanced_overall.append(results['Enhanced']['weighted avg'][metric] * 100)
        else:
            enhanced_overall.append(0)
    
    x = np.arange(len(metrics))
    bars1 = axes[0, 1].bar(x - width/2, original_overall, width, label='기존 모델', color='lightblue', alpha=0.7)
    bars2 = axes[0, 1].bar(x + width/2, enhanced_overall, width, label='개선된 모델', color='lightcoral', alpha=0.7)
    
    axes[0, 1].set_title('전체 성능 비교')
    axes[0, 1].set_xlabel('메트릭')
    axes[0, 1].set_ylabel('값 (%)')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(metrics)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 값 표시
    for bar in bars1:
        height = bar.get_height()
        axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + 1,
                       f'{height:.1f}%', ha='center', va='bottom')
    
    for bar in bars2:
        height = bar.get_height()
        axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + 1,
                       f'{height:.1f}%', ha='center', va='bottom')
    
    # 3. 성능 향상률
    improvement_rates = []
    for i, vowel in enumerate(vowels):
        if original_accuracies[i] > 0:
            improvement = ((enhanced_accuracies[i] - original_accuracies[i]) / original_accuracies[i]) * 100
            improvement_rates.append(improvement)
        else:
            improvement_rates.append(0)
    
    bars = axes[1, 0].bar(vowels, improvement_rates, color='lightgreen', alpha=0.7)
    axes[1, 0].set_title('성능 향상률')
    axes[1, 0].set_xlabel('모음')
    axes[1, 0].set_ylabel('향상률 (%)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 값 표시
    for bar, rate in zip(bars, improvement_rates):
        height = bar.get_height()
        axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + 0.5,
                       f'{rate:.1f}%', ha='center', va='bottom', fontsize=8)
    
    # 4. 평균 성능 비교
    avg_original = np.mean(original_accuracies)
    avg_enhanced = np.mean(enhanced_accuracies)
    
    models = ['기존 모델', '개선된 모델']
    avg_performances = [avg_original, avg_enhanced]
    
    bars = axes[1, 1].bar(models, avg_performances, color=['lightblue', 'lightcoral'], alpha=0.7)
    axes[1, 1].set_title('평균 모음 성능')
    axes[1, 1].set_ylabel('평균 정확도 (%)')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 값 표시
    for bar, perf in zip(bars, avg_performances):
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 1,
                       f'{perf:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('vowel_performance_comparison.png', dpi=300, bbox_inches='tight')
    print("📊 성능 비교 차트가 'vowel_performance_comparison.png'에 저장되었습니다.")

def print_comparison_summary(results):
    """비교 결과 요약 출력"""
    print("\n📋 모음 성능 비교 요약")
    print("=" * 50)
    
    if len(results) < 2:
        print("⚠️ 비교할 모델이 충분하지 않습니다.")
        return
    
    vowels = ['ㅏ', 'ㅑ', 'ㅓ', 'ㅕ', 'ㅗ', 'ㅛ', 'ㅜ', 'ㅠ', 'ㅡ', 'ㅣ']
    
    print("🔍 모음별 성능 비교:")
    print("-" * 40)
    
    for vowel in vowels:
        original_precision = results['Original'].get(vowel, {}).get('precision', 0) * 100
        enhanced_precision = results['Enhanced'].get(vowel, {}).get('precision', 0) * 100
        
        improvement = enhanced_precision - original_precision
        
        print(f"{vowel}:")
        print(f"  기존 모델: {original_precision:.1f}%")
        print(f"  개선된 모델: {enhanced_precision:.1f}%")
        print(f"  향상: {improvement:+.1f}%")
        print()
    
    # 전체 성능 비교
    original_overall = results['Original'].get('weighted avg', {})
    enhanced_overall = results['Enhanced'].get('weighted avg', {})
    
    print("📈 전체 성능 비교:")
    print("-" * 40)
    
    metrics = ['precision', 'recall', 'f1-score']
    for metric in metrics:
        original_val = original_overall.get(metric, 0) * 100
        enhanced_val = enhanced_overall.get(metric, 0) * 100
        improvement = enhanced_val - original_val
        
        print(f"{metric.title()}:")
        print(f"  기존 모델: {original_val:.1f}%")
        print(f"  개선된 모델: {enhanced_val:.1f}%")
        print(f"  향상: {improvement:+.1f}%")
        print()
    
    # 평균 향상률
    vowel_improvements = []
    for vowel in vowels:
        original_precision = results['Original'].get(vowel, {}).get('precision', 0) * 100
        enhanced_precision = results['Enhanced'].get(vowel, {}).get('precision', 0) * 100
        
        if original_precision > 0:
            improvement_rate = ((enhanced_precision - original_precision) / original_precision) * 100
            vowel_improvements.append(improvement_rate)
    
    avg_improvement = np.mean(vowel_improvements)
    print(f"🎯 평균 성능 향상률: {avg_improvement:.1f}%")

def main():
    """메인 함수"""
    print("📊 모음 성능 비교 시작")
    print("=" * 50)
    
    # 1. 모델 로드
    original_model, enhanced_model = load_models()
    
    if original_model is None and enhanced_model is None:
        print("❌ 비교할 모델이 없습니다.")
        return
    
    # 2. 테스트 데이터셋 로드
    print("\n📊 테스트 데이터셋 로딩 중...")
    test_dataset = VowelImprovedDataset(
        data_path="../SignGlove_HW/datasets/unified",
        sequence_length=20,
        apply_vowel_enhancement=False  # 원본 데이터로 테스트
    )
    
    # 테스트 데이터로더
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 3. 성능 비교
    results = compare_vowel_performance(
        original_model, enhanced_model, test_loader, test_dataset.all_classes
    )
    
    # 4. 결과 시각화
    visualize_comparison(results, test_dataset.all_classes)
    
    # 5. 결과 요약
    print_comparison_summary(results)
    
    print(f"\n✅ 모음 성능 비교 완료!")

if __name__ == "__main__":
    main()


