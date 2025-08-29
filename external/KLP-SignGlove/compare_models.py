#!/usr/bin/env python3
"""
모델 성능 비교 분석
이전 모델과 과적합 방지 모델의 성능을 비교
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def load_results(filename):
    """결과 파일 로드"""
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)

def compare_models():
    """모델 성능 비교"""
    print("🔍 모델 성능 비교 분석")
    print("=" * 60)
    
    # 결과 로드
    try:
        old_results = load_results('strict_validation_results.json')
        new_results = load_results('anti_overfitting_test_results.json')
    except FileNotFoundError as e:
        print(f"❌ 결과 파일을 찾을 수 없습니다: {e}")
        return
    
    # 비교 테이블 생성
    print("\n📊 모델 성능 비교")
    print("-" * 80)
    print(f"{'검증 타입':<15} {'이전 모델':<20} {'과적합 방지 모델':<20} {'개선도':<10}")
    print("-" * 80)
    
    improvements = []
    
    for old, new in zip(old_results, new_results):
        validation_type = old['validation_type']
        old_acc = old['overall_accuracy']
        new_acc = new['overall_accuracy']
        old_conf = old['avg_confidence']
        new_conf = new['avg_confidence']
        old_score = old['overfitting_score']
        new_score = new['overfitting_score']
        
        # 정확도 개선도
        acc_improvement = ((new_acc - old_acc) / old_acc) * 100 if old_acc > 0 else 0
        
        print(f"{validation_type:<15} {old_acc:.4f} ({old_acc*100:.1f}%) {'':<5} {new_acc:.4f} ({new_acc*100:.1f}%) {'':<5} {acc_improvement:+.1f}%")
        
        improvements.append({
            'validation_type': validation_type,
            'old_accuracy': old_acc,
            'new_accuracy': new_acc,
            'old_confidence': old_conf,
            'new_confidence': new_conf,
            'old_overfitting_score': old_score,
            'new_overfitting_score': new_score,
            'accuracy_improvement': acc_improvement
        })
    
    print("\n📈 신뢰도 비교")
    print("-" * 80)
    print(f"{'검증 타입':<15} {'이전 모델':<20} {'과적합 방지 모델':<20} {'개선도':<10}")
    print("-" * 80)
    
    for imp in improvements:
        conf_improvement = ((new_conf - old_conf) / old_conf) * 100 if old_conf > 0 else 0
        print(f"{imp['validation_type']:<15} {imp['old_confidence']:.4f} ({imp['old_confidence']*100:.1f}%) {'':<5} {imp['new_confidence']:.4f} ({imp['new_confidence']*100:.1f}%) {'':<5} {conf_improvement:+.1f}%")
    
    print("\n🏆 과적합 점수 비교")
    print("-" * 80)
    print(f"{'검증 타입':<15} {'이전 모델':<20} {'과적합 방지 모델':<20} {'개선도':<10}")
    print("-" * 80)
    
    for imp in improvements:
        score_improvement = imp['old_overfitting_score'] - imp['new_overfitting_score']
        print(f"{imp['validation_type']:<15} {imp['old_overfitting_score']}/6 {'':<15} {imp['new_overfitting_score']}/6 {'':<15} {score_improvement:+d}점")
    
    # 시각화
    create_comparison_plots(improvements)
    
    # 종합 평가
    print("\n🎯 종합 평가")
    print("=" * 60)
    
    total_old_score = sum(imp['old_overfitting_score'] for imp in improvements)
    total_new_score = sum(imp['new_overfitting_score'] for imp in improvements)
    
    print(f"📊 전체 과적합 점수:")
    print(f"  이전 모델: {total_old_score}/18 ({total_old_score/18*100:.1f}%)")
    print(f"  과적합 방지 모델: {total_new_score}/18 ({total_new_score/18*100:.1f}%)")
    print(f"  개선도: {total_old_score - total_new_score}점 감소")
    
    avg_old_acc = np.mean([imp['old_accuracy'] for imp in improvements])
    avg_new_acc = np.mean([imp['new_accuracy'] for imp in improvements])
    
    print(f"\n📊 평균 정확도:")
    print(f"  이전 모델: {avg_old_acc:.4f} ({avg_old_acc*100:.1f}%)")
    print(f"  과적합 방지 모델: {avg_new_acc:.4f} ({avg_new_acc*100:.1f}%)")
    print(f"  변화: {((avg_new_acc - avg_old_acc) / avg_old_acc) * 100:+.1f}%")
    
    avg_old_conf = np.mean([imp['old_confidence'] for imp in improvements])
    avg_new_conf = np.mean([imp['new_confidence'] for imp in improvements])
    
    print(f"\n📊 평균 신뢰도:")
    print(f"  이전 모델: {avg_old_conf:.4f} ({avg_old_conf*100:.1f}%)")
    print(f"  과적합 방지 모델: {avg_new_conf:.4f} ({avg_new_conf*100:.1f}%)")
    print(f"  변화: {((avg_new_conf - avg_old_conf) / avg_old_conf) * 100:+.1f}%")
    
    # 개선 효과 평가
    print(f"\n✅ 과적합 방지 효과:")
    if total_new_score < total_old_score:
        print(f"  🎉 과적합이 성공적으로 감소했습니다!")
        print(f"  📉 과적합 점수가 {total_old_score - total_new_score}점 감소")
        
        if avg_new_acc > 0.85:
            print(f"  🎯 정확도는 여전히 높은 수준을 유지합니다 ({avg_new_acc*100:.1f}%)")
        else:
            print(f"  ⚠️ 정확도가 다소 감소했습니다 ({avg_new_acc*100:.1f}%)")
        
        if avg_new_conf < 0.8:
            print(f"  🎯 신뢰도가 현실적인 수준으로 조정되었습니다 ({avg_new_conf*100:.1f}%)")
        else:
            print(f"  ⚠️ 신뢰도가 여전히 높습니다 ({avg_new_conf*100:.1f}%)")
    else:
        print(f"  ❌ 과적합 감소 효과가 제한적입니다.")
    
    # 결과 저장
    comparison_results = {
        'improvements': improvements,
        'summary': {
            'total_old_overfitting_score': total_old_score,
            'total_new_overfitting_score': total_new_score,
            'avg_old_accuracy': float(avg_old_acc),
            'avg_new_accuracy': float(avg_new_acc),
            'avg_old_confidence': float(avg_old_conf),
            'avg_new_confidence': float(avg_new_conf)
        }
    }
    
    with open('model_comparison_results.json', 'w', encoding='utf-8') as f:
        json.dump(comparison_results, f, ensure_ascii=False, indent=2)
    
    print(f"\n📁 비교 결과가 'model_comparison_results.json'에 저장되었습니다.")

def create_comparison_plots(improvements):
    """비교 차트 생성"""
    print("\n📊 비교 차트 생성 중...")
    
    # 데이터 준비
    validation_types = [imp['validation_type'] for imp in improvements]
    old_accuracies = [imp['old_accuracy'] for imp in improvements]
    new_accuracies = [imp['new_accuracy'] for imp in improvements]
    old_confidences = [imp['old_confidence'] for imp in improvements]
    new_confidences = [imp['new_confidence'] for imp in improvements]
    old_scores = [imp['old_overfitting_score'] for imp in improvements]
    new_scores = [imp['new_overfitting_score'] for imp in improvements]
    
    # 차트 생성
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('모델 성능 비교 분석', fontsize=16, fontweight='bold')
    
    # 1. 정확도 비교
    x = np.arange(len(validation_types))
    width = 0.35
    
    axes[0, 0].bar(x - width/2, old_accuracies, width, label='이전 모델', alpha=0.8, color='red')
    axes[0, 0].bar(x + width/2, new_accuracies, width, label='과적합 방지 모델', alpha=0.8, color='blue')
    axes[0, 0].set_xlabel('검증 타입')
    axes[0, 0].set_ylabel('정확도')
    axes[0, 0].set_title('정확도 비교')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(validation_types)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 신뢰도 비교
    axes[0, 1].bar(x - width/2, old_confidences, width, label='이전 모델', alpha=0.8, color='red')
    axes[0, 1].bar(x + width/2, new_confidences, width, label='과적합 방지 모델', alpha=0.8, color='blue')
    axes[0, 1].set_xlabel('검증 타입')
    axes[0, 1].set_ylabel('신뢰도')
    axes[0, 1].set_title('신뢰도 비교')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(validation_types)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 과적합 점수 비교
    axes[1, 0].bar(x - width/2, old_scores, width, label='이전 모델', alpha=0.8, color='red')
    axes[1, 0].bar(x + width/2, new_scores, width, label='과적합 방지 모델', alpha=0.8, color='blue')
    axes[1, 0].set_xlabel('검증 타입')
    axes[1, 0].set_ylabel('과적합 점수 (0-6)')
    axes[1, 0].set_title('과적합 점수 비교')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(validation_types)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim(0, 6)
    
    # 4. 개선도 비교
    accuracy_improvements = [imp['accuracy_improvement'] for imp in improvements]
    axes[1, 1].bar(x, accuracy_improvements, alpha=0.8, color='green')
    axes[1, 1].set_xlabel('검증 타입')
    axes[1, 1].set_ylabel('정확도 개선도 (%)')
    axes[1, 1].set_title('정확도 개선도')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(validation_types)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('model_comparison_analysis.png', dpi=300, bbox_inches='tight')
    print("📊 비교 차트가 'model_comparison_analysis.png'에 저장되었습니다.")

if __name__ == "__main__":
    compare_models()



