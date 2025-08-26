#!/usr/bin/env python3
"""
과적합 분석 스크립트
F1-Score 1.000이 과적합인지 확인하는 종합 분석
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore')

def analyze_overfitting_indicators():
    """과적합 지표 분석"""
    
    print("🔍 과적합 분석 시작")
    print("=" * 50)
    
    # 결과 파일 로드
    with open('improved_preprocessing_classification_report.json', 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    # 훈련 곡선 데이터 확인
    try:
        with open('improved_preprocessing_training_curves.png', 'r') as f:
            print("✅ 훈련 곡선 파일 존재")
    except:
        print("⚠️ 훈련 곡선 파일 없음")
    
    # 클래스별 성능 데이터 추출
    detailed_report = results['detailed_report']
    class_names = [k for k in detailed_report.keys() if k not in ['accuracy', 'macro avg', 'weighted avg']]
    
    # F1-Score 1.000 클래스들 식별
    perfect_classes = []
    non_perfect_classes = []
    
    for class_name in class_names:
        if class_name in detailed_report:
            f1_score = detailed_report[class_name]['f1-score']
            precision = detailed_report[class_name]['precision']
            recall = detailed_report[class_name]['recall']
            support = detailed_report[class_name]['support']
            
            if f1_score == 1.0:
                perfect_classes.append({
                    'class': class_name,
                    'f1_score': f1_score,
                    'precision': precision,
                    'recall': recall,
                    'support': support
                })
            else:
                non_perfect_classes.append({
                    'class': class_name,
                    'f1_score': f1_score,
                    'precision': precision,
                    'recall': recall,
                    'support': support
                })
    
    print(f"\n📊 F1-Score 1.000 클래스 분석:")
    print(f"  완벽한 성능 클래스: {len(perfect_classes)}개")
    print(f"  일반 성능 클래스: {len(non_perfect_classes)}개")
    
    # 1. Support (데이터 수) 분석
    print(f"\n📈 데이터 수 분석:")
    perfect_supports = [cls['support'] for cls in perfect_classes]
    non_perfect_supports = [cls['support'] for cls in non_perfect_classes]
    
    print(f"  완벽한 성능 클래스 평균 데이터 수: {np.mean(perfect_supports):.1f}")
    print(f"  일반 성능 클래스 평균 데이터 수: {np.mean(non_perfect_supports):.1f}")
    
    if np.mean(perfect_supports) < np.mean(non_perfect_supports):
        print(f"  ⚠️ 과적합 의심: 완벽한 성능 클래스의 데이터 수가 더 적음")
    else:
        print(f"  ✅ 데이터 수는 과적합과 무관")
    
    # 2. Precision vs Recall 패턴 분석
    print(f"\n📊 Precision vs Recall 패턴 분석:")
    
    perfect_precisions = [cls['precision'] for cls in perfect_classes]
    perfect_recalls = [cls['recall'] for cls in perfect_classes]
    
    print(f"  완벽한 성능 클래스:")
    print(f"    평균 Precision: {np.mean(perfect_precisions):.3f}")
    print(f"    평균 Recall: {np.mean(perfect_recalls):.3f}")
    
    # 3. 클래스별 상세 분석
    print(f"\n🔍 완벽한 성능 클래스 상세 분석:")
    print("-" * 60)
    print(f"{'Class':<4} {'F1':<6} {'Precision':<10} {'Recall':<8} {'Support':<8} {'Pattern':<15}")
    print("-" * 60)
    
    for cls in perfect_classes:
        pattern = "Perfect" if cls['precision'] == 1.0 and cls['recall'] == 1.0 else "High"
        print(f"{cls['class']:<4} {cls['f1_score']:<6.3f} {cls['precision']:<10.3f} {cls['recall']:<8.3f} {cls['support']:<8.0f} {pattern:<15}")
    
    # 4. 과적합 가능성 평가
    print(f"\n⚠️ 과적합 가능성 평가:")
    
    overfitting_risk = 0
    risk_factors = []
    
    # Factor 1: 데이터 수가 적은 클래스가 완벽한 성능
    low_support_perfect = [cls for cls in perfect_classes if cls['support'] <= 3]
    if len(low_support_perfect) > 0:
        overfitting_risk += 2
        risk_factors.append(f"데이터 수가 적은 클래스({len(low_support_perfect)}개)가 완벽한 성능")
    
    # Factor 2: Precision과 Recall이 모두 1.0인 경우
    perfect_both = [cls for cls in perfect_classes if cls['precision'] == 1.0 and cls['recall'] == 1.0]
    if len(perfect_both) > 0:
        overfitting_risk += 1
        risk_factors.append(f"Precision과 Recall이 모두 1.0인 클래스({len(perfect_both)}개)")
    
    # Factor 3: 일반적인 성능 클래스와의 격차
    avg_perfect_f1 = np.mean([cls['f1_score'] for cls in perfect_classes])
    avg_non_perfect_f1 = np.mean([cls['f1_score'] for cls in non_perfect_classes])
    gap = avg_perfect_f1 - avg_non_perfect_f1
    
    if gap > 0.3:
        overfitting_risk += 1
        risk_factors.append(f"성능 격차가 큼 (평균 격차: {gap:.3f})")
    
    # 과적합 위험도 평가
    print(f"  과적합 위험도: {overfitting_risk}/4")
    
    if overfitting_risk >= 3:
        print(f"  🔴 높은 과적합 위험")
    elif overfitting_risk >= 2:
        print(f"  🟡 중간 과적합 위험")
    else:
        print(f"  🟢 낮은 과적합 위험")
    
    if risk_factors:
        print(f"  위험 요인:")
        for factor in risk_factors:
            print(f"    - {factor}")
    
    # 5. 추가 검증 방법 제안
    print(f"\n💡 추가 검증 방법:")
    print(f"  1. 교차 검증 (Cross-Validation) 수행")
    print(f"  2. 다른 데이터셋에서 테스트")
    print(f"  3. 데이터 증강 후 재학습")
    print(f"  4. 모델 복잡도 감소")
    
    return perfect_classes, non_perfect_classes, overfitting_risk

def create_overfitting_analysis_visualization(perfect_classes, non_perfect_classes):
    """과적합 분석 시각화"""
    print(f"\n📊 과적합 분석 시각화 생성 중...")
    
    plt.figure(figsize=(15, 10))
    
    # 1. 데이터 수 vs F1-Score
    plt.subplot(2, 3, 1)
    
    perfect_supports = [cls['support'] for cls in perfect_classes]
    perfect_f1s = [cls['f1_score'] for cls in perfect_classes]
    non_perfect_supports = [cls['support'] for cls in non_perfect_classes]
    non_perfect_f1s = [cls['f1_score'] for cls in non_perfect_classes]
    
    plt.scatter(perfect_supports, perfect_f1s, c='red', s=100, alpha=0.7, label='Perfect (F1=1.0)')
    plt.scatter(non_perfect_supports, non_perfect_f1s, c='blue', s=100, alpha=0.7, label='Non-Perfect')
    
    plt.xlabel('Support (데이터 수)')
    plt.ylabel('F1-Score')
    plt.title('데이터 수 vs F1-Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Precision vs Recall
    plt.subplot(2, 3, 2)
    
    perfect_precisions = [cls['precision'] for cls in perfect_classes]
    perfect_recalls = [cls['recall'] for cls in perfect_classes]
    non_perfect_precisions = [cls['precision'] for cls in non_perfect_classes]
    non_perfect_recalls = [cls['recall'] for cls in non_perfect_classes]
    
    plt.scatter(perfect_precisions, perfect_recalls, c='red', s=100, alpha=0.7, label='Perfect')
    plt.scatter(non_perfect_precisions, non_perfect_recalls, c='blue', s=100, alpha=0.7, label='Non-Perfect')
    
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.title('Precision vs Recall')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. 성능 분포 히스토그램
    plt.subplot(2, 3, 3)
    
    all_f1_scores = perfect_f1s + non_perfect_f1s
    plt.hist(all_f1_scores, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(x=1.0, color='red', linestyle='--', label='Perfect Score')
    plt.xlabel('F1-Score')
    plt.ylabel('Number of Classes')
    plt.title('F1-Score Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. 클래스별 데이터 수 비교
    plt.subplot(2, 3, 4)
    
    all_supports = perfect_supports + non_perfect_supports
    all_labels = ['Perfect'] * len(perfect_supports) + ['Non-Perfect'] * len(non_perfect_supports)
    
    perfect_mean = np.mean(perfect_supports)
    non_perfect_mean = np.mean(non_perfect_supports)
    
    plt.bar(['Perfect', 'Non-Perfect'], [perfect_mean, non_perfect_mean], 
            color=['red', 'blue'], alpha=0.7)
    plt.ylabel('Average Support')
    plt.title('Average Data Count by Performance')
    plt.grid(True, alpha=0.3)
    
    # 5. 성능 격차 분석
    plt.subplot(2, 3, 5)
    
    avg_perfect_f1 = np.mean(perfect_f1s)
    avg_non_perfect_f1 = np.mean(non_perfect_f1s)
    
    plt.bar(['Perfect Classes', 'Non-Perfect Classes'], [avg_perfect_f1, avg_non_perfect_f1],
            color=['red', 'blue'], alpha=0.7)
    plt.ylabel('Average F1-Score')
    plt.title('Average Performance Comparison')
    plt.ylim(0, 1.1)
    plt.grid(True, alpha=0.3)
    
    # 6. 과적합 위험 지표
    plt.subplot(2, 3, 6)
    
    # 위험 지표 계산
    low_support_ratio = len([s for s in perfect_supports if s <= 3]) / len(perfect_supports)
    perfect_both_ratio = len([cls for cls in perfect_classes if cls['precision'] == 1.0 and cls['recall'] == 1.0]) / len(perfect_classes)
    
    risk_indicators = ['Low Support\n(≤3 samples)', 'Perfect P&R\n(1.0/1.0)', 'High Gap\n(>0.3)']
    risk_values = [low_support_ratio, perfect_both_ratio, 1.0 if avg_perfect_f1 - avg_non_perfect_f1 > 0.3 else 0.0]
    
    colors = ['red' if v > 0.5 else 'orange' if v > 0.2 else 'green' for v in risk_values]
    
    plt.bar(risk_indicators, risk_values, color=colors, alpha=0.7)
    plt.ylabel('Risk Ratio')
    plt.title('Overfitting Risk Indicators')
    plt.ylim(0, 1.1)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('overfitting_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✅ 과적합 분석 시각화 저장됨: overfitting_analysis.png")

def generate_overfitting_report(perfect_classes, non_perfect_classes, overfitting_risk):
    """과적합 분석 보고서 생성"""
    
    report = {
        'analysis_summary': {
            'total_classes': len(perfect_classes) + len(non_perfect_classes),
            'perfect_classes_count': len(perfect_classes),
            'non_perfect_classes_count': len(non_perfect_classes),
            'overfitting_risk_score': overfitting_risk,
            'risk_level': 'High' if overfitting_risk >= 3 else 'Medium' if overfitting_risk >= 2 else 'Low'
        },
        'perfect_classes': perfect_classes,
        'non_perfect_classes': non_perfect_classes,
        'statistical_analysis': {
            'perfect_avg_support': np.mean([cls['support'] for cls in perfect_classes]),
            'non_perfect_avg_support': np.mean([cls['support'] for cls in non_perfect_classes]),
            'perfect_avg_f1': np.mean([cls['f1_score'] for cls in perfect_classes]),
            'non_perfect_avg_f1': np.mean([cls['f1_score'] for cls in non_perfect_classes]),
            'performance_gap': np.mean([cls['f1_score'] for cls in perfect_classes]) - np.mean([cls['f1_score'] for cls in non_perfect_classes])
        },
        'recommendations': [
            '교차 검증을 통한 모델 안정성 확인',
            '데이터 증강을 통한 일반화 성능 향상',
            '모델 복잡도 감소 고려',
            '정규화 기법 강화',
            '다른 데이터셋에서의 검증 수행'
        ]
    }
    
    with open('overfitting_analysis_report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"\n📄 과적합 분석 보고서 저장됨: overfitting_analysis_report.json")

def main():
    """메인 함수"""
    perfect_classes, non_perfect_classes, overfitting_risk = analyze_overfitting_indicators()
    
    create_overfitting_analysis_visualization(perfect_classes, non_perfect_classes)
    generate_overfitting_report(perfect_classes, non_perfect_classes, overfitting_risk)
    
    print(f"\n🎯 과적합 분석 완료!")
    print(f"📊 주요 결론:")
    
    if overfitting_risk >= 3:
        print(f"  🔴 과적합 위험이 높습니다. 추가 검증이 필요합니다.")
    elif overfitting_risk >= 2:
        print(f"  🟡 과적합 위험이 있습니다. 주의가 필요합니다.")
    else:
        print(f"  🟢 과적합 위험이 낮습니다. 현재 성능이 신뢰할 수 있습니다.")

if __name__ == "__main__":
    main()
