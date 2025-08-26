#!/usr/bin/env python3
"""
클래스별 정확도 상세 분석
각 클래스의 정확도, precision, recall, F1-score를 명확하게 표시
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def analyze_class_accuracies():
    """클래스별 정확도 분석"""
    
    # 결과 파일 로드
    with open('improved_preprocessing_classification_report.json', 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    print("🎯 클래스별 정확도 상세 분석")
    print("=" * 60)
    
    # 전체 정확도
    overall_accuracy = results['overall_accuracy']
    print(f"📊 전체 정확도: {overall_accuracy:.2%}")
    print()
    
    # 클래스별 성능 데이터 추출
    detailed_report = results['detailed_report']
    class_names = [k for k in detailed_report.keys() if k not in ['accuracy', 'macro avg', 'weighted avg']]
    
    # 데이터프레임 생성
    class_data = []
    for class_name in class_names:
        if class_name in detailed_report:
            data = detailed_report[class_name]
            class_data.append({
                'Class': class_name,
                'Precision': data['precision'],
                'Recall': data['recall'],
                'F1-Score': data['f1-score'],
                'Support': data['support']
            })
    
    df = pd.DataFrame(class_data)
    
    # 정확도 계산 (Precision을 정확도로 사용)
    df['Accuracy'] = df['Precision']
    
    # 성능 등급 분류
    def get_performance_grade(f1_score):
        if f1_score >= 0.9:
            return '🟢 Excellent'
        elif f1_score >= 0.8:
            return '🟡 Good'
        elif f1_score >= 0.7:
            return '🟠 Fair'
        else:
            return '🔴 Poor'
    
    df['Grade'] = df['F1-Score'].apply(get_performance_grade)
    
    # 성능별 정렬
    df_sorted = df.sort_values('F1-Score', ascending=False)
    
    # 상세 테이블 출력
    print("📋 클래스별 상세 성능 (F1-Score 순)")
    print("-" * 80)
    print(f"{'Class':<4} {'Accuracy':<10} {'Precision':<10} {'Recall':<8} {'F1-Score':<9} {'Support':<8} {'Grade':<12}")
    print("-" * 80)
    
    for _, row in df_sorted.iterrows():
        print(f"{row['Class']:<4} {row['Accuracy']:<10.3f} {row['Precision']:<10.3f} {row['Recall']:<8.3f} {row['F1-Score']:<9.3f} {row['Support']:<8.0f} {row['Grade']:<12}")
    
    print("-" * 80)
    
    # 성능 통계
    print(f"\n📊 성능 통계:")
    print(f"  평균 정확도: {df['Accuracy'].mean():.3f}")
    print(f"  평균 F1-Score: {df['F1-Score'].mean():.3f}")
    print(f"  최고 F1-Score: {df['F1-Score'].max():.3f} ({df_sorted.iloc[0]['Class']})")
    print(f"  최저 F1-Score: {df['F1-Score'].min():.3f} ({df_sorted.iloc[-1]['Class']})")
    
    # 성능 등급별 분포
    grade_counts = df['Grade'].value_counts()
    print(f"\n📈 성능 등급별 분포:")
    for grade, count in grade_counts.items():
        print(f"  {grade}: {count}개 클래스")
    
    # 높은 성능 클래스들
    excellent_classes = df[df['F1-Score'] >= 0.9]['Class'].tolist()
    poor_classes = df[df['F1-Score'] < 0.7]['Class'].tolist()
    
    print(f"\n🏆 우수 성능 클래스들 (F1-Score ≥ 0.9):")
    print(f"  {', '.join(excellent_classes)}")
    
    if poor_classes:
        print(f"\n⚠️ 개선 필요 클래스들 (F1-Score < 0.7):")
        print(f"  {', '.join(poor_classes)}")
    
    # 시각화 생성
    create_class_accuracy_visualizations(df, df_sorted)
    
    return df, df_sorted

def create_class_accuracy_visualizations(df, df_sorted):
    """클래스별 정확도 시각화"""
    print(f"\n📊 시각화 생성 중...")
    
    # 1. 클래스별 F1-Score 막대 그래프
    plt.figure(figsize=(16, 10))
    
    plt.subplot(2, 2, 1)
    classes = df_sorted['Class']
    f1_scores = df_sorted['F1-Score']
    
    # 색상 매핑
    colors = []
    for f1 in f1_scores:
        if f1 >= 0.9:
            colors.append('green')
        elif f1 >= 0.8:
            colors.append('orange')
        elif f1 >= 0.7:
            colors.append('yellow')
        else:
            colors.append('red')
    
    bars = plt.bar(range(len(classes)), f1_scores, color=colors, alpha=0.7)
    plt.title('클래스별 F1-Score', fontsize=14, fontweight='bold')
    plt.xlabel('클래스')
    plt.ylabel('F1-Score')
    plt.xticks(range(len(classes)), classes, rotation=45)
    plt.ylim(0, 1.1)
    
    # 성능 구분선 추가
    plt.axhline(y=0.9, color='green', linestyle='--', alpha=0.7, label='Excellent (0.9)')
    plt.axhline(y=0.8, color='orange', linestyle='--', alpha=0.7, label='Good (0.8)')
    plt.axhline(y=0.7, color='red', linestyle='--', alpha=0.7, label='Fair (0.7)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Precision vs Recall 산점도
    plt.subplot(2, 2, 2)
    plt.scatter(df['Precision'], df['Recall'], s=100, alpha=0.7, c=df['F1-Score'], cmap='RdYlGn')
    
    # 클래스명 표시
    for i, class_name in enumerate(df['Class']):
        plt.annotate(class_name, (df['Precision'].iloc[i], df['Recall'].iloc[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.title('Precision vs Recall (색상: F1-Score)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.colorbar(label='F1-Score')
    
    # 3. 성능 등급별 분포 파이 차트
    plt.subplot(2, 2, 3)
    grade_counts = df['Grade'].value_counts()
    colors_grade = ['green', 'orange', 'yellow', 'red']
    plt.pie(grade_counts.values, labels=grade_counts.index, autopct='%1.1f%%', 
            colors=colors_grade[:len(grade_counts)], startangle=90)
    plt.title('성능 등급별 분포', fontsize=14, fontweight='bold')
    
    # 4. 클래스별 정확도 비교 (Accuracy vs F1-Score)
    plt.subplot(2, 2, 4)
    plt.scatter(df['Accuracy'], df['F1-Score'], s=100, alpha=0.7)
    
    # 클래스명 표시
    for i, class_name in enumerate(df['Class']):
        plt.annotate(class_name, (df['Accuracy'].iloc[i], df['F1-Score'].iloc[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    plt.xlabel('Accuracy (Precision)')
    plt.ylabel('F1-Score')
    plt.title('Accuracy vs F1-Score', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # 대각선 추가 (Accuracy = F1-Score)
    min_val = min(df['Accuracy'].min(), df['F1-Score'].min())
    max_val = max(df['Accuracy'].max(), df['F1-Score'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Accuracy = F1-Score')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('class_accuracy_detailed_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✅ 클래스별 정확도 시각화 저장됨: class_accuracy_detailed_analysis.png")
    
    # 추가: 성능 순위 테이블 생성
    create_performance_ranking_table(df_sorted)

def create_performance_ranking_table(df_sorted):
    """성능 순위 테이블 생성"""
    print(f"\n📊 성능 순위 테이블 생성 중...")
    
    # 순위 추가
    df_ranked = df_sorted.copy()
    df_ranked['Rank'] = range(1, len(df_ranked) + 1)
    
    # 순위별 색상 지정
    def get_rank_color(rank):
        if rank <= 5:
            return '🥇'  # Top 5
        elif rank <= 10:
            return '🥈'  # Top 10
        elif rank <= 15:
            return '🥉'  # Top 15
        else:
            return '📊'  # Others
    
    df_ranked['Rank_Icon'] = df_ranked['Rank'].apply(get_rank_color)
    
    # 순위 테이블 출력
    print(f"\n🏆 클래스별 성능 순위 (상위 10개)")
    print("-" * 70)
    print(f"{'Rank':<4} {'Icon':<4} {'Class':<4} {'F1-Score':<9} {'Accuracy':<10} {'Grade':<12}")
    print("-" * 70)
    
    for i, row in df_ranked.head(10).iterrows():
        print(f"{row['Rank']:<4} {row['Rank_Icon']:<4} {row['Class']:<4} {row['F1-Score']:<9.3f} {row['Accuracy']:<10.3f} {row['Grade']:<12}")
    
    print("-" * 70)
    
    # 하위 5개 클래스
    print(f"\n⚠️ 성능 개선 필요 클래스들 (하위 5개)")
    print("-" * 70)
    print(f"{'Rank':<4} {'Icon':<4} {'Class':<4} {'F1-Score':<9} {'Accuracy':<10} {'Grade':<12}")
    print("-" * 70)
    
    for i, row in df_ranked.tail(5).iterrows():
        print(f"{row['Rank']:<4} {row['Rank_Icon']:<4} {row['Class']:<4} {row['F1-Score']:<9.3f} {row['Accuracy']:<10.3f} {row['Grade']:<12}")
    
    print("-" * 70)
    
    # 순위별 통계
    top_5_avg = df_ranked.head(5)['F1-Score'].mean()
    bottom_5_avg = df_ranked.tail(5)['F1-Score'].mean()
    
    print(f"\n📈 순위별 평균 성능:")
    print(f"  상위 5개 클래스 평균 F1-Score: {top_5_avg:.3f}")
    print(f"  하위 5개 클래스 평균 F1-Score: {bottom_5_avg:.3f}")
    print(f"  성능 격차: {top_5_avg - bottom_5_avg:.3f}")
    
    # 결과 저장
    ranking_summary = {
        'top_5_classes': df_ranked.head(5)['Class'].tolist(),
        'bottom_5_classes': df_ranked.tail(5)['Class'].tolist(),
        'top_5_avg_f1': top_5_avg,
        'bottom_5_avg_f1': bottom_5_avg,
        'performance_gap': top_5_avg - bottom_5_avg,
        'excellent_classes': df_ranked[df_ranked['F1-Score'] >= 0.9]['Class'].tolist(),
        'poor_classes': df_ranked[df_ranked['F1-Score'] < 0.7]['Class'].tolist()
    }
    
    with open('class_accuracy_ranking_summary.json', 'w', encoding='utf-8') as f:
        json.dump(ranking_summary, f, ensure_ascii=False, indent=2)
    
    print(f"\n📄 순위 요약 저장됨: class_accuracy_ranking_summary.json")

def main():
    """메인 함수"""
    df, df_sorted = analyze_class_accuracies()
    
    print(f"\n🎉 클래스별 정확도 분석 완료!")
    print(f"📊 총 24개 클래스 중:")
    print(f"  🟢 우수 성능 (≥90%): {len(df[df['F1-Score'] >= 0.9])}개")
    print(f"  🟡 양호 성능 (80-90%): {len(df[(df['F1-Score'] >= 0.8) & (df['F1-Score'] < 0.9)])}개")
    print(f"  🟠 보통 성능 (70-80%): {len(df[(df['F1-Score'] >= 0.7) & (df['F1-Score'] < 0.8)])}개")
    print(f"  🔴 개선 필요 (<70%): {len(df[df['F1-Score'] < 0.7])}개")

if __name__ == "__main__":
    main()
