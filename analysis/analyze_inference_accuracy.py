#!/usr/bin/env python3
"""
추론 결과 정확도 분석
실제 파일명과 예측 결과를 비교하여 정확도 계산
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
import re
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

class InferenceAccuracyAnalyzer:
    """추론 정확도 분석기"""
    
    def __init__(self):
        self.class_names = ['ㄱ', 'ㄴ', 'ㄷ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅅ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ',
                           'ㅏ', 'ㅑ', 'ㅓ', 'ㅕ', 'ㅗ', 'ㅛ', 'ㅜ', 'ㅠ', 'ㅡ', 'ㅣ']
        print('🔍 추론 정확도 분석기 초기화 완료')
    
    def load_inference_results(self, results_file='corrected_filtered_results.json'):
        """추론 결과 로드"""
        try:
            with open(results_file, 'r', encoding='utf-8') as f:
                results = json.load(f)
            print(f'✅ 추론 결과 로드 완료: {len(results)}개 파일')
            return results
        except Exception as e:
            print(f'❌ 결과 파일 로드 실패: {e}')
            return None
    
    def extract_true_class(self, filename):
        """파일명에서 실제 클래스 추출"""
        # 파일명 패턴: real_data_filtered/ㅁ/1/episode_20250819_201111_ㅁ_1.csv
        match = re.search(r'/([ㄱ-ㅎㅏ-ㅣ])/\d+/episode_', filename)
        if match:
            return match.group(1)
        return None
    
    def analyze_accuracy(self, results):
        """정확도 분석"""
        print('\n📊 추론 정확도 분석 시작...')
        
        # 결과 데이터프레임 생성
        data = []
        for result in results:
            filename = result['file']
            predicted = result['predicted_class']
            confidence = result['confidence']
            true_class = self.extract_true_class(filename)
            
            if true_class:
                data.append({
                    'filename': filename,
                    'true_class': true_class,
                    'predicted_class': predicted,
                    'confidence': confidence,
                    'correct': true_class == predicted
                })
        
        df = pd.DataFrame(data)
        
        if df.empty:
            print('❌ 분석할 데이터가 없습니다.')
            return None
        
        print(f'📁 분석 대상: {len(df)}개 파일')
        
        # 전체 정확도
        overall_accuracy = df['correct'].mean()
        print(f'🎯 전체 정확도: {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)')
        
        # 클래스별 정확도
        class_accuracy = df.groupby('true_class')['correct'].agg(['mean', 'count']).round(4)
        class_accuracy.columns = ['accuracy', 'count']
        class_accuracy = class_accuracy.sort_values('accuracy', ascending=False)
        
        print('\n📊 클래스별 정확도:')
        print(class_accuracy)
        
        # 혼동 행렬 생성
        confusion_matrix = pd.crosstab(df['true_class'], df['predicted_class'], margins=True)
        
        # 신뢰도 분석
        confidence_analysis = df.groupby('correct')['confidence'].agg(['mean', 'std', 'count']).round(4)
        confidence_analysis.columns = ['mean_confidence', 'std_confidence', 'count']
        
        print('\n📊 신뢰도 분석:')
        print(confidence_analysis)
        
        return {
            'df': df,
            'overall_accuracy': overall_accuracy,
            'class_accuracy': class_accuracy,
            'confusion_matrix': confusion_matrix,
            'confidence_analysis': confidence_analysis
        }
    
    def plot_accuracy_analysis(self, analysis_results, save_path='inference_accuracy_analysis.png'):
        """정확도 분석 시각화"""
        if not analysis_results:
            return
        
        df = analysis_results['df']
        class_accuracy = analysis_results['class_accuracy']
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('SignGlove Inference Accuracy Analysis', fontsize=16, fontweight='bold')
        
        # 1. 클래스별 정확도
        colors = ['green' if acc > 0.8 else 'orange' if acc > 0.6 else 'red' for acc in class_accuracy['accuracy']]
        bars = ax1.bar(range(len(class_accuracy)), class_accuracy['accuracy'], color=colors, alpha=0.7)
        ax1.set_title('Class-wise Accuracy', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Class')
        ax1.set_ylabel('Accuracy')
        ax1.set_xticks(range(len(class_accuracy)))
        ax1.set_xticklabels(class_accuracy.index, rotation=45)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1.0)
        
        # 정확도 값 표시
        for i, (bar, acc) in enumerate(zip(bars, class_accuracy['accuracy'])):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{acc:.3f}', ha='center', va='bottom', fontsize=8)
        
        # 2. 정확/오답 신뢰도 분포
        correct_conf = df[df['correct']]['confidence']
        incorrect_conf = df[~df['correct']]['confidence']
        
        ax2.hist(correct_conf, bins=20, alpha=0.7, label='Correct Predictions', color='green', density=True)
        ax2.hist(incorrect_conf, bins=20, alpha=0.7, label='Incorrect Predictions', color='red', density=True)
        ax2.set_title('Confidence Distribution by Accuracy', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Confidence')
        ax2.set_ylabel('Density')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 클래스별 샘플 수와 정확도
        ax3.scatter(class_accuracy['count'], class_accuracy['accuracy'], 
                   s=100, alpha=0.7, c=colors)
        ax3.set_title('Sample Count vs Accuracy', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Sample Count')
        ax3.set_ylabel('Accuracy')
        ax3.grid(True, alpha=0.3)
        
        # 클래스명 표시
        for i, (idx, row) in enumerate(class_accuracy.iterrows()):
            ax3.annotate(idx, (row['count'], row['accuracy']), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # 4. 전체 성능 요약
        overall_acc = analysis_results['overall_accuracy']
        total_files = len(df)
        correct_files = df['correct'].sum()
        incorrect_files = total_files - correct_files
        
        summary_text = f"""
Overall Accuracy: {overall_acc:.4f} ({overall_acc*100:.2f}%)

Total Files: {total_files}
Correct: {correct_files}
Incorrect: {incorrect_files}

High Accuracy Classes (>80%): {len(class_accuracy[class_accuracy['accuracy'] > 0.8])}
Medium Accuracy Classes (60-80%): {len(class_accuracy[(class_accuracy['accuracy'] >= 0.6) & (class_accuracy['accuracy'] <= 0.8)])}
Low Accuracy Classes (<60%): {len(class_accuracy[class_accuracy['accuracy'] < 0.6])}
        """
        
        ax4.axis('off')
        ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=12,
                verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'✅ 정확도 분석 시각화 저장: {save_path}')
        plt.show()
    
    def generate_detailed_report(self, analysis_results, save_path='inference_accuracy_report.txt'):
        """상세 정확도 보고서 생성"""
        if not analysis_results:
            return
        
        df = analysis_results['df']
        class_accuracy = analysis_results['class_accuracy']
        confusion_matrix = analysis_results['confusion_matrix']
        confidence_analysis = analysis_results['confidence_analysis']
        
        report = []
        report.append("=" * 70)
        report.append("SIGNGLOVE INFERENCE ACCURACY REPORT")
        report.append("=" * 70)
        report.append("")
        
        # 전체 성능
        overall_acc = analysis_results['overall_accuracy']
        report.append("📊 OVERALL PERFORMANCE:")
        report.append(f"  Total Files: {len(df)}")
        report.append(f"  Overall Accuracy: {overall_acc:.4f} ({overall_acc*100:.2f}%)")
        report.append(f"  Correct Predictions: {df['correct'].sum()}")
        report.append(f"  Incorrect Predictions: {len(df) - df['correct'].sum()}")
        report.append("")
        
        # 클래스별 성능
        report.append("🎯 CLASS-WISE PERFORMANCE:")
        for class_name in class_accuracy.index:
            acc = class_accuracy.loc[class_name, 'accuracy']
            count = class_accuracy.loc[class_name, 'count']
            status = "✅" if acc > 0.8 else "⚠️" if acc > 0.6 else "❌"
            report.append(f"  {status} {class_name}: {acc:.4f} ({acc*100:.1f}%) - {count} samples")
        report.append("")
        
        # 신뢰도 분석
        report.append("📈 CONFIDENCE ANALYSIS:")
        for correct in [True, False]:
            status = "Correct" if correct else "Incorrect"
            mean_conf = confidence_analysis.loc[correct, 'mean_confidence']
            std_conf = confidence_analysis.loc[correct, 'std_confidence']
            count = confidence_analysis.loc[correct, 'count']
            report.append(f"  {status}: {mean_conf:.4f} ± {std_conf:.4f} ({count} samples)")
        report.append("")
        
        # 문제점 분석
        report.append("🔍 ISSUE ANALYSIS:")
        low_acc_classes = class_accuracy[class_accuracy['accuracy'] < 0.6]
        if not low_acc_classes.empty:
            report.append("  ❌ Low Accuracy Classes (<60%):")
            for class_name in low_acc_classes.index:
                acc = low_acc_classes.loc[class_name, 'accuracy']
                report.append(f"    - {class_name}: {acc:.4f} ({acc*100:.1f}%)")
        else:
            report.append("  ✅ All classes have acceptable accuracy (>60%)")
        
        # 혼동 분석
        report.append("")
        report.append("🔄 CONFUSION ANALYSIS:")
        # 가장 많이 혼동되는 클래스들
        incorrect_predictions = df[~df['correct']]
        if not incorrect_predictions.empty:
            confusion_pairs = incorrect_predictions.groupby(['true_class', 'predicted_class']).size()
            confusion_pairs = confusion_pairs.sort_values(ascending=False)
            
            report.append("  Most Common Confusions:")
            for (true, pred), count in confusion_pairs.head(5).items():
                report.append(f"    {true} → {pred}: {count} times")
        report.append("")
        
        # 권장사항
        report.append("💡 RECOMMENDATIONS:")
        if overall_acc < 0.8:
            report.append("  - Consider model retraining with more data")
            report.append("  - Check data preprocessing consistency")
        
        if len(low_acc_classes) > 0:
            report.append("  - Focus on improving low-accuracy classes")
            report.append("  - Consider class-specific data augmentation")
        
        if confidence_analysis.loc[False, 'mean_confidence'] > 0.7:
            report.append("  - High confidence incorrect predictions detected")
            report.append("  - Consider confidence calibration")
        
        report.append("")
        report.append("=" * 70)
        
        # 파일 저장
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        
        print(f'✅ 상세 정확도 보고서 저장: {save_path}')
        
        # 콘솔 출력
        for line in report:
            print(line)

def main():
    """메인 실행 함수"""
    print('🔍 추론 정확도 분석 시스템')
    print('=' * 60)
    
    # 분석기 초기화
    analyzer = InferenceAccuracyAnalyzer()
    
    # 추론 결과 로드
    results = analyzer.load_inference_results()
    
    if results is None:
        print('❌ 추론 결과를 로드할 수 없습니다.')
        return
    
    # 정확도 분석
    analysis_results = analyzer.analyze_accuracy(results)
    
    if analysis_results is None:
        print('❌ 정확도 분석을 수행할 수 없습니다.')
        return
    
    # 시각화
    print('\n📊 정확도 분석 시각화 생성 중...')
    analyzer.plot_accuracy_analysis(analysis_results, 'inference_accuracy_analysis.png')
    
    # 상세 보고서
    print('\n📋 상세 정확도 보고서 생성 중...')
    analyzer.generate_detailed_report(analysis_results, 'inference_accuracy_report.txt')
    
    print('\n🎉 추론 정확도 분석이 완료되었습니다!')
    print('📁 생성된 파일들:')
    print('  - inference_accuracy_analysis.png')
    print('  - inference_accuracy_report.txt')

if __name__ == "__main__":
    main()
