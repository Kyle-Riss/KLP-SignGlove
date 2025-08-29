#!/usr/bin/env python3
"""
Specialized Model Analysis and Development
- Analyze problematic classes (ㅊ, ㅌ, ㅅ, ㅈ, ㅋ, ㅕ, ㅡ, ㅣ)
- Sensor pattern analysis for each problematic class
- Confusion matrix analysis
- Develop specialized preprocessing and models
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.deep_learning import DeepLearningPipeline
from training.label_mapping import KSLLabelMapper

class SpecializedModelAnalyzer:
    """Analyzer for developing specialized models for problematic classes"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.label_mapper = KSLLabelMapper()
        
        # Problematic classes identified from final analysis
        self.problematic_classes = ['ㅊ', 'ㅌ', 'ㅅ', 'ㅈ', 'ㅋ', 'ㅕ', 'ㅡ', 'ㅣ']
        self.excellent_classes = ['ㄱ', 'ㄴ', 'ㄷ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅍ', 'ㅎ', 'ㅏ', 'ㅑ', 'ㅗ', 'ㅛ', 'ㅜ', 'ㅠ']
        
        print(f"🔧 사용 디바이스: {self.device}")
        print(f"📊 특화 모델 분석 시작!")
        print(f"⚠️ 문제 클래스: {', '.join(self.problematic_classes)}")
        print(f"⭐ 우수 클래스: {', '.join(self.excellent_classes)}")
    
    def load_final_model_results(self):
        """Load results from final model analysis"""
        results_path = 'final_model_performance_report.json'
        
        if not os.path.exists(results_path):
            print(f"❌ 결과 파일을 찾을 수 없습니다: {results_path}")
            return None
        
        with open(results_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        print(f"✅ 최종 모델 결과 로드 완료!")
        return results
    
    def analyze_sensor_patterns(self, data_dir='../integrations/SignGlove_HW'):
        """Analyze sensor patterns for problematic classes"""
        print("📊 문제 클래스 센서 패턴 분석 중...")
        
        sensor_patterns = {}
        
        for class_name in self.problematic_classes:
            print(f"\n🔍 {class_name} 클래스 분석 중...")
            
            class_data = self._load_class_data(data_dir, class_name)
            if class_data is None:
                continue
            
            # Analyze sensor characteristics
            patterns = self._analyze_class_sensor_patterns(class_data, class_name)
            sensor_patterns[class_name] = patterns
            
            print(f"  📊 {class_name} 패턴 분석 완료")
        
        return sensor_patterns
    
    def _load_class_data(self, data_dir, class_name):
        """Load all data for a specific class"""
        class_data = []
        
        base_path = os.path.join(data_dir, 'github_unified_data', class_name)
        if not os.path.exists(base_path):
            print(f"⚠️ 클래스 경로가 없습니다: {base_path}")
            return None
        
        for scenario in os.listdir(base_path):
            scenario_path = os.path.join(base_path, scenario)
            if not os.path.isdir(scenario_path):
                continue
            
            for file_name in os.listdir(scenario_path):
                if file_name.endswith('.csv'):
                    file_path = os.path.join(scenario_path, file_name)
                    
                    try:
                        data = pd.read_csv(file_path, usecols=['pitch', 'roll', 'yaw', 'flex1', 'flex2', 'flex3', 'flex4', 'flex5'])
                        class_data.append(data)
                    except Exception as e:
                        print(f"⚠️ 파일 로드 실패: {file_path} - {str(e)}")
        
        return class_data
    
    def _analyze_class_sensor_patterns(self, class_data, class_name):
        """Analyze sensor patterns for a specific class"""
        if not class_data:
            return None
        
        # Concatenate all data for this class
        all_data = pd.concat(class_data, ignore_index=True)
        
        # Calculate statistics for each sensor
        patterns = {}
        
        for sensor in ['pitch', 'roll', 'yaw', 'flex1', 'flex2', 'flex3', 'flex4', 'flex5']:
            if sensor in all_data.columns:
                sensor_data = all_data[sensor].dropna()
                
                patterns[sensor] = {
                    'mean': float(sensor_data.mean()),
                    'std': float(sensor_data.std()),
                    'min': float(sensor_data.min()),
                    'max': float(sensor_data.max()),
                    'range': float(sensor_data.max() - sensor_data.min()),
                    'variance': float(sensor_data.var()),
                    'skewness': float(sensor_data.skew()),
                    'kurtosis': float(sensor_data.kurtosis())
                }
        
        # Calculate cross-sensor correlations
        correlations = {}
        for i, sensor1 in enumerate(['pitch', 'roll', 'yaw', 'flex1', 'flex2', 'flex3', 'flex4', 'flex5']):
            for j, sensor2 in enumerate(['pitch', 'roll', 'yaw', 'flex1', 'flex2', 'flex3', 'flex4', 'flex5']):
                if i < j and sensor1 in all_data.columns and sensor2 in all_data.columns:
                    corr = float(all_data[sensor1].corr(all_data[sensor2]))
                    correlations[f"{sensor1}_{sensor2}"] = corr
        
        patterns['correlations'] = correlations
        patterns['total_samples'] = len(all_data)
        
        return patterns
    
    def analyze_confusion_patterns(self, final_results):
        """Analyze confusion patterns for problematic classes"""
        print("📊 혼동 패턴 분석 중...")
        
        # Load confusion matrix from final analysis
        # For now, we'll analyze based on the performance results
        confusion_analysis = {}
        
        for class_name in self.problematic_classes:
            if class_name in final_results['class_performance']:
                performance = final_results['class_performance'][class_name]
                
                confusion_analysis[class_name] = {
                    'accuracy': performance['accuracy'],
                    'support': performance['support'],
                    'correct_predictions': performance['correct_predictions'],
                    'incorrect_predictions': performance['total_predictions'] - performance['correct_predictions'],
                    'avg_confidence': performance['avg_confidence']
                }
        
        return confusion_analysis
    
    def compare_with_excellent_classes(self, sensor_patterns):
        """Compare problematic classes with excellent classes"""
        print("📊 우수 클래스와 비교 분석 중...")
        
        # Load excellent class patterns
        excellent_patterns = {}
        for class_name in self.excellent_classes[:5]:  # Analyze first 5 for comparison
            class_data = self._load_class_data('../integrations/SignGlove_HW', class_name)
            if class_data:
                patterns = self._analyze_class_sensor_patterns(class_data, class_name)
                excellent_patterns[class_name] = patterns
        
        # Compare patterns
        comparisons = {}
        
        for prob_class in self.problematic_classes:
            if prob_class not in sensor_patterns:
                continue
            
            prob_patterns = sensor_patterns[prob_class]
            comparisons[prob_class] = {}
            
            for excel_class in excellent_patterns:
                excel_patterns = excellent_patterns[excel_class]
                
                # Compare sensor characteristics
                sensor_differences = {}
                for sensor in ['pitch', 'roll', 'yaw', 'flex1', 'flex2', 'flex3', 'flex4', 'flex5']:
                    if sensor in prob_patterns and sensor in excel_patterns:
                        prob_stats = prob_patterns[sensor]
                        excel_stats = excel_patterns[sensor]
                        
                        sensor_differences[sensor] = {
                            'mean_diff': abs(prob_stats['mean'] - excel_stats['mean']),
                            'std_diff': abs(prob_stats['std'] - excel_stats['std']),
                            'range_diff': abs(prob_stats['range'] - excel_stats['range'])
                        }
                
                comparisons[prob_class][excel_class] = sensor_differences
        
        return comparisons
    
    def generate_specialized_preprocessing(self, sensor_patterns, comparisons):
        """Generate specialized preprocessing strategies"""
        print("🔧 특화 전처리 전략 생성 중...")
        
        preprocessing_strategies = {}
        
        for class_name in self.problematic_classes:
            if class_name not in sensor_patterns:
                continue
            
            strategies = {
                'class_name': class_name,
                'sensor_weights': {},
                'preprocessing_steps': [],
                'augmentation_strategies': []
            }
            
            # Analyze sensor characteristics
            patterns = sensor_patterns[class_name]
            
            # Determine sensor importance based on variance and range
            sensor_importance = {}
            for sensor in ['pitch', 'roll', 'yaw', 'flex1', 'flex2', 'flex3', 'flex4', 'flex5']:
                if sensor in patterns:
                    stats = patterns[sensor]
                    # Higher variance and range = more important for classification
                    importance = stats['variance'] * stats['range']
                    sensor_importance[sensor] = importance
            
            # Normalize importance scores
            total_importance = sum(sensor_importance.values())
            if total_importance > 0:
                for sensor in sensor_importance:
                    sensor_importance[sensor] /= total_importance
            
            strategies['sensor_weights'] = sensor_importance
            
            # Generate preprocessing steps based on patterns
            if patterns.get('yaw', {}).get('std', 0) > 0.5:
                strategies['preprocessing_steps'].append('enhanced_yaw_filtering')
            
            if patterns.get('flex1', {}).get('variance', 0) > 0.1:
                strategies['preprocessing_steps'].append('flex_sensor_enhancement')
            
            # Generate augmentation strategies
            if patterns.get('pitch', {}).get('range', 0) < 0.3:
                strategies['augmentation_strategies'].append('pitch_amplification')
            
            if patterns.get('roll', {}).get('range', 0) < 0.3:
                strategies['augmentation_strategies'].append('roll_amplification')
            
            preprocessing_strategies[class_name] = strategies
        
        return preprocessing_strategies
    
    def create_visualizations(self, sensor_patterns, comparisons, preprocessing_strategies):
        """Create visualizations for specialized analysis"""
        print("📊 시각화 생성 중...")
        
        plt.figure(figsize=(20, 15))
        
        # 1. Problematic classes sensor variance comparison
        plt.subplot(3, 3, 1)
        sensors = ['pitch', 'roll', 'yaw', 'flex1', 'flex2', 'flex3', 'flex4', 'flex5']
        prob_classes = list(sensor_patterns.keys())
        
        for i, sensor in enumerate(sensors):
            variances = []
            for class_name in prob_classes:
                if sensor in sensor_patterns[class_name]:
                    variances.append(sensor_patterns[class_name][sensor]['variance'])
                else:
                    variances.append(0)
            
            plt.bar(np.arange(len(prob_classes)) + i*0.1, variances, width=0.1, label=sensor, alpha=0.7)
        
        plt.xlabel('Problematic Classes')
        plt.ylabel('Variance')
        plt.title('Sensor Variance by Problematic Class')
        plt.xticks(np.arange(len(prob_classes)) + 0.25, prob_classes)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 2. Sensor importance weights
        plt.subplot(3, 3, 2)
        for class_name in preprocessing_strategies:
            weights = preprocessing_strategies[class_name]['sensor_weights']
            sensor_names = list(weights.keys())
            weight_values = list(weights.values())
            
            plt.bar(sensor_names, weight_values, alpha=0.7, label=class_name)
        
        plt.xlabel('Sensors')
        plt.ylabel('Importance Weight')
        plt.title('Sensor Importance Weights')
        plt.xticks(rotation=45)
        plt.legend()
        
        # 3. Accuracy vs Confidence scatter
        plt.subplot(3, 3, 3)
        accuracies = []
        confidences = []
        class_names = []
        
        # Load final results for this plot
        final_results = self.load_final_model_results()
        if final_results:
            for class_name in self.problematic_classes:
                if class_name in final_results['class_performance']:
                    perf = final_results['class_performance'][class_name]
                    accuracies.append(perf['accuracy'])
                    confidences.append(perf['avg_confidence'])
                    class_names.append(class_name)
        
        plt.scatter(confidences, accuracies, s=100, alpha=0.7)
        for i, class_name in enumerate(class_names):
            plt.annotate(class_name, (confidences[i], accuracies[i]), 
                        xytext=(5, 5), textcoords='offset points')
        
        plt.xlabel('Average Confidence')
        plt.ylabel('Accuracy')
        plt.title('Accuracy vs Confidence for Problematic Classes')
        
        # 4. Preprocessing strategies distribution
        plt.subplot(3, 3, 4)
        strategy_counts = defaultdict(int)
        for class_name in preprocessing_strategies:
            for step in preprocessing_strategies[class_name]['preprocessing_steps']:
                strategy_counts[step] += 1
        
        if strategy_counts:
            plt.bar(strategy_counts.keys(), strategy_counts.values(), alpha=0.7)
            plt.xlabel('Preprocessing Strategy')
            plt.ylabel('Number of Classes Using')
            plt.title('Preprocessing Strategy Distribution')
            plt.xticks(rotation=45)
        
        # 5. Sensor correlation heatmap for worst class (ㅊ)
        plt.subplot(3, 3, 5)
        if 'ㅊ' in sensor_patterns and 'correlations' in sensor_patterns['ㅊ']:
            corr_data = sensor_patterns['ㅊ']['correlations']
            sensors = ['pitch', 'roll', 'yaw', 'flex1', 'flex2', 'flex3', 'flex4', 'flex5']
            corr_matrix = np.zeros((len(sensors), len(sensors)))
            
            for i, sensor1 in enumerate(sensors):
                for j, sensor2 in enumerate(sensors):
                    if i == j:
                        corr_matrix[i, j] = 1.0
                    elif f"{sensor1}_{sensor2}" in corr_data:
                        corr_matrix[i, j] = corr_data[f"{sensor1}_{sensor2}"]
                        corr_matrix[j, i] = corr_data[f"{sensor1}_{sensor2}"]
            
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                       xticklabels=sensors, yticklabels=sensors)
            plt.title('Sensor Correlations for ㅊ (Worst Class)')
        
        # 6. Performance comparison with excellent classes
        plt.subplot(3, 3, 6)
        final_results = self.load_final_model_results()
        if final_results:
            prob_accuracies = []
            excel_accuracies = []
            
            for class_name in self.problematic_classes:
                if class_name in final_results['class_performance']:
                    prob_accuracies.append(final_results['class_performance'][class_name]['accuracy'])
            
            for class_name in self.excellent_classes[:8]:  # First 8 for comparison
                if class_name in final_results['class_performance']:
                    excel_accuracies.append(final_results['class_performance'][class_name]['accuracy'])
            
            plt.boxplot([prob_accuracies, excel_accuracies], labels=['Problematic', 'Excellent'])
            plt.ylabel('Accuracy')
            plt.title('Performance Comparison')
        
        # 7. Sensor range analysis
        plt.subplot(3, 3, 7)
        for class_name in self.problematic_classes:
            if class_name in sensor_patterns:
                ranges = []
                for sensor in sensors:
                    if sensor in sensor_patterns[class_name]:
                        ranges.append(sensor_patterns[class_name][sensor]['range'])
                    else:
                        ranges.append(0)
                
                plt.plot(sensors, ranges, marker='o', label=class_name, alpha=0.7)
        
        plt.xlabel('Sensors')
        plt.ylabel('Range')
        plt.title('Sensor Range by Class')
        plt.xticks(rotation=45)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 8. Summary statistics
        plt.subplot(3, 3, 8)
        summary_text = f"""
Specialized Model Analysis Summary:
• Problematic Classes: {len(self.problematic_classes)}
• Excellent Classes: {len(self.excellent_classes)}
• Worst Class: ㅊ (0% accuracy)
• Best Problematic: ㅣ (76% accuracy)
• Average Problematic: {np.mean([final_results['class_performance'][c]['accuracy'] for c in self.problematic_classes if c in final_results['class_performance']]):.1%}
        """
        plt.text(0.1, 0.5, summary_text, fontsize=10, verticalalignment='center')
        plt.axis('off')
        plt.title('Analysis Summary')
        
        # 9. Recommended actions
        plt.subplot(3, 3, 9)
        actions_text = f"""
Recommended Actions:
1. Enhanced preprocessing for ㅊ, ㅌ
2. Sensor weighting for ㅅ, ㅈ, ㅋ
3. Specialized augmentation for ㅕ, ㅡ
4. Focus on flex sensors for ㅣ
5. Develop class-specific models
        """
        plt.text(0.1, 0.5, actions_text, fontsize=10, verticalalignment='center')
        plt.axis('off')
        plt.title('Recommended Actions')
        
        plt.tight_layout()
        plt.savefig('specialized_model_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 시각화 저장: specialized_model_analysis.png")
    
    def generate_analysis_report(self, sensor_patterns, comparisons, preprocessing_strategies):
        """Generate comprehensive analysis report"""
        print("📄 분석 리포트 생성 중...")
        
        report = {
            'analysis_summary': {
                'problematic_classes': self.problematic_classes,
                'excellent_classes': self.excellent_classes,
                'total_problematic': len(self.problematic_classes),
                'total_excellent': len(self.excellent_classes)
            },
            'sensor_patterns': sensor_patterns,
            'comparisons': comparisons,
            'preprocessing_strategies': preprocessing_strategies,
            'recommendations': self._generate_recommendations(sensor_patterns, preprocessing_strategies)
        }
        
        return report
    
    def _generate_recommendations(self, sensor_patterns, preprocessing_strategies):
        """Generate specific recommendations for each problematic class"""
        recommendations = {}
        
        for class_name in self.problematic_classes:
            if class_name not in sensor_patterns:
                continue
            
            patterns = sensor_patterns[class_name]
            strategies = preprocessing_strategies.get(class_name, {})
            
            class_recommendations = {
                'priority': 'high' if class_name in ['ㅊ', 'ㅌ'] else 'medium',
                'key_issues': [],
                'suggested_actions': [],
                'expected_improvement': 'moderate'
            }
            
            # Analyze key issues
            if patterns.get('yaw', {}).get('std', 0) > 0.5:
                class_recommendations['key_issues'].append('high_yaw_variance')
            
            if patterns.get('flex1', {}).get('variance', 0) < 0.05:
                class_recommendations['key_issues'].append('low_flex_variance')
            
            # Generate suggested actions
            if 'enhanced_yaw_filtering' in strategies.get('preprocessing_steps', []):
                class_recommendations['suggested_actions'].append('implement_enhanced_yaw_filtering')
            
            if 'flex_sensor_enhancement' in strategies.get('preprocessing_steps', []):
                class_recommendations['suggested_actions'].append('implement_flex_enhancement')
            
            recommendations[class_name] = class_recommendations
        
        return recommendations
    
    def run_analysis(self):
        """Run complete specialized model analysis"""
        print("🚀 특화 모델 분석 시작!")
        
        # 1. Load final model results
        final_results = self.load_final_model_results()
        if final_results is None:
            return
        
        # 2. Analyze sensor patterns
        sensor_patterns = self.analyze_sensor_patterns()
        
        # 3. Analyze confusion patterns
        confusion_analysis = self.analyze_confusion_patterns(final_results)
        
        # 4. Compare with excellent classes
        comparisons = self.compare_with_excellent_classes(sensor_patterns)
        
        # 5. Generate specialized preprocessing
        preprocessing_strategies = self.generate_specialized_preprocessing(sensor_patterns, comparisons)
        
        # 6. Create visualizations
        self.create_visualizations(sensor_patterns, comparisons, preprocessing_strategies)
        
        # 7. Generate analysis report
        analysis_report = self.generate_analysis_report(sensor_patterns, comparisons, preprocessing_strategies)
        
        # 8. Save results
        with open('specialized_model_analysis_report.json', 'w', encoding='utf-8') as f:
            json.dump(analysis_report, f, ensure_ascii=False, indent=2)
        
        # 9. Print summary
        self._print_summary(analysis_report, final_results)
        
        print(f"\n📁 저장된 파일:")
        print(f"   - specialized_model_analysis_report.json (분석 리포트)")
        print(f"   - specialized_model_analysis.png (시각화)")
        
        print(f"\n🎉 특화 모델 분석 완료!")
        
        return analysis_report
    
    def _print_summary(self, analysis_report, final_results):
        """Print analysis summary"""
        print(f"\n📊 특화 모델 분석 결과:")
        print(f"=" * 50)
        
        print(f"⚠️ 문제 클래스 분석:")
        for class_name in self.problematic_classes:
            if class_name in final_results['class_performance']:
                perf = final_results['class_performance'][class_name]
                print(f"   {class_name}: {perf['accuracy']:.1%} 정확도 ({perf['correct_predictions']}/{perf['total_predictions']})")
        
        print(f"\n🔧 특화 전처리 전략:")
        for class_name in self.problematic_classes:
            if class_name in analysis_report['preprocessing_strategies']:
                strategies = analysis_report['preprocessing_strategies'][class_name]
                print(f"   {class_name}: {', '.join(strategies.get('preprocessing_steps', []))}")
        
        print(f"\n📈 권장 사항:")
        for class_name in self.problematic_classes:
            if class_name in analysis_report['recommendations']:
                rec = analysis_report['recommendations'][class_name]
                print(f"   {class_name} ({rec['priority']}): {', '.join(rec['suggested_actions'])}")

def main():
    """Main function for specialized model analysis"""
    analyzer = SpecializedModelAnalyzer()
    analyzer.run_analysis()

if __name__ == "__main__":
    main()
