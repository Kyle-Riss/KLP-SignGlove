#!/usr/bin/env python3
"""
Final Model Performance Analysis
- Comprehensive analysis of cross-validation model
- Class-wise accuracy analysis
- Confusion matrix visualization
- Performance comparison with previous models
- Detailed performance metrics
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.deep_learning import DeepLearningPipeline
from training.label_mapping import KSLLabelMapper

class FinalModelAnalyzer:
    """Comprehensive final model performance analyzer"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.label_mapper = KSLLabelMapper()
        
        print(f"🔧 사용 디바이스: {self.device}")
        print(f"📊 최종 모델 성능 분석 시작!")
    
    def load_cross_validation_model(self):
        """Load the best cross-validation model"""
        model_path = 'cross_validation_model.pth'
        
        if not os.path.exists(model_path):
            print(f"❌ 모델 파일을 찾을 수 없습니다: {model_path}")
            return None
        
        # Load model checkpoint
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # Create model with same architecture
        model = DeepLearningPipeline(
            input_features=8,
            hidden_dim=48,
            num_layers=1,
            num_classes=24,
            dropout=0.5
        )
        
        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        # Load CV results
        cv_results = checkpoint.get('cv_results', {})
        
        print(f"✅ 교차 검증 모델 로드 완료!")
        print(f"📊 최고 폴드: {checkpoint.get('best_fold_idx', 'N/A')}")
        print(f"📊 CV 결과: {cv_results.get('mean_validation_accuracy', 'N/A'):.4f}")
        
        return model, cv_results
    
    def create_test_dataset(self, data_dir='../integrations/SignGlove_HW'):
        """Create comprehensive test dataset"""
        print("📊 테스트 데이터셋 생성 중...")
        
        data = []
        labels = []
        file_paths = []
        class_names = []
        
        base_path = os.path.join(data_dir, 'github_unified_data')
        
        for class_name in os.listdir(base_path):
            class_path = os.path.join(base_path, class_name)
            if not os.path.isdir(class_path):
                continue
            
            try:
                class_label = self.label_mapper.get_label_id(class_name)
            except:
                continue
            
            for scenario in os.listdir(class_path):
                scenario_path = os.path.join(class_path, scenario)
                if not os.path.isdir(scenario_path):
                    continue
                
                for file_name in os.listdir(scenario_path):
                    if file_name.endswith('.csv'):
                        file_path = os.path.join(scenario_path, file_name)
                        
                        # Preprocess file
                        processed_data = self._preprocess_file(file_path)
                        if processed_data is not None:
                            data.append(processed_data)
                            labels.append(class_label)
                            file_paths.append(file_path)
                            class_names.append(class_name)
        
        print(f"📊 테스트 데이터셋 생성 완료: {len(data)}개 파일")
        
        # Analyze class distribution
        class_counts = defaultdict(int)
        for class_name in class_names:
            class_counts[class_name] += 1
        
        print(f"📊 클래스별 파일 수:")
        for class_name in sorted(class_counts.keys()):
            print(f"  {class_name}: {class_counts[class_name]}개")
        
        return data, labels, file_paths, class_names
    
    def _preprocess_file(self, file_path):
        """Preprocess single file"""
        try:
            # Load only necessary columns
            usecols = ['pitch', 'roll', 'yaw', 'flex1', 'flex2', 'flex3', 'flex4', 'flex5']
            data = pd.read_csv(file_path, usecols=usecols)
            
            # Enhanced yaw correction
            if 'yaw' in data.columns:
                yaw_detrended = data['yaw'] - data['yaw'].rolling(window=10, center=True).mean()
                yaw_detrended = yaw_detrended.fillna(method='bfill').fillna(method='ffill')
                yaw_detrended = yaw_detrended.rolling(window=3, center=True).mean().fillna(method='bfill').fillna(method='ffill')
                data['yaw'] = yaw_detrended
            
            # Length normalization
            target_length = 200
            if len(data) != target_length:
                if len(data) < target_length:
                    indices = np.linspace(0, len(data)-1, target_length)
                    data_interpolated = []
                    for col in data.columns:
                        col_data = data[col].values
                        interpolated = np.interp(indices, np.arange(len(col_data)), col_data)
                        data_interpolated.append(interpolated)
                    data = pd.DataFrame(np.column_stack(data_interpolated), columns=data.columns)
                else:
                    data = data.iloc[::len(data)//target_length][:target_length]
            
            return data.values.astype(np.float32)
            
        except Exception as e:
            print(f"⚠️ 전처리 실패: {file_path} - {str(e)}")
            return None
    
    def evaluate_model(self, model, data, labels, class_names):
        """Comprehensive model evaluation"""
        print("🔍 모델 평가 중...")
        
        model.eval()
        all_predictions = []
        all_targets = []
        all_confidences = []
        all_file_paths = []
        
        # Create data loader
        dataset = list(zip(data, labels))
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
        
        with torch.no_grad():
            for batch_data, batch_labels in dataloader:
                # Convert to tensors
                batch_tensors = []
                for d in batch_data:
                    d_tensor = torch.FloatTensor(d)
                    batch_tensors.append(d_tensor)
                
                # Pad batch
                max_len = max(len(d) for d in batch_tensors)
                padded_batch = []
                for d_tensor in batch_tensors:
                    if len(d_tensor) < max_len:
                        padding = torch.zeros(max_len - len(d_tensor), d_tensor.shape[1])
                        d_tensor = torch.cat([d_tensor, padding], dim=0)
                    padded_batch.append(d_tensor)
                
                batch_tensor = torch.stack(padded_batch).to(self.device)
                batch_labels = torch.LongTensor(batch_labels).to(self.device)
                
                # Forward pass
                output = model(batch_tensor)
                
                if isinstance(output, dict):
                    logits = output['class_logits']
                else:
                    logits = output
                
                # Get predictions and confidences
                probabilities = torch.softmax(logits, dim=1)
                confidences, predictions = torch.max(probabilities, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(batch_labels.cpu().numpy())
                all_confidences.extend(confidences.cpu().numpy())
        
        # Calculate overall accuracy
        overall_accuracy = accuracy_score(all_targets, all_predictions)
        
        print(f"📊 전체 정확도: {overall_accuracy:.4f}")
        
        return {
            'predictions': all_predictions,
            'targets': all_targets,
            'confidences': all_confidences,
            'overall_accuracy': overall_accuracy
        }
    
    def analyze_class_performance(self, evaluation_results, class_names):
        """Analyze class-wise performance"""
        print("📊 클래스별 성능 분석 중...")
        
        predictions = evaluation_results['predictions']
        targets = evaluation_results['targets']
        confidences = evaluation_results['confidences']
        
        # Get class names
        all_class_names = [self.label_mapper.get_class_name(i) for i in range(24)]
        
        # Calculate class-wise metrics
        class_metrics = {}
        
        for class_idx in range(24):
            class_name = all_class_names[class_idx]
            
            # Get indices for this class
            class_indices = [i for i, t in enumerate(targets) if t == class_idx]
            
            if len(class_indices) == 0:
                class_metrics[class_name] = {
                    'accuracy': 0.0,
                    'support': 0,
                    'avg_confidence': 0.0,
                    'correct_predictions': 0,
                    'total_predictions': 0
                }
                continue
            
            # Calculate metrics
            class_predictions = [predictions[i] for i in class_indices]
            class_targets = [targets[i] for i in class_indices]
            class_confidences = [confidences[i] for i in class_indices]
            
            correct = sum(1 for p, t in zip(class_predictions, class_targets) if p == t)
            total = len(class_indices)
            accuracy = correct / total if total > 0 else 0.0
            avg_confidence = np.mean(class_confidences) if class_confidences else 0.0
            
            class_metrics[class_name] = {
                'accuracy': float(accuracy),
                'support': int(total),
                'avg_confidence': float(avg_confidence),
                'correct_predictions': int(correct),
                'total_predictions': int(total)
            }
        
        return class_metrics
    
    def create_confusion_matrix(self, evaluation_results):
        """Create and visualize confusion matrix"""
        print("📊 혼동 행렬 생성 중...")
        
        predictions = evaluation_results['predictions']
        targets = evaluation_results['targets']
        
        # Get class names
        class_names = [self.label_mapper.get_class_name(i) for i in range(24)]
        
        # Create confusion matrix
        cm = confusion_matrix(targets, predictions, labels=range(24))
        
        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_normalized = np.nan_to_num(cm_normalized)
        
        return cm, cm_normalized, class_names
    
    def generate_performance_report(self, evaluation_results, class_metrics, cv_results):
        """Generate comprehensive performance report"""
        print("📄 성능 리포트 생성 중...")
        
        # Overall metrics
        overall_accuracy = evaluation_results['overall_accuracy']
        avg_confidence = np.mean(evaluation_results['confidences'])
        
        # Class performance summary
        accuracies = [metrics['accuracy'] for metrics in class_metrics.values()]
        supports = [metrics['support'] for metrics in class_metrics.values()]
        confidences = [metrics['avg_confidence'] for metrics in class_metrics.values()]
        
        # Find best and worst performing classes
        class_names = list(class_metrics.keys())
        best_class_idx = np.argmax(accuracies)
        worst_class_idx = np.argmin(accuracies)
        
        # Performance categories
        excellent_classes = [name for name, metrics in class_metrics.items() if metrics['accuracy'] >= 0.95]
        good_classes = [name for name, metrics in class_metrics.items() if 0.8 <= metrics['accuracy'] < 0.95]
        poor_classes = [name for name, metrics in class_metrics.items() if metrics['accuracy'] < 0.8]
        
        report = {
            'overall_performance': {
                'accuracy': float(overall_accuracy),
                'avg_confidence': float(avg_confidence),
                'total_samples': len(evaluation_results['targets'])
            },
            'cross_validation_results': cv_results,
            'class_performance': class_metrics,
            'performance_summary': {
                'mean_class_accuracy': float(np.mean(accuracies)),
                'std_class_accuracy': float(np.std(accuracies)),
                'min_class_accuracy': float(np.min(accuracies)),
                'max_class_accuracy': float(np.max(accuracies)),
                'excellent_classes_count': len(excellent_classes),
                'good_classes_count': len(good_classes),
                'poor_classes_count': len(poor_classes)
            },
            'best_performing_class': {
                'name': class_names[best_class_idx],
                'accuracy': float(accuracies[best_class_idx]),
                'support': int(supports[best_class_idx])
            },
            'worst_performing_class': {
                'name': class_names[worst_class_idx],
                'accuracy': float(accuracies[worst_class_idx]),
                'support': int(supports[worst_class_idx])
            },
            'class_categories': {
                'excellent': excellent_classes,
                'good': good_classes,
                'poor': poor_classes
            }
        }
        
        return report
    
    def create_visualizations(self, evaluation_results, class_metrics, cm, cm_normalized, class_names):
        """Create comprehensive visualizations"""
        print("📊 시각화 생성 중...")
        
        plt.figure(figsize=(20, 15))
        
        # 1. Class-wise accuracy
        plt.subplot(3, 3, 1)
        class_names_list = list(class_metrics.keys())
        accuracies = [class_metrics[name]['accuracy'] for name in class_names_list]
        supports = [class_metrics[name]['support'] for name in class_names_list]
        
        colors = ['green' if acc >= 0.9 else 'orange' if acc >= 0.8 else 'red' for acc in accuracies]
        bars = plt.bar(range(len(class_names_list)), accuracies, color=colors, alpha=0.7)
        plt.xlabel('Classes')
        plt.ylabel('Accuracy')
        plt.title('Class-wise Accuracy')
        plt.xticks(range(len(class_names_list)), class_names_list, rotation=45, ha='right')
        plt.ylim(0, 1)
        
        # Add value labels on bars
        for i, (bar, acc, sup) in enumerate(zip(bars, accuracies, supports)):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{acc:.3f}\n({sup})', ha='center', va='bottom', fontsize=8)
        
        # 2. Confusion matrix
        plt.subplot(3, 3, 2)
        sns.heatmap(cm_normalized, annot=False, cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Normalized Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        # 3. Accuracy distribution
        plt.subplot(3, 3, 3)
        plt.hist(accuracies, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
        plt.xlabel('Accuracy')
        plt.ylabel('Number of Classes')
        plt.title('Accuracy Distribution')
        plt.axvline(np.mean(accuracies), color='red', linestyle='--', label=f'Mean: {np.mean(accuracies):.3f}')
        plt.legend()
        
        # 4. Support vs Accuracy
        plt.subplot(3, 3, 4)
        plt.scatter(supports, accuracies, alpha=0.6, s=50)
        plt.xlabel('Support (Number of Samples)')
        plt.ylabel('Accuracy')
        plt.title('Support vs Accuracy')
        
        # Add trend line
        z = np.polyfit(supports, accuracies, 1)
        p = np.poly1d(z)
        plt.plot(supports, p(supports), "r--", alpha=0.8)
        
        # 5. Confidence analysis
        plt.subplot(3, 3, 5)
        confidences = [class_metrics[name]['avg_confidence'] for name in class_names_list]
        plt.scatter(accuracies, confidences, alpha=0.6, s=50)
        plt.xlabel('Accuracy')
        plt.ylabel('Average Confidence')
        plt.title('Accuracy vs Confidence')
        
        # Add diagonal line
        plt.plot([0, 1], [0, 1], 'r--', alpha=0.5, label='Perfect Calibration')
        plt.legend()
        
        # 6. Performance categories
        plt.subplot(3, 3, 6)
        excellent_count = len([acc for acc in accuracies if acc >= 0.95])
        good_count = len([acc for acc in accuracies if 0.8 <= acc < 0.95])
        poor_count = len([acc for acc in accuracies if acc < 0.8])
        
        categories = ['Excellent\n(≥95%)', 'Good\n(80-95%)', 'Poor\n(<80%)']
        counts = [excellent_count, good_count, poor_count]
        colors = ['green', 'orange', 'red']
        
        plt.pie(counts, labels=categories, colors=colors, autopct='%1.1f%%', startangle=90)
        plt.title('Performance Categories')
        
        # 7. Cross-validation comparison
        plt.subplot(3, 3, 7)
        cv_accuracies = [0.875, 0.9, 0.8917, 0.9167, 0.9]  # From CV results
        fold_indices = range(1, 6)
        
        plt.bar(fold_indices, cv_accuracies, alpha=0.7, color='lightblue')
        plt.axhline(y=np.mean(cv_accuracies), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(cv_accuracies):.3f}')
        plt.xlabel('Fold')
        plt.ylabel('Validation Accuracy')
        plt.title('Cross-Validation Performance')
        plt.legend()
        
        # 8. Model comparison
        plt.subplot(3, 3, 8)
        models = ['Cross-Validation', 'Stratified', 'File-Based', 'Improved Preprocessing']
        accuracies_comp = [evaluation_results['overall_accuracy'], 0.9778, 0.9111, 0.8667]  # From previous results
        
        colors_comp = ['green', 'blue', 'orange', 'red']
        bars_comp = plt.bar(models, accuracies_comp, color=colors_comp, alpha=0.7)
        plt.ylabel('Accuracy')
        plt.title('Model Performance Comparison')
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0, 1)
        
        # Add value labels
        for bar, acc in zip(bars_comp, accuracies_comp):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{acc:.3f}', ha='center', va='bottom')
        
        # 9. Summary statistics
        plt.subplot(3, 3, 9)
        # Find best and worst performing classes
        best_class_idx = np.argmax(accuracies)
        worst_class_idx = np.argmin(accuracies)
        
        summary_text = f"""
Final Model Performance Summary:
• Overall Accuracy: {evaluation_results['overall_accuracy']:.4f}
• Mean Class Accuracy: {np.mean(accuracies):.4f}
• Std Class Accuracy: {np.std(accuracies):.4f}
• Best Class: {class_names_list[best_class_idx]} ({accuracies[best_class_idx]:.4f})
• Worst Class: {class_names_list[worst_class_idx]} ({accuracies[worst_class_idx]:.4f})
• Excellent Classes: {len([acc for acc in accuracies if acc >= 0.95])}
• Good Classes: {len([acc for acc in accuracies if 0.8 <= acc < 0.95])}
• Poor Classes: {len([acc for acc in accuracies if acc < 0.8])}
        """
        plt.text(0.1, 0.5, summary_text, fontsize=10, verticalalignment='center')
        plt.axis('off')
        plt.title('Performance Summary')
        
        plt.tight_layout()
        plt.savefig('final_model_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 시각화 저장: final_model_analysis.png")
    
    def run_analysis(self):
        """Run complete final model analysis"""
        print("🚀 최종 모델 성능 분석 시작!")
        
        # 1. Load model
        model_result = self.load_cross_validation_model()
        if model_result is None:
            return
        
        model, cv_results = model_result
        
        # 2. Create test dataset
        data, labels, file_paths, class_names = self.create_test_dataset()
        
        # 3. Evaluate model
        evaluation_results = self.evaluate_model(model, data, labels, class_names)
        
        # 4. Analyze class performance
        class_metrics = self.analyze_class_performance(evaluation_results, class_names)
        
        # 5. Create confusion matrix
        cm, cm_normalized, cm_class_names = self.create_confusion_matrix(evaluation_results)
        
        # 6. Generate performance report
        performance_report = self.generate_performance_report(evaluation_results, class_metrics, cv_results)
        
        # 7. Create visualizations
        self.create_visualizations(evaluation_results, class_metrics, cm, cm_normalized, cm_class_names)
        
        # 8. Save results
        with open('final_model_performance_report.json', 'w', encoding='utf-8') as f:
            json.dump(performance_report, f, ensure_ascii=False, indent=2)
        
        # 9. Print summary
        self._print_summary(performance_report)
        
        print(f"\n📁 저장된 파일:")
        print(f"   - final_model_performance_report.json (상세 리포트)")
        print(f"   - final_model_analysis.png (시각화)")
        
        print(f"\n🎉 최종 모델 성능 분석 완료!")
        
        return performance_report
    
    def _print_summary(self, report):
        """Print analysis summary"""
        print(f"\n📊 최종 모델 성능 분석 결과:")
        print(f"=" * 50)
        
        overall = report['overall_performance']
        summary = report['performance_summary']
        best_class = report['best_performing_class']
        worst_class = report['worst_performing_class']
        
        print(f"🎯 전체 정확도: {overall['accuracy']:.4f}")
        print(f"📊 평균 클래스 정확도: {summary['mean_class_accuracy']:.4f} ± {summary['std_class_accuracy']:.4f}")
        print(f"📈 최고 클래스: {best_class['name']} ({best_class['accuracy']:.4f})")
        print(f"📉 최저 클래스: {worst_class['name']} ({worst_class['accuracy']:.4f})")
        print(f"⭐ 우수 클래스: {summary['excellent_classes_count']}개 (≥95%)")
        print(f"✅ 양호 클래스: {summary['good_classes_count']}개 (80-95%)")
        print(f"⚠️ 부족 클래스: {summary['poor_classes_count']}개 (<80%)")
        
        print(f"\n📊 성능 카테고리:")
        print(f"   우수: {', '.join(report['class_categories']['excellent'])}")
        print(f"   양호: {', '.join(report['class_categories']['good'])}")
        print(f"   부족: {', '.join(report['class_categories']['poor'])}")

def main():
    """Main function for final model analysis"""
    analyzer = FinalModelAnalyzer()
    analyzer.run_analysis()

if __name__ == "__main__":
    main()
