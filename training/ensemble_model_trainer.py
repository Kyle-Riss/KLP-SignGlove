#!/usr/bin/env python3
"""
Ensemble Model Trainer
- Combines Cross-Validation Model (77.33%) and Specialized Model (13.00%)
- Dynamic weighting based on class performance
- Ensemble prediction and evaluation
- Performance comparison and analysis
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
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.deep_learning import DeepLearningPipeline
from training.label_mapping import KSLLabelMapper

class EnsembleModel:
    """Ensemble model combining cross-validation and specialized models"""
    
    def __init__(self, cv_model_path='cross_validation_model.pth', 
                 specialized_model_path='specialized_model.pth'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.label_mapper = KSLLabelMapper()
        
        # Load models
        self.cv_model = self._load_cv_model(cv_model_path)
        self.specialized_model = self._load_specialized_model(specialized_model_path)
        
        # Problematic classes for specialized weighting
        self.problematic_classes = ['ㅊ', 'ㅌ', 'ㅅ', 'ㅈ', 'ㅋ', 'ㅕ', 'ㅡ', 'ㅣ']
        self.problematic_indices = [self.label_mapper.get_label_id(c) for c in self.problematic_classes]
        
        print(f"🔧 사용 디바이스: {self.device}")
        print(f"📊 앙상블 모델 초기화 완료!")
        print(f"⚠️ 문제 클래스: {', '.join(self.problematic_classes)}")
    
    def _load_cv_model(self, model_path):
        """Load cross-validation model"""
        if not os.path.exists(model_path):
            print(f"❌ CV 모델 파일을 찾을 수 없습니다: {model_path}")
            return None
        
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        model = DeepLearningPipeline(
            input_features=8,
            hidden_dim=48,
            num_layers=1,
            num_classes=24,
            dropout=0.5
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        print(f"✅ 교차 검증 모델 로드 완료!")
        return model
    
    def _load_specialized_model(self, model_path):
        """Load specialized model"""
        if not os.path.exists(model_path):
            print(f"❌ 특화 모델 파일을 찾을 수 없습니다: {model_path}")
            return None
        
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # Create specialized model with same architecture
        from training.specialized_model_trainer import SpecializedModel
        model = SpecializedModel(
            input_features=8,
            hidden_dim=64,
            num_classes=24
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        print(f"✅ 특화 모델 로드 완료!")
        return model
    
    def predict_ensemble(self, data, dynamic_weighting=True):
        """Make ensemble prediction with dynamic weighting"""
        if self.cv_model is None or self.specialized_model is None:
            print("❌ 모델이 로드되지 않았습니다!")
            return None
        
        # Get predictions from both models
        with torch.no_grad():
            cv_output = self.cv_model(data)
            specialized_output = self.specialized_model(data)
            
            # Handle different output formats
            if isinstance(cv_output, dict):
                cv_output = cv_output['class_logits']
            if isinstance(specialized_output, dict):
                specialized_output = specialized_output['class_logits']
            
            # Convert to probabilities
            cv_probs = torch.softmax(cv_output, dim=1)
            specialized_probs = torch.softmax(specialized_output, dim=1)
            
            if dynamic_weighting:
                # Dynamic weighting based on class
                ensemble_probs = self._dynamic_weighting(cv_probs, specialized_probs)
            else:
                # Simple averaging
                ensemble_probs = (cv_probs + specialized_probs) / 2
            
            # Get predictions
            predictions = torch.argmax(ensemble_probs, dim=1)
            confidences = torch.max(ensemble_probs, dim=1)[0]
            
            return {
                'predictions': predictions,
                'confidences': confidences,
                'cv_probs': cv_probs,
                'specialized_probs': specialized_probs,
                'ensemble_probs': ensemble_probs
            }
    
    def _dynamic_weighting(self, cv_probs, specialized_probs):
        """Dynamic weighting based on class performance"""
        batch_size = cv_probs.shape[0]
        ensemble_probs = torch.zeros_like(cv_probs)
        
        for i in range(batch_size):
            # Get class predictions
            cv_pred = torch.argmax(cv_probs[i])
            spec_pred = torch.argmax(specialized_probs[i])
            
            # Determine weights based on class
            if cv_pred.item() in self.problematic_indices:
                # For problematic classes, give more weight to specialized model
                cv_weight = 0.3
                spec_weight = 0.7
            else:
                # For excellent classes, give more weight to CV model
                cv_weight = 0.8
                spec_weight = 0.2
            
            # Weighted combination
            ensemble_probs[i] = cv_weight * cv_probs[i] + spec_weight * specialized_probs[i]
        
        return ensemble_probs
    
    def evaluate_ensemble(self, test_loader):
        """Evaluate ensemble model performance"""
        print("🔍 앙상블 모델 평가 중...")
        
        all_predictions = []
        all_targets = []
        all_confidences = []
        all_cv_predictions = []
        all_spec_predictions = []
        
        self.cv_model.eval()
        self.specialized_model.eval()
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                # Get ensemble predictions
                ensemble_result = self.predict_ensemble(data, dynamic_weighting=True)
                
                if ensemble_result is None:
                    continue
                
                # Get individual model predictions
                cv_pred = torch.argmax(ensemble_result['cv_probs'], dim=1)
                spec_pred = torch.argmax(ensemble_result['specialized_probs'], dim=1)
                
                all_predictions.extend(ensemble_result['predictions'].cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                all_confidences.extend(ensemble_result['confidences'].cpu().numpy())
                all_cv_predictions.extend(cv_pred.cpu().numpy())
                all_spec_predictions.extend(spec_pred.cpu().numpy())
        
        # Calculate accuracies
        ensemble_accuracy = accuracy_score(all_targets, all_predictions)
        cv_accuracy = accuracy_score(all_targets, all_cv_predictions)
        spec_accuracy = accuracy_score(all_targets, all_spec_predictions)
        
        print(f"📊 앙상블 정확도: {ensemble_accuracy:.4f}")
        print(f"📊 CV 모델 정확도: {cv_accuracy:.4f}")
        print(f"📊 특화 모델 정확도: {spec_accuracy:.4f}")
        
        return {
            'ensemble_accuracy': ensemble_accuracy,
            'cv_accuracy': cv_accuracy,
            'specialized_accuracy': spec_accuracy,
            'predictions': all_predictions,
            'targets': all_targets,
            'confidences': all_confidences,
            'cv_predictions': all_cv_predictions,
            'specialized_predictions': all_spec_predictions
        }

class EnsembleDataset(Dataset):
    """Dataset for ensemble evaluation"""
    
    def __init__(self, data_dir='../integrations/SignGlove_HW'):
        self.data_dir = data_dir
        self.label_mapper = KSLLabelMapper()
        
        self.data, self.labels, self.file_paths = self._load_data()
        
        print(f"📊 앙상블 평가 데이터셋 생성 완료: {len(self.data)}개 파일")
    
    def _load_data(self):
        """Load all data for evaluation"""
        data = []
        labels = []
        file_paths = []
        
        base_path = os.path.join(self.data_dir, 'github_unified_data')
        
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
                        
                        # Simple preprocessing for evaluation
                        processed_data = self._preprocess_file(file_path)
                        if processed_data is not None:
                            data.append(processed_data)
                            labels.append(class_label)
                            file_paths.append(file_path)
        
        return data, labels, file_paths
    
    def _preprocess_file(self, file_path):
        """Simple preprocessing for evaluation"""
        try:
            usecols = ['pitch', 'roll', 'yaw', 'flex1', 'flex2', 'flex3', 'flex4', 'flex5']
            data = pd.read_csv(file_path, usecols=usecols)
            
            # Basic yaw correction
            if 'yaw' in data.columns:
                yaw_detrended = data['yaw'] - data['yaw'].rolling(window=10, center=True).mean()
                yaw_detrended = yaw_detrended.fillna(method='bfill').fillna(method='ffill')
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
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data = torch.FloatTensor(self.data[idx])
        label = torch.LongTensor([self.labels[idx]])[0]
        return data, label

class EnsembleAnalyzer:
    """Analyzer for ensemble model performance"""
    
    def __init__(self):
        self.label_mapper = KSLLabelMapper()
        self.problematic_classes = ['ㅊ', 'ㅌ', 'ㅅ', 'ㅈ', 'ㅋ', 'ㅕ', 'ㅡ', 'ㅣ']
        
        print(f"📊 앙상블 분석기 초기화 완료!")
    
    def analyze_class_performance(self, evaluation_results):
        """Analyze class-wise performance"""
        print("📊 클래스별 성능 분석 중...")
        
        predictions = evaluation_results['predictions']
        targets = evaluation_results['targets']
        cv_predictions = evaluation_results['cv_predictions']
        specialized_predictions = evaluation_results['specialized_predictions']
        
        class_metrics = {}
        
        for class_idx in range(24):
            class_name = self.label_mapper.get_class_name(class_idx)
            
            # Get indices for this class
            class_indices = [i for i, t in enumerate(targets) if t == class_idx]
            
            if len(class_indices) == 0:
                class_metrics[class_name] = {
                    'ensemble_accuracy': 0.0,
                    'cv_accuracy': 0.0,
                    'specialized_accuracy': 0.0,
                    'support': 0
                }
                continue
            
            # Calculate accuracies
            class_targets = [targets[i] for i in class_indices]
            class_ensemble_preds = [predictions[i] for i in class_indices]
            class_cv_preds = [cv_predictions[i] for i in class_indices]
            class_spec_preds = [specialized_predictions[i] for i in class_indices]
            
            ensemble_correct = sum(1 for p, t in zip(class_ensemble_preds, class_targets) if p == t)
            cv_correct = sum(1 for p, t in zip(class_cv_preds, class_targets) if p == t)
            spec_correct = sum(1 for p, t in zip(class_spec_preds, class_targets) if p == t)
            total = len(class_indices)
            
            class_metrics[class_name] = {
                'ensemble_accuracy': ensemble_correct / total if total > 0 else 0.0,
                'cv_accuracy': cv_correct / total if total > 0 else 0.0,
                'specialized_accuracy': spec_correct / total if total > 0 else 0.0,
                'support': total
            }
        
        return class_metrics
    
    def create_visualizations(self, evaluation_results, class_metrics):
        """Create ensemble analysis visualizations"""
        print("📊 시각화 생성 중...")
        
        plt.figure(figsize=(20, 15))
        
        # 1. Model comparison
        plt.subplot(3, 3, 1)
        models = ['Ensemble', 'CV Model', 'Specialized']
        accuracies = [
            evaluation_results['ensemble_accuracy'],
            evaluation_results['cv_accuracy'],
            evaluation_results['specialized_accuracy']
        ]
        colors = ['green', 'blue', 'orange']
        
        bars = plt.bar(models, accuracies, color=colors, alpha=0.7)
        plt.ylabel('Accuracy')
        plt.title('Model Performance Comparison')
        plt.ylim(0, 1)
        
        # Add value labels
        for bar, acc in zip(bars, accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{acc:.3f}', ha='center', va='bottom')
        
        # 2. Class-wise ensemble vs CV accuracy
        plt.subplot(3, 3, 2)
        class_names = list(class_metrics.keys())
        ensemble_accs = [class_metrics[name]['ensemble_accuracy'] for name in class_names]
        cv_accs = [class_metrics[name]['cv_accuracy'] for name in class_names]
        
        x = np.arange(len(class_names))
        width = 0.35
        
        plt.bar(x - width/2, ensemble_accs, width, label='Ensemble', alpha=0.7)
        plt.bar(x + width/2, cv_accs, width, label='CV Model', alpha=0.7)
        
        plt.xlabel('Classes')
        plt.ylabel('Accuracy')
        plt.title('Ensemble vs CV Model by Class')
        plt.xticks(x, class_names, rotation=45, ha='right')
        plt.legend()
        
        # 3. Problematic classes improvement
        plt.subplot(3, 3, 3)
        prob_classes = [c for c in class_names if c in self.problematic_classes]
        prob_ensemble = [class_metrics[c]['ensemble_accuracy'] for c in prob_classes]
        prob_cv = [class_metrics[c]['cv_accuracy'] for c in prob_classes]
        prob_spec = [class_metrics[c]['specialized_accuracy'] for c in prob_classes]
        
        x = np.arange(len(prob_classes))
        width = 0.25
        
        plt.bar(x - width, prob_ensemble, width, label='Ensemble', alpha=0.7)
        plt.bar(x, prob_cv, width, label='CV Model', alpha=0.7)
        plt.bar(x + width, prob_spec, width, label='Specialized', alpha=0.7)
        
        plt.xlabel('Problematic Classes')
        plt.ylabel('Accuracy')
        plt.title('Problematic Classes Performance')
        plt.xticks(x, prob_classes, rotation=45, ha='right')
        plt.legend()
        
        # 4. Accuracy improvement heatmap
        plt.subplot(3, 3, 4)
        improvement_data = []
        for class_name in class_names:
            ensemble_acc = class_metrics[class_name]['ensemble_accuracy']
            cv_acc = class_metrics[class_name]['cv_accuracy']
            improvement = ensemble_acc - cv_acc
            improvement_data.append(improvement)
        
        colors_improvement = ['red' if imp < 0 else 'green' for imp in improvement_data]
        bars = plt.bar(class_names, improvement_data, color=colors_improvement, alpha=0.7)
        plt.xlabel('Classes')
        plt.ylabel('Accuracy Improvement')
        plt.title('Ensemble Improvement over CV Model')
        plt.xticks(rotation=45, ha='right')
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # 5. Confusion matrix for ensemble
        plt.subplot(3, 3, 5)
        cm = confusion_matrix(evaluation_results['targets'], evaluation_results['predictions'], labels=range(24))
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_normalized = np.nan_to_num(cm_normalized)
        
        sns.heatmap(cm_normalized, annot=False, cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Ensemble Model Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        # 6. Performance distribution
        plt.subplot(3, 3, 6)
        plt.hist(ensemble_accs, bins=10, alpha=0.7, color='green', edgecolor='black', label='Ensemble')
        plt.hist(cv_accs, bins=10, alpha=0.7, color='blue', edgecolor='black', label='CV Model')
        plt.xlabel('Accuracy')
        plt.ylabel('Number of Classes')
        plt.title('Accuracy Distribution')
        plt.legend()
        
        # 7. Support vs Accuracy
        plt.subplot(3, 3, 7)
        supports = [class_metrics[name]['support'] for name in class_names]
        plt.scatter(supports, ensemble_accs, alpha=0.6, s=50, label='Ensemble')
        plt.scatter(supports, cv_accs, alpha=0.6, s=50, label='CV Model')
        plt.xlabel('Support (Number of Samples)')
        plt.ylabel('Accuracy')
        plt.title('Support vs Accuracy')
        plt.legend()
        
        # 8. Summary statistics
        plt.subplot(3, 3, 8)
        summary_text = f"""
Ensemble Model Summary:
• Ensemble Accuracy: {evaluation_results['ensemble_accuracy']:.4f}
• CV Model Accuracy: {evaluation_results['cv_accuracy']:.4f}
• Specialized Accuracy: {evaluation_results['specialized_accuracy']:.4f}
• Improvement: {evaluation_results['ensemble_accuracy'] - evaluation_results['cv_accuracy']:.4f}
• Problematic Classes: {len(self.problematic_classes)}
• Total Classes: 24
        """
        plt.text(0.1, 0.5, summary_text, fontsize=10, verticalalignment='center')
        plt.axis('off')
        plt.title('Ensemble Summary')
        
        # 9. Best performing classes
        plt.subplot(3, 3, 9)
        sorted_classes = sorted(class_metrics.items(), key=lambda x: x[1]['ensemble_accuracy'], reverse=True)
        top_classes = sorted_classes[:8]
        
        class_names_top = [c[0] for c in top_classes]
        accuracies_top = [c[1]['ensemble_accuracy'] for c in top_classes]
        
        plt.bar(class_names_top, accuracies_top, alpha=0.7, color='green')
        plt.xlabel('Classes')
        plt.ylabel('Ensemble Accuracy')
        plt.title('Top 8 Performing Classes')
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig('ensemble_model_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 시각화 저장: ensemble_model_analysis.png")
    
    def generate_ensemble_report(self, evaluation_results, class_metrics):
        """Generate comprehensive ensemble report"""
        print("📄 앙상블 리포트 생성 중...")
        
        # Calculate improvements
        improvements = {}
        for class_name in class_metrics:
            ensemble_acc = class_metrics[class_name]['ensemble_accuracy']
            cv_acc = class_metrics[class_name]['cv_accuracy']
            improvements[class_name] = ensemble_acc - cv_acc
        
        # Find best and worst improvements
        sorted_improvements = sorted(improvements.items(), key=lambda x: x[1], reverse=True)
        best_improvement = sorted_improvements[0]
        worst_improvement = sorted_improvements[-1]
        
        # Problematic classes analysis
        problematic_performance = {}
        for class_name in self.problematic_classes:
            if class_name in class_metrics:
                problematic_performance[class_name] = {
                    'ensemble_accuracy': class_metrics[class_name]['ensemble_accuracy'],
                    'cv_accuracy': class_metrics[class_name]['cv_accuracy'],
                    'specialized_accuracy': class_metrics[class_name]['specialized_accuracy'],
                    'improvement': improvements[class_name]
                }
        
        report = {
            'overall_performance': {
                'ensemble_accuracy': evaluation_results['ensemble_accuracy'],
                'cv_accuracy': evaluation_results['cv_accuracy'],
                'specialized_accuracy': evaluation_results['specialized_accuracy'],
                'improvement': evaluation_results['ensemble_accuracy'] - evaluation_results['cv_accuracy']
            },
            'class_performance': class_metrics,
            'improvements': improvements,
            'best_improvement': {
                'class': best_improvement[0],
                'improvement': best_improvement[1]
            },
            'worst_improvement': {
                'class': worst_improvement[0],
                'improvement': worst_improvement[1]
            },
            'problematic_classes_analysis': problematic_performance
        }
        
        return report

def main():
    """Main function for ensemble model training and evaluation"""
    print("📊 앙상블 모델 시스템 시작!")
    
    # 1. Create ensemble model
    print("\n1️⃣ 앙상블 모델 생성 중...")
    ensemble_model = EnsembleModel()
    
    # 2. Create evaluation dataset
    print("\n2️⃣ 평가 데이터셋 생성 중...")
    dataset = EnsembleDataset()
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    # 3. Evaluate ensemble
    print("\n3️⃣ 앙상블 모델 평가 중...")
    evaluation_results = ensemble_model.evaluate_ensemble(dataloader)
    
    # 4. Analyze performance
    print("\n4️⃣ 성능 분석 중...")
    analyzer = EnsembleAnalyzer()
    class_metrics = analyzer.analyze_class_performance(evaluation_results)
    
    # 5. Create visualizations
    print("\n5️⃣ 시각화 생성 중...")
    analyzer.create_visualizations(evaluation_results, class_metrics)
    
    # 6. Generate report
    print("\n6️⃣ 리포트 생성 중...")
    ensemble_report = analyzer.generate_ensemble_report(evaluation_results, class_metrics)
    
    # 7. Save results
    with open('ensemble_model_report.json', 'w', encoding='utf-8') as f:
        json.dump(ensemble_report, f, ensure_ascii=False, indent=2)
    
    # 8. Print summary
    print(f"\n📊 앙상블 모델 결과:")
    print(f"=" * 50)
    print(f"🎯 앙상블 정확도: {evaluation_results['ensemble_accuracy']:.4f}")
    print(f"📊 CV 모델 정확도: {evaluation_results['cv_accuracy']:.4f}")
    print(f"📊 특화 모델 정확도: {evaluation_results['specialized_accuracy']:.4f}")
    print(f"📈 개선도: {evaluation_results['ensemble_accuracy'] - evaluation_results['cv_accuracy']:.4f}")
    
    print(f"\n⚠️ 문제 클래스 성능:")
    for class_name in analyzer.problematic_classes:
        if class_name in class_metrics:
            metrics = class_metrics[class_name]
            improvement = metrics['ensemble_accuracy'] - metrics['cv_accuracy']
            print(f"   {class_name}: {metrics['ensemble_accuracy']:.3f} (개선: {improvement:+.3f})")
    
    print(f"\n📁 저장된 파일:")
    print(f"   - ensemble_model_report.json (앙상블 리포트)")
    print(f"   - ensemble_model_analysis.png (시각화)")
    
    print(f"\n🎉 앙상블 모델 시스템 완료!")

if __name__ == "__main__":
    main()
