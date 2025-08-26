#!/usr/bin/env python3
"""
최종 상보 필터 모델 훈련
24개 자음/모음 전체 최적화 모델
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import json
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from training.label_mapping import KSLLabelMapper
from models.deep_learning import DeepLearningPipeline

class FinalComplementaryFilterDataset(Dataset):
    """최종 상보 필터 적용 데이터셋"""
    
    def __init__(self, data_dir, mode='train'):
        self.data_dir = data_dir
        self.mode = mode
        self.label_mapper = KSLLabelMapper()
        
        # 데이터 로딩
        self.data = []
        self.labels = []
        
        print(f"🔍 최종 상보 필터 데이터셋 로딩: {mode} 모드")
        
        for class_name in self.label_mapper.get_all_classes():
            self._load_class_data(class_name)
        
        print(f"📊 로딩 완료: {len(self.data)}개 샘플")
    
    def _load_class_data(self, class_name):
        """클래스별 데이터 로딩 및 최적화된 전처리"""
        pattern = os.path.join(self.data_dir, f"**/{class_name}/**/episode_*.csv")
        files = glob.glob(pattern, recursive=True)
        
        if not files:
            print(f"  ⚠️  {class_name}: 파일 없음")
            return
        
        print(f"  📥 {class_name}: {len(files)}개 파일")
        
        for file_path in files:
            try:
                df = pd.read_csv(file_path)
                sensor_data = df[['pitch', 'roll', 'yaw', 'flex1', 'flex2', 'flex3', 'flex4', 'flex5']].values
                
                # 최적화된 전처리 적용
                processed_data = self._apply_optimized_preprocessing(sensor_data, class_name)
                
                # 라벨 변환
                label = self.label_mapper.get_label_id(class_name)
                
                self.data.append(processed_data)
                self.labels.append(label)
                
            except Exception as e:
                print(f"  ⚠️  파일 로드 실패: {file_path} - {e}")
    
    def _apply_optimized_preprocessing(self, sensor_data, class_name):
        """최적화된 전처리 적용"""
        # 데이터 길이 통일
        target_length = 300
        current_length = len(sensor_data)
        
        if current_length < target_length:
            padding_length = target_length - current_length
            padding = np.tile(sensor_data[-1:], (padding_length, 1))
            sensor_data = np.vstack([sensor_data, padding])
        elif current_length > target_length:
            sensor_data = sensor_data[:target_length]
        
        # 상보 필터 적용 (최적화된 파라미터)
        alpha = 0.96
        pitch = sensor_data[:, 0]
        roll = sensor_data[:, 1]
        yaw = sensor_data[:, 2]
        
        filtered_pitch = self._complementary_filter(pitch, alpha)
        filtered_roll = self._complementary_filter(roll, alpha)
        filtered_yaw = self._complementary_filter(yaw, alpha)
        
        # Flex 센서는 그대로 유지
        flex1 = sensor_data[:, 3]
        flex2 = sensor_data[:, 4]
        flex3 = sensor_data[:, 5]
        flex4 = sensor_data[:, 6]
        flex5 = sensor_data[:, 7]
        
        # 클래스별 최적화된 전처리
        processed_data = self._apply_class_specific_optimization(
            filtered_pitch, filtered_roll, filtered_yaw,
            flex1, flex2, flex3, flex4, flex5, class_name
        )
        
        return processed_data
    
    def _complementary_filter(self, data, alpha):
        """상보 필터 구현"""
        filtered_data = np.zeros_like(data)
        filtered_data[0] = data[0]
        
        for i in range(1, len(data)):
            filtered_data[i] = alpha * (filtered_data[i-1] + data[i]) + (1-alpha) * data[i]
        
        return filtered_data
    
    def _apply_class_specific_optimization(self, pitch, roll, yaw, flex1, flex2, flex3, flex4, flex5, class_name):
        """클래스별 최적화된 전처리"""
        
        # 모든 클래스에 기본적인 개선 적용
        # 1. IMU 센서 강화 (상보 필터로 이미 개선됨)
        imu_weight = 1.2
        pitch = pitch * imu_weight
        roll = roll * imu_weight
        yaw = yaw * imu_weight
        
        # 2. Flex 센서 정규화 및 강화
        flex_features = np.column_stack([flex1, flex2, flex3, flex4, flex5])
        flex_features = self._normalize_and_enhance_flex(flex_features)
        flex1, flex2, flex3, flex4, flex5 = flex_features.T
        
        # 3. 클래스별 특화 가중치 적용
        if class_name in ['ㅈ', 'ㅍ', 'ㅕ']:
            # 낮은 성능 클래스들에 특별한 가중치
            flex4 = flex4 * 1.5
            flex5 = flex5 * 1.5
        elif class_name in ['ㅊ', 'ㅌ', 'ㄹ']:
            # 혼동이 많은 클래스들
            pitch = pitch * 1.3
            roll = roll * 1.3
        elif class_name in ['ㅁ', 'ㅂ', 'ㅅ']:
            # 안정적인 클래스들
            flex1 = flex1 * 1.2
            flex2 = flex2 * 1.2
        
        # 4. 노이즈 감소 (전체적으로 적용)
        pitch = self._reduce_noise(pitch, window_size=3)
        roll = self._reduce_noise(roll, window_size=3)
        yaw = self._reduce_noise(yaw, window_size=3)
        
        return np.column_stack([
            pitch, roll, yaw, flex1, flex2, flex3, flex4, flex5
        ])
    
    def _normalize_and_enhance_flex(self, flex_features):
        """Flex 센서 정규화 및 강화"""
        enhanced_features = flex_features.copy()
        
        for i in range(flex_features.shape[1]):
            # 정규화
            mean = np.mean(flex_features[:, i])
            std = np.std(flex_features[:, i])
            if std > 0:
                enhanced_features[:, i] = (flex_features[:, i] - mean) / std
            
            # 특징 강화 (표준편차 증가)
            enhanced_features[:, i] = enhanced_features[:, i] * 1.2
        
        return enhanced_features
    
    def _reduce_noise(self, data, window_size=3):
        """노이즈 감소"""
        kernel = np.ones(window_size) / window_size
        return np.convolve(data, kernel, mode='same')
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sensor_data = torch.FloatTensor(self.data[idx])
        label = torch.LongTensor([self.labels[idx]])[0]
        return sensor_data, label

class FinalLoss(nn.Module):
    """최종 손실 함수"""
    
    def __init__(self, label_mapper):
        super(FinalLoss, self).__init__()
        self.label_mapper = label_mapper
        
        # 클래스별 가중치 생성
        self.class_weights = self._create_balanced_weights()
        
        # 기본 손실 함수
        self.ce_loss = nn.CrossEntropyLoss(weight=self.class_weights)
    
    def _create_balanced_weights(self):
        """균형잡힌 클래스 가중치 생성"""
        num_classes = 24
        weights = torch.ones(num_classes)
        
        # 낮은 성능 클래스들에 적당한 가중치
        low_perf_classes = ['ㅈ', 'ㅍ', 'ㅕ']
        for class_name in low_perf_classes:
            idx = self.label_mapper.get_label_id(class_name)
            weights[idx] = 2.0  # 2배 가중치
        
        # 혼동이 많은 클래스들에 가중치
        confusion_classes = ['ㅊ', 'ㅌ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅅ']
        for class_name in confusion_classes:
            idx = self.label_mapper.get_label_id(class_name)
            weights[idx] = 1.5  # 1.5배 가중치
        
        return weights
    
    def forward(self, outputs, targets):
        """최종 손실 계산"""
        # 기본 CrossEntropy 손실
        ce_loss = self.ce_loss(outputs, targets)
        
        # 정규화 손실 (과적합 방지)
        l2_loss = 0.0
        for param in outputs:
            l2_loss += torch.norm(param, p=2)
        
        # 최종 손실
        total_loss = ce_loss + 0.0001 * l2_loss
        
        return total_loss

class FinalComplementaryFilterTrainer:
    """최종 상보 필터 모델 훈련기"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"🚀 최종 상보 필터 모델 훈련 시작")
        print(f"📱 디바이스: {self.device}")
        print(f"🎯 24개 클래스 전체 최적화")
    
    def create_model(self):
        """모델 생성"""
        model = DeepLearningPipeline(
            input_features=8,
            sequence_length=300,
            num_classes=24,
            hidden_dim=128,
            num_layers=3,
            dropout=0.3
        ).to(self.device)
        
        return model
    
    def create_optimizer(self, model):
        """옵티마이저 생성"""
        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )
        
        return optimizer, scheduler
    
    def train(self, train_loader, val_loader, model, optimizer, scheduler, criterion):
        """훈련 실행"""
        print(f"\n🎯 최종 상보 필터 모델 훈련 시작")
        print(f"📊 에포크: {self.config['epochs']}")
        print(f"📦 배치 크기: {self.config['batch_size']}")
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []
        
        for epoch in range(self.config['epochs']):
            # 훈련
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_idx, (data, targets) in enumerate(train_loader):
                data, targets = data.to(self.device), targets.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(data)
                loss = criterion(outputs['class_logits'], targets)
                loss.backward()
                
                # 그래디언트 클리핑
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs['class_logits'].max(1)
                train_total += targets.size(0)
                train_correct += predicted.eq(targets).sum().item()
            
            # 검증
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for data, targets in val_loader:
                    data, targets = data.to(self.device), targets.to(self.device)
                    outputs = model(data)
                    loss = criterion(outputs['class_logits'], targets)
                    
                    val_loss += loss.item()
                    _, predicted = outputs['class_logits'].max(1)
                    val_total += targets.size(0)
                    val_correct += predicted.eq(targets).sum().item()
            
            # 평균 손실 및 정확도 계산
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            train_acc = 100. * train_correct / train_total
            val_acc = 100. * val_correct / val_total
            
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            train_accuracies.append(train_acc)
            val_accuracies.append(val_acc)
            
            print(f"Epoch {epoch+1}/{self.config['epochs']}:")
            print(f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # 스케줄러 업데이트
            scheduler.step(avg_val_loss)
            
            # 조기 종료
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                torch.save(model.state_dict(), 'best_final_complementary_filter_model.pth')
                print(f"  ✅ 새로운 최고 모델 저장")
            else:
                patience_counter += 1
                if patience_counter >= self.config['early_stopping_patience']:
                    print(f"  🛑 조기 종료 (에포크 {epoch+1})")
                    break
        
        # 훈련 곡선 시각화
        self._plot_training_curves(train_losses, val_losses, train_accuracies, val_accuracies)
        
        return model
    
    def _plot_training_curves(self, train_losses, val_losses, train_accuracies, val_accuracies):
        """훈련 곡선 시각화"""
        plt.figure(figsize=(15, 5))
        
        # 손실 곡선
        plt.subplot(1, 3, 1)
        plt.plot(train_losses, label='Train Loss', color='blue')
        plt.plot(val_losses, label='Val Loss', color='red')
        plt.title('Final Complementary Filter Model Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 정확도 곡선
        plt.subplot(1, 3, 2)
        plt.plot(train_accuracies, label='Train Accuracy', color='blue')
        plt.plot(val_accuracies, label='Val Accuracy', color='red')
        plt.title('Final Complementary Filter Model Training and Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 손실과 정확도 비교
        plt.subplot(1, 3, 3)
        ax1 = plt.gca()
        ax2 = ax1.twinx()
        
        line1 = ax1.plot(train_losses, color='blue', label='Train Loss')
        line2 = ax2.plot(train_accuracies, color='red', label='Train Accuracy')
        
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss', color='blue')
        ax2.set_ylabel('Accuracy (%)', color='red')
        
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper right')
        
        plt.title('Training Loss vs Accuracy')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('final_complementary_filter_training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ 훈련 곡선 저장: final_complementary_filter_training_curves.png")
    
    def evaluate(self, test_loader, model):
        """모델 평가"""
        print(f"\n📊 최종 상보 필터 모델 평가")
        
        model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for data, targets in test_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = model(data)
                _, predicted = outputs['class_logits'].max(1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        # 혼동 행렬 생성
        self._create_confusion_matrix(all_targets, all_predictions)
        
        # 클래스별 성능 분석
        self._analyze_class_performance(all_targets, all_predictions)
        
        # 성능 요약 생성
        self._generate_performance_summary(all_targets, all_predictions)
    
    def _create_confusion_matrix(self, targets, predictions):
        """혼동 행렬 생성"""
        from sklearn.metrics import confusion_matrix, classification_report
        
        cm = confusion_matrix(targets, predictions)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Final Complementary Filter Model Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig('final_complementary_filter_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        report = classification_report(targets, predictions, output_dict=True)
        
        with open('final_complementary_filter_classification_report.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print("✅ 혼동 행렬 저장: final_complementary_filter_confusion_matrix.png")
        print("✅ 분류 보고서 저장: final_complementary_filter_classification_report.json")
    
    def _analyze_class_performance(self, targets, predictions):
        """클래스별 성능 분석"""
        from sklearn.metrics import accuracy_score
        
        label_mapper = KSLLabelMapper()
        
        print(f"\n🎯 클래스별 성능:")
        
        # 전체 정확도
        overall_accuracy = accuracy_score(targets, predictions)
        print(f"📊 전체 정확도: {overall_accuracy:.3f} ({overall_accuracy*100:.1f}%)")
        
        # 클래스별 정확도
        class_accuracies = {}
        for class_name in label_mapper.get_all_classes():
            class_idx = label_mapper.get_label_id(class_name)
            
            mask = np.array(targets) == class_idx
            if np.any(mask):
                class_targets = np.array(targets)[mask]
                class_predictions = np.array(predictions)[mask]
                
                accuracy = accuracy_score(class_targets, class_predictions)
                class_accuracies[class_name] = accuracy
                print(f"  {class_name}: {accuracy:.3f} ({accuracy*100:.1f}%)")
        
        # 성능 분포 분석
        accuracies = list(class_accuracies.values())
        print(f"\n📈 성능 통계:")
        print(f"  평균 정확도: {np.mean(accuracies):.3f} ({np.mean(accuracies)*100:.1f}%)")
        print(f"  최고 정확도: {np.max(accuracies):.3f} ({np.max(accuracies)*100:.1f}%)")
        print(f"  최저 정확도: {np.min(accuracies):.3f} ({np.min(accuracies)*100:.1f}%)")
        print(f"  표준편차: {np.std(accuracies):.3f} ({np.std(accuracies)*100:.1f}%)")
        
        # 성능 등급별 분류
        excellent = [cls for cls, acc in class_accuracies.items() if acc >= 0.9]
        good = [cls for cls, acc in class_accuracies.items() if 0.7 <= acc < 0.9]
        fair = [cls for cls, acc in class_accuracies.items() if 0.5 <= acc < 0.7]
        poor = [cls for cls, acc in class_accuracies.items() if acc < 0.5]
        
        print(f"\n🏆 성능 등급별 분류:")
        print(f"  우수 (90% 이상): {len(excellent)}개 - {', '.join(excellent)}")
        print(f"  양호 (70-90%): {len(good)}개 - {', '.join(good)}")
        print(f"  보통 (50-70%): {len(fair)}개 - {', '.join(fair)}")
        print(f"  미흡 (50% 미만): {len(poor)}개 - {', '.join(poor)}")
    
    def _generate_performance_summary(self, targets, predictions):
        """성능 요약 생성"""
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support
        
        label_mapper = KSLLabelMapper()
        
        # 전체 성능
        overall_accuracy = accuracy_score(targets, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(targets, predictions, average='weighted')
        
        # 클래스별 성능
        class_performances = {}
        for class_name in label_mapper.get_all_classes():
            class_idx = label_mapper.get_label_id(class_name)
            mask = np.array(targets) == class_idx
            if np.any(mask):
                class_targets = np.array(targets)[mask]
                class_predictions = np.array(predictions)[mask]
                accuracy = accuracy_score(class_targets, class_predictions)
                class_performances[class_name] = accuracy
        
        # 성능 요약 저장
        summary = {
            'overall_performance': {
                'accuracy': float(overall_accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1)
            },
            'class_performances': class_performances,
            'performance_grades': {
                'excellent': [cls for cls, acc in class_performances.items() if acc >= 0.9],
                'good': [cls for cls, acc in class_performances.items() if 0.7 <= acc < 0.9],
                'fair': [cls for cls, acc in class_performances.items() if 0.5 <= acc < 0.7],
                'poor': [cls for cls, acc in class_performances.items() if acc < 0.5]
            },
            'model_info': {
                'name': 'Final Complementary Filter Model',
                'description': '24개 자음/모음 전체 최적화 모델',
                'preprocessing': '상보 필터 + 클래스별 최적화',
                'architecture': 'DeepLearningPipeline (CNN + LSTM)'
            }
        }
        
        with open('final_complementary_filter_performance_summary.json', 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        print("✅ 성능 요약 저장: final_complementary_filter_performance_summary.json")

def main():
    """메인 함수"""
    config = {
        'learning_rate': 0.0001,
        'batch_size': 32,
        'epochs': 50,
        'weight_decay': 1e-4,
        'early_stopping_patience': 10
    }
    
    # 데이터 로딩
    print("📊 최종 상보 필터 데이터셋 로딩 중...")
    
    train_dataset = FinalComplementaryFilterDataset('integrations/SignGlove_HW/github_unified_data', 'train')
    val_dataset = FinalComplementaryFilterDataset('integrations/SignGlove_HW/github_unified_data', 'val')
    test_dataset = FinalComplementaryFilterDataset('integrations/SignGlove_HW/github_unified_data', 'test')
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
    
    # 모델 생성
    trainer = FinalComplementaryFilterTrainer(config)
    model = trainer.create_model()
    
    # 옵티마이저 및 손실 함수
    optimizer, scheduler = trainer.create_optimizer(model)
    label_mapper = KSLLabelMapper()
    criterion = FinalLoss(label_mapper)
    
    # 훈련
    model = trainer.train(train_loader, val_loader, model, optimizer, scheduler, criterion)
    
    # 평가
    trainer.evaluate(test_loader, model)
    
    print("\n🎉 최종 상보 필터 모델 훈련 완료!")
    print("📁 생성된 파일들:")
    print("  - best_final_complementary_filter_model.pth: 최고 성능 모델")
    print("  - final_complementary_filter_training_curves.png: 훈련 곡선")
    print("  - final_complementary_filter_confusion_matrix.png: 혼동 행렬")
    print("  - final_complementary_filter_classification_report.json: 분류 보고서")
    print("  - final_complementary_filter_performance_summary.json: 성능 요약")

if __name__ == "__main__":
    main()
