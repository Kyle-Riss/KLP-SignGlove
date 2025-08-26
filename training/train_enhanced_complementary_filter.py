#!/usr/bin/env python3
"""
향상된 상보 필터 모델 훈련
낮은 성능 클래스 (ㅈ, ㅍ, ㅕ) 특화 개선 모델
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

class EnhancedComplementaryFilterDataset(Dataset):
    """향상된 상보 필터 적용 데이터셋"""
    
    def __init__(self, data_dir, mode='train'):
        self.data_dir = data_dir
        self.mode = mode
        self.label_mapper = KSLLabelMapper()
        self.low_performance_classes = ['ㅈ', 'ㅍ', 'ㅕ']
        
        # 데이터 로딩
        self.data = []
        self.labels = []
        
        print(f"🔍 향상된 상보 필터 데이터셋 로딩: {mode} 모드")
        
        for class_name in self.label_mapper.get_all_classes():
            self._load_class_data(class_name)
        
        print(f"📊 로딩 완료: {len(self.data)}개 샘플")
    
    def _load_class_data(self, class_name):
        """클래스별 데이터 로딩 및 향상된 전처리"""
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
                
                # 향상된 전처리 적용
                processed_data = self._apply_enhanced_preprocessing(sensor_data, class_name)
                
                # 라벨 변환
                label = self.label_mapper.get_label_id(class_name)
                
                self.data.append(processed_data)
                self.labels.append(label)
                
            except Exception as e:
                print(f"  ⚠️  파일 로드 실패: {file_path} - {e}")
    
    def _apply_enhanced_preprocessing(self, sensor_data, class_name):
        """향상된 전처리 적용"""
        # 데이터 길이 통일
        target_length = 300
        current_length = len(sensor_data)
        
        if current_length < target_length:
            padding_length = target_length - current_length
            padding = np.tile(sensor_data[-1:], (padding_length, 1))
            sensor_data = np.vstack([sensor_data, padding])
        elif current_length > target_length:
            sensor_data = sensor_data[:target_length]
        
        # 상보 필터 적용
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
        
        # 낮은 성능 클래스 특화 전처리
        if class_name in self.low_performance_classes:
            processed_data = self._apply_specialized_preprocessing(
                filtered_pitch, filtered_roll, filtered_yaw,
                flex1, flex2, flex3, flex4, flex5, class_name
            )
        else:
            # 일반 클래스는 표준 전처리
            processed_data = np.column_stack([
                filtered_pitch, filtered_roll, filtered_yaw,
                flex1, flex2, flex3, flex4, flex5
            ])
        
        return processed_data
    
    def _complementary_filter(self, data, alpha):
        """상보 필터 구현"""
        filtered_data = np.zeros_like(data)
        filtered_data[0] = data[0]
        
        for i in range(1, len(data)):
            filtered_data[i] = alpha * (filtered_data[i-1] + data[i]) + (1-alpha) * data[i]
        
        return filtered_data
    
    def _apply_specialized_preprocessing(self, pitch, roll, yaw, flex1, flex2, flex3, flex4, flex5, class_name):
        """낮은 성능 클래스 특화 전처리"""
        
        # 1. 센서 가중치 적용 (분석 결과 기반)
        if class_name == 'ㅈ':
            # ㅈ: flex4, flex5가 주요 특징
            flex4_weight = 2.0
            flex5_weight = 2.0
            imu_weight = 1.5  # IMU 센서도 강화
        elif class_name == 'ㅍ':
            # ㅍ: flex2, flex3, flex4, flex5가 주요 특징
            flex2_weight = 2.0
            flex3_weight = 2.0
            flex4_weight = 2.0
            flex5_weight = 2.0
            imu_weight = 1.2
        elif class_name == 'ㅕ':
            # ㅕ: flex2, flex4가 주요 특징
            flex2_weight = 2.0
            flex4_weight = 2.0
            imu_weight = 1.3
        
        # 2. 센서별 가중치 적용
        pitch = pitch * imu_weight
        roll = roll * imu_weight
        yaw = yaw * imu_weight
        
        if class_name == 'ㅈ':
            flex4 = flex4 * flex4_weight
            flex5 = flex5 * flex5_weight
        elif class_name == 'ㅍ':
            flex2 = flex2 * flex2_weight
            flex3 = flex3 * flex3_weight
            flex4 = flex4 * flex4_weight
            flex5 = flex5 * flex5_weight
        elif class_name == 'ㅕ':
            flex2 = flex2 * flex2_weight
            flex4 = flex4 * flex4_weight
        
        # 3. 노이즈 감소 (품질 점수가 낮은 센서들)
        if class_name == 'ㅍ':
            # ㅍ의 yaw 품질이 낮음
            yaw = self._reduce_noise(yaw, window_size=5)
        
        if class_name == 'ㅕ':
            # ㅕ의 flex2 품질이 상대적으로 낮음
            flex2 = self._reduce_noise(flex2, window_size=3)
        
        # 4. 특징 강화 (클래스 간 차이를 더 명확하게)
        if class_name in ['ㅈ', 'ㅍ', 'ㅕ']:
            # flex 센서들의 차이를 더 명확하게 만들기
            flex_features = np.column_stack([flex1, flex2, flex3, flex4, flex5])
            flex_features = self._enhance_differences(flex_features)
            flex1, flex2, flex3, flex4, flex5 = flex_features.T
        
        return np.column_stack([
            pitch, roll, yaw, flex1, flex2, flex3, flex4, flex5
        ])
    
    def _reduce_noise(self, data, window_size=3):
        """노이즈 감소"""
        # 간단한 이동 평균 필터
        kernel = np.ones(window_size) / window_size
        return np.convolve(data, kernel, mode='same')
    
    def _enhance_differences(self, flex_features):
        """flex 센서 간 차이 강화"""
        # 각 flex 센서의 표준편차를 증가시켜 차이를 더 명확하게
        enhanced_features = flex_features.copy()
        
        for i in range(flex_features.shape[1]):
            std = np.std(flex_features[:, i])
            mean = np.mean(flex_features[:, i])
            enhanced_features[:, i] = (flex_features[:, i] - mean) * 1.5 + mean
        
        return enhanced_features
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sensor_data = torch.FloatTensor(self.data[idx])
        label = torch.LongTensor([self.labels[idx]])[0]
        return sensor_data, label

class EnhancedLoss(nn.Module):
    """향상된 손실 함수"""
    
    def __init__(self, low_performance_classes, label_mapper):
        super(EnhancedLoss, self).__init__()
        self.low_performance_classes = low_performance_classes
        self.label_mapper = label_mapper
        
        # 낮은 성능 클래스들의 인덱스
        self.low_perf_indices = [self.label_mapper.get_label_id(cls) for cls in low_performance_classes]
        
        # 클래스 가중치 생성
        self.class_weights = self._create_class_weights()
        
        # 기본 손실 함수
        self.ce_loss = nn.CrossEntropyLoss(weight=self.class_weights)
    
    def _create_class_weights(self):
        """클래스 가중치 생성"""
        num_classes = 24
        weights = torch.ones(num_classes)
        
        # 낮은 성능 클래스들에 높은 가중치
        for idx in self.low_perf_indices:
            weights[idx] = 3.0  # 3배 가중치
        
        return weights
    
    def forward(self, outputs, targets):
        """향상된 손실 계산"""
        # 기본 CrossEntropy 손실
        ce_loss = self.ce_loss(outputs, targets)
        
        # 낮은 성능 클래스들 간의 혼동 패널티
        confusion_penalty = self._confusion_penalty(outputs, targets)
        
        # 최종 손실
        total_loss = ce_loss + 0.1 * confusion_penalty
        
        return total_loss
    
    def _confusion_penalty(self, outputs, targets):
        """낮은 성능 클래스들 간 혼동 패널티"""
        penalty = 0.0
        
        # 낮은 성능 클래스들의 출력 확률
        low_perf_outputs = outputs[:, self.low_perf_indices]
        
        # 낮은 성능 클래스들 간의 유사성을 줄이기
        for i in range(len(self.low_perf_indices)):
            for j in range(i+1, len(self.low_perf_indices)):
                # 두 클래스의 출력 차이를 최대화
                diff = torch.abs(low_perf_outputs[:, i] - low_perf_outputs[:, j])
                penalty += torch.exp(-torch.mean(diff))
        
        return penalty

class EnhancedComplementaryFilterTrainer:
    """향상된 상보 필터 모델 훈련기"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"🚀 향상된 상보 필터 모델 훈련 시작")
        print(f"📱 디바이스: {self.device}")
        print(f"🎯 낮은 성능 클래스 특화 개선")
    
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
        print(f"\n🎯 향상된 상보 필터 모델 훈련 시작")
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
                torch.save(model.state_dict(), 'best_enhanced_complementary_filter_model.pth')
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
        plt.title('Enhanced Complementary Filter Model Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 정확도 곡선
        plt.subplot(1, 3, 2)
        plt.plot(train_accuracies, label='Train Accuracy', color='blue')
        plt.plot(val_accuracies, label='Val Accuracy', color='red')
        plt.title('Enhanced Complementary Filter Model Training and Validation Accuracy')
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
        plt.savefig('enhanced_complementary_filter_training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ 훈련 곡선 저장: enhanced_complementary_filter_training_curves.png")
    
    def evaluate(self, test_loader, model):
        """모델 평가"""
        print(f"\n📊 향상된 상보 필터 모델 평가")
        
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
    
    def _create_confusion_matrix(self, targets, predictions):
        """혼동 행렬 생성"""
        from sklearn.metrics import confusion_matrix, classification_report
        
        cm = confusion_matrix(targets, predictions)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Enhanced Complementary Filter Model Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig('enhanced_complementary_filter_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        report = classification_report(targets, predictions, output_dict=True)
        
        with open('enhanced_complementary_filter_classification_report.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print("✅ 혼동 행렬 저장: enhanced_complementary_filter_confusion_matrix.png")
        print("✅ 분류 보고서 저장: enhanced_complementary_filter_classification_report.json")
    
    def _analyze_class_performance(self, targets, predictions):
        """클래스별 성능 분석"""
        from sklearn.metrics import accuracy_score
        
        label_mapper = KSLLabelMapper()
        low_performance_classes = ['ㅈ', 'ㅍ', 'ㅕ']
        
        print(f"\n🎯 클래스별 성능:")
        
        # 전체 정확도
        overall_accuracy = accuracy_score(targets, predictions)
        print(f"📊 전체 정확도: {overall_accuracy:.3f} ({overall_accuracy*100:.1f}%)")
        
        # 클래스별 정확도
        for class_name in label_mapper.get_all_classes():
            class_idx = label_mapper.get_label_id(class_name)
            
            mask = np.array(targets) == class_idx
            if np.any(mask):
                class_targets = np.array(targets)[mask]
                class_predictions = np.array(predictions)[mask]
                
                accuracy = accuracy_score(class_targets, class_predictions)
                status = "🔥" if class_name in low_performance_classes else ""
                print(f"  {class_name}: {accuracy:.3f} ({accuracy*100:.1f}%) {status}")
        
        # 낮은 성능 클래스 개선 분석
        print(f"\n📈 낮은 성능 클래스 개선 분석:")
        low_perf_accuracies = []
        for class_name in low_performance_classes:
            class_idx = label_mapper.get_label_id(class_name)
            mask = np.array(targets) == class_idx
            if np.any(mask):
                class_targets = np.array(targets)[mask]
                class_predictions = np.array(predictions)[mask]
                accuracy = accuracy_score(class_targets, class_predictions)
                low_perf_accuracies.append(accuracy)
                print(f"  {class_name}: {accuracy:.3f} ({accuracy*100:.1f}%)")
        
        if low_perf_accuracies:
            avg_low_perf = np.mean(low_perf_accuracies)
            print(f"  평균 낮은 성능 클래스 정확도: {avg_low_perf:.3f} ({avg_low_perf*100:.1f}%)")

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
    print("📊 향상된 상보 필터 데이터셋 로딩 중...")
    
    train_dataset = EnhancedComplementaryFilterDataset('integrations/SignGlove_HW/github_unified_data', 'train')
    val_dataset = EnhancedComplementaryFilterDataset('integrations/SignGlove_HW/github_unified_data', 'val')
    test_dataset = EnhancedComplementaryFilterDataset('integrations/SignGlove_HW/github_unified_data', 'test')
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
    
    # 모델 생성
    trainer = EnhancedComplementaryFilterTrainer(config)
    model = trainer.create_model()
    
    # 옵티마이저 및 손실 함수
    optimizer, scheduler = trainer.create_optimizer(model)
    label_mapper = KSLLabelMapper()
    criterion = EnhancedLoss(['ㅈ', 'ㅍ', 'ㅕ'], label_mapper)
    
    # 훈련
    model = trainer.train(train_loader, val_loader, model, optimizer, scheduler, criterion)
    
    # 평가
    trainer.evaluate(test_loader, model)
    
    print("\n🎉 향상된 상보 필터 모델 훈련 완료!")
    print("📁 생성된 파일들:")
    print("  - best_enhanced_complementary_filter_model.pth: 최고 성능 모델")
    print("  - enhanced_complementary_filter_training_curves.png: 훈련 곡선")
    print("  - enhanced_complementary_filter_confusion_matrix.png: 혼동 행렬")
    print("  - enhanced_complementary_filter_classification_report.json: 분류 보고서")

if __name__ == "__main__":
    main()
