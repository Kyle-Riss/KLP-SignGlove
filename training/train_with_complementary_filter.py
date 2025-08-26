#!/usr/bin/env python3
"""
상보 필터 적용 모델 훈련 스크립트
상보 필터로 전처리된 데이터로 24개 자음/모음 분류 모델 훈련
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

class ComplementaryFilterDataset(Dataset):
    """상보 필터 적용 데이터셋"""
    
    def __init__(self, data_dir, mode='train'):
        self.data_dir = data_dir
        self.mode = mode
        self.label_mapper = KSLLabelMapper()
        
        # 데이터 로딩
        self.data = []
        self.labels = []
        
        print(f"🔍 상보 필터 적용 데이터셋 로딩: {mode} 모드")
        
        for class_name in self.label_mapper.get_all_classes():
            self._load_class_data(class_name)
        
        print(f"📊 로딩 완료: {len(self.data)}개 샘플")
    
    def _load_class_data(self, class_name):
        """클래스별 데이터 로딩 및 상보 필터 적용"""
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
                
                # 상보 필터 적용
                processed_data = self._apply_complementary_filter(sensor_data)
                
                # 라벨 변환
                label = self.label_mapper.get_label_id(class_name)
                
                self.data.append(processed_data)
                self.labels.append(label)
                
            except Exception as e:
                print(f"  ⚠️  파일 로드 실패: {file_path} - {e}")
    
    def _apply_complementary_filter(self, sensor_data):
        """상보 필터 적용"""
        # 데이터 길이 통일 (300으로 패딩 또는 자르기)
        target_length = 300
        current_length = len(sensor_data)
        
        if current_length < target_length:
            # 패딩
            padding_length = target_length - current_length
            padding = np.tile(sensor_data[-1:], (padding_length, 1))
            sensor_data = np.vstack([sensor_data, padding])
        elif current_length > target_length:
            # 자르기
            sensor_data = sensor_data[:target_length]
        
        # 상보 필터 파라미터
        alpha = 0.96  # 상보 필터 계수
        
        # IMU 센서 데이터 추출
        pitch = sensor_data[:, 0]
        roll = sensor_data[:, 1]
        yaw = sensor_data[:, 2]
        
        # 상보 필터 적용
        filtered_pitch = self._complementary_filter(pitch, alpha)
        filtered_roll = self._complementary_filter(roll, alpha)
        filtered_yaw = self._complementary_filter(yaw, alpha)
        
        # Flex 센서는 그대로 유지
        flex1 = sensor_data[:, 3]
        flex2 = sensor_data[:, 4]
        flex3 = sensor_data[:, 5]
        flex4 = sensor_data[:, 6]
        flex5 = sensor_data[:, 7]
        
        # 상보 필터 적용된 데이터 조합
        processed_data = np.column_stack([
            filtered_pitch,
            filtered_roll,
            filtered_yaw,
            flex1, flex2, flex3, flex4, flex5
        ])
        
        return processed_data
    
    def _complementary_filter(self, data, alpha):
        """상보 필터 구현"""
        filtered_data = np.zeros_like(data)
        
        # 초기값 설정
        filtered_data[0] = data[0]
        
        # 상보 필터 적용
        for i in range(1, len(data)):
            # 상보 필터 공식: filtered = alpha * (prev + gyro) + (1-alpha) * accel
            filtered_data[i] = alpha * (filtered_data[i-1] + data[i]) + (1-alpha) * data[i]
        
        return filtered_data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sensor_data = torch.FloatTensor(self.data[idx])
        label = torch.LongTensor([self.labels[idx]])[0]
        return sensor_data, label

class ComplementaryFilterTrainer:
    """상보 필터 적용 모델 훈련기"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"🚀 상보 필터 적용 모델 훈련 시작")
        print(f"📱 디바이스: {self.device}")
        print(f"🎯 상보 필터로 개선된 센서 데이터 활용")
    
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
        print(f"\n🎯 상보 필터 모델 훈련 시작")
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
                torch.save(model.state_dict(), 'best_complementary_filter_model.pth')
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
        plt.title('Complementary Filter Model Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 정확도 곡선
        plt.subplot(1, 3, 2)
        plt.plot(train_accuracies, label='Train Accuracy', color='blue')
        plt.plot(val_accuracies, label='Val Accuracy', color='red')
        plt.title('Complementary Filter Model Training and Validation Accuracy')
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
        plt.savefig('complementary_filter_training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ 훈련 곡선 저장: complementary_filter_training_curves.png")
    
    def evaluate(self, test_loader, model):
        """모델 평가"""
        print(f"\n📊 상보 필터 모델 평가")
        
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
        plt.title('Complementary Filter Model Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig('complementary_filter_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        report = classification_report(targets, predictions, output_dict=True)
        
        with open('complementary_filter_classification_report.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print("✅ 혼동 행렬 저장: complementary_filter_confusion_matrix.png")
        print("✅ 분류 보고서 저장: complementary_filter_classification_report.json")
    
    def _analyze_class_performance(self, targets, predictions):
        """클래스별 성능 분석"""
        from sklearn.metrics import accuracy_score
        
        label_mapper = KSLLabelMapper()
        
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
                print(f"  {class_name}: {accuracy:.3f} ({accuracy*100:.1f}%)")
        
        # 성능 분포 분석
        class_accuracies = []
        for class_name in label_mapper.get_all_classes():
            class_idx = label_mapper.get_label_id(class_name)
            mask = np.array(targets) == class_idx
            if np.any(mask):
                class_targets = np.array(targets)[mask]
                class_predictions = np.array(predictions)[mask]
                accuracy = accuracy_score(class_targets, class_predictions)
                class_accuracies.append(accuracy)
        
        if class_accuracies:
            print(f"\n📈 성능 통계:")
            print(f"  평균 정확도: {np.mean(class_accuracies):.3f} ({np.mean(class_accuracies)*100:.1f}%)")
            print(f"  최고 정확도: {np.max(class_accuracies):.3f} ({np.max(class_accuracies)*100:.1f}%)")
            print(f"  최저 정확도: {np.min(class_accuracies):.3f} ({np.min(class_accuracies)*100:.1f}%)")
            print(f"  표준편차: {np.std(class_accuracies):.3f} ({np.std(class_accuracies)*100:.1f}%)")
            
            # 성능이 좋은 클래스들
            high_performance = [acc for acc in class_accuracies if acc > 0.8]
            print(f"  높은 성능 클래스 수 (80% 이상): {len(high_performance)}개")
            
            # 성능이 낮은 클래스들
            low_performance = [acc for acc in class_accuracies if acc < 0.5]
            print(f"  낮은 성능 클래스 수 (50% 미만): {len(low_performance)}개")

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
    print("📊 상보 필터 적용 데이터셋 로딩 중...")
    
    train_dataset = ComplementaryFilterDataset('integrations/SignGlove_HW/github_unified_data', 'train')
    val_dataset = ComplementaryFilterDataset('integrations/SignGlove_HW/github_unified_data', 'val')
    test_dataset = ComplementaryFilterDataset('integrations/SignGlove_HW/github_unified_data', 'test')
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
    
    # 모델 생성
    trainer = ComplementaryFilterTrainer(config)
    model = trainer.create_model()
    
    # 옵티마이저 및 손실 함수
    optimizer, scheduler = trainer.create_optimizer(model)
    criterion = nn.CrossEntropyLoss()
    
    # 훈련
    model = trainer.train(train_loader, val_loader, model, optimizer, scheduler, criterion)
    
    # 평가
    trainer.evaluate(test_loader, model)
    
    print("\n🎉 상보 필터 적용 모델 훈련 완료!")
    print("📁 생성된 파일들:")
    print("  - best_complementary_filter_model.pth: 최고 성능 모델")
    print("  - complementary_filter_training_curves.png: 훈련 곡선")
    print("  - complementary_filter_confusion_matrix.png: 혼동 행렬")
    print("  - complementary_filter_classification_report.json: 분류 보고서")

if __name__ == "__main__":
    main()
