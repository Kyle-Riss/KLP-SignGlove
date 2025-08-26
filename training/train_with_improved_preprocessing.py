#!/usr/bin/env python3
"""
개선된 전처리 파이프라인을 사용한 모델 학습
데이터 품질 분석 결과를 바탕으로 한 최적화된 학습 시스템
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import json
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.deep_learning import DeepLearningPipeline
from training.label_mapping import KSLLabelMapper
from training.improved_preprocessing_pipeline import ImprovedPreprocessingPipeline, ImprovedDataset

class ImprovedPreprocessingTrainer:
    """개선된 전처리 파이프라인을 사용한 트레이너"""
    
    def __init__(self, config=None):
        self.config = config or self._get_default_config()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.label_mapper = KSLLabelMapper()
        
        print(f"🚀 개선된 전처리 학습기 초기화")
        print(f"📱 디바이스: {self.device}")
        print(f"⚙️ 설정: {self.config}")
    
    def _get_default_config(self):
        """기본 설정"""
        return {
            'data_dir': 'integrations/SignGlove_HW/github_unified_data',
            'batch_size': 32,
            'learning_rate': 0.001,
            'epochs': 100,
            'early_stopping_patience': 15,
            'weight_decay': 1e-4,
            'dropout': 0.3,
            'hidden_dim': 128,
            'num_layers': 2,
            'train_split': 0.7,
            'val_split': 0.15,
            'test_split': 0.15,
            'use_class_weights': True,
            'gradient_clipping': True,
            'max_grad_norm': 1.0
        }
    
    def create_model(self):
        """모델 생성"""
        model = DeepLearningPipeline(
            input_features=8,
            sequence_length=300,
            num_classes=24,
            hidden_dim=self.config['hidden_dim'],
            num_layers=self.config['num_layers'],
            dropout=self.config['dropout']
        )
        
        return model.to(self.device)
    
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
    
    def create_class_weights(self):
        """클래스 가중치 생성"""
        if not self.config['use_class_weights']:
            return None
        
        # 클래스별 일관성 점수 기반 가중치
        consistency_scores = {
            'ㄱ': 0.179, 'ㄴ': 0.660, 'ㄷ': 0.661, 'ㄹ': 0.855, 'ㅁ': 0.558,
            'ㅂ': 0.777, 'ㅅ': 0.805, 'ㅇ': 0.824, 'ㅈ': 0.796, 'ㅊ': 0.772,
            'ㅋ': 0.730, 'ㅌ': 0.863, 'ㅍ': 0.734, 'ㅎ': 0.743, 'ㅏ': 0.680,
            'ㅑ': 0.760, 'ㅓ': 1.503, 'ㅕ': 0.652, 'ㅗ': 0.682, 'ㅛ': 0.787,
            'ㅜ': 0.685, 'ㅠ': 0.616, 'ㅡ': 0.635, 'ㅣ': 0.705
        }
        
        # 일관성 점수를 가중치로 변환 (낮은 일관성 = 높은 가중치)
        weights = []
        for class_name in self.label_mapper.get_all_classes():
            consistency = consistency_scores.get(class_name, 0.5)
            # 일관성 점수가 낮을수록 가중치를 높게
            weight = 1.0 / (consistency + 0.1)
            weights.append(weight)
        
        # 가중치 정규화
        weights = np.array(weights)
        weights = weights / np.mean(weights)
        
        return torch.FloatTensor(weights).to(self.device)
    
    def create_dataloaders(self):
        """데이터로더 생성"""
        print(f"📊 데이터셋 로드 중...")
        
        # 전처리 파이프라인 초기화
        preprocessor = ImprovedPreprocessingPipeline()
        
        # 전체 데이터셋 생성
        full_dataset = ImprovedDataset(self.config['data_dir'], preprocessor)
        
        # 데이터 분할
        total_size = len(full_dataset)
        train_size = int(self.config['train_split'] * total_size)
        val_size = int(self.config['val_split'] * total_size)
        test_size = total_size - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = random_split(
            full_dataset, [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        print(f"  📈 훈련 데이터: {len(train_dataset)}")
        print(f"  📊 검증 데이터: {len(val_dataset)}")
        print(f"  🧪 테스트 데이터: {len(test_dataset)}")
        
        # 데이터로더 생성
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=0,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )
        
        return train_loader, val_loader, test_loader
    
    def train_epoch(self, model, train_loader, criterion, optimizer):
        """한 에포크 훈련"""
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, labels) in enumerate(train_loader):
            data, labels = data.to(self.device), labels.squeeze().to(self.device)
            
            optimizer.zero_grad()
            
            # 모델 출력
            outputs = model(data)
            logits = outputs['class_logits']
            
            # 손실 계산
            loss = criterion(logits, labels)
            
            # 역전파
            loss.backward()
            
            # 그래디언트 클리핑
            if self.config['gradient_clipping']:
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.config['max_grad_norm'])
            
            optimizer.step()
            
            # 통계 업데이트
            total_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100 * correct / total
        
        return avg_loss, accuracy
    
    def validate_epoch(self, model, val_loader, criterion):
        """검증 에포크"""
        model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, labels in val_loader:
                data, labels = data.to(self.device), labels.squeeze().to(self.device)
                
                outputs = model(data)
                logits = outputs['class_logits']
                
                loss = criterion(logits, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(logits.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100 * correct / total
        
        return avg_loss, accuracy
    
    def train(self):
        """전체 훈련 과정"""
        print(f"🎯 훈련 시작!")
        
        # 모델, 옵티마이저, 손실 함수 생성
        model = self.create_model()
        optimizer, scheduler = self.create_optimizer(model)
        
        class_weights = self.create_class_weights()
        if class_weights is not None:
            criterion = nn.CrossEntropyLoss(weight=class_weights)
            print(f"📊 클래스 가중치 적용됨")
        else:
            criterion = nn.CrossEntropyLoss()
        
        # 데이터로더 생성
        train_loader, val_loader, test_loader = self.create_dataloaders()
        
        # 훈련 기록
        train_losses = []
        train_accuracies = []
        val_losses = []
        val_accuracies = []
        
        best_val_loss = float('inf')
        best_model_state = None
        patience_counter = 0
        
        print(f"\n📈 훈련 진행 상황:")
        print(f"{'Epoch':<6} {'Train Loss':<12} {'Train Acc':<12} {'Val Loss':<12} {'Val Acc':<12}")
        print("-" * 60)
        
        for epoch in range(self.config['epochs']):
            # 훈련
            train_loss, train_acc = self.train_epoch(model, train_loader, criterion, optimizer)
            
            # 검증
            val_loss, val_acc = self.validate_epoch(model, val_loader, criterion)
            
            # 스케줄러 업데이트
            scheduler.step(val_loss)
            
            # 기록 저장
            train_losses.append(train_loss)
            train_accuracies.append(train_acc)
            val_losses.append(val_loss)
            val_accuracies.append(val_acc)
            
            # 출력
            print(f"{epoch+1:<6} {train_loss:<12.4f} {train_acc:<12.2f} {val_loss:<12.4f} {val_acc:<12.2f}")
            
            # 최고 모델 저장
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
            
            # 조기 종료
            if patience_counter >= self.config['early_stopping_patience']:
                print(f"\n⏹️ 조기 종료: {self.config['early_stopping_patience']} 에포크 동안 개선 없음")
                break
        
        # 최고 모델 복원
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            print(f"💾 최고 모델 복원됨 (검증 손실: {best_val_loss:.4f})")
        
        # 훈련 곡선 저장
        self._save_training_curves(train_losses, train_accuracies, val_losses, val_accuracies)
        
        # 최종 평가
        test_loss, test_acc, test_report = self.evaluate(model, test_loader, criterion)
        
        # 모델 저장
        torch.save(model.state_dict(), 'best_improved_preprocessing_model.pth')
        
        print(f"\n🎉 훈련 완료!")
        print(f"📊 최종 테스트 정확도: {test_acc:.2f}%")
        print(f"💾 모델 저장됨: best_improved_preprocessing_model.pth")
        
        return model, test_acc, test_report
    
    def evaluate(self, model, test_loader, criterion):
        """모델 평가"""
        model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for data, labels in test_loader:
                data, labels = data.to(self.device), labels.squeeze().to(self.device)
                
                outputs = model(data)
                logits = outputs['class_logits']
                
                loss = criterion(logits, labels)
                total_loss += loss.item()
                
                _, predicted = torch.max(logits.data, 1)
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(test_loader)
        accuracy = 100 * sum(1 for p, l in zip(all_predictions, all_labels) if p == l) / len(all_labels)
        
        # 분류 보고서 생성
        class_names = self.label_mapper.get_all_classes()
        report = classification_report(all_labels, all_predictions, target_names=class_names, output_dict=True)
        
        # 혼동 행렬 생성
        self._create_confusion_matrix(all_labels, all_predictions, class_names)
        
        # 클래스별 성능 분석
        self._analyze_class_performance(report)
        
        return avg_loss, accuracy, report
    
    def _save_training_curves(self, train_losses, train_accuracies, val_losses, val_accuracies):
        """훈련 곡선 저장"""
        plt.figure(figsize=(15, 5))
        
        # 손실 곡선
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Train Loss', color='blue')
        plt.plot(val_losses, label='Validation Loss', color='red')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 정확도 곡선
        plt.subplot(1, 2, 2)
        plt.plot(train_accuracies, label='Train Accuracy', color='blue')
        plt.plot(val_accuracies, label='Validation Accuracy', color='red')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('improved_preprocessing_training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 훈련 곡선 저장됨: improved_preprocessing_training_curves.png")
    
    def _create_confusion_matrix(self, true_labels, predictions, class_names):
        """혼동 행렬 생성"""
        cm = confusion_matrix(true_labels, predictions)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix - Improved Preprocessing Model')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig('improved_preprocessing_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 혼동 행렬 저장됨: improved_preprocessing_confusion_matrix.png")
    
    def _analyze_class_performance(self, report):
        """클래스별 성능 분석"""
        class_names = self.label_mapper.get_all_classes()
        
        # 클래스별 정확도 추출
        class_accuracies = {}
        for i, class_name in enumerate(class_names):
            if str(i) in report:
                class_accuracies[class_name] = report[str(i)]['precision']
            else:
                class_accuracies[class_name] = 0.0
        
        # 성능별 클래스 분류
        high_performance = {k: v for k, v in class_accuracies.items() if v >= 0.9}
        medium_performance = {k: v for k, v in class_accuracies.items() if 0.7 <= v < 0.9}
        low_performance = {k: v for k, v in class_accuracies.items() if v < 0.7}
        
        # 결과 저장
        results = {
            'overall_accuracy': report['accuracy'],
            'class_accuracies': class_accuracies,
            'high_performance_classes': list(high_performance.keys()),
            'medium_performance_classes': list(medium_performance.keys()),
            'low_performance_classes': list(low_performance.keys()),
            'detailed_report': report
        }
        
        with open('improved_preprocessing_classification_report.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        # 성능 요약 출력
        print(f"\n📊 클래스별 성능 분석:")
        print(f"  🟢 높은 성능 (≥90%): {len(high_performance)}개 클래스")
        print(f"  🟡 중간 성능 (70-90%): {len(medium_performance)}개 클래스")
        print(f"  🔴 낮은 성능 (<70%): {len(low_performance)}개 클래스")
        
        if low_performance:
            print(f"  ⚠️ 낮은 성능 클래스들: {list(low_performance.keys())}")
        
        print(f"📄 상세 보고서 저장됨: improved_preprocessing_classification_report.json")

def main():
    """메인 함수"""
    print(f"🎯 개선된 전처리 파이프라인을 사용한 모델 학습 시작")
    
    # 트레이너 초기화
    trainer = ImprovedPreprocessingTrainer()
    
    # 훈련 실행
    model, test_acc, test_report = trainer.train()
    
    print(f"\n🎉 학습 완료!")
    print(f"📊 최종 테스트 정확도: {test_acc:.2f}%")

if __name__ == "__main__":
    main()
