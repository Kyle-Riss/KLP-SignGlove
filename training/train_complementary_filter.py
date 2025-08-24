#!/usr/bin/env python3
"""
상보필터 전용 모델 학습 스크립트
새로운 unified 데이터셋을 사용하여 상보필터 데이터에 최적화된 모델 학습
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from datetime import datetime

# 프로젝트 루트 추가
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.deep_learning import DeepLearningPipeline
from training.label_mapping import KSLLabelMapper
from training.dataset import KSLCsvDataset

class ComplementaryFilterTrainer:
    """상보필터 전용 모델 학습기"""
    
    def __init__(self, config: dict = None):
        self.config = self._load_config(config)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.label_mapper = KSLLabelMapper()
        
        print(f"🚀 상보필터 전용 모델 학습기 초기화")
        print(f"📱 장치: {self.device}")
        print(f"🎯 클래스 수: {self.label_mapper.get_num_classes()}")
    
    def _load_config(self, config: dict = None) -> dict:
        """설정 로드"""
        default_config = {
            'data_dir': 'integrations/SignGlove_HW',
            'model_save_path': 'best_complementary_model.pth',
            'window_size': 20,
            'stride': 10,
            'batch_size': 32,
            'learning_rate': 0.001,
            'epochs': 50,
            'train_split': 0.7,
            'val_split': 0.15,
            'test_split': 0.15,
            'early_stopping_patience': 10,
            'save_best_only': True
        }
        
        if config:
            default_config.update(config)
        
        return default_config
    
    def create_complementary_dataset(self) -> KSLCsvDataset:
        """상보필터 데이터셋 생성"""
        print(f"📁 데이터셋 생성 중: {self.config['data_dir']}")
        
        # unified 데이터 파일들만 사용
        data_dir = Path(self.config['data_dir'])
        unified_files = [f for f in data_dir.glob("*_unified_data_*.csv")]
        
        if not unified_files:
            raise FileNotFoundError(f"Unified 데이터 파일을 찾을 수 없습니다: {data_dir}")
        
        print(f"📊 발견된 unified 파일: {len(unified_files)}개")
        
        # 임시 디렉토리에 unified 파일들만 복사
        temp_dir = Path("temp_complementary_data")
        temp_dir.mkdir(exist_ok=True)
        
        for file_path in unified_files:
            # 파일 복사
            import shutil
            shutil.copy2(file_path, temp_dir / file_path.name)
        
        # 데이터셋 생성
        dataset = KSLCsvDataset(
            csv_dir=str(temp_dir),
            window_size=self.config['window_size'],
            stride=self.config['stride'],
            transform=None,
            use_labeling=True
        )
        
        print(f"✅ 데이터셋 생성 완료: {len(dataset)}개 샘플")
        
        # 클래스 분포 출력
        class_dist = dataset.get_class_distribution()
        print(f"📊 클래스 분포:")
        for class_name, count in class_dist.items():
            print(f"  {class_name}: {count}개")
        
        return dataset
    
    def train_model(self, dataset: KSLCsvDataset):
        """모델 학습"""
        print("🚀 상보필터 전용 모델 학습 시작...")
        
        # 데이터 분할
        total_size = len(dataset)
        train_size = int(self.config['train_split'] * total_size)
        val_size = int(self.config['val_split'] * total_size)
        test_size = total_size - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        print(f"📊 데이터 분할:")
        print(f"  훈련: {len(train_dataset)}개")
        print(f"  검증: {len(val_dataset)}개")
        print(f"  테스트: {len(test_dataset)}개")
        
        # 데이터 로더 생성
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config['batch_size'], 
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.config['batch_size'], 
            shuffle=False
        )
        test_loader = DataLoader(
            test_dataset, 
            batch_size=self.config['batch_size'], 
            shuffle=False
        )
        
        # 모델 초기화
        num_classes = self.label_mapper.get_num_classes()
        model = DeepLearningPipeline(
            input_features=8,  # flex5 + orientation3 (상보필터)
            sequence_length=self.config['window_size'],
            num_classes=num_classes,
            hidden_dim=128,
            num_layers=2,
            dropout=0.3
        ).to(self.device)
        
        print(f"🧠 모델 구조:")
        print(f"  입력 특성: 8개 (flex5 + orientation3)")
        print(f"  시퀀스 길이: {self.config['window_size']}")
        print(f"  클래스 수: {num_classes}")
        print(f"  파라미터 수: {sum(p.numel() for p in model.parameters()):,}")
        
        # 손실 함수와 옵티마이저
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.config['learning_rate'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        # 학습 기록
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []
        best_val_loss = float('inf')
        patience_counter = 0
        
        print(f"\n🎯 학습 시작 (에포크: {self.config['epochs']})")
        print("=" * 80)
        
        for epoch in range(self.config['epochs']):
            # 훈련
            train_loss, train_acc = self._train_epoch(model, train_loader, criterion, optimizer)
            
            # 검증
            val_loss, val_acc = self._validate_epoch(model, val_loader, criterion)
            
            # 스케줄러 업데이트
            scheduler.step(val_loss)
            
            # 기록
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accuracies.append(train_acc)
            val_accuracies.append(val_acc)
            
            # 출력
            print(f"Epoch {epoch+1:2d}/{self.config['epochs']:2d} | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | "
                  f"LR: {optimizer.param_groups[0]['lr']:.6f}")
            
            # 최고 모델 저장
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
                if self.config['save_best_only']:
                    torch.save(model, self.config['model_save_path'])
                    print(f"  💾 최고 모델 저장: {self.config['model_save_path']}")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= self.config['early_stopping_patience']:
                print(f"  ⏹️  Early stopping (patience: {self.config['early_stopping_patience']})")
                break
        
        # 최종 모델 저장 (save_best_only가 False인 경우)
        if not self.config['save_best_only']:
            torch.save(model, self.config['model_save_path'])
            print(f"💾 최종 모델 저장: {self.config['model_save_path']}")
        
        # 학습 곡선 저장
        self._save_training_curves(train_losses, val_losses, train_accuracies, val_accuracies)
        
        # 테스트 평가
        print(f"\n🧪 테스트 평가:")
        test_loss, test_acc = self._validate_epoch(model, test_loader, criterion)
        print(f"  테스트 손실: {test_loss:.4f}")
        print(f"  테스트 정확도: {test_acc:.2f}%")
        
        # 상세 평가
        self._detailed_evaluation(model, test_loader)
        
        print(f"\n🎉 상보필터 전용 모델 학습 완료!")
        return model
    
    def _train_epoch(self, model, train_loader, criterion, optimizer):
        """한 에포크 훈련"""
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            output = model(data)
            
            if isinstance(output, dict):
                output = output['class_logits']
            
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100 * correct / total
        return avg_loss, accuracy
    
    def _validate_epoch(self, model, val_loader, criterion):
        """한 에포크 검증"""
        model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                
                if isinstance(output, dict):
                    output = output['class_logits']
                
                loss = criterion(output, target)
                total_loss += loss.item()
                
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100 * correct / total
        return avg_loss, accuracy
    
    def _detailed_evaluation(self, model, test_loader):
        """상세 평가 (분류 보고서, 혼동 행렬)"""
        model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                
                if isinstance(output, dict):
                    output = output['class_logits']
                
                _, predicted = torch.max(output.data, 1)
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        # 분류 보고서
        class_names = [self.label_mapper.get_class_name(i) for i in range(self.label_mapper.get_num_classes())]
        report = classification_report(all_targets, all_predictions, target_names=class_names)
        print(f"\n📋 분류 보고서:")
        print(report)
        
        # 혼동 행렬
        cm = confusion_matrix(all_targets, all_predictions)
        self._plot_confusion_matrix(cm, class_names, "complementary_confusion_matrix.png")
        
        # 클래스별 정확도
        print(f"\n📊 클래스별 정확도:")
        for i, class_name in enumerate(class_names):
            class_mask = np.array(all_targets) == i
            if np.sum(class_mask) > 0:
                class_acc = np.sum(np.array(all_predictions)[class_mask] == i) / np.sum(class_mask) * 100
                print(f"  {class_name}: {class_acc:.1f}%")
    
    def _plot_confusion_matrix(self, cm, class_names, filename):
        """혼동 행렬 시각화"""
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('상보필터 모델 혼동 행렬')
        plt.xlabel('예측')
        plt.ylabel('실제')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📊 혼동 행렬 저장: {filename}")
    
    def _save_training_curves(self, train_losses, val_losses, train_accuracies, val_accuracies):
        """학습 곡선 저장"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # 손실 곡선
        ax1.plot(train_losses, label='Train Loss', color='blue')
        ax1.plot(val_losses, label='Val Loss', color='red')
        ax1.set_title('상보필터 모델 학습 손실')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # 정확도 곡선
        ax2.plot(train_accuracies, label='Train Accuracy', color='blue')
        ax2.plot(val_accuracies, label='Val Accuracy', color='red')
        ax2.set_title('상보필터 모델 학습 정확도')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('complementary_training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📈 학습 곡선 저장: complementary_training_curves.png")

def main():
    """메인 실행 함수"""
    print("🎯 상보필터 전용 모델 학습")
    print("=" * 60)
    
    # 학습기 초기화
    trainer = ComplementaryFilterTrainer()
    
    try:
        # 데이터셋 생성
        dataset = trainer.create_complementary_dataset()
        
        # 모델 학습
        model = trainer.train_model(dataset)
        
        print(f"\n✅ 상보필터 전용 모델 학습 완료!")
        print(f"📁 모델 파일: {trainer.config['model_save_path']}")
        
    except Exception as e:
        print(f"❌ 학습 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
