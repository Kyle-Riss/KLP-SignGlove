"""
Unified 데이터셋을 사용한 모델 재학습
SignGlove_HW unified 데이터셋으로 새로운 모델 학습
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import pandas as pd
import numpy as np
from pathlib import Path
import glob
import re
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import json

# 프로젝트 루트 추가
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.deep_learning import DeepLearningPipeline
from training.dataset import KSLCsvDataset
from training.label_mapping import KSLLabelMapper

class UnifiedModelTrainer:
    """Unified 데이터셋을 사용한 모델 학습기"""
    
    def __init__(self, data_dir: str = "integrations/SignGlove_HW"):
        self.data_dir = data_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.label_mapper = KSLLabelMapper()
        
        # 학습 설정
        self.config = {
            'batch_size': 32,
            'learning_rate': 0.001,
            'epochs': 50,
            'window_size': 20,
            'stride': 10,
            'train_split': 0.8,
            'val_split': 0.1,
            'test_split': 0.1
        }
        
        print(f"🔧 학습 설정:")
        for key, value in self.config.items():
            print(f"  {key}: {value}")
        print(f"🖥️  장치: {self.device}")
    
    def load_unified_data(self) -> List[str]:
        """Unified 데이터셋 파일 목록 로드"""
        print("📁 Unified 데이터셋 로드 중...")
        
        # 모든 CSV 파일 찾기 (기존 + Unified)
        csv_files = []
        
        # 기존 샘플 데이터
        sample_files = glob.glob(os.path.join(self.data_dir, "*_sample_data.csv"))
        csv_files.extend(sample_files)
        print(f"  📄 기존 샘플 데이터: {len(sample_files)}개")
        
        # Unified 데이터
        unified_files = glob.glob(os.path.join(self.data_dir, "*_unified_data_*.csv"))
        csv_files.extend(unified_files)
        print(f"  📄 Unified 데이터: {len(unified_files)}개")
        
        # Madgwick 데이터는 제거됨
        print(f"  📄 Madgwick 데이터: 0개 (제거됨)")
        
        print(f"✅ 총 {len(csv_files)}개 데이터 파일 로드 완료")
        
        # 클래스별 통계
        class_stats = {}
        for file_path in csv_files:
            filename = os.path.basename(file_path)
            
            # 파일명에서 클래스 추출
            for class_name in self.label_mapper.class_to_id.keys():
                if class_name in filename:
                    if class_name not in class_stats:
                        class_stats[class_name] = 0
                    class_stats[class_name] += 1
                    break
        
        print("📊 클래스별 데이터 통계:")
        for class_name, count in sorted(class_stats.items()):
            print(f"  {class_name}: {count}개 파일")
        
        return csv_files
    
    def create_balanced_dataset(self, csv_files: List[str]) -> KSLCsvDataset:
        """균형잡힌 데이터셋 생성"""
        print("⚖️ 균형잡힌 데이터셋 생성 중...")
        
        # 클래스별로 파일 그룹화
        class_files = {}
        for file_path in csv_files:
            filename = os.path.basename(file_path)
            
            for class_name in self.label_mapper.class_to_id.keys():
                if class_name in filename:
                    if class_name not in class_files:
                        class_files[class_name] = []
                    class_files[class_name].append(file_path)
                    break
        
        # 각 클래스당 최대 파일 수 제한 (균형 유지)
        max_files_per_class = min(len(files) for files in class_files.values())
        print(f"  📊 클래스당 최대 파일 수: {max_files_per_class}")
        
        balanced_files = []
        for class_name, files in class_files.items():
            selected_files = files[:max_files_per_class]
            balanced_files.extend(selected_files)
            print(f"  {class_name}: {len(selected_files)}개 선택")
        
        print(f"✅ 균형잡힌 데이터셋: {len(balanced_files)}개 파일")
        
        # 임시 디렉토리에 파일 복사
        temp_dir = "temp_unified_data"
        os.makedirs(temp_dir, exist_ok=True)
        
        for file_path in balanced_files:
            filename = os.path.basename(file_path)
            temp_path = os.path.join(temp_dir, filename)
            import shutil
            shutil.copy2(file_path, temp_path)
        
        # 데이터셋 생성
        dataset = KSLCsvDataset(
            csv_dir=temp_dir,
            window_size=self.config['window_size'],
            stride=self.config['stride'],
            use_labeling=True
        )
        
        return dataset
    
    def train_model(self, dataset: KSLCsvDataset):
        """모델 학습"""
        print("🚀 모델 학습 시작...")
        
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
            input_features=8,  # pitch, roll, yaw + 5 flex sensors
            sequence_length=self.config['window_size'],
            num_classes=num_classes,
            hidden_dim=128,
            num_layers=2,
            dropout=0.3
        ).to(self.device)
        
        # 손실 함수와 옵티마이저
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.config['learning_rate'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        
        # 학습 기록
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []
        best_val_loss = float('inf')
        best_model_path = "best_unified_model.pth"
        
        print("🎯 학습 시작...")
        for epoch in range(self.config['epochs']):
            # 훈련
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = model(data)
                
                # 모델 출력이 딕셔너리인 경우 처리
                if isinstance(output, dict):
                    output = output['class_logits']
                
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                train_total += target.size(0)
                train_correct += (predicted == target).sum().item()
                
                if batch_idx % 10 == 0:
                    print(f"  Epoch {epoch+1}/{self.config['epochs']}, "
                          f"Batch {batch_idx}/{len(train_loader)}, "
                          f"Loss: {loss.item():.4f}")
            
            # 검증
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output = model(data)
                    
                    if isinstance(output, dict):
                        output = output['class_logits']
                    
                    loss = criterion(output, target)
                    val_loss += loss.item()
                    _, predicted = torch.max(output.data, 1)
                    val_total += target.size(0)
                    val_correct += (predicted == target).sum().item()
            
            # 평균 손실과 정확도 계산
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            train_accuracy = 100 * train_correct / train_total
            val_accuracy = 100 * val_correct / val_total
            
            # 기록 저장
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            train_accuracies.append(train_accuracy)
            val_accuracies.append(val_accuracy)
            
            # 스케줄러 업데이트
            scheduler.step(avg_val_loss)
            
            # 최고 모델 저장
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), best_model_path)
                print(f"  💾 최고 모델 저장: {best_model_path}")
            
            print(f"  Epoch {epoch+1}/{self.config['epochs']}:")
            print(f"    훈련 손실: {avg_train_loss:.4f}, 정확도: {train_accuracy:.2f}%")
            print(f"    검증 손실: {avg_val_loss:.4f}, 정확도: {val_accuracy:.2f}%")
        
        # 학습 곡선 시각화
        self.plot_training_curves(train_losses, val_losses, train_accuracies, val_accuracies)
        
        # 테스트 성능 평가
        self.evaluate_model(model, test_loader, "테스트")
        
        return model
    
    def evaluate_model(self, model, data_loader, split_name: str):
        """모델 평가"""
        print(f"📊 {split_name} 성능 평가...")
        
        model.eval()
        all_predictions = []
        all_targets = []
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                
                if isinstance(output, dict):
                    output = output['class_logits']
                
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        accuracy = 100 * correct / total
        print(f"  📈 {split_name} 정확도: {accuracy:.2f}%")
        
        # 분류 보고서
        class_names = [self.label_mapper.get_class_name(i) for i in range(len(self.label_mapper.class_to_id))]
        report = classification_report(all_targets, all_predictions, target_names=class_names)
        print(f"  📋 분류 보고서:\n{report}")
        
        # 혼동 행렬
        cm = confusion_matrix(all_targets, all_predictions)
        self.plot_confusion_matrix(cm, class_names, f"{split_name}_confusion_matrix.png")
        
        return accuracy
    
    def plot_training_curves(self, train_losses, val_losses, train_accuracies, val_accuracies):
        """학습 곡선 시각화"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # 손실 곡선
        ax1.plot(train_losses, label='훈련 손실')
        ax1.plot(val_losses, label='검증 손실')
        ax1.set_title('학습 손실')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # 정확도 곡선
        ax2.plot(train_accuracies, label='훈련 정확도')
        ax2.plot(val_accuracies, label='검증 정확도')
        ax2.set_title('학습 정확도')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('unified_training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("📊 학습 곡선 저장: unified_training_curves.png")
    
    def plot_confusion_matrix(self, cm, class_names, filename):
        """혼동 행렬 시각화"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('혼동 행렬')
        plt.xlabel('예측')
        plt.ylabel('실제')
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📊 혼동 행렬 저장: {filename}")

def main():
    """메인 실행 함수"""
    print("🎯 SignGlove_HW Unified 데이터셋 모델 학습")
    print("=" * 60)
    
    trainer = UnifiedModelTrainer()
    
    # 1. 데이터 로드
    csv_files = trainer.load_unified_data()
    
    if not csv_files:
        print("❌ 데이터 파일이 없습니다.")
        return
    
    # 2. 균형잡힌 데이터셋 생성
    dataset = trainer.create_balanced_dataset(csv_files)
    
    if len(dataset) == 0:
        print("❌ 데이터셋이 비어있습니다.")
        return
    
    # 3. 모델 학습
    model = trainer.train_model(dataset)
    
    print("🎉 Unified 모델 학습 완료!")
    print("📁 생성된 파일:")
    print("  - best_unified_model.pth: 최고 성능 모델")
    print("  - unified_training_curves.png: 학습 곡선")
    print("  - 테스트_confusion_matrix.png: 혼동 행렬")

if __name__ == "__main__":
    main()
