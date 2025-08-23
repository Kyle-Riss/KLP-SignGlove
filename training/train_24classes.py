"""
24개 클래스 (14개 자음 + 10개 모음) 모델 학습
SignGlove_HW unified 데이터셋으로 24개 클래스 모두 지원하는 모델 학습
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

class TwentyFourClassTrainer:
    """24개 클래스 모델 학습기"""
    
    def __init__(self, data_dir: str = "integrations/SignGlove_HW"):
        self.data_dir = data_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.label_mapper = KSLLabelMapper()  # 24개 클래스 지원
        
        # 학습 설정
        self.config = {
            'batch_size': 32,
            'learning_rate': 0.001,
            'epochs': 100,  # 더 많은 에포크
            'window_size': 20,
            'stride': 10,
            'train_split': 0.8,
            'val_split': 0.1,
            'test_split': 0.1
        }
        
        print(f"🎯 24개 클래스 모델 학습기 초기화")
        print(f"🔧 학습 설정:")
        for key, value in self.config.items():
            print(f"  {key}: {value}")
        print(f"🖥️  장치: {self.device}")
        print(f"📊 지원 클래스: {self.label_mapper.get_num_classes()}개")
        print(f"  🔤 자음: {len(self.label_mapper.get_consonants())}개")
        print(f"  🅰️ 모음: {len(self.label_mapper.get_vowels())}개")
    
    def load_all_unified_data(self) -> List[str]:
        """모든 Unified 데이터셋 파일 목록 로드"""
        print("📁 24개 클래스 Unified 데이터셋 로드 중...")
        
        # Unified 데이터만 사용 (기존 샘플 데이터 제외)
        unified_files = glob.glob(os.path.join(self.data_dir, "*_unified_data_*.csv"))
        
        print(f"  📄 Unified 데이터: {len(unified_files)}개")
        
        # 클래스별 통계
        class_stats = {}
        for file_path in unified_files:
            filename = os.path.basename(file_path)
            
            # 파일명에서 클래스 추출
            for class_name in self.label_mapper.class_to_id.keys():
                if class_name in filename:
                    if class_name not in class_stats:
                        class_stats[class_name] = 0
                    class_stats[class_name] += 1
                    break
        
        print("📊 클래스별 데이터 통계:")
        consonants = []
        vowels = []
        for class_name, count in sorted(class_stats.items()):
            char_type = "자음" if self.label_mapper.is_consonant(class_name) else "모음"
            print(f"  {class_name} ({char_type}): {count}개 파일")
            if char_type == "자음":
                consonants.append((class_name, count))
            else:
                vowels.append((class_name, count))
        
        print(f"\n📈 요약:")
        print(f"  🔤 자음: {len(consonants)}개 클래스")
        print(f"  🅰️ 모음: {len(vowels)}개 클래스")
        print(f"  📄 총 파일: {len(unified_files)}개")
        
        return unified_files
    
    def create_balanced_24class_dataset(self, csv_files: List[str]) -> KSLCsvDataset:
        """24개 클래스 균형잡힌 데이터셋 생성"""
        print("⚖️ 24개 클래스 균형잡힌 데이터셋 생성 중...")
        
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
            char_type = "자음" if self.label_mapper.is_consonant(class_name) else "모음"
            print(f"  {class_name} ({char_type}): {len(selected_files)}개 선택")
        
        print(f"✅ 균형잡힌 데이터셋: {len(balanced_files)}개 파일")
        
        # 임시 디렉토리에 파일 복사
        temp_dir = "temp_24class_data"
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
    
    def train_24class_model(self, dataset: KSLCsvDataset):
        """24개 클래스 모델 학습"""
        print("🚀 24개 클래스 모델 학습 시작...")
        
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
        
        # 모델 초기화 (24개 클래스)
        num_classes = self.label_mapper.get_num_classes()
        model = DeepLearningPipeline(
            input_features=8,  # pitch, roll, yaw + 5 flex sensors
            sequence_length=self.config['window_size'],
            num_classes=num_classes,  # 24개 클래스
            hidden_dim=256,  # 더 큰 모델
            num_layers=3,    # 더 깊은 모델
            dropout=0.3
        ).to(self.device)
        
        print(f"🤖 모델 구조:")
        print(f"  입력 특성: 8개")
        print(f"  시퀀스 길이: {self.config['window_size']}")
        print(f"  출력 클래스: {num_classes}개")
        print(f"  히든 차원: 256")
        print(f"  레이어 수: 3")
        
        # 손실 함수와 옵티마이저
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.config['learning_rate'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10
        )
        
        # 학습 기록
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []
        best_val_loss = float('inf')
        best_model_path = "best_24class_model.pth"
        
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
                
                if batch_idx % 20 == 0:
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
        self.evaluate_24class_model(model, test_loader, "테스트")
        
        return model
    
    def evaluate_24class_model(self, model, data_loader, split_name: str):
        """24개 클래스 모델 평가"""
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
        
        # 클래스별 정확도
        class_names = [self.label_mapper.get_class_name(i) for i in range(len(self.label_mapper.class_to_id))]
        
        # 자음과 모음 분리
        consonant_indices = [i for i, name in enumerate(class_names) if self.label_mapper.is_consonant(name)]
        vowel_indices = [i for i, name in enumerate(class_names) if self.label_mapper.is_vowel(name)]
        
        consonant_accuracy = 0
        vowel_accuracy = 0
        
        if consonant_indices:
            consonant_correct = sum(1 for i, (pred, true) in enumerate(zip(all_predictions, all_targets)) 
                                  if true in consonant_indices and pred == true)
            consonant_total = sum(1 for true in all_targets if true in consonant_indices)
            if consonant_total > 0:
                consonant_accuracy = 100 * consonant_correct / consonant_total
        
        if vowel_indices:
            vowel_correct = sum(1 for i, (pred, true) in enumerate(zip(all_predictions, all_targets)) 
                              if true in vowel_indices and pred == true)
            vowel_total = sum(1 for true in all_targets if true in vowel_indices)
            if vowel_total > 0:
                vowel_accuracy = 100 * vowel_correct / vowel_total
        
        print(f"  🔤 자음 정확도: {consonant_accuracy:.2f}%")
        print(f"  🅰️ 모음 정확도: {vowel_accuracy:.2f}%")
        
        # 분류 보고서
        report = classification_report(all_targets, all_predictions, target_names=class_names)
        print(f"  📋 분류 보고서:\n{report}")
        
        # 혼동 행렬
        cm = confusion_matrix(all_targets, all_predictions)
        self.plot_24class_confusion_matrix(cm, class_names, f"{split_name}_24class_confusion_matrix.png")
        
        return accuracy
    
    def plot_training_curves(self, train_losses, val_losses, train_accuracies, val_accuracies):
        """학습 곡선 시각화"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # 손실 곡선
        ax1.plot(train_losses, label='훈련 손실')
        ax1.plot(val_losses, label='검증 손실')
        ax1.set_title('24개 클래스 학습 손실')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # 정확도 곡선
        ax2.plot(train_accuracies, label='훈련 정확도')
        ax2.plot(val_accuracies, label='검증 정확도')
        ax2.set_title('24개 클래스 학습 정확도')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('24class_training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("📊 학습 곡선 저장: 24class_training_curves.png")
    
    def plot_24class_confusion_matrix(self, cm, class_names, filename):
        """24개 클래스 혼동 행렬 시각화"""
        plt.figure(figsize=(20, 16))
        
        # 자음과 모음 색상 구분
        colors = []
        for name in class_names:
            if self.label_mapper.is_consonant(name):
                colors.append('Blues')  # 자음은 파란색
            else:
                colors.append('Reds')   # 모음은 빨간색
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('24개 클래스 혼동 행렬 (자음 + 모음)')
        plt.xlabel('예측')
        plt.ylabel('실제')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📊 혼동 행렬 저장: {filename}")

def main():
    """메인 실행 함수"""
    print("🎯 24개 클래스 (14개 자음 + 10개 모음) 모델 학습")
    print("=" * 70)
    
    trainer = TwentyFourClassTrainer()
    
    # 1. 데이터 로드
    csv_files = trainer.load_all_unified_data()
    
    if not csv_files:
        print("❌ 데이터 파일이 없습니다.")
        return
    
    # 2. 균형잡힌 데이터셋 생성
    dataset = trainer.create_balanced_24class_dataset(csv_files)
    
    if len(dataset) == 0:
        print("❌ 데이터셋이 비어있습니다.")
        return
    
    # 3. 24개 클래스 모델 학습
    model = trainer.train_24class_model(dataset)
    
    print("🎉 24개 클래스 모델 학습 완료!")
    print("📁 생성된 파일:")
    print("  - best_24class_model.pth: 24개 클래스 모델")
    print("  - 24class_training_curves.png: 학습 곡선")
    print("  - 테스트_24class_confusion_matrix.png: 혼동 행렬")

if __name__ == "__main__":
    main()

