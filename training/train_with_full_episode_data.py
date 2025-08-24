#!/usr/bin/env python3
"""
풍부한 Episode 데이터를 사용한 모델 재학습 스크립트
600개의 episode CSV 파일을 활용한 고품질 모델 학습
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import glob
import time
import json
from sklearn.metrics import confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

# 프로젝트 루트 추가
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.deep_learning import DeepLearningPipeline
from training.label_mapping import KSLLabelMapper

class EpisodeDataset(Dataset):
    """Episode 데이터를 사용하는 데이터셋"""
    
    def __init__(self, data_dir, window_size=20, stride=5, augment=True, target_samples_per_class=2000):
        self.data_dir = data_dir
        self.window_size = window_size
        self.stride = stride
        self.augment = augment
        self.target_samples_per_class = target_samples_per_class
        self.label_mapper = KSLLabelMapper()
        
        self.data = []
        self.labels = []
        
        print("🔄 풍부한 Episode 데이터셋 로딩 중...")
        self._load_episode_data()
        
    def _load_episode_data(self):
        """Episode 데이터 로딩"""
        # 모든 episode CSV 파일 찾기
        data_files = glob.glob(os.path.join(self.data_dir, "**/episode_*.csv"), recursive=True)
        
        if not data_files:
            raise ValueError("episode CSV 파일을 찾을 수 없습니다.")
        
        print(f"📁 발견된 Episode 파일: {len(data_files)}개")
        
        # 클래스별 데이터 수집
        class_data = {}
        
        for file_path in data_files:
            # 파일 경로에서 클래스명 추출
            # 예: .../ㄱ/1/episode_20250819_190506_ㄱ_1.csv
            path_parts = file_path.split('/')
            class_name = None
            
            # 경로에서 한글 클래스명 찾기
            for part in path_parts:
                if part in self.label_mapper.get_consonants() or part in self.label_mapper.get_vowels():
                    class_name = part
                    break
            
            if not class_name:
                print(f"⚠️ 클래스명을 찾을 수 없음: {file_path}")
                continue
            
            if class_name not in class_data:
                class_data[class_name] = []
            
            try:
                df = pd.read_csv(file_path)
                
                # 컬럼명 확인 및 정리
                available_cols = df.columns.tolist()
                
                # Flex 센서 컬럼 찾기
                flex_cols = []
                for i in range(1, 6):
                    flex_patterns = [f'flex{i}', f'Flex{i}', f'FLEX{i}']
                    for pattern in flex_patterns:
                        if pattern in available_cols:
                            flex_cols.append(pattern)
                            break
                
                # Orientation 컬럼 찾기
                pitch_col = None
                roll_col = None
                yaw_col = None
                
                for col in available_cols:
                    if 'pitch' in col.lower():
                        pitch_col = col
                    elif 'roll' in col.lower():
                        roll_col = col
                    elif 'yaw' in col.lower():
                        yaw_col = col
                
                if not flex_cols or not all([pitch_col, roll_col, yaw_col]):
                    print(f"⚠️ 필요한 컬럼이 없음: {file_path}")
                    print(f"  Flex: {flex_cols}, Pitch: {pitch_col}, Roll: {roll_col}, Yaw: {yaw_col}")
                    continue
                
                target_cols = flex_cols + [pitch_col, roll_col, yaw_col]
                arr = df[target_cols].values
                
                # 윈도우 기반 데이터 생성
                for start in range(0, len(arr)-self.window_size+1, self.stride):
                    window = arr[start:start+self.window_size]
                    class_data[class_name].append(window)
                    
            except Exception as e:
                print(f"⚠️ 파일 처리 실패: {file_path} - {e}")
        
        # 데이터 증강 및 균형 조정
        for class_name, windows in class_data.items():
            class_id = self.label_mapper.get_label_id(class_name)
            
            if class_id == -1:
                print(f"⚠️ 알 수 없는 클래스: {class_name}")
                continue
            
            print(f"📊 {class_name} 클래스: 원본 {len(windows)}개 윈도우")
            
            # 데이터 증강
            if self.augment and len(windows) < self.target_samples_per_class:
                augmented_windows = self._augment_data(windows, self.target_samples_per_class)
                print(f"  → 증강 후 {len(augmented_windows)}개 윈도우")
            else:
                augmented_windows = windows
            
            # 데이터 추가
            self.data.extend(augmented_windows)
            self.labels.extend([class_id] * len(augmented_windows))
        
        print(f"✅ 총 {len(self.data)}개 윈도우 로드 완료")
        
        # 클래스 분포 출력
        unique, counts = np.unique(self.labels, return_counts=True)
        print(f"📊 최종 클래스 분포:")
        for class_id, count in zip(unique, counts):
            class_name = self.label_mapper.get_class_name(class_id)
            print(f"  {class_name}: {count}개")
    
    def _augment_data(self, windows, target_count):
        """데이터 증강"""
        if len(windows) == 0:
            return []
        
        augmented = windows.copy()
        
        while len(augmented) < target_count:
            # 랜덤하게 원본 윈도우 선택 (인덱스로 선택)
            original_idx = np.random.randint(0, len(windows))
            original = windows[original_idx]
            
            # 다양한 증강 기법 적용
            augmented_window = self._apply_augmentation(original)
            augmented.append(augmented_window)
        
        return augmented[:target_count]  # 정확히 target_count개만 반환
    
    def _apply_augmentation(self, window):
        """개별 윈도우에 증강 적용"""
        # 노이즈 추가
        noise_factor = np.random.uniform(0.01, 0.03)
        noise = np.random.normal(0, noise_factor, window.shape)
        augmented = window + noise
        
        # 시간 축에서의 작은 변형
        if np.random.random() < 0.2:
            # 시간 축에서 랜덤하게 일부 프레임 제거/복제
            time_shift = np.random.randint(-1, 2)
            if time_shift > 0:
                # 프레임 복제
                augmented = np.concatenate([augmented, augmented[-time_shift:]], axis=0)[:self.window_size]
            elif time_shift < 0:
                # 프레임 제거
                augmented = augmented[abs(time_shift):]
                if len(augmented) < self.window_size:
                    # 부족한 부분을 마지막 프레임으로 채움
                    padding = np.tile(augmented[-1:], (self.window_size - len(augmented), 1))
                    augmented = np.concatenate([augmented, padding], axis=0)
        
        # 센서 값 범위 내에서 클리핑
        # Flex 센서 (500-1000 범위)
        augmented[:, :5] = np.clip(augmented[:, :5], 500, 1000)
        # Orientation (-180~180 범위)
        augmented[:, 5:] = np.clip(augmented[:, 5:], -180, 180)
        
        return augmented
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        return sample.astype(np.float32), label

class EpisodeTrainer:
    """Episode 데이터를 사용하는 학습기"""
    
    def __init__(self, config=None):
        self.config = self._get_default_config()
        if config:
            self.config.update(config)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.label_mapper = KSLLabelMapper()
        
        print(f"🚀 Episode 데이터 학습기 초기화")
        print(f"  장치: {self.device}")
        print(f"  클래스 수: {self.label_mapper.get_num_classes()}")
        print(f"  설정: {self.config}")
    
    def _get_default_config(self):
        """기본 설정"""
        return {
            'data_dir': 'integrations/SignGlove_HW/github_unified_data',
            'window_size': 20,
            'stride': 5,
            'batch_size': 32,  # 더 큰 배치 크기
            'learning_rate': 0.001,  # 더 높은 학습률
            'epochs': 150,  # 더 많은 에포크
            'train_split': 0.7,
            'val_split': 0.15,
            'test_split': 0.15,
            'early_stopping_patience': 20,  # 더 긴 인내심
            'save_best_only': True,
            'augment': True,
            'target_samples_per_class': 2000,  # 더 많은 목표 샘플
            'weight_decay': 1e-4,
            'scheduler_patience': 15,
            'scheduler_factor': 0.7
        }
    
    def create_dataset(self) -> EpisodeDataset:
        """Episode 데이터셋 생성"""
        dataset = EpisodeDataset(
            data_dir=self.config['data_dir'],
            window_size=self.config['window_size'],
            stride=self.config['stride'],
            augment=self.config['augment'],
            target_samples_per_class=self.config['target_samples_per_class']
        )
        return dataset
    
    def train_model(self, dataset: EpisodeDataset):
        """모델 학습"""
        print("🚀 풍부한 Episode 데이터로 모델 학습 시작...")
        
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
            shuffle=True,
            num_workers=0
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.config['batch_size'], 
            shuffle=False,
            num_workers=0
        )
        test_loader = DataLoader(
            test_dataset, 
            batch_size=self.config['batch_size'], 
            shuffle=False,
            num_workers=0
        )
        
        # 모델 초기화
        model = DeepLearningPipeline(
            input_features=8,
            sequence_length=self.config['window_size'],
            num_classes=self.label_mapper.get_num_classes(),
            hidden_dim=256,  # 더 큰 hidden dimension
            num_layers=4,    # 더 깊은 네트워크
            dropout=0.4      # 더 높은 dropout
        ).to(self.device)
        
        # 손실 함수 및 옵티마이저
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(
            model.parameters(), 
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        
        # 학습률 스케줄러
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=self.config['scheduler_factor'],
            patience=self.config['scheduler_patience']
        )
        
        # 학습 루프
        best_val_loss = float('inf')
        patience_counter = 0
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []
        
        print(f"\n🎯 학습 시작 (총 {self.config['epochs']} 에포크)")
        print("=" * 60)
        
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
                
                if isinstance(outputs, dict):
                    outputs = outputs['class_logits']
                
                loss = criterion(outputs, targets)
                loss.backward()
                
                # 그래디언트 클리핑
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += targets.size(0)
                train_correct += (predicted == targets).sum().item()
                
                if batch_idx % 100 == 0:
                    print(f"  에포크 {epoch+1}/{self.config['epochs']} - 배치 {batch_idx}/{len(train_loader)} - 손실: {loss.item():.4f}")
            
            # 검증
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for data, targets in val_loader:
                    data, targets = data.to(self.device), targets.to(self.device)
                    outputs = model(data)
                    
                    if isinstance(outputs, dict):
                        outputs = outputs['class_logits']
                    
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()
                    
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += targets.size(0)
                    val_correct += (predicted == targets).sum().item()
            
            # 평균 손실 및 정확도 계산
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            train_accuracy = 100 * train_correct / train_total
            val_accuracy = 100 * val_correct / val_total
            
            # 스케줄러 업데이트
            scheduler.step(avg_val_loss)
            
            # 결과 저장
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            train_accuracies.append(train_accuracy)
            val_accuracies.append(val_accuracy)
            
            # 진행 상황 출력
            print(f"에포크 {epoch+1}/{self.config['epochs']}:")
            print(f"  훈련 손실: {avg_train_loss:.4f}, 정확도: {train_accuracy:.2f}%")
            print(f"  검증 손실: {avg_val_loss:.4f}, 정확도: {val_accuracy:.2f}%")
            print(f"  학습률: {optimizer.param_groups[0]['lr']:.6f}")
            
            # 최고 모델 저장
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                
                if self.config['save_best_only']:
                    torch.save(model, 'best_episode_model.pth')
                    print(f"  ✅ 새로운 최고 모델 저장!")
            else:
                patience_counter += 1
                print(f"  ⏳ 인내심: {patience_counter}/{self.config['early_stopping_patience']}")
            
            # Early stopping
            if patience_counter >= self.config['early_stopping_patience']:
                print(f"  🛑 Early stopping at epoch {epoch+1}")
                break
            
            print("-" * 40)
        
        # 최종 모델 저장
        torch.save(model, 'final_episode_model.pth')
        
        # 학습 곡선 플롯
        self._plot_training_curves(train_losses, val_losses, train_accuracies, val_accuracies)
        
        # 테스트 성능 평가
        self._evaluate_model(model, test_loader)
        
        return model
    
    def _plot_training_curves(self, train_losses, val_losses, train_accuracies, val_accuracies):
        """학습 곡선 플롯"""
        plt.figure(figsize=(15, 6))
        
        # 손실 곡선
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='훈련 손실', color='blue')
        plt.plot(val_losses, label='검증 손실', color='red')
        plt.title('학습 및 검증 손실', fontsize=14, fontweight='bold')
        plt.xlabel('에포크')
        plt.ylabel('손실')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 정확도 곡선
        plt.subplot(1, 2, 2)
        plt.plot(train_accuracies, label='훈련 정확도', color='blue')
        plt.plot(val_accuracies, label='검증 정확도', color='red')
        plt.title('학습 및 검증 정확도', fontsize=14, fontweight='bold')
        plt.xlabel('에포크')
        plt.ylabel('정확도 (%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('episode_training_curves.png', dpi=300, bbox_inches='tight')
        print("✅ 학습 곡선 저장: episode_training_curves.png")
        plt.show()
    
    def _evaluate_model(self, model, test_loader):
        """모델 평가"""
        print(f"\n🎯 최종 모델 평가")
        print("=" * 60)
        
        model.eval()
        test_correct = 0
        test_total = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for data, targets in test_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = model(data)
                
                if isinstance(outputs, dict):
                    outputs = outputs['class_logits']
                
                _, predicted = torch.max(outputs.data, 1)
                test_total += targets.size(0)
                test_correct += (predicted == targets).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        test_accuracy = 100 * test_correct / test_total
        print(f"📊 테스트 정확도: {test_accuracy:.2f}%")
        
        # 혼동 행렬 생성
        self._plot_confusion_matrix(all_targets, all_predictions)
        
        # 클래스별 성능
        self._print_class_performance(all_targets, all_predictions)
    
    def _plot_confusion_matrix(self, targets, predictions):
        """혼동 행렬 플롯"""
        cm = confusion_matrix(targets, predictions)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=[self.label_mapper.get_class_name(i) for i in range(len(cm))],
                   yticklabels=[self.label_mapper.get_class_name(i) for i in range(len(cm))])
        plt.title('혼동 행렬', fontsize=16, fontweight='bold')
        plt.xlabel('예측 클래스')
        plt.ylabel('실제 클래스')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig('episode_confusion_matrix.png', dpi=300, bbox_inches='tight')
        print("✅ 혼동 행렬 저장: episode_confusion_matrix.png")
        plt.show()
    
    def _print_class_performance(self, targets, predictions):
        """클래스별 성능 출력"""
        print(f"\n📊 클래스별 성능:")
        print("-" * 60)
        
        report = classification_report(targets, predictions, 
                                     target_names=[self.label_mapper.get_class_name(i) for i in range(self.label_mapper.get_num_classes())],
                                     output_dict=True)
        
        for class_name in report.keys():
            if class_name in ['accuracy', 'macro avg', 'weighted avg']:
                continue
            
            metrics = report[class_name]
            print(f"{class_name:<4}: 정확도={metrics['precision']:.3f}, 재현율={metrics['recall']:.3f}, F1={metrics['f1-score']:.3f}")

def main():
    """메인 함수"""
    print("🚀 풍부한 Episode 데이터로 모델 재학습 시작")
    print("=" * 60)
    
    # 학습기 초기화
    trainer = EpisodeTrainer()
    
    # 데이터셋 생성
    dataset = trainer.create_dataset()
    
    # 모델 학습
    model = trainer.train_model(dataset)
    
    print(f"\n🎉 Episode 데이터 모델 학습 완료!")
    print(f"✅ 최고 모델: best_episode_model.pth")
    print(f"✅ 최종 모델: final_episode_model.pth")

if __name__ == "__main__":
    main()
