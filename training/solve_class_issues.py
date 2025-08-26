#!/usr/bin/env python3
"""
클래스 문제 해결 스크립트
과적합 클래스 + 실패한 클래스 (ㅊ, ㅕ) 해결 방안
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import glob
import json
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, RobustScaler
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# 프로젝트 루트 추가
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.deep_learning import DeepLearningPipeline
from training.label_mapping import KSLLabelMapper

class ProblemSolverDataset(Dataset):
    """클래스 문제 해결을 위한 데이터셋"""
    
    def __init__(self, data_dir, window_size=20, stride=5, augment=True):
        self.data_dir = data_dir
        self.window_size = window_size
        self.stride = stride
        self.augment = augment
        self.label_mapper = KSLLabelMapper()
        
        # 문제가 있는 클래스들
        self.overfitting_classes = ['ㄱ', 'ㄴ', 'ㅂ', 'ㅇ', 'ㅎ', 'ㅏ', 'ㅣ']  # 1.000 성능
        self.failed_classes = ['ㅊ', 'ㅕ']  # 0.000 성능
        
        self.data = []
        self.labels = []
        self.scenario_sources = []
        
        print("🔄 클래스 문제 해결 데이터셋 로딩 중...")
        self._load_episode_data()
        self._analyze_class_distribution()
    
    def _load_episode_data(self):
        """Episode 데이터 로딩"""
        data_files = glob.glob(os.path.join(self.data_dir, "**/episode_*.csv"), recursive=True)
        
        print(f"📁 발견된 Episode 파일: {len(data_files)}개")
        
        for file_path in data_files:
            try:
                # 클래스 이름과 시나리오 추출
                class_name, scenario_id = self._extract_class_and_scenario(file_path)
                if class_name not in self.label_mapper.class_to_id:
                    continue
                
                class_id = self.label_mapper.class_to_id[class_name]
                
                # CSV 파일 읽기
                df = pd.read_csv(file_path)
                
                # 센서 데이터 추출 및 전처리
                sensor_data = self._preprocess_sensor_data(df, class_name)
                
                # 윈도우 생성
                windows = self._create_windows(sensor_data)
                
                # 데이터 추가
                for window in windows:
                    self.data.append(window)
                    self.labels.append(class_id)
                    self.scenario_sources.append(f"{class_name}_{scenario_id}")
                
            except Exception as e:
                print(f"⚠️  파일 로드 실패: {file_path} - {e}")
        
        print(f"✅ 총 {len(self.data)}개 윈도우 로드 완료")
    
    def _preprocess_sensor_data(self, df, class_name):
        """센서 데이터 전처리"""
        sensor_data = df[['pitch', 'roll', 'yaw', 'flex1', 'flex2', 'flex3', 'flex4', 'flex5']].values
        
        # 클래스별 맞춤 전처리
        if class_name in self.failed_classes:
            # 실패한 클래스: 이상값 제거 및 정규화
            sensor_data = self._clean_failed_class_data(sensor_data, class_name)
        elif class_name in self.overfitting_classes:
            # 과적합 클래스: 약한 정규화
            sensor_data = self._normalize_overfitting_class_data(sensor_data)
        
        return sensor_data
    
    def _clean_failed_class_data(self, data, class_name):
        """실패한 클래스 데이터 정리"""
        # ㅕ 클래스의 flex2 센서 문제 해결
        if class_name == 'ㅕ':
            # flex2 센서의 비정상적인 값들 처리
            flex2_col = 4  # flex2 인덱스
            flex2_values = data[:, flex2_col]
            
            # 비정상적으로 낮은 값들 (0 근처) 제거
            normal_mask = flex2_values > 100  # 정상 범위
            if np.sum(normal_mask) > len(data) * 0.5:  # 50% 이상이 정상이면
                data = data[normal_mask]
        
        # ㅊ 클래스의 시나리오별 불일치 해결
        if class_name == 'ㅊ':
            # flex5 센서의 높은 변동성 감소
            flex5_col = 7  # flex5 인덱스
            flex5_values = data[:, flex5_col]
            
            # 이상값 제거 (3 표준편차 이상)
            mean_flex5 = np.mean(flex5_values)
            std_flex5 = np.std(flex5_values)
            normal_mask = np.abs(flex5_values - mean_flex5) <= 3 * std_flex5
            data = data[normal_mask]
        
        return data
    
    def _normalize_overfitting_class_data(self, data):
        """과적합 클래스 데이터 정규화"""
        # 약한 노이즈 추가로 과적합 방지
        if np.random.random() < 0.1:  # 10% 확률로 노이즈 추가
            noise = np.random.normal(0, 0.001, data.shape)
            data = data + noise
        
        return data
    
    def _create_windows(self, sensor_data):
        """윈도우 생성"""
        windows = []
        for i in range(0, len(sensor_data) - self.window_size + 1, self.stride):
            window = sensor_data[i:i + self.window_size]
            windows.append(window)
        return windows
    
    def _extract_class_and_scenario(self, file_path):
        """파일 경로에서 클래스 이름과 시나리오 추출"""
        parts = file_path.split(os.sep)
        
        # 클래스 이름 찾기
        class_name = None
        for part in parts:
            if part in self.label_mapper.class_to_id:
                class_name = part
                break
        
        # 시나리오 ID 찾기 (숫자 폴더)
        scenario_id = None
        for part in parts:
            if part.isdigit() and 1 <= int(part) <= 5:
                scenario_id = part
                break
        
        return class_name, scenario_id
    
    def _analyze_class_distribution(self):
        """클래스 분포 분석"""
        class_counts = Counter(self.labels)
        print(f"\n📊 클래스별 분포 분석:")
        
        for class_id, count in sorted(class_counts.items()):
            class_name = list(self.label_mapper.class_to_id.keys())[class_id]
            status = ""
            if class_name in self.overfitting_classes:
                status = " (과적합)"
            elif class_name in self.failed_classes:
                status = " (실패)"
            print(f"  {class_name}: {count}개{status}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        
        # 클래스별 맞춤형 데이터 증강
        if self.augment:
            sample = self._apply_smart_augmentation(sample, label)
        
        return sample.astype(np.float32), label
    
    def _apply_smart_augmentation(self, sample, label):
        """스마트 데이터 증강"""
        class_name = list(self.label_mapper.class_to_id.keys())[label]
        
        if class_name in self.failed_classes:
            # 실패한 클래스: 강한 증강
            if np.random.random() < 0.6:  # 60% 확률
                # 가우시안 노이즈
                noise = np.random.normal(0, 0.005, sample.shape)
                sample = sample + noise
                
                # 시간 이동
                if np.random.random() < 0.4:
                    shift = np.random.randint(-2, 3)
                    if shift != 0:
                        sample = np.roll(sample, shift, axis=0)
                        if shift > 0:
                            sample[:shift] = sample[shift]
                        else:
                            sample[shift:] = sample[shift-1]
                
                # 스케일링
                if np.random.random() < 0.3:
                    scale = np.random.uniform(0.95, 1.05)
                    sample = sample * scale
        
        elif class_name in self.overfitting_classes:
            # 과적합 클래스: 약한 증강
            if np.random.random() < 0.2:  # 20% 확률
                # 작은 노이즈만
                noise = np.random.normal(0, 0.001, sample.shape)
                sample = sample + noise
        
        else:
            # 일반 클래스: 중간 증강
            if np.random.random() < 0.4:  # 40% 확률
                noise = np.random.normal(0, 0.003, sample.shape)
                sample = sample + noise
                
                if np.random.random() < 0.3:
                    shift = np.random.randint(-1, 2)
                    if shift != 0:
                        sample = np.roll(sample, shift, axis=0)
                        if shift > 0:
                            sample[:shift] = sample[shift]
                        else:
                            sample[shift:] = sample[shift-1]
        
        return sample
    
    def get_scenario_sources(self):
        """시나리오 출처 반환"""
        return self.scenario_sources

class ProblemSolverTrainer:
    """클래스 문제 해결 학습기"""
    
    def __init__(self, config=None):
        self.config = self._get_default_config()
        if config:
            self.config.update(config)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.label_mapper = KSLLabelMapper()
        
        print(f"🚀 클래스 문제 해결 학습기 초기화")
        print(f"  장치: {self.device}")
        print(f"  클래스 수: {self.label_mapper.get_num_classes()}")
        print(f"  설정: {self.config}")
    
    def _get_default_config(self):
        """클래스 문제 해결 설정"""
        return {
            'data_dir': 'integrations/SignGlove_HW/github_unified_data',
            'window_size': 20,
            'stride': 5,
            'batch_size': 32,
            'learning_rate': 0.0001,
            'epochs': 50,
            'train_scenarios': [1, 2, 3],
            'val_scenarios': [4],
            'test_scenarios': [5],
            'early_stopping_patience': 10,
            'save_best_only': True,
            'augment': True,
            'weight_decay': 1e-4,
            'scheduler_patience': 8,
            'scheduler_factor': 0.8,
            'use_class_weights': True,
            'use_focal_loss': False,  # 실패한 클래스에 집중
            'focal_alpha': 1.0,
            'focal_gamma': 2.0
        }
    
    def create_dataset(self) -> ProblemSolverDataset:
        """클래스 문제 해결 데이터셋 생성"""
        dataset = ProblemSolverDataset(
            data_dir=self.config['data_dir'],
            window_size=self.config['window_size'],
            stride=self.config['stride'],
            augment=self.config['augment']
        )
        return dataset
    
    def split_dataset_by_scenarios(self, dataset: ProblemSolverDataset):
        """시나리오 단위로 데이터 분할"""
        print("🔒 시나리오 단위 데이터 분할")
        
        # 시나리오별로 데이터 인덱스 그룹화
        scenario_groups = {}
        for i, scenario in enumerate(dataset.get_scenario_sources()):
            if scenario not in scenario_groups:
                scenario_groups[scenario] = []
            scenario_groups[scenario].append(i)
        
        # 시나리오별로 분할
        train_indices = []
        val_indices = []
        test_indices = []
        
        for scenario, indices in scenario_groups.items():
            scenario_num = int(scenario.split('_')[1])
            
            if scenario_num in self.config['train_scenarios']:
                train_indices.extend(indices)
            elif scenario_num in self.config['val_scenarios']:
                val_indices.extend(indices)
            elif scenario_num in self.config['test_scenarios']:
                test_indices.extend(indices)
        
        print(f"📊 시나리오 분할 결과:")
        print(f"  훈련 시나리오: {self.config['train_scenarios']}")
        print(f"  검증 시나리오: {self.config['val_scenarios']}")
        print(f"  테스트 시나리오: {self.config['test_scenarios']}")
        print(f"  훈련 샘플: {len(train_indices)}개")
        print(f"  검증 샘플: {len(val_indices)}개")
        print(f"  테스트 샘플: {len(test_indices)}개")
        
        return train_indices, val_indices, test_indices
    
    def create_class_weights(self, dataset, train_indices):
        """클래스별 가중치 생성"""
        if not self.config['use_class_weights']:
            return None
        
        # 훈련 데이터의 클래스별 분포
        train_labels = [dataset.labels[i] for i in train_indices]
        class_counts = Counter(train_labels)
        
        # 클래스별 가중치 계산
        total_samples = len(train_labels)
        class_weights = {}
        
        for class_id in range(self.label_mapper.get_num_classes()):
            class_name = list(self.label_mapper.class_to_id.keys())[class_id]
            
            if class_name in ['ㅊ', 'ㅕ']:  # 실패한 클래스
                # 높은 가중치
                class_weights[class_id] = total_samples / (len(class_counts) * max(class_counts.get(class_id, 1), 1)) * 3.0
            elif class_name in ['ㄱ', 'ㄴ', 'ㅂ', 'ㅇ', 'ㅎ', 'ㅏ', 'ㅣ']:  # 과적합 클래스
                # 낮은 가중치
                class_weights[class_id] = total_samples / (len(class_counts) * max(class_counts.get(class_id, 1), 1)) * 0.5
            else:
                # 일반 가중치
                class_weights[class_id] = total_samples / (len(class_counts) * max(class_counts.get(class_id, 1), 1))
        
        print(f"\n⚖️  클래스별 가중치:")
        for class_id, weight in sorted(class_weights.items()):
            class_name = list(self.label_mapper.class_to_id.keys())[class_id]
            count = class_counts.get(class_id, 0)
            print(f"  {class_name}: {count}개, 가중치 {weight:.3f}")
        
        return class_weights
    
    def train_model(self, dataset: ProblemSolverDataset):
        """클래스 문제 해결 모델 학습"""
        print("🚀 클래스 문제 해결 모델 학습 시작...")
        
        # 시나리오 단위 데이터 분할
        train_indices, val_indices, test_indices = self.split_dataset_by_scenarios(dataset)
        
        # 클래스 가중치 생성
        class_weights = self.create_class_weights(dataset, train_indices)
        
        # 데이터 로더 생성
        train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
        val_sampler = torch.utils.data.SubsetRandomSampler(val_indices)
        test_sampler = torch.utils.data.SubsetRandomSampler(test_indices)
        
        train_loader = DataLoader(
            dataset, 
            batch_size=self.config['batch_size'], 
            sampler=train_sampler,
            num_workers=0
        )
        val_loader = DataLoader(
            dataset, 
            batch_size=self.config['batch_size'], 
            sampler=val_sampler,
            num_workers=0
        )
        test_loader = DataLoader(
            dataset, 
            batch_size=self.config['batch_size'], 
            sampler=test_sampler,
            num_workers=0
        )
        
        # 모델 초기화 - 균형잡힌 모델
        model = DeepLearningPipeline(
            input_features=8,
            hidden_dim=64,  # 중간 크기
            num_classes=self.label_mapper.get_num_classes(),
            sequence_length=self.config['window_size'],
            num_layers=2,  # 2 레이어
            dropout=0.3  # 중간 dropout
        ).to(self.device)
        
        # 손실 함수 (가중치 적용)
        if class_weights:
            weight_tensor = torch.FloatTensor([class_weights[i] for i in range(self.label_mapper.get_num_classes())]).to(self.device)
            criterion = nn.CrossEntropyLoss(weight=weight_tensor)
            print("🎯 가중치가 적용된 Cross Entropy Loss 사용")
        else:
            criterion = nn.CrossEntropyLoss()
            print("🎯 일반 Cross Entropy Loss 사용")
        
        # 옵티마이저 및 스케줄러
        optimizer = optim.AdamW(
            model.parameters(), 
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
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
        train_accs = []
        val_accs = []
        
        print(f"🎯 학습 시작 (최대 {self.config['epochs']} 에포크)")
        print(f"🛑 조기 종료: 검증 손실 기준, 인내심 {self.config['early_stopping_patience']}")
        
        for epoch in range(self.config['epochs']):
            # 훈련
            model.train()
            train_loss = 0
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
            val_loss = 0
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
            
            # 메트릭 계산
            train_loss = train_loss / len(train_loader)
            val_loss = val_loss / len(val_loader)
            train_acc = 100. * train_correct / train_total
            val_acc = 100. * val_correct / val_total
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accs.append(train_acc)
            val_accs.append(val_acc)
            
            # 스케줄러 업데이트
            scheduler.step(val_loss)
            
            # 모델 저장 (검증 손실 기준)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), 'best_problem_solver_model.pth')
                print(f"✅ 최고 모델 저장: best_problem_solver_model.pth (검증 손실: {val_loss:.4f}, 정확도: {val_acc:.2f}%)")
            else:
                patience_counter += 1
            
            print(f"Epoch {epoch+1}/{self.config['epochs']}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # 조기 종료 (검증 손실 기준)
            if patience_counter >= self.config['early_stopping_patience']:
                print(f"🛑 조기 종료! 검증 손실이 {self.config['early_stopping_patience']} 에포크 동안 개선되지 않음")
                print(f"📊 최고 성능: 검증 손실 {best_val_loss:.4f} (에포크 {epoch+1-self.config['early_stopping_patience']})")
                break
        
        # 학습 곡선 저장
        self._save_training_curves(train_losses, val_losses, train_accs, val_accs)
        
        # 테스트 성능 평가
        self._evaluate_model(model, test_loader)
        
        return model
    
    def _save_training_curves(self, train_losses, val_losses, train_accs, val_accs):
        """학습 곡선 저장"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # 손실 곡선
        ax1.plot(train_losses, label='Train Loss', color='blue')
        ax1.plot(val_losses, label='Val Loss', color='red')
        ax1.set_title('Problem Solver Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # 정확도 곡선
        ax2.plot(train_accs, label='Train Acc', color='blue')
        ax2.plot(val_accs, label='Val Acc', color='red')
        ax2.set_title('Problem Solver Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('problem_solver_training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("📈 학습 곡선 저장: problem_solver_training_curves.png")
    
    def _evaluate_model(self, model, test_loader):
        """모델 평가"""
        model.eval()
        test_correct = 0
        test_total = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for data, targets in test_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = model(data)
                _, predicted = outputs['class_logits'].max(1)
                
                test_total += targets.size(0)
                test_correct += predicted.eq(targets).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        test_acc = 100. * test_correct / test_total
        print(f"\n📊 테스트 성능:")
        print(f"  정확도: {test_acc:.2f}% ({test_correct}/{test_total})")
        
        # 클래스별 성능
        print(f"\n📋 클래스별 성능:")
        
        # 실제 예측된 클래스만 사용
        unique_classes = sorted(list(set(all_targets + all_predictions)))
        class_names = [list(self.label_mapper.class_to_id.keys())[i] for i in unique_classes]
        
        report = classification_report(all_targets, all_predictions, 
                                     labels=unique_classes,
                                     target_names=class_names,
                                     output_dict=True)
        
        for class_name in class_names:
            if class_name in report:
                precision = report[class_name]['precision']
                recall = report[class_name]['recall']
                f1 = report[class_name]['f1-score']
                status = ""
                if class_name in ['ㄱ', 'ㄴ', 'ㅂ', 'ㅇ', 'ㅎ', 'ㅏ', 'ㅣ']:
                    status = " (과적합)"
                elif class_name in ['ㅊ', 'ㅕ']:
                    status = " (실패)"
                print(f"  {class_name}: 정확도={precision:.3f}, 재현율={recall:.3f}, F1={f1:.3f}{status}")
        
        # 혼동 행렬 저장
        self._save_confusion_matrix(all_targets, all_predictions)
    
    def _save_confusion_matrix(self, targets, predictions):
        """혼동 행렬 저장"""
        cm = confusion_matrix(targets, predictions)
        
        # 실제 예측된 클래스만 사용
        unique_classes = sorted(list(set(targets + predictions)))
        class_names = [list(self.label_mapper.class_to_id.keys())[i] for i in unique_classes]
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names,
                   yticklabels=class_names)
        plt.title('Problem Solver Model Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig('problem_solver_테스트_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("📊 혼동 행렬 저장: problem_solver_테스트_confusion_matrix.png")

def main():
    """메인 함수"""
    print("🚀 클래스 문제 해결 시작")
    print("=" * 60)
    
    # 라벨 매퍼 초기화
    label_mapper = KSLLabelMapper()
    print("🎯 24개 클래스 라벨 매퍼 초기화 완료")
    print(f"  📝 자음: {label_mapper.get_consonants()}")
    print(f"  📝 모음: {label_mapper.get_vowels()}")
    
    # 학습기 초기화
    trainer = ProblemSolverTrainer()
    
    # 데이터셋 생성
    dataset = trainer.create_dataset()
    
    # 모델 학습
    model = trainer.train_model(dataset)
    
    print("🎉 클래스 문제 해결 모델 학습 완료!")
    print("\n✅ 클래스 문제 해결 완료!")
    print("📁 생성된 파일들:")
    print("  - best_problem_solver_model.pth")
    print("  - problem_solver_training_curves.png")
    print("  - problem_solver_테스트_confusion_matrix.png")

if __name__ == "__main__":
    main()
