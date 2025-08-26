#!/usr/bin/env python3
"""
앙상블 모델 훈련 스크립트
특화 모델 + 일반 모델 조합으로 전체 24개 클래스 성능 향상
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
from pathlib import Path
import glob
import json
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# 프로젝트 루트 추가
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from training.label_mapping import KSLLabelMapper
from models.deep_learning import DeepLearningPipeline

class EnsembleDataset(Dataset):
    """앙상블용 데이터셋 - 전체 24개 클래스"""
    
    def __init__(self, data_dir, mode='train'):
        self.data_dir = data_dir
        self.mode = mode
        self.label_mapper = KSLLabelMapper()
        
        # 혼동 클래스 그룹 정의 (특화 모델이 처리할 클래스들)
        self.confusion_classes = ['ㅊ', 'ㅑ', 'ㅕ', 'ㅌ', 'ㄹ']
        
        # 데이터 로딩
        self.data = []
        self.labels = []
        self.scenarios = []
        self.is_confusion_class = []  # 혼동 클래스 여부
        
        print(f"🔍 앙상블 모델 데이터 로딩: 전체 24개 클래스")
        
        # 모든 클래스 데이터 로딩
        all_classes = self.label_mapper.get_all_classes()
        
        for class_name in all_classes:
            self._load_class_data(class_name)
        
        print(f"📊 로딩 완료: {len(self.data)}개 샘플")
        print(f"🎯 혼동 클래스: {sum(self.is_confusion_class)}개")
        print(f"🎯 일반 클래스: {len(self.data) - sum(self.is_confusion_class)}개")
    
    def _load_class_data(self, class_name):
        """클래스별 데이터 로딩"""
        pattern = os.path.join(self.data_dir, f"**/{class_name}/**/episode_*.csv")
        files = glob.glob(pattern, recursive=True)
        
        print(f"  {class_name}: {len(files)}개 파일")
        
        for file_path in files:
            try:
                # 시나리오 추출
                scenario = self._extract_scenario(file_path)
                
                # CSV 파일 읽기
                df = pd.read_csv(file_path)
                sensor_data = df[['pitch', 'roll', 'yaw', 'flex1', 'flex2', 'flex3', 'flex4', 'flex5']].values
                
                # 전처리 적용
                processed_data = self._preprocess_data(sensor_data, class_name)
                
                # 라벨 변환
                label = self.label_mapper.get_label_id(class_name)
                
                # 혼동 클래스 여부 확인
                is_confusion = class_name in self.confusion_classes
                
                self.data.append(processed_data)
                self.labels.append(label)
                self.scenarios.append(scenario)
                self.is_confusion_class.append(is_confusion)
                
            except Exception as e:
                print(f"⚠️  파일 로드 실패: {file_path} - {e}")
    
    def _extract_scenario(self, file_path):
        """파일 경로에서 시나리오 추출"""
        parts = file_path.split(os.sep)
        for part in parts:
            if part.isdigit() and 1 <= int(part) <= 5:
                return int(part)
        return None
    
    def _preprocess_data(self, sensor_data, class_name):
        """데이터 전처리"""
        processed_data = sensor_data.copy()
        
        # 데이터 길이 통일 (300으로 패딩 또는 자르기)
        target_length = 300
        current_length = len(processed_data)
        
        if current_length < target_length:
            # 패딩
            padding_length = target_length - current_length
            padding = np.tile(processed_data[-1:], (padding_length, 1))
            processed_data = np.vstack([processed_data, padding])
        elif current_length > target_length:
            # 자르기
            processed_data = processed_data[:target_length]
        
        # 혼동 클래스에 대해서는 특화된 전처리 적용
        if class_name in self.confusion_classes:
            processed_data = self._specialized_preprocessing(processed_data, class_name)
        else:
            # 일반 클래스는 기본 전처리
            processed_data = self._general_preprocessing(processed_data)
        
        return processed_data
    
    def _specialized_preprocessing(self, sensor_data, class_name):
        """혼동 클래스 특화 전처리"""
        processed_data = sensor_data.copy()
        
        # ㅕ 클래스의 flex2 센서 고장 데이터 정제
        if class_name == 'ㅕ':
            flex2_values = sensor_data[:, 4]
            mask = (flex2_values >= 0) & (flex2_values <= 100)
            if np.any(mask):
                processed_data[mask, 4] = 750 + (flex2_values[mask] / 100) * 50
        
        # 센서별 가중치 적용
        processed_data[:, 0] *= 3.0  # pitch 가중치
        processed_data[:, 1] *= 3.0  # roll 가중치
        processed_data[:, 2] *= 1.5  # yaw 가중치
        processed_data[:, 3:8] *= 0.5  # flex 센서 가중치 감소
        
        return processed_data
    
    def _general_preprocessing(self, sensor_data):
        """일반 클래스 전처리"""
        # 기본 정규화
        processed_data = sensor_data.copy()
        
        # 표준화
        mean = np.mean(processed_data, axis=0)
        std = np.std(processed_data, axis=0)
        processed_data = (processed_data - mean) / (std + 1e-8)
        
        return processed_data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sensor_data = torch.FloatTensor(self.data[idx])
        label = torch.LongTensor([self.labels[idx]])[0]
        is_confusion = torch.BoolTensor([self.is_confusion_class[idx]])[0]
        return sensor_data, label, is_confusion

class EnsembleModel(nn.Module):
    """앙상블 모델 - 특화 모델 + 일반 모델"""
    
    def __init__(self, num_classes=24):
        super(EnsembleModel, self).__init__()
        
        # 일반 모델 (전체 클래스용)
        self.general_model = DeepLearningPipeline(
            input_features=8,
            sequence_length=300,
            num_classes=num_classes,
            hidden_dim=128,
            num_layers=3,
            dropout=0.3
        )
        
        # 특화 모델 (혼동 클래스용)
        self.specialized_model = DeepLearningPipeline(
            input_features=8,
            sequence_length=300,
            num_classes=num_classes,
            hidden_dim=128,
            num_layers=3,
            dropout=0.4
        )
        
        # 앙상블 가중치 학습
        self.ensemble_weights = nn.Parameter(torch.tensor([0.5, 0.5]))
        
        # 혼동 클래스 인덱스
        self.confusion_indices = [3, 9, 11, 15, 17]  # ㄹ, ㅊ, ㅌ, ㅑ, ㅕ
    
    def forward(self, x, is_confusion=None):
        """
        Args:
            x: 입력 데이터
            is_confusion: 혼동 클래스 여부 (배치별)
        """
        batch_size = x.size(0)
        
        # 일반 모델 예측
        general_output = self.general_model(x)
        general_logits = general_output['class_logits']
        
        # 특화 모델 예측
        specialized_output = self.specialized_model(x)
        specialized_logits = specialized_output['class_logits']
        
        # 앙상블 가중치 적용
        if is_confusion is not None:
            # 혼동 클래스는 특화 모델에 더 높은 가중치
            confusion_weight = torch.sigmoid(self.ensemble_weights[1]).item()
            general_weight = torch.sigmoid(self.ensemble_weights[0]).item()
            
            # 배치별 가중치 계산
            general_weights = torch.where(is_confusion, 
                                        torch.full_like(is_confusion, general_weight, dtype=torch.float),
                                        torch.full_like(is_confusion, confusion_weight, dtype=torch.float))
            specialized_weights = torch.where(is_confusion, 
                                            torch.full_like(is_confusion, confusion_weight, dtype=torch.float),
                                            torch.full_like(is_confusion, general_weight, dtype=torch.float))
            
            # 가중 평균
            ensemble_logits = (general_weights.unsqueeze(1) * general_logits + 
                             specialized_weights.unsqueeze(1) * specialized_logits)
        else:
            # 기본 앙상블 (동일 가중치)
            ensemble_logits = 0.5 * general_logits + 0.5 * specialized_logits
        
        return {
            'ensemble_logits': ensemble_logits,
            'general_logits': general_logits,
            'specialized_logits': specialized_logits,
            'ensemble_weights': self.ensemble_weights
        }

class EnsembleTrainer:
    """앙상블 모델 훈련기"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"🚀 앙상블 모델 훈련 시작")
        print(f"📱 디바이스: {self.device}")
        print(f"🎯 전체 24개 클래스 + 혼동 클래스 특화")
    
    def create_model(self):
        """앙상블 모델 생성"""
        model = EnsembleModel(num_classes=24).to(self.device)
        
        # 사전 훈련된 모델 로드 (있는 경우)
        if os.path.exists('best_problem_solver_model.pth'):
            print("📥 기존 모델 로드 중...")
            try:
                model.general_model.load_state_dict(torch.load('best_problem_solver_model.pth', map_location=self.device))
                print("✅ 일반 모델 로드 완료")
            except:
                print("⚠️  일반 모델 로드 실패")
        
        if os.path.exists('best_specialized_model.pth'):
            print("📥 특화 모델 로드 중...")
            try:
                model.specialized_model.load_state_dict(torch.load('best_specialized_model.pth', map_location=self.device))
                print("✅ 특화 모델 로드 완료")
            except:
                print("⚠️  특화 모델 로드 실패")
        
        return model
    
    def create_optimizer(self, model):
        """옵티마이저 생성"""
        # 앙상블 가중치만 학습
        optimizer = optim.AdamW([
            {'params': model.ensemble_weights, 'lr': self.config['learning_rate']},
            {'params': model.general_model.parameters(), 'lr': self.config['learning_rate'] * 0.1},
            {'params': model.specialized_model.parameters(), 'lr': self.config['learning_rate'] * 0.1}
        ], weight_decay=self.config['weight_decay'])
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )
        
        return optimizer, scheduler
    
    def train(self, train_loader, val_loader, model, optimizer, scheduler, criterion):
        """훈련 실행"""
        print(f"\n🎯 앙상블 모델 훈련 시작")
        print(f"📊 에포크: {self.config['epochs']}")
        print(f"📦 배치 크기: {self.config['batch_size']}")
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        train_losses = []
        val_losses = []
        
        for epoch in range(self.config['epochs']):
            # 훈련
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_idx, (data, targets, is_confusion) in enumerate(train_loader):
                data, targets = data.to(self.device), targets.to(self.device)
                is_confusion = is_confusion.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(data, is_confusion)
                loss = criterion(outputs['ensemble_logits'], targets)
                loss.backward()
                
                # 그래디언트 클리핑
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs['ensemble_logits'].max(1)
                train_total += targets.size(0)
                train_correct += predicted.eq(targets).sum().item()
            
            # 검증
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for data, targets, is_confusion in val_loader:
                    data, targets = data.to(self.device), targets.to(self.device)
                    is_confusion = is_confusion.to(self.device)
                    outputs = model(data, is_confusion)
                    loss = criterion(outputs['ensemble_logits'], targets)
                    
                    val_loss += loss.item()
                    _, predicted = outputs['ensemble_logits'].max(1)
                    val_total += targets.size(0)
                    val_correct += predicted.eq(targets).sum().item()
            
            # 평균 손실 계산
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            train_acc = 100. * train_correct / train_total
            val_acc = 100. * val_correct / val_total
            
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            
            # 앙상블 가중치 출력
            weights = torch.sigmoid(model.ensemble_weights)
            
            print(f"Epoch {epoch+1}/{self.config['epochs']}:")
            print(f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            print(f"  Ensemble Weights: General={weights[0]:.3f}, Specialized={weights[1]:.3f}")
            
            # 스케줄러 업데이트
            scheduler.step(avg_val_loss)
            
            # 조기 종료
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                torch.save(model.state_dict(), 'best_ensemble_model.pth')
                print(f"  ✅ 새로운 최고 모델 저장")
            else:
                patience_counter += 1
                if patience_counter >= self.config['early_stopping_patience']:
                    print(f"  🛑 조기 종료 (에포크 {epoch+1})")
                    break
        
        # 훈련 곡선 시각화
        self._plot_training_curves(train_losses, val_losses)
        
        return model
    
    def _plot_training_curves(self, train_losses, val_losses):
        """훈련 곡선 시각화"""
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Train Loss', color='blue')
        plt.plot(val_losses, label='Val Loss', color='red')
        plt.title('Ensemble Model Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('ensemble_training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ 훈련 곡선 저장: ensemble_training_curves.png")
    
    def evaluate(self, test_loader, model):
        """모델 평가"""
        print(f"\n📊 앙상블 모델 평가")
        
        model.eval()
        all_predictions = []
        all_targets = []
        all_is_confusion = []
        
        with torch.no_grad():
            for data, targets, is_confusion in test_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                is_confusion = is_confusion.to(self.device)
                outputs = model(data, is_confusion)
                _, predicted = outputs['ensemble_logits'].max(1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_is_confusion.extend(is_confusion.cpu().numpy())
        
        # 혼동 행렬 생성
        self._create_confusion_matrix(all_targets, all_predictions)
        
        # 클래스별 성능 분석
        self._analyze_class_performance(all_targets, all_predictions, all_is_confusion)
    
    def _create_confusion_matrix(self, targets, predictions):
        """혼동 행렬 생성"""
        from sklearn.metrics import confusion_matrix, classification_report
        
        # 혼동 행렬
        cm = confusion_matrix(targets, predictions)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Ensemble Model Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig('ensemble_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 분류 보고서
        report = classification_report(targets, predictions, output_dict=True)
        
        with open('ensemble_classification_report.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print("✅ 혼동 행렬 저장: ensemble_confusion_matrix.png")
        print("✅ 분류 보고서 저장: ensemble_classification_report.json")
    
    def _analyze_class_performance(self, targets, predictions, is_confusion):
        """클래스별 성능 분석"""
        from sklearn.metrics import accuracy_score
        
        label_mapper = KSLLabelMapper()
        
        # 혼동 클래스별 성능
        confusion_classes = ['ㅊ', 'ㅑ', 'ㅕ', 'ㅌ', 'ㄹ']
        
        print(f"\n🎯 혼동 클래스별 성능:")
        
        for class_name in confusion_classes:
            class_idx = label_mapper.get_label_id(class_name)
            
            # 해당 클래스의 데이터만 추출
            mask = np.array(targets) == class_idx
            if np.any(mask):
                class_targets = np.array(targets)[mask]
                class_predictions = np.array(predictions)[mask]
                
                accuracy = accuracy_score(class_targets, class_predictions)
                print(f"  {class_name}: {accuracy:.3f} ({accuracy*100:.1f}%)")
        
        # 일반 클래스 vs 혼동 클래스 성능 비교
        confusion_mask = np.array(is_confusion)
        general_mask = ~confusion_mask
        
        if np.any(confusion_mask):
            confusion_acc = accuracy_score(np.array(targets)[confusion_mask], 
                                         np.array(predictions)[confusion_mask])
            print(f"\n📊 혼동 클래스 평균 정확도: {confusion_acc:.3f} ({confusion_acc*100:.1f}%)")
        
        if np.any(general_mask):
            general_acc = accuracy_score(np.array(targets)[general_mask], 
                                       np.array(predictions)[general_mask])
            print(f"📊 일반 클래스 평균 정확도: {general_acc:.3f} ({general_acc*100:.1f}%)")

def main():
    """메인 함수"""
    # 설정
    config = {
        'learning_rate': 0.0001,
        'batch_size': 32,
        'epochs': 50,
        'weight_decay': 1e-4,
        'early_stopping_patience': 10
    }
    
    # 데이터 로딩
    print("📊 데이터 로딩 중...")
    
    train_dataset = EnsembleDataset('integrations/SignGlove_HW/github_unified_data', 'train')
    val_dataset = EnsembleDataset('integrations/SignGlove_HW/github_unified_data', 'val')
    test_dataset = EnsembleDataset('integrations/SignGlove_HW/github_unified_data', 'test')
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
    
    # 모델 생성
    trainer = EnsembleTrainer(config)
    model = trainer.create_model()
    
    # 옵티마이저 및 손실 함수
    optimizer, scheduler = trainer.create_optimizer(model)
    criterion = nn.CrossEntropyLoss()
    
    # 훈련
    model = trainer.train(train_loader, val_loader, model, optimizer, scheduler, criterion)
    
    # 평가
    trainer.evaluate(test_loader, model)
    
    print("\n🎉 앙상블 모델 훈련 완료!")

if __name__ == "__main__":
    main()
