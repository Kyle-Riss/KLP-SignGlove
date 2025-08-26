#!/usr/bin/env python3
"""
특화된 모델 훈련 스크립트
ㅊ-ㅑ, ㅕ-ㅌ-ㄹ 클래스 혼동 문제 해결
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

class SpecializedDataset(Dataset):
    """특화된 데이터셋 - 혼동 클래스들에 대한 특별 처리"""
    
    def __init__(self, data_dir, mode='train', target_classes=None):
        self.data_dir = data_dir
        self.mode = mode
        self.label_mapper = KSLLabelMapper()
        
        # 혼동 클래스 그룹 정의
        self.confusion_groups = {
            'cha_ya': ['ㅊ', 'ㅑ'],      # 코사인 유사도 0.998
            'yeo_teul_rieul': ['ㅕ', 'ㅌ', 'ㄹ']  # 코사인 유사도 0.997-1.000
        }
        
        # 특화 가중치 정의
        self.sensor_weights = {
            'pitch_roll_weight': 5.0,    # pitch/roll 가중치 증가
            'flex_weight': 0.3,          # flex 센서 가중치 감소
            'yaw_weight': 2.0            # yaw 가중치 증가
        }
        
        # 데이터 로딩
        self.data = []
        self.labels = []
        self.scenarios = []
        
        if target_classes is None:
            target_classes = ['ㅊ', 'ㅑ', 'ㅕ', 'ㅌ', 'ㄹ']
        
        print(f"🔍 특화 모델 데이터 로딩: {target_classes}")
        
        for class_name in target_classes:
            self._load_class_data(class_name)
        
        print(f"📊 로딩 완료: {len(self.data)}개 샘플")
    
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
                
                # 특화된 전처리 적용
                processed_data = self._specialized_preprocessing(sensor_data, class_name)
                
                # 라벨 변환
                label = self.label_mapper.get_label_id(class_name)
                
                self.data.append(processed_data)
                self.labels.append(label)
                self.scenarios.append(scenario)
                
            except Exception as e:
                print(f"⚠️  파일 로드 실패: {file_path} - {e}")
    
    def _extract_scenario(self, file_path):
        """파일 경로에서 시나리오 추출"""
        parts = file_path.split(os.sep)
        for part in parts:
            if part.isdigit() and 1 <= int(part) <= 5:
                return int(part)
        return None
    
    def _specialized_preprocessing(self, sensor_data, class_name):
        """특화된 전처리"""
        processed_data = sensor_data.copy()
        
        # ㅕ 클래스의 flex2 센서 고장 데이터 정제
        if class_name == 'ㅕ':
            processed_data = self._clean_yeo_flex2_data(processed_data)
        
        # 센서별 가중치 적용
        processed_data = self._apply_sensor_weights(processed_data, class_name)
        
        # 클래스별 특화 정규화
        processed_data = self._class_specific_normalization(processed_data, class_name)
        
        return processed_data
    
    def _clean_yeo_flex2_data(self, sensor_data):
        """ㅕ 클래스의 flex2 센서 데이터 정제"""
        # flex2 > 100 값들을 제거하고 정규화
        flex2_values = sensor_data[:, 4]
        
        # flex2가 0-100 범위인 경우 (센서 고장) 정규화
        mask = (flex2_values >= 0) & (flex2_values <= 100)
        if np.any(mask):
            # 정상 범위로 변환 (750-800)
            sensor_data[mask, 4] = 750 + (flex2_values[mask] / 100) * 50
        
        return sensor_data
    
    def _apply_sensor_weights(self, sensor_data, class_name):
        """센서별 가중치 적용"""
        weighted_data = sensor_data.copy()
        
        # pitch, roll 가중치 증가 (혼동 클래스 구분에 중요)
        weighted_data[:, 0] *= self.sensor_weights['pitch_roll_weight']  # pitch
        weighted_data[:, 1] *= self.sensor_weights['pitch_roll_weight']  # roll
        
        # yaw 가중치 증가
        weighted_data[:, 2] *= self.sensor_weights['yaw_weight']  # yaw
        
        # flex 센서 가중치 감소
        weighted_data[:, 3:8] *= self.sensor_weights['flex_weight']  # flex1-5
        
        return weighted_data
    
    def _class_specific_normalization(self, sensor_data, class_name):
        """클래스별 특화 정규화"""
        normalized_data = sensor_data.copy()
        
        # ㅊ, ㅑ 클래스: pitch/roll 차이 강조
        if class_name in ['ㅊ', 'ㅑ']:
            # pitch 차이 강조 (ㅊ: 25.993°, ㅑ: 18.193°)
            if class_name == 'ㅊ':
                normalized_data[:, 0] = (normalized_data[:, 0] - 18.193) * 2.0  # ㅑ 기준으로 차이 강조
            else:  # ㅑ
                normalized_data[:, 0] = (normalized_data[:, 0] - 25.993) * 2.0  # ㅊ 기준으로 차이 강조
            
            # roll 차이 강조 (ㅊ: -38.625°, ㅑ: 53.565°)
            if class_name == 'ㅊ':
                normalized_data[:, 1] = (normalized_data[:, 1] - 53.565) * 2.0
            else:  # ㅑ
                normalized_data[:, 1] = (normalized_data[:, 1] - (-38.625)) * 2.0
        
        # ㅕ, ㅌ, ㄹ 클래스: pitch/roll 차이 강조
        elif class_name in ['ㅕ', 'ㅌ', 'ㄹ']:
            # pitch 차이 강조 (ㅕ: 57.278°, ㅌ: 67.103°, ㄹ: 58.028°)
            if class_name == 'ㅕ':
                normalized_data[:, 0] = (normalized_data[:, 0] - 67.103) * 3.0  # ㅌ 기준
            elif class_name == 'ㅌ':
                normalized_data[:, 0] = (normalized_data[:, 0] - 57.278) * 3.0  # ㅕ 기준
            else:  # ㄹ
                normalized_data[:, 0] = (normalized_data[:, 0] - 57.278) * 3.0  # ㅕ 기준
            
            # roll 차이 강조 (ㅕ: -71.097°, ㅌ: -93.691°, ㄹ: -68.003°)
            if class_name == 'ㅕ':
                normalized_data[:, 1] = (normalized_data[:, 1] - (-93.691)) * 3.0
            elif class_name == 'ㅌ':
                normalized_data[:, 1] = (normalized_data[:, 1] - (-71.097)) * 3.0
            else:  # ㄹ
                normalized_data[:, 1] = (normalized_data[:, 1] - (-71.097)) * 3.0
        
        return normalized_data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sensor_data = torch.FloatTensor(self.data[idx])
        label = torch.LongTensor([self.labels[idx]])[0]
        return sensor_data, label

class SpecializedLoss(nn.Module):
    """특화된 손실 함수 - 혼동 클래스 간 차이 강조"""
    
    def __init__(self, confusion_groups, alpha=0.3):
        super(SpecializedLoss, self).__init__()
        self.confusion_groups = confusion_groups
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, outputs, targets, features=None):
        # 기본 CrossEntropy 손실
        ce_loss = self.ce_loss(outputs, targets)
        
        # 혼동 클래스 간 차이 강조 손실
        confusion_loss = self._confusion_penalty(outputs, targets)
        
        # 최종 손실
        total_loss = ce_loss + self.alpha * confusion_loss
        
        return total_loss
    
    def _confusion_penalty(self, outputs, targets):
        """혼동 클래스 간 페널티"""
        penalty = 0.0
        
        # ㅊ-ㅑ 그룹
        cha_idx = 9   # ㅊ 인덱스
        ya_idx = 15   # ㅑ 인덱스
        
        cha_mask = (targets == cha_idx)
        ya_mask = (targets == ya_idx)
        
        if torch.any(cha_mask) and torch.any(ya_mask):
            cha_outputs = outputs[cha_mask]
            ya_outputs = outputs[ya_mask]
            
            # 배치 크기가 다를 수 있으므로 평균으로 정규화
            cha_mean = torch.mean(cha_outputs, dim=0, keepdim=True)
            ya_mean = torch.mean(ya_outputs, dim=0, keepdim=True)
            
            # ㅊ-ㅑ 출력 차이를 최대화
            cha_ya_diff = torch.mean(torch.abs(cha_mean - ya_mean))
            penalty += torch.exp(-cha_ya_diff)  # 차이가 작을수록 페널티 증가
        
        # ㅕ-ㅌ-ㄹ 그룹
        yeo_idx = 17  # ㅕ 인덱스
        teul_idx = 11  # ㅌ 인덱스
        rieul_idx = 3  # ㄹ 인덱스
        
        yeo_mask = (targets == yeo_idx)
        teul_mask = (targets == teul_idx)
        rieul_mask = (targets == rieul_idx)
        
        if torch.any(yeo_mask) and torch.any(teul_mask):
            yeo_outputs = outputs[yeo_mask]
            teul_outputs = outputs[teul_mask]
            yeo_mean = torch.mean(yeo_outputs, dim=0, keepdim=True)
            teul_mean = torch.mean(teul_outputs, dim=0, keepdim=True)
            yeo_teul_diff = torch.mean(torch.abs(yeo_mean - teul_mean))
            penalty += torch.exp(-yeo_teul_diff)
        
        if torch.any(yeo_mask) and torch.any(rieul_mask):
            yeo_outputs = outputs[yeo_mask]
            rieul_outputs = outputs[rieul_mask]
            yeo_mean = torch.mean(yeo_outputs, dim=0, keepdim=True)
            rieul_mean = torch.mean(rieul_outputs, dim=0, keepdim=True)
            yeo_rieul_diff = torch.mean(torch.abs(yeo_mean - rieul_mean))
            penalty += torch.exp(-yeo_rieul_diff)
        
        if torch.any(teul_mask) and torch.any(rieul_mask):
            teul_outputs = outputs[teul_mask]
            rieul_outputs = outputs[rieul_mask]
            teul_mean = torch.mean(teul_outputs, dim=0, keepdim=True)
            rieul_mean = torch.mean(rieul_outputs, dim=0, keepdim=True)
            teul_rieul_diff = torch.mean(torch.abs(teul_mean - rieul_mean))
            penalty += torch.exp(-teul_rieul_diff)
        
        return penalty

class SpecializedTrainer:
    """특화된 모델 훈련기"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"🚀 특화 모델 훈련 시작")
        print(f"📱 디바이스: {self.device}")
        print(f"🎯 혼동 클래스: ㅊ-ㅑ, ㅕ-ㅌ-ㄹ")
    
    def create_model(self):
        """특화된 모델 생성"""
        model = DeepLearningPipeline(
            input_features=8,
            sequence_length=300,
            num_classes=24,
            hidden_dim=128,  # 더 큰 히든 차원
            num_layers=3,    # 더 깊은 네트워크
            dropout=0.4      # 더 강한 드롭아웃
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
        print(f"\n🎯 특화 모델 훈련 시작")
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
            
            for batch_idx, (data, targets) in enumerate(train_loader):
                data, targets = data.to(self.device), targets.to(self.device)
                
                optimizer.zero_grad()
                model_output = model(data)
                outputs = model_output['class_logits']
                loss = criterion(outputs, targets)
                loss.backward()
                
                # 그래디언트 클리핑
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
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
                    model_output = model(data)
                    outputs = model_output['class_logits']
                    loss = criterion(outputs, targets)
                    
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += targets.size(0)
                    val_correct += predicted.eq(targets).sum().item()
            
            # 평균 손실 계산
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            train_acc = 100. * train_correct / train_total
            val_acc = 100. * val_correct / val_total
            
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            
            print(f"Epoch {epoch+1}/{self.config['epochs']}:")
            print(f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # 스케줄러 업데이트
            scheduler.step(avg_val_loss)
            
            # 조기 종료
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                torch.save(model.state_dict(), 'best_specialized_model.pth')
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
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('specialized_training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ 훈련 곡선 저장: specialized_training_curves.png")
    
    def evaluate(self, test_loader, model):
        """모델 평가"""
        print(f"\n📊 특화 모델 평가")
        
        model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for data, targets in test_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                model_output = model(data)
                outputs = model_output['class_logits']
                _, predicted = outputs.max(1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        # 혼동 행렬 생성
        self._create_confusion_matrix(all_targets, all_predictions)
        
        # 클래스별 성능 분석
        self._analyze_class_performance(all_targets, all_predictions)
    
    def _create_confusion_matrix(self, targets, predictions):
        """혼동 행렬 생성"""
        from sklearn.metrics import confusion_matrix, classification_report
        
        # 혼동 행렬
        cm = confusion_matrix(targets, predictions)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Specialized Model Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig('specialized_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 분류 보고서
        report = classification_report(targets, predictions, output_dict=True)
        
        with open('specialized_classification_report.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print("✅ 혼동 행렬 저장: specialized_confusion_matrix.png")
        print("✅ 분류 보고서 저장: specialized_classification_report.json")
    
    def _analyze_class_performance(self, targets, predictions):
        """클래스별 성능 분석"""
        from sklearn.metrics import accuracy_score
        
        # 라벨 매퍼 생성
        label_mapper = KSLLabelMapper()
        
        # 혼동 클래스별 성능
        confusion_classes = {
            'cha_ya': ['ㅊ', 'ㅑ'],
            'yeo_teul_rieul': ['ㅕ', 'ㅌ', 'ㄹ']
        }
        
        print(f"\n🎯 혼동 클래스별 성능:")
        
        for group_name, classes in confusion_classes.items():
            print(f"\n📊 {group_name} 그룹:")
            
            for class_name in classes:
                class_idx = label_mapper.get_label_id(class_name)
                
                # 해당 클래스의 데이터만 추출
                mask = np.array(targets) == class_idx
                if np.any(mask):
                    class_targets = np.array(targets)[mask]
                    class_predictions = np.array(predictions)[mask]
                    
                    accuracy = accuracy_score(class_targets, class_predictions)
                    print(f"  {class_name}: {accuracy:.3f} ({accuracy*100:.1f}%)")

def main():
    """메인 함수"""
    # 설정
    config = {
        'learning_rate': 0.0001,
        'batch_size': 32,
        'epochs': 100,
        'weight_decay': 1e-4,
        'early_stopping_patience': 15
    }
    
    # 데이터 로딩
    print("📊 데이터 로딩 중...")
    
    # 혼동 클래스들만 포함
    target_classes = ['ㅊ', 'ㅑ', 'ㅕ', 'ㅌ', 'ㄹ']
    
    train_dataset = SpecializedDataset('integrations/SignGlove_HW/github_unified_data', 'train', target_classes)
    val_dataset = SpecializedDataset('integrations/SignGlove_HW/github_unified_data', 'val', target_classes)
    test_dataset = SpecializedDataset('integrations/SignGlove_HW/github_unified_data', 'test', target_classes)
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
    
    # 모델 생성
    trainer = SpecializedTrainer(config)
    model = trainer.create_model()
    
    # 옵티마이저 및 손실 함수
    optimizer, scheduler = trainer.create_optimizer(model)
    criterion = SpecializedLoss(confusion_groups={
        'cha_ya': ['ㅊ', 'ㅑ'],
        'yeo_teul_rieul': ['ㅕ', 'ㅌ', 'ㄹ']
    })
    
    # 훈련
    model = trainer.train(train_loader, val_loader, model, optimizer, scheduler, criterion)
    
    # 평가
    trainer.evaluate(test_loader, model)
    
    print("\n🎉 특화 모델 훈련 완료!")

if __name__ == "__main__":
    main()
