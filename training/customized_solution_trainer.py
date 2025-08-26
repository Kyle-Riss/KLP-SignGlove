#!/usr/bin/env python3
"""
맞춤형 솔루션 훈련 스크립트
24개 자음/모음 클래스별 특성에 맞는 맞춤형 접근법
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

class CustomizedDataset(Dataset):
    """맞춤형 데이터셋 - 클래스별 특화 전처리"""
    
    def __init__(self, data_dir, mode='train'):
        self.data_dir = data_dir
        self.mode = mode
        self.label_mapper = KSLLabelMapper()
        
        # 분석 결과 기반 클래스 그룹화
        self.class_groups = self._define_class_groups()
        
        # 데이터 로딩
        self.data = []
        self.labels = []
        self.class_groups_idx = []
        
        print(f"🔍 맞춤형 데이터셋 로딩: {mode} 모드")
        
        for class_name in self.label_mapper.get_all_classes():
            self._load_class_data(class_name)
        
        print(f"📊 로딩 완료: {len(self.data)}개 샘플")
    
    def _define_class_groups(self):
        """분석 결과 기반 클래스 그룹 정의"""
        # 높은 유사도 클래스 그룹 (분석 결과 기반)
        high_similarity_groups = [
            ['ㄱ', 'ㄴ', 'ㄷ', 'ㄹ'],  # 자음 그룹 1
            ['ㅁ', 'ㅂ', 'ㅅ', 'ㅇ'],  # 자음 그룹 2
            ['ㅈ', 'ㅊ', 'ㅋ', 'ㅌ'],  # 자음 그룹 3
            ['ㅍ', 'ㅎ'],              # 자음 그룹 4
            ['ㅏ', 'ㅑ', 'ㅓ', 'ㅕ'],  # 모음 그룹 1
            ['ㅗ', 'ㅛ', 'ㅜ', 'ㅠ'],  # 모음 그룹 2
            ['ㅡ', 'ㅣ']               # 모음 그룹 3
        ]
        
        # 클래스별 특성 기반 그룹
        class_groups = {
            'high_similarity': high_similarity_groups,
            'pitch_dominant': ['ㅎ', 'ㅗ', 'ㅛ', 'ㅡ'],  # pitch 값이 높은 클래스들
            'roll_dominant': ['ㅂ', 'ㅍ', 'ㅏ', 'ㅑ'],   # roll 값이 높은 클래스들
            'flex_sensitive': ['ㄷ', 'ㄹ', 'ㅌ', 'ㅕ'],  # flex 센서 변화가 큰 클래스들
            'stable_pattern': ['ㄱ', 'ㅅ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ']  # 안정적인 패턴 클래스들
        }
        
        return class_groups
    
    def _load_class_data(self, class_name):
        """클래스별 데이터 로딩"""
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
                
                # 클래스별 맞춤형 전처리 적용
                processed_data = self._customized_preprocessing(sensor_data, class_name)
                
                # 라벨 변환
                label = self.label_mapper.get_label_id(class_name)
                
                # 클래스 그룹 인덱스 결정
                group_idx = self._get_class_group_index(class_name)
                
                self.data.append(processed_data)
                self.labels.append(label)
                self.class_groups_idx.append(group_idx)
                
            except Exception as e:
                print(f"  ⚠️  파일 로드 실패: {file_path} - {e}")
    
    def _get_class_group_index(self, class_name):
        """클래스 그룹 인덱스 결정"""
        # 높은 유사도 그룹에서 찾기
        for i, group in enumerate(self.class_groups['high_similarity']):
            if class_name in group:
                return i
        
        # 특성 기반 그룹에서 찾기
        if class_name in self.class_groups['pitch_dominant']:
            return 10
        elif class_name in self.class_groups['roll_dominant']:
            return 11
        elif class_name in self.class_groups['flex_sensitive']:
            return 12
        elif class_name in self.class_groups['stable_pattern']:
            return 13
        
        return 14  # 기본 그룹
    
    def _customized_preprocessing(self, sensor_data, class_name):
        """클래스별 맞춤형 전처리"""
        processed_data = sensor_data.copy()
        
        # 데이터 길이 통일
        target_length = 300
        current_length = len(processed_data)
        
        if current_length < target_length:
            padding_length = target_length - current_length
            padding = np.tile(processed_data[-1:], (padding_length, 1))
            processed_data = np.vstack([processed_data, padding])
        elif current_length > target_length:
            processed_data = processed_data[:target_length]
        
        # 클래스별 특화 전처리
        if class_name in self.class_groups['pitch_dominant']:
            processed_data = self._pitch_dominant_preprocessing(processed_data)
        elif class_name in self.class_groups['roll_dominant']:
            processed_data = self._roll_dominant_preprocessing(processed_data)
        elif class_name in self.class_groups['flex_sensitive']:
            processed_data = self._flex_sensitive_preprocessing(processed_data)
        elif class_name in self.class_groups['stable_pattern']:
            processed_data = self._stable_pattern_preprocessing(processed_data)
        else:
            processed_data = self._general_preprocessing(processed_data)
        
        return processed_data
    
    def _pitch_dominant_preprocessing(self, sensor_data):
        """pitch 중심 전처리"""
        processed_data = sensor_data.copy()
        
        # pitch 센서 강화
        processed_data[:, 0] *= 2.0  # pitch 가중치 증가
        
        # 다른 센서들 정규화
        for i in range(1, 8):
            mean_val = np.mean(processed_data[:, i])
            std_val = np.std(processed_data[:, i])
            processed_data[:, i] = (processed_data[:, i] - mean_val) / (std_val + 1e-8)
        
        return processed_data
    
    def _roll_dominant_preprocessing(self, sensor_data):
        """roll 중심 전처리"""
        processed_data = sensor_data.copy()
        
        # roll 센서 강화
        processed_data[:, 1] *= 2.0  # roll 가중치 증가
        
        # 다른 센서들 정규화
        for i in [0, 2, 3, 4, 5, 6, 7]:  # pitch, yaw, flex 센서들
            mean_val = np.mean(processed_data[:, i])
            std_val = np.std(processed_data[:, i])
            processed_data[:, i] = (processed_data[:, i] - mean_val) / (std_val + 1e-8)
        
        return processed_data
    
    def _flex_sensitive_preprocessing(self, sensor_data):
        """flex 센서 민감 전처리"""
        processed_data = sensor_data.copy()
        
        # flex 센서들 강화
        processed_data[:, 3:8] *= 1.5  # flex1-5 가중치 증가
        
        # IMU 센서들 정규화
        for i in range(3):  # pitch, roll, yaw
            mean_val = np.mean(processed_data[:, i])
            std_val = np.std(processed_data[:, i])
            processed_data[:, i] = (processed_data[:, i] - mean_val) / (std_val + 1e-8)
        
        return processed_data
    
    def _stable_pattern_preprocessing(self, sensor_data):
        """안정적 패턴 전처리"""
        processed_data = sensor_data.copy()
        
        # 모든 센서 균등하게 처리
        for i in range(8):
            mean_val = np.mean(processed_data[:, i])
            std_val = np.std(processed_data[:, i])
            processed_data[:, i] = (processed_data[:, i] - mean_val) / (std_val + 1e-8)
        
        # 노이즈 감소
        processed_data += np.random.normal(0, 0.01, processed_data.shape)
        
        return processed_data
    
    def _general_preprocessing(self, sensor_data):
        """일반 전처리"""
        processed_data = sensor_data.copy()
        
        # 표준화
        for i in range(8):
            mean_val = np.mean(processed_data[:, i])
            std_val = np.std(processed_data[:, i])
            processed_data[:, i] = (processed_data[:, i] - mean_val) / (std_val + 1e-8)
        
        return processed_data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sensor_data = torch.FloatTensor(self.data[idx])
        label = torch.LongTensor([self.labels[idx]])[0]
        group_idx = torch.LongTensor([self.class_groups_idx[idx]])[0]
        return sensor_data, label, group_idx

class AttentionEnhancedModel(nn.Module):
    """어텐션 메커니즘 강화 모델"""
    
    def __init__(self, num_classes=24, num_groups=15):
        super(AttentionEnhancedModel, self).__init__()
        
        # 기본 CNN-LSTM 파이프라인
        self.base_model = DeepLearningPipeline(
            input_features=8,
            sequence_length=300,
            num_classes=num_classes,
            hidden_dim=128,
            num_layers=3,
            dropout=0.3
        )
        
        # 클래스 그룹별 어텐션 가중치
        self.group_attention_weights = nn.Parameter(torch.ones(num_groups, 8))
        
        # 클래스별 어텐션 메커니즘
        self.class_attention = nn.MultiheadAttention(
            embed_dim=64,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # 클래스별 특화 분류기
        self.class_specific_classifiers = nn.ModuleDict({
            'pitch_dominant': nn.Linear(64, num_classes),
            'roll_dominant': nn.Linear(64, num_classes),
            'flex_sensitive': nn.Linear(64, num_classes),
            'stable_pattern': nn.Linear(64, num_classes),
            'general': nn.Linear(64, num_classes)
        })
        
        # 최종 앙상블 가중치
        self.ensemble_weights = nn.Parameter(torch.ones(5))
    
    def forward(self, x, group_idx=None):
        batch_size = x.size(0)
        
        # 그룹별 센서 가중치 적용
        if group_idx is not None:
            attention_weights = self.group_attention_weights[group_idx]  # [batch_size, 8]
            x = x * attention_weights.unsqueeze(1)  # [batch_size, seq_len, 8]
        
        # 기본 모델 통과
        base_output = self.base_model(x)
        features = base_output['features']  # [batch_size, 64]
        
        # 특징을 시퀀스 형태로 변환 (어텐션을 위해)
        features = features.unsqueeze(1)  # [batch_size, 1, 64]
        
        # 어텐션 메커니즘 적용
        attended_features, _ = self.class_attention(features, features, features)
        
        # 평균 풀링
        pooled_features = torch.mean(attended_features, dim=1)  # [batch_size, hidden_dim]
        
        # 클래스별 특화 분류기 적용
        classifier_outputs = {}
        for name, classifier in self.class_specific_classifiers.items():
            classifier_outputs[name] = classifier(pooled_features)
        
        # 앙상블 가중치 적용
        ensemble_weights = torch.softmax(self.ensemble_weights, dim=0)
        
        # 가중 평균으로 최종 예측
        final_logits = torch.zeros_like(classifier_outputs['general'])
        for i, (name, output) in enumerate(classifier_outputs.items()):
            final_logits += ensemble_weights[i] * output
        
        return {
            'class_logits': final_logits,
            'base_logits': base_output['class_logits'],
            'ensemble_weights': ensemble_weights,
            'classifier_outputs': classifier_outputs
        }

class CustomizedLoss(nn.Module):
    """맞춤형 손실 함수"""
    
    def __init__(self, num_classes=24):
        super(CustomizedLoss, self).__init__()
        self.num_classes = num_classes
        
        # 기본 손실 함수
        self.ce_loss = nn.CrossEntropyLoss()
        
        # 클래스별 가중치 (분석 결과 기반)
        self.class_weights = self._create_class_weights()
        
        # 유사도 페널티
        self.similarity_penalty_weight = 0.1
    
    def _create_class_weights(self):
        """분석 결과 기반 클래스 가중치 생성"""
        # 높은 유사도 클래스들에 더 높은 가중치
        high_similarity_classes = ['ㄱ', 'ㄴ', 'ㄷ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅅ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ']
        weights = torch.ones(self.num_classes)
        
        for class_name in high_similarity_classes:
            class_idx = KSLLabelMapper().get_label_id(class_name)
            weights[class_idx] = 1.5  # 높은 유사도 클래스 가중치 증가
        
        return weights
    
    def forward(self, outputs, targets, group_idx=None):
        """맞춤형 손실 계산"""
        # 기본 분류 손실
        ce_loss = self.ce_loss(outputs['class_logits'], targets)
        
        # 가중치 적용
        weighted_loss = ce_loss * self.class_weights[targets].mean()
        
        # 유사도 페널티 (높은 유사도 클래스들 간의 구분 강화)
        similarity_penalty = self._calculate_similarity_penalty(outputs['class_logits'], targets)
        
        total_loss = weighted_loss + self.similarity_penalty_weight * similarity_penalty
        
        return total_loss
    
    def _calculate_similarity_penalty(self, logits, targets):
        """유사도 페널티 계산"""
        # 높은 유사도 클래스 그룹들
        high_similarity_groups = [
            [0, 1, 2, 3],   # ㄱ, ㄴ, ㄷ, ㄹ
            [4, 5, 6, 7],   # ㅁ, ㅂ, ㅅ, ㅇ
            [8, 9, 10, 11], # ㅈ, ㅊ, ㅋ, ㅌ
        ]
        
        penalty = 0.0
        
        for group in high_similarity_groups:
            # 해당 그룹의 클래스들에 대한 예측 확률
            group_probs = torch.softmax(logits[:, group], dim=1)
            
            # 엔트로피 최소화 (확실한 예측 유도)
            entropy = -torch.sum(group_probs * torch.log(group_probs + 1e-8), dim=1)
            penalty += torch.mean(entropy)
        
        return penalty

class CustomizedTrainer:
    """맞춤형 훈련기"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"🚀 맞춤형 솔루션 훈련 시작")
        print(f"📱 디바이스: {self.device}")
        print(f"🎯 클래스별 특화 접근법 적용")
    
    def create_model(self):
        """맞춤형 모델 생성"""
        model = AttentionEnhancedModel(num_classes=24, num_groups=15).to(self.device)
        return model
    
    def create_optimizer(self, model):
        """옵티마이저 생성"""
        optimizer = optim.AdamW([
            {'params': model.base_model.parameters(), 'lr': self.config['learning_rate']},
            {'params': model.group_attention_weights, 'lr': self.config['learning_rate'] * 2.0},
            {'params': model.class_attention.parameters(), 'lr': self.config['learning_rate']},
            {'params': model.class_specific_classifiers.parameters(), 'lr': self.config['learning_rate']},
            {'params': model.ensemble_weights, 'lr': self.config['learning_rate'] * 3.0}
        ], weight_decay=self.config['weight_decay'])
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        
        return optimizer, scheduler
    
    def train(self, train_loader, val_loader, model, optimizer, scheduler, criterion):
        """훈련 실행"""
        print(f"\n🎯 맞춤형 모델 훈련 시작")
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
            
            for batch_idx, (data, targets, group_idx) in enumerate(train_loader):
                data, targets = data.to(self.device), targets.to(self.device)
                group_idx = group_idx.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(data, group_idx)
                loss = criterion(outputs, targets, group_idx)
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
                for data, targets, group_idx in val_loader:
                    data, targets = data.to(self.device), targets.to(self.device)
                    group_idx = group_idx.to(self.device)
                    outputs = model(data, group_idx)
                    loss = criterion(outputs, targets, group_idx)
                    
                    val_loss += loss.item()
                    _, predicted = outputs['class_logits'].max(1)
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
            ensemble_weights = outputs['ensemble_weights'].detach().cpu().numpy()
            
            print(f"Epoch {epoch+1}/{self.config['epochs']}:")
            print(f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            print(f"  Ensemble Weights: {ensemble_weights}")
            
            # 스케줄러 업데이트
            scheduler.step(avg_val_loss)
            
            # 조기 종료
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                torch.save(model.state_dict(), 'best_customized_model.pth')
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
        plt.title('Customized Model Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('customized_training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ 훈련 곡선 저장: customized_training_curves.png")
    
    def evaluate(self, test_loader, model):
        """모델 평가"""
        print(f"\n📊 맞춤형 모델 평가")
        
        model.eval()
        all_predictions = []
        all_targets = []
        all_group_indices = []
        
        with torch.no_grad():
            for data, targets, group_idx in test_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                group_idx = group_idx.to(self.device)
                outputs = model(data, group_idx)
                _, predicted = outputs['class_logits'].max(1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_group_indices.extend(group_idx.cpu().numpy())
        
        # 혼동 행렬 생성
        self._create_confusion_matrix(all_targets, all_predictions)
        
        # 클래스별 성능 분석
        self._analyze_class_performance(all_targets, all_predictions, all_group_indices)
    
    def _create_confusion_matrix(self, targets, predictions):
        """혼동 행렬 생성"""
        from sklearn.metrics import confusion_matrix, classification_report
        
        cm = confusion_matrix(targets, predictions)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Customized Model Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig('customized_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        report = classification_report(targets, predictions, output_dict=True)
        
        with open('customized_classification_report.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print("✅ 혼동 행렬 저장: customized_confusion_matrix.png")
        print("✅ 분류 보고서 저장: customized_classification_report.json")
    
    def _analyze_class_performance(self, targets, predictions, group_indices):
        """클래스별 성능 분석"""
        from sklearn.metrics import accuracy_score
        
        label_mapper = KSLLabelMapper()
        
        print(f"\n🎯 클래스별 성능:")
        
        for class_name in label_mapper.get_all_classes():
            class_idx = label_mapper.get_label_id(class_name)
            
            mask = np.array(targets) == class_idx
            if np.any(mask):
                class_targets = np.array(targets)[mask]
                class_predictions = np.array(predictions)[mask]
                
                accuracy = accuracy_score(class_targets, class_predictions)
                print(f"  {class_name}: {accuracy:.3f} ({accuracy*100:.1f}%)")
        
        # 그룹별 성능 분석
        unique_groups = np.unique(group_indices)
        print(f"\n📊 그룹별 성능:")
        
        for group in unique_groups:
            mask = np.array(group_indices) == group
            if np.any(mask):
                group_targets = np.array(targets)[mask]
                group_predictions = np.array(predictions)[mask]
                
                accuracy = accuracy_score(group_targets, group_predictions)
                print(f"  그룹 {group}: {accuracy:.3f} ({accuracy*100:.1f}%)")

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
    print("📊 맞춤형 데이터셋 로딩 중...")
    
    train_dataset = CustomizedDataset('integrations/SignGlove_HW/github_unified_data', 'train')
    val_dataset = CustomizedDataset('integrations/SignGlove_HW/github_unified_data', 'val')
    test_dataset = CustomizedDataset('integrations/SignGlove_HW/github_unified_data', 'test')
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
    
    # 모델 생성
    trainer = CustomizedTrainer(config)
    model = trainer.create_model()
    
    # 옵티마이저 및 손실 함수
    optimizer, scheduler = trainer.create_optimizer(model)
    criterion = CustomizedLoss(num_classes=24)
    
    # 훈련
    model = trainer.train(train_loader, val_loader, model, optimizer, scheduler, criterion)
    
    # 평가
    trainer.evaluate(test_loader, model)
    
    print("\n🎉 맞춤형 솔루션 훈련 완료!")

if __name__ == "__main__":
    main()
