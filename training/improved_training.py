#!/usr/bin/env python3
"""
개선된 모델 학습
과적합 방지 및 성능 향상을 위한 개선된 학습 시스템
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import json
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# 모델 경로 추가
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'models'))
from improved_model_architecture import ImprovedSGRU, AttentionLSTM, RGRU, create_model_ensemble

class ImprovedTrainingSystem:
    """개선된 학습 시스템"""
    
    def __init__(self, model_type='RGRU', device=None):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_type = model_type
        self.class_names = ['ㄱ', 'ㄴ', 'ㄷ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅅ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ',
                           'ㅏ', 'ㅑ', 'ㅓ', 'ㅕ', 'ㅗ', 'ㅛ', 'ㅜ', 'ㅠ', 'ㅡ', 'ㅣ']
        
        # 모델 생성
        self.model = self._create_model()
        self.model.to(self.device)
        
        # 학습 히스토리
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'epochs': []
        }
        
        print(f'✅ 개선된 학습 시스템 초기화 완료: {model_type}')
        print(f'🖥️  디바이스: {self.device}')
    
    def _create_model(self):
        """모델 생성"""
        if self.model_type == 'ImprovedSGRU':
            return ImprovedSGRU(input_size=8, hidden_size=64, num_classes=24, dropout=0.3)
        elif self.model_type == 'AttentionLSTM':
            return AttentionLSTM(input_size=8, hidden_size=128, num_classes=24, dropout=0.4)
        elif self.model_type == 'RGRU':
            return RGRU(input_size=8, hidden_size=96, num_classes=24, dropout=0.5)
        elif self.model_type == 'Ensemble':
            models = [
                ImprovedSGRU(input_size=8, hidden_size=64, num_classes=24, dropout=0.3),
                AttentionLSTM(input_size=8, hidden_size=128, num_classes=24, dropout=0.4),
                RGRU(input_size=8, hidden_size=96, num_classes=24, dropout=0.5)
            ]
            return create_model_ensemble(models, weights=[0.4, 0.3, 0.3])
        else:
            raise ValueError(f'Unknown model type: {self.model_type}')
    
    def load_and_preprocess_data(self, data_dir='../real_data_filtered'):
        """데이터 로드 및 전처리"""
        print(f'📁 데이터 로드 중: {data_dir}')
        
        X, y = [], []
        class_mapping = {name: idx for idx, name in enumerate(self.class_names)}
        
        # 각 클래스별로 데이터 로드
        for class_name in self.class_names:
            class_dir = Path(data_dir) / class_name
            if not class_dir.exists():
                print(f'⚠️ 클래스 디렉토리를 찾을 수 없습니다: {class_dir}')
                continue
            
            class_idx = class_mapping[class_name]
            class_files = 0
            
            # 각 각도별로 데이터 로드
            for angle in range(1, 6):
                angle_dir = class_dir / str(angle)
                if angle_dir.exists():
                    csv_files = list(angle_dir.glob("*.csv"))
                    for file_path in csv_files:
                        try:
                            df = pd.read_csv(file_path)
                            data = df.iloc[:, :8].values.astype(np.float32)
                            
                            # 시퀀스 길이 통일 (패딩)
                            if len(data) < 300:
                                padding = np.tile(data[-1:], (300 - len(data), 1))
                                data = np.vstack([data, padding])
                            else:
                                data = data[:300]
                            
                            X.append(data)
                            y.append(class_idx)
                            class_files += 1
                            
                        except Exception as e:
                            print(f'⚠️ 파일 처리 실패: {file_path} - {e}')
            
            print(f'  📊 {class_name}: {class_files}개 파일')
        
        if not X:
            print('❌ 데이터를 로드할 수 없습니다.')
            return None, None, None, None
        
        X = np.array(X)
        y = np.array(y)
        
        print(f'✅ 데이터 로드 완료: {len(X)}개 샘플, {len(self.class_names)}개 클래스')
        print(f'📏 데이터 형태: {X.shape}')
        
        # 훈련/검증/테스트 분할
        X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42, stratify=y_temp)
        
        print(f'📊 데이터 분할:')
        print(f'  훈련: {len(X_train)}개')
        print(f'  검증: {len(X_val)}개')
        print(f'  테스트: {len(X_test)}개')
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def create_data_loaders(self, X_train, X_val, X_test, y_train, y_val, y_test, batch_size=32):
        """데이터 로더 생성"""
        # 텐서 변환
        X_train_tensor = torch.FloatTensor(X_train)
        X_val_tensor = torch.FloatTensor(X_val)
        X_test_tensor = torch.FloatTensor(X_test)
        y_train_tensor = torch.LongTensor(y_train)
        y_val_tensor = torch.LongTensor(y_val)
        y_test_tensor = torch.LongTensor(y_test)
        
        # 데이터셋 생성
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        
        # 데이터 로더 생성
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader, test_loader
    
    def train_model(self, train_loader, val_loader, epochs=100, learning_rate=0.001, patience=15):
        """모델 학습"""
        print(f'🚀 모델 학습 시작: {self.model_type}')
        print(f'📊 에포크: {epochs}, 학습률: {learning_rate}, 조기 종료: {patience}')
        
        # 손실 함수 및 옵티마이저
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=0.01)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
        
        # 조기 종료
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        # 학습 루프
        for epoch in range(epochs):
            start_time = time.time()
            
            # 훈련 단계
            train_loss, train_acc = self._train_epoch(train_loader, criterion, optimizer)
            
            # 검증 단계
            val_loss, val_acc = self._validate_epoch(val_loader, criterion)
            
            # 학습률 스케줄링
            scheduler.step(val_loss)
            
            # 히스토리 저장
            self.history['epochs'].append(epoch + 1)
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            # 조기 종료 체크
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
            
            # 진행 상황 출력
            epoch_time = time.time() - start_time
            print(f'Epoch {epoch+1:3d}/{epochs}: '
                  f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
                  f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, '
                  f'Time: {epoch_time:.1f}s')
            
            # 조기 종료
            if patience_counter >= patience:
                print(f'🛑 조기 종료: {patience} 에포크 동안 개선 없음')
                break
        
        # 최고 모델 복원
        if best_model_state:
            self.model.load_state_dict(best_model_state)
            print(f'✅ 최고 모델 복원 (검증 손실: {best_val_loss:.4f})')
        
        return self.history
    
    def _train_epoch(self, train_loader, criterion, optimizer):
        """훈련 에포크"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            output = self.model(data)
            loss = criterion(output, target)
            loss.backward()
            
            # 그래디언트 클리핑
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
        
        return total_loss / len(train_loader), correct / total
    
    def _validate_epoch(self, val_loader, criterion):
        """검증 에포크"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = criterion(output, target)
                
                total_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        return total_loss / len(val_loader), correct / total
    
    def evaluate_model(self, test_loader):
        """모델 평가"""
        print('📊 모델 평가 중...')
        
        self.model.eval()
        all_predictions = []
        all_targets = []
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                _, predicted = output.max(1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        accuracy = correct / total
        
        print(f'✅ 테스트 정확도: {accuracy:.4f}')
        
        # 분류 보고서
        report = classification_report(all_targets, all_predictions, 
                                     target_names=self.class_names, digits=4)
        print('\n📋 분류 보고서:')
        print(report)
        
        return accuracy, all_predictions, all_targets
    
    def save_model(self, save_path='models/improved_model.pth'):
        """모델 저장"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_type': self.model_type,
            'history': self.history,
            'class_names': self.class_names
        }, save_path)
        
        print(f'✅ 모델 저장: {save_path}')
    
    def plot_training_curves(self, save_path='improved_training_curves.png'):
        """학습 곡선 플롯"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Improved {self.model_type} Training Curves', fontsize=16, fontweight='bold')
        
        epochs = self.history['epochs']
        
        # 1. 손실 곡선
        ax1.plot(epochs, self.history['train_loss'], 'b-', label='Training Loss', linewidth=2)
        ax1.plot(epochs, self.history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
        ax1.set_title('Loss Curves', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 정확도 곡선
        ax2.plot(epochs, self.history['train_acc'], 'b-', label='Training Accuracy', linewidth=2)
        ax2.plot(epochs, self.history['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
        ax2.set_title('Accuracy Curves', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1.0)
        
        # 3. 과적합 분석
        overfitting = np.array(self.history['val_loss']) - np.array(self.history['train_loss'])
        ax3.plot(epochs, overfitting, 'g-', label='Overfitting Gap', linewidth=2)
        ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax3.set_title('Overfitting Analysis', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Validation Loss - Training Loss')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. 학습 안정성
        train_loss_smooth = np.convolve(self.history['train_loss'], np.ones(5)/5, mode='valid')
        val_loss_smooth = np.convolve(self.history['val_loss'], np.ones(5)/5, mode='valid')
        smooth_epochs = epochs[2:-2]
        
        ax4.plot(smooth_epochs, train_loss_smooth, 'b-', label='Training Loss (Smoothed)', linewidth=2)
        ax4.plot(smooth_epochs, val_loss_smooth, 'r-', label='Validation Loss (Smoothed)', linewidth=2)
        ax4.set_title('Smoothed Loss Curves', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Loss')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'✅ 학습 곡선 저장: {save_path}')
        plt.show()

def main():
    """메인 실행 함수"""
    print('🚀 개선된 모델 학습 시스템')
    print('=' * 60)
    
    # 학습 시스템 초기화
    training_system = ImprovedTrainingSystem(model_type='RGRU')
    
    # 데이터 로드
    data = training_system.load_and_preprocess_data('../real_data_filtered')
    if data is None:
        print('❌ 데이터 로드 실패')
        return
    
    X_train, X_val, X_test, y_train, y_val, y_test = data
    
    # 데이터 로더 생성
    train_loader, val_loader, test_loader = training_system.create_data_loaders(
        X_train, X_val, X_test, y_train, y_val, y_test, batch_size=32
    )
    
    # 모델 학습
    history = training_system.train_model(
        train_loader, val_loader, 
        epochs=100, learning_rate=0.001, patience=15
    )
    
    # 모델 평가
    accuracy, predictions, targets = training_system.evaluate_model(test_loader)
    
    # 모델 저장
    training_system.save_model('models/improved_regularized_model.pth')
    
    # 학습 곡선 플롯
    training_system.plot_training_curves('improved_training_curves.png')
    
    print('\n🎉 개선된 모델 학습 완료!')
    print(f'📊 최종 테스트 정확도: {accuracy:.4f}')

if __name__ == "__main__":
    main()
