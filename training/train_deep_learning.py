import argparse
import os
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import time

from training.dataset import KSLCsvDataset
from models.deep_learning import DeepLearningPipeline, CNNLSTMAdvanced

class DeepLearningTrainer:
    def __init__(self, config_path=None, csv_dir=None):
        # 설정 로드
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)['deep_learning']
        else:
            # 기본 설정
            self.config = {
                'model_type': 'DeepLearningPipeline',
                'input_features': 8,
                'sequence_length': 20,
                'num_classes': 5,
                'hidden_dim': 128,
                'num_layers': 2,
                'dropout': 0.3,
                'training': {
                    'batch_size': 32,
                    'learning_rate': 0.001,
                    'epochs': 100,
                    'patience': 10,
                    'weight_decay': 1e-5
                },
                'data': {
                    'window_size': 20,
                    'stride': 10,
                    'train_split': 0.8,
                    'val_split': 0.1,
                    'test_split': 0.1
                }
            }
        
        self.csv_dir = csv_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"사용 장치: {self.device}")
        
        # 모델 초기화
        self.model = self._create_model()
        
        # 데이터셋 및 데이터로더 초기화
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        
        # 학습 이력
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
    def _create_model(self):
        """모델 생성"""
        if self.config['model_type'] == 'CNNLSTMAdvanced':
            model = CNNLSTMAdvanced(
                input_features=self.config['input_features'],
                sequence_length=self.config['sequence_length'],
                num_classes=self.config['num_classes']
            )
        else:
            model = DeepLearningPipeline(
                input_features=self.config['input_features'],
                sequence_length=self.config['sequence_length'],
                num_classes=self.config['num_classes'],
                hidden_dim=self.config['hidden_dim'],
                num_layers=self.config['num_layers'],
                dropout=self.config['dropout']
            )
        
        return model.to(self.device)
    
    def prepare_data(self, csv_dir=None):
        """데이터 준비 및 분할"""
        if csv_dir:
            self.csv_dir = csv_dir
            
        if not self.csv_dir:
            raise ValueError("CSV 디렉토리가 지정되지 않았습니다.")
        
        # 데이터셋 생성
        dataset = KSLCsvDataset(
            self.csv_dir,
            window_size=self.config['data']['window_size'],
            stride=self.config['data']['stride']
        )
        
        # 데이터셋 정보 출력
        dataset.print_dataset_info()
        
        # 데이터 분할
        total_size = len(dataset)
        train_size = int(self.config['data']['train_split'] * total_size)
        val_size = int(self.config['data']['val_split'] * total_size)
        test_size = total_size - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [train_size, val_size, test_size]
        )
        
        # 데이터로더 생성
        batch_size = self.config['training']['batch_size']
        
        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
        )
        self.val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False
        )
        
        print(f"데이터 분할 완료:")
        print(f"  학습: {len(train_dataset)}개 ({train_size})")
        print(f"  검증: {len(val_dataset)}개 ({val_size})")
        print(f"  테스트: {len(test_dataset)}개 ({test_size})")
        
    def train(self):
        """모델 학습"""
        if not self.train_loader:
            raise ValueError("데이터가 준비되지 않았습니다. prepare_data()를 먼저 호출하세요.")
        
        # 옵티마이저 및 스케줄러 설정
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=float(self.config['training']['learning_rate']),
            weight_decay=float(self.config['training']['weight_decay'])
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
        )
        
        criterion = nn.CrossEntropyLoss()
        
        # Early stopping 설정
        best_val_loss = float('inf')
        patience_counter = 0
        patience = self.config['training']['patience']
        
        epochs = self.config['training']['epochs']
        
        print(f"\n=== 딥러닝 모델 학습 시작 ===")
        print(f"에포크: {epochs}, 배치 크기: {self.config['training']['batch_size']}")
        print(f"학습률: {self.config['training']['learning_rate']}")
        
        for epoch in range(epochs):
            # 학습 단계
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_idx, (X, y) in enumerate(tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{epochs}")):
                X, y = X.to(self.device), y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(X)
                loss = criterion(outputs['class_logits'], y)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs['class_logits'].data, 1)
                train_total += y.size(0)
                train_correct += (predicted == y).sum().item()
            
            # 검증 단계
            val_loss, val_acc = self.evaluate(self.val_loader)
            
            # 스케줄러 업데이트
            scheduler.step(val_loss)
            
            # 평균 계산
            avg_train_loss = train_loss / len(self.train_loader)
            train_acc = 100.0 * train_correct / train_total
            
            # 이력 저장
            self.train_losses.append(avg_train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            print(f"Epoch {epoch+1}/{epochs}:")
            print(f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
            
            # Early stopping 체크
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # 베스트 모델 저장
                torch.save(self.model.state_dict(), 'best_dl_model.pth')
                print(f"  ✓ 베스트 모델 저장됨 (Val Loss: {best_val_loss:.4f})")
            else:
                patience_counter += 1
                print(f"  Patience: {patience_counter}/{patience}")
                
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered at epoch {epoch+1}")
                break
            
            print("-" * 50)
        
        # 베스트 모델 로드
        self.model.load_state_dict(torch.load('best_dl_model.pth'))
        print("베스트 모델 로드 완료")
    
    def evaluate(self, dataloader):
        """모델 평가"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(self.device), y.to(self.device)
                outputs = self.model(X)
                loss = criterion(outputs['class_logits'], y)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs['class_logits'].data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
        
        avg_loss = total_loss / len(dataloader)
        accuracy = 100.0 * correct / total
        
        return avg_loss, accuracy
    
    def test(self):
        """테스트 평가 및 상세 결과"""
        if not self.test_loader:
            raise ValueError("테스트 데이터가 준비되지 않았습니다.")
        
        print("\n=== 테스트 평가 ===")
        test_loss, test_acc = self.evaluate(self.test_loader)
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_acc:.2f}%")
        
        # 상세 분류 리포트
        y_true = []
        y_pred = []
        
        self.model.eval()
        with torch.no_grad():
            for X, y in self.test_loader:
                X, y = X.to(self.device), y.to(self.device)
                outputs = self.model(X)
                _, predicted = torch.max(outputs['class_logits'].data, 1)
                
                y_true.extend(y.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())
        
        # 분류 리포트
        class_names = ['ㄱ', 'ㄴ', 'ㄷ', 'ㄹ', 'ㅁ']
        print("\n분류 리포트:")
        print(classification_report(y_true, y_pred, target_names=class_names))
        
        # 혼동 행렬
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix - Deep Learning Model')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('dl_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return test_acc
    
    def plot_training_history(self):
        """학습 이력 시각화"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss 그래프
        ax1.plot(self.train_losses, label='Train Loss', color='blue')
        ax1.plot(self.val_losses, label='Validation Loss', color='red')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy 그래프
        ax2.plot(self.train_accuracies, label='Train Accuracy', color='blue')
        ax2.plot(self.val_accuracies, label='Validation Accuracy', color='red')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('dl_training_history.png', dpi=300, bbox_inches='tight')
        plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='딥러닝 모델 학습')
    parser.add_argument('--csv_dir', type=str, required=True, help='CSV 데이터 디렉토리')
    parser.add_argument('--config', type=str, default='configs/model.yaml', help='설정 파일 경로')
    parser.add_argument('--epochs', type=int, help='에포크 수 (설정 파일 오버라이드)')
    parser.add_argument('--batch_size', type=int, help='배치 크기 (설정 파일 오버라이드)')
    parser.add_argument('--lr', type=float, help='학습률 (설정 파일 오버라이드)')
    
    args = parser.parse_args()
    
    # 트레이너 초기화
    trainer = DeepLearningTrainer(config_path=args.config, csv_dir=args.csv_dir)
    
    # 명령행 인자로 설정 오버라이드
    if args.epochs:
        trainer.config['training']['epochs'] = args.epochs
    if args.batch_size:
        trainer.config['training']['batch_size'] = args.batch_size
    if args.lr:
        trainer.config['training']['learning_rate'] = args.lr
    
    try:
        # 데이터 준비
        trainer.prepare_data()
        
        # 모델 학습
        start_time = time.time()
        trainer.train()
        end_time = time.time()
        
        print(f"\n학습 완료! 소요 시간: {end_time - start_time:.2f}초")
        
        # 테스트 평가
        test_accuracy = trainer.test()
        
        # 학습 이력 시각화
        trainer.plot_training_history()
        
        print(f"\n=== 최종 결과 ===")
        print(f"테스트 정확도: {test_accuracy:.2f}%")
        print(f"모델 저장됨: best_dl_model.pth")
        
    except Exception as e:
        print(f"오류 발생: {e}")
        import traceback
        traceback.print_exc()
