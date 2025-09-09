#!/usr/bin/env python3
"""
SignGlove 완벽 검증 시스템
600 epochs + 5-fold 교차검증으로 과적합 진단
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import pickle
from tqdm import tqdm

# 한글 폰트 설정
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Liberation Sans', 'Arial', 'Helvetica']

class RobustSignGloveDataset(Dataset):
    """강화된 SignGlove 데이터셋"""
    
    def __init__(self, data_path="datasets/unified", max_samples_per_class=100):
        self.data_path = data_path
        self.max_samples_per_class = max_samples_per_class
        self.X, self.y = self.load_data()
        self.label_encoder = LabelEncoder()
        self.y_encoded = self.label_encoder.fit_transform(self.y)
        
    def load_data(self):
        """SignGlove H5 데이터 로드"""
        print("📊 SignGlove 데이터 로드 중...")
        
        X, y = [], []
        
        try:
            # 클래스별 데이터 수집
            for class_name in os.listdir(self.data_path):
                class_dir = os.path.join(self.data_path, class_name)
                if not os.path.isdir(class_dir):
                    continue
                
                print(f"  📁 클래스 {class_name} 처리 중...")
                class_samples = 0
                
                for session in os.listdir(class_dir):
                    if class_samples >= self.max_samples_per_class:
                        break
                        
                    session_path = os.path.join(class_dir, session)
                    if not os.path.isdir(session_path):
                        continue
                    
                    for file_name in os.listdir(session_path):
                        if class_samples >= self.max_samples_per_class:
                            break
                            
                        if file_name.endswith('.h5'):
                            file_path = os.path.join(session_path, file_name)
                            
                            try:
                                with h5py.File(file_path, 'r') as f:
                                    if 'sensor_data' in f:
                                        sensor_data = f['sensor_data'][:]
                                        
                                        # 데이터 검증: (300, 8) 형태 확인
                                        if sensor_data.shape == (300, 8):
                                            # NaN 처리
                                            sensor_data = np.nan_to_num(sensor_data, nan=0.0)
                                            
                                            X.append(sensor_data)
                                            y.append(class_name)
                                            class_samples += 1
                                            
                                            if class_samples % 20 == 0:
                                                print(f"    ✅ {class_samples}개 샘플 로드됨")
                            except Exception as e:
                                print(f"    ❌ 파일 로드 실패: {file_name} - {e}")
                                continue
                
                print(f"  ✅ 클래스 {class_name}: {class_samples}개 샘플")
            
            X = np.array(X)
            y = np.array(y)
            
            print(f"📊 총 {X.shape[0]}개 샘플, {X.shape[1]} 시퀀스, {X.shape[2]} 특성")
            print(f"📊 클래스 수: {len(np.unique(y))}")
            
            return X, y
            
        except Exception as e:
            print(f"❌ 데이터 로드 실패: {e}")
            return np.array([]), np.array([])
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.X[idx]), torch.LongTensor([self.y_encoded[idx]])

class RobustSignSpeakModels:
    """강화된 SignSpeak 모델들 (600 epochs 대응)"""
    
    @staticmethod
    def create_robust_gru(input_size=8, hidden_size=64, num_classes=24):
        """강화된 GRU 모델"""
        class RobustGRU(nn.Module):
            def __init__(self, input_size=8, hidden_size=64, num_classes=24):
                super().__init__()
                self.gru = nn.GRU(input_size, hidden_size, batch_first=True, dropout=0.2)
                self.dropout = nn.Dropout(0.3)
                self.batch_norm = nn.BatchNorm1d(hidden_size)
                self.classifier = nn.Sequential(
                    nn.Linear(hidden_size, hidden_size // 2),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.BatchNorm1d(hidden_size // 2),
                    nn.Linear(hidden_size // 2, num_classes)
                )
            
            def forward(self, x):
                gru_out, _ = self.gru(x)
                last_output = gru_out[:, -1, :]
                last_output = self.dropout(last_output)
                last_output = self.batch_norm(last_output)
                return self.classifier(last_output)
        
        return RobustGRU(input_size, hidden_size, num_classes)
    
    @staticmethod
    def create_robust_lstm(input_size=8, hidden_size=64, num_classes=24):
        """강화된 LSTM 모델"""
        class RobustLSTM(nn.Module):
            def __init__(self, input_size=8, hidden_size=64, num_classes=24):
                super().__init__()
                self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, dropout=0.2)
                self.dropout = nn.Dropout(0.3)
                self.batch_norm = nn.BatchNorm1d(hidden_size)
                self.classifier = nn.Sequential(
                    nn.Linear(hidden_size, hidden_size // 2),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.BatchNorm1d(hidden_size // 2),
                    nn.Linear(hidden_size // 2, num_classes)
                )
            
            def forward(self, x):
                lstm_out, _ = self.lstm(x)
                last_output = lstm_out[:, -1, :]
                last_output = self.dropout(last_output)
                last_output = self.batch_norm(last_output)
                return self.classifier(last_output)
        
        return RobustLSTM(input_size, hidden_size, num_classes)
    
    @staticmethod
    def create_robust_transformer(input_size=8, d_model=64, num_classes=24):
        """강화된 Transformer 모델"""
        class RobustTransformer(nn.Module):
            def __init__(self, input_size=8, d_model=64, num_classes=24):
                super().__init__()
                self.input_projection = nn.Linear(input_size, d_model)
                self.pos_encoding = nn.Parameter(torch.randn(1, 300, d_model))
                self.dropout = nn.Dropout(0.1)
                
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=d_model, 
                    nhead=8, 
                    dim_feedforward=d_model * 4,
                    dropout=0.1,
                    batch_first=True,
                    norm_first=True
                )
                self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)
                
                self.classifier = nn.Sequential(
                    nn.Linear(d_model, d_model // 2),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.BatchNorm1d(d_model // 2),
                    nn.Linear(d_model // 2, num_classes)
                )
            
            def forward(self, x):
                x = self.input_projection(x)
                x = x + self.pos_encoding[:, :x.size(1), :]
                x = self.dropout(x)
                encoded = self.transformer(x)
                last_output = encoded[:, -1, :]
                return self.classifier(last_output)
        
        return RobustTransformer(input_size, d_model, num_classes)

class RobustTrainer:
    """강화된 훈련 시스템 (600 epochs + 과적합 진단)"""
    
    def __init__(self, dataset, batch_size=32, learning_rate=0.001):
        self.dataset = dataset
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🖥️ 사용 디바이스: {self.device}")
        
    def train_with_kfold(self, model, epochs=600, n_splits=5):
        """K-fold 교차검증으로 훈련"""
        print(f"🎯 K-fold 교차검증 훈련 시작 ({n_splits}-fold, {epochs} epochs)")
        
        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        fold_results = []
        
        # 데이터 준비
        X = np.array([self.dataset[i][0].numpy() for i in range(len(self.dataset))])
        y = np.array([self.dataset[i][1].numpy()[0] for i in range(len(self.dataset))])
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
            print(f"\n🔄 Fold {fold + 1}/{n_splits} 훈련 중...")
            
            # 데이터 분할
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # 데이터 로더 생성
            train_dataset = torch.utils.data.TensorDataset(
                torch.FloatTensor(X_train), torch.LongTensor(y_train)
            )
            val_dataset = torch.utils.data.TensorDataset(
                torch.FloatTensor(X_val), torch.LongTensor(y_val)
            )
            
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
            
            print(f"  📊 훈련 데이터: {len(train_dataset)}개, 검증 데이터: {len(val_dataset)}개")
            
            # 모델 훈련
            fold_result = self.train_single_fold(
                model, train_loader, val_loader, epochs, fold
            )
            
            fold_results.append(fold_result)
            
            # 모델 재초기화 (다음 fold를 위해)
            if fold < n_splits - 1:
                model = model.__class__()
        
        return fold_results
    
    def train_single_fold(self, model, train_loader, val_loader, epochs, fold):
        """단일 fold 훈련"""
        model = model.to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=20, factor=0.5, verbose=True
        )
        
        # 훈련 기록
        train_losses, val_losses, val_accuracies = [], [], []
        best_val_acc = 0.0
        best_epoch = 0
        
        # 과적합 진단을 위한 훈련 정확도도 기록
        train_accuracies = []
        
        print(f"    🎯 Fold {fold + 1} 훈련 시작 ({epochs} epochs)")
        
        for epoch in tqdm(range(epochs), desc=f"Fold {fold + 1}"):
            # 훈련 단계
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                
                # 그래디언트 클리핑
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                train_loss += loss.item()
                
                # 훈련 정확도 계산
                _, predicted = torch.max(outputs, 1)
                train_total += batch_y.size(0)
                train_correct += (predicted == batch_y).sum().item()
            
            train_loss /= len(train_loader)
            train_acc = train_correct / train_total
            
            # 검증 단계
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device)
                    
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
                    
                    _, predicted = torch.max(outputs, 1)
                    val_total += batch_y.size(0)
                    val_correct += (predicted == batch_y).sum().item()
            
            val_loss /= len(val_loader)
            val_acc = val_correct / val_total
            
            # 기록
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            val_accuracies.append(val_acc)
            train_accuracies.append(train_acc)
            
            # 학습률 조정
            scheduler.step(val_loss)
            
            # 최고 성능 모델 저장
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                torch.save(model.state_dict(), f'best_model_fold_{fold + 1}.pth')
            
            # 진행 상황 출력
            if (epoch + 1) % 50 == 0:
                print(f"      Epoch {epoch+1}/{epochs}: "
                      f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.3f}, "
                      f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.3f}")
        
        print(f"    ✅ Fold {fold + 1} 완료! 최고 검증 정확도: {best_val_acc:.3f} (Epoch {best_epoch + 1})")
        
        # 과적합 진단
        overfitting_score = self.diagnose_overfitting(train_accuracies, val_accuracies)
        
        return {
            'fold': fold + 1,
            'best_val_acc': best_val_acc,
            'best_epoch': best_epoch,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accuracies': train_accuracies,
            'val_accuracies': val_accuracies,
            'overfitting_score': overfitting_score,
            'final_train_acc': train_accuracies[-1],
            'final_val_acc': val_accuracies[-1]
        }
    
    def diagnose_overfitting(self, train_accs, val_accs):
        """과적합 진단"""
        if len(train_accs) < 10:
            return "insufficient_data"
        
        # 마지막 100 epochs의 과적합 진단
        recent_train = train_accs[-100:]
        recent_val = val_accs[-100:]
        
        # 훈련 정확도가 계속 증가하는지 확인
        train_trend = np.polyfit(range(len(recent_train)), recent_train, 1)[0]
        
        # 검증 정확도가 감소하는지 확인
        val_trend = np.polyfit(range(len(recent_val)), recent_val, 1)[0]
        
        # 과적합 지수 계산
        train_val_gap = np.mean(recent_train) - np.mean(recent_val)
        
        if train_trend > 0.001 and val_trend < -0.001:
            return "severe_overfitting"
        elif train_trend > 0.0005 and val_trend < 0:
            return "moderate_overfitting"
        elif train_val_gap > 0.1:
            return "mild_overfitting"
        else:
            return "no_overfitting"

def run_robust_validation():
    """강화된 검증 시스템 실행"""
    print("🚀 SignGlove 완벽 검증 시스템 시작!")
    print("=" * 60)
    
    # 1. 데이터셋 로드
    print("📊 데이터셋 로드 중...")
    dataset = RobustSignGloveDataset(max_samples_per_class=50)
    
    if len(dataset) == 0:
        print("❌ 데이터 로드 실패")
        return
    
    # 2. 강화된 모델 생성
    print("\n🤖 강화된 SignSpeak 모델 생성 중...")
    models = {
        'GRU': RobustSignSpeakModels.create_robust_gru(),
        'LSTM': RobustSignSpeakModels.create_robust_lstm(),
        'Transformer': RobustSignSpeakModels.create_robust_transformer()
    }
    
    # 3. K-fold 교차검증 훈련
    print("\n" + "=" * 60)
    print("🎯 K-fold 교차검증 훈련 시작")
    
    all_results = {}
    
    for model_name, model in models.items():
        print(f"\n🔬 {model_name} 모델 K-fold 훈련 중...")
        
        # 훈련
        trainer = RobustTrainer(dataset, batch_size=32, learning_rate=0.001)
        fold_results = trainer.train_with_kfold(model, epochs=600, n_splits=5)
        
        # 결과 저장
        all_results[model_name] = fold_results
        
        # 성능 요약
        fold_accuracies = [result['best_val_acc'] for result in fold_results]
        mean_acc = np.mean(fold_accuracies)
        std_acc = np.std(fold_accuracies)
        
        print(f"  ✅ {model_name} K-fold 완료!")
        print(f"  📊 평균 정확도: {mean_acc:.3f} ± {std_acc:.3f}")
        print(f"  📊 최고 정확도: {max(fold_accuracies):.3f}")
        print(f"  📊 최저 정확도: {min(fold_accuracies):.3f}")
    
    # 4. 결과 분석 및 시각화
    print("\n" + "=" * 60)
    print("📊 결과 분석 및 시각화 생성 중...")
    
    create_robust_visualization(all_results)
    save_robust_results(all_results)
    
    print("\n🎯 완벽한 검증 완료!")
    print("\n📁 생성된 파일들:")
    print("  - best_model_fold_*.pth (각 fold별 최고 모델)")
    print("  - robust_validation_results.png")
    print("  - robust_training_results.pkl")

def create_robust_visualization(all_results):
    """강화된 시각화 생성"""
    print("  📊 강화된 시각화 생성 중...")
    
    try:
        # 1. K-fold 성능 비교
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('SignGlove 강화된 검증 결과 (600 epochs + 5-fold CV)', fontsize=16, fontweight='bold')
        
        model_names = list(all_results.keys())
        
        for i, model_name in enumerate(model_names):
            fold_results = all_results[model_name]
            
            # Fold별 성능
            fold_numbers = [result['fold'] for result in fold_results]
            fold_accuracies = [result['best_val_acc'] for result in fold_results]
            
            ax1 = axes[0, i]
            bars = ax1.bar(fold_numbers, fold_accuracies, color='skyblue', alpha=0.8)
            ax1.set_title(f'{model_name} - Fold별 성능')
            ax1.set_xlabel('Fold')
            ax1.set_ylabel('Best Validation Accuracy')
            ax1.set_ylim(0, 1)
            
            # 값 표시
            for bar, acc in zip(bars, fold_accuracies):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                        f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
            
            # 훈련 과정 비교 (첫 번째 fold)
            if fold_results:
                first_fold = fold_results[0]
                ax2 = axes[1, i]
                
                epochs = range(1, len(first_fold['train_accuracies']) + 1)
                ax2.plot(epochs, first_fold['train_accuracies'], label='Train', color='blue', alpha=0.7)
                ax2.plot(epochs, first_fold['val_accuracies'], label='Validation', color='red', alpha=0.7)
                
                ax2.set_title(f'{model_name} - 훈련 과정 (Fold 1)')
                ax2.set_xlabel('Epoch')
                ax2.set_ylabel('Accuracy')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                ax2.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig('robust_validation_results.png', dpi=300, bbox_inches='tight')
        print("    ✅ 강화된 검증 결과 차트 저장: robust_validation_results.png")
        
        # 2. 과적합 진단 요약
        create_overfitting_summary(all_results)
        
    except Exception as e:
        print(f"    ❌ 시각화 생성 실패: {e}")

def create_overfitting_summary(all_results):
    """과적합 진단 요약"""
    print("  📋 과적합 진단 요약 생성 중...")
    
    try:
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.axis('tight')
        ax.axis('off')
        
        # 과적합 진단 결과 요약
        summary_data = []
        for model_name, fold_results in all_results.items():
            for fold_result in fold_results:
                summary_data.append([
                    f"{model_name}",
                    f"Fold {fold_result['fold']}",
                    f"{fold_result['best_val_acc']:.3f}",
                    f"{fold_result['final_train_acc']:.3f}",
                    f"{fold_result['final_val_acc']:.3f}",
                    f"{fold_result['overfitting_score']}",
                    f"{fold_result['best_epoch'] + 1}"
                ])
        
        columns = ['Model', 'Fold', 'Best Val Acc', 'Final Train Acc', 'Final Val Acc', 'Overfitting', 'Best Epoch']
        
        table = ax.table(cellText=summary_data, colLabels=columns, 
                        cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
        
        # 테이블 스타일링
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        
        # 헤더 스타일링
        for i in range(len(columns)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        plt.title('SignGlove 강화된 검증 결과 요약 (600 epochs + 5-fold CV)', fontsize=16, fontweight='bold', pad=20)
        plt.savefig('robust_validation_summary.png', dpi=300, bbox_inches='tight')
        print("    ✅ 과적합 진단 요약 저장: robust_validation_summary.png")
        
    except Exception as e:
        print(f"    ❌ 과적합 진단 요약 생성 실패: {e}")

def save_robust_results(all_results):
    """강화된 결과 저장"""
    print("  💾 강화된 결과 저장 중...")
    
    try:
        # Pickle 파일로 저장
        with open('robust_training_results.pkl', 'wb') as f:
            pickle.dump(all_results, f)
        
        # CSV 요약 저장
        summary_data = []
        for model_name, fold_results in all_results.items():
            for fold_result in fold_results:
                summary_data.append({
                    'Model': model_name,
                    'Fold': fold_result['fold'],
                    'Best_Val_Accuracy': fold_result['best_val_acc'],
                    'Best_Epoch': fold_result['best_epoch'] + 1,
                    'Final_Train_Accuracy': fold_result['final_train_acc'],
                    'Final_Val_Accuracy': fold_result['final_val_acc'],
                    'Overfitting_Diagnosis': fold_result['overfitting_score']
                })
        
        df_summary = pd.DataFrame(summary_data)
        df_summary.to_csv('robust_validation_summary.csv', index=False)
        
        print("    ✅ 강화된 결과 저장 완료:")
        print("      - robust_training_results.pkl")
        print("      - robust_validation_summary.csv")
        
    except Exception as e:
        print(f"    ❌ 강화된 결과 저장 실패: {e}")

def main():
    """메인 실행 함수"""
    run_robust_validation()

if __name__ == "__main__":
    main()

