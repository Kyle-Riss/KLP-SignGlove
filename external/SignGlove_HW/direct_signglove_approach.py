#!/usr/bin/env python3
"""
SignGlove 직접 접근법
모델 입력 크기를 8로 수정하여 SignGlove 데이터 직접 사용
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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import pickle

# 한글 폰트 설정
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Liberation Sans', 'Arial', 'Helvetica']

class SignGloveDataset(Dataset):
    """SignGlove 데이터를 위한 PyTorch Dataset"""
    
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

class ModifiedSignSpeakModels:
    """수정된 SignSpeak 모델들 (입력 크기: 8)"""
    
    @staticmethod
    def create_modified_gru(input_size=8, hidden_size=64, num_classes=24):
        """수정된 GRU 모델 (8개 입력)"""
        class ModifiedGRU(nn.Module):
            def __init__(self, input_size=8, hidden_size=64, num_classes=24):
                super().__init__()
                self.gru = nn.GRU(input_size, hidden_size, batch_first=True, dropout=0.2)
                self.dropout = nn.Dropout(0.3)
                self.classifier = nn.Sequential(
                    nn.Linear(hidden_size, hidden_size // 2),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(hidden_size // 2, num_classes)
                )
            
            def forward(self, x):
                gru_out, _ = self.gru(x)
                # 마지막 시퀀스 출력 사용
                last_output = gru_out[:, -1, :]
                last_output = self.dropout(last_output)
                return self.classifier(last_output)
        
        return ModifiedGRU(input_size, hidden_size, num_classes)
    
    @staticmethod
    def create_modified_lstm(input_size=8, hidden_size=64, num_classes=24):
        """수정된 LSTM 모델 (8개 입력)"""
        class ModifiedLSTM(nn.Module):
            def __init__(self, input_size=8, hidden_size=64, num_classes=24):
                super().__init__()
                self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, dropout=0.2)
                self.dropout = nn.Dropout(0.3)
                self.classifier = nn.Sequential(
                    nn.Linear(hidden_size, hidden_size // 2),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(hidden_size // 2, num_classes)
                )
            
            def forward(self, x):
                lstm_out, _ = self.lstm(x)
                # 마지막 시퀀스 출력 사용
                last_output = lstm_out[:, -1, :]
                last_output = self.dropout(last_output)
                return self.classifier(last_output)
        
        return ModifiedLSTM(input_size, hidden_size, num_classes)
    
    @staticmethod
    def create_modified_transformer(input_size=8, d_model=64, num_classes=24):
        """수정된 Transformer 모델 (8개 입력)"""
        class ModifiedTransformer(nn.Module):
            def __init__(self, input_size=8, d_model=64, num_classes=24):
                super().__init__()
                self.input_projection = nn.Linear(input_size, d_model)
                self.pos_encoding = nn.Parameter(torch.randn(1, 300, d_model))
                
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=d_model, 
                    nhead=8, 
                    dim_feedforward=d_model * 4,
                    dropout=0.1,
                    batch_first=True
                )
                self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
                
                self.classifier = nn.Sequential(
                    nn.Linear(d_model, d_model // 2),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(d_model // 2, num_classes)
                )
            
            def forward(self, x):
                # 입력을 d_model 차원으로 투영
                x = self.input_projection(x)
                # 위치 인코딩 추가
                x = x + self.pos_encoding[:, :x.size(1), :]
                # Transformer 인코더
                encoded = self.transformer(x)
                # 마지막 시퀀스 출력 사용
                last_output = encoded[:, -1, :]
                return self.classifier(last_output)
        
        return ModifiedTransformer(input_size, d_model, num_classes)

class SignGloveTrainer:
    """SignGlove 데이터로 모델 훈련"""
    
    def __init__(self, dataset, batch_size=32, learning_rate=0.001):
        self.dataset = dataset
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🖥️ 사용 디바이스: {self.device}")
        
        # 데이터 로더 생성
        self.train_loader, self.val_loader = self.create_data_loaders()
        
    def create_data_loaders(self):
        """훈련/검증 데이터 로더 생성"""
        # 데이터 분할
        train_size = int(0.8 * len(self.dataset))
        val_size = len(self.dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(self.dataset, [train_size, val_size])
        
        # 데이터 로더 생성
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        print(f"📊 훈련 데이터: {len(train_dataset)}개, 검증 데이터: {len(val_dataset)}개")
        return train_loader, val_loader
    
    def train_model(self, model, epochs=100, early_stopping_patience=15):
        """모델 훈련"""
        model = model.to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)
        
        # 훈련 기록
        train_losses, val_losses, val_accuracies = [], [], []
        best_val_acc = 0.0
        patience_counter = 0
        
        print(f"🎯 모델 훈련 시작 ({epochs} epochs)")
        
        for epoch in range(epochs):
            # 훈련 단계
            model.train()
            train_loss = 0.0
            
            for batch_X, batch_y in self.train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.squeeze().to(self.device)
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(self.train_loader)
            
            # 검증 단계
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for batch_X, batch_y in self.val_loader:
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.squeeze().to(self.device)
                    
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
                    
                    _, predicted = torch.max(outputs, 1)
                    total += batch_y.size(0)
                    correct += (predicted == batch_y).sum().item()
            
            val_loss /= len(self.val_loader)
            val_acc = correct / total
            
            # 기록
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            val_accuracies.append(val_acc)
            
            # 학습률 조정
            scheduler.step(val_loss)
            
            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                # 최고 모델 저장
                torch.save(model.state_dict(), 'best_model.pth')
            else:
                patience_counter += 1
            
            # 진행 상황 출력
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{epochs}: "
                      f"Train Loss: {train_loss:.4f}, "
                      f"Val Loss: {val_loss:.4f}, "
                      f"Val Acc: {val_acc:.3f}")
            
            # Early stopping 체크
            if patience_counter >= early_stopping_patience:
                print(f"  🛑 Early stopping at epoch {epoch+1}")
                break
        
        print(f"✅ 훈련 완료! 최고 검증 정확도: {best_val_acc:.3f}")
        
        # 최고 모델 로드
        model.load_state_dict(torch.load('best_model.pth'))
        
        return {
            'model': model,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_accuracies': val_accuracies,
            'best_val_acc': best_val_acc
        }
    
    def evaluate_model(self, model, test_loader=None):
        """모델 평가"""
        if test_loader is None:
            test_loader = self.val_loader
        
        model.eval()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.squeeze()
                
                outputs = model(batch_X)
                _, predicted = torch.max(outputs, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(batch_y.numpy())
        
        # 성능 계산
        accuracy = accuracy_score(all_labels, all_predictions)
        report = classification_report(all_labels, all_predictions, output_dict=True)
        
        print(f"📊 최종 테스트 정확도: {accuracy:.3f}")
        
        return accuracy, report

def run_complete_training():
    """전체 훈련 프로세스 실행"""
    print("🚀 SignGlove 직접 접근법 훈련 시작!")
    print("=" * 60)
    
    # 1. 데이터셋 로드
    print("📊 데이터셋 로드 중...")
    dataset = SignGloveDataset(max_samples_per_class=50)
    
    if len(dataset) == 0:
        print("❌ 데이터 로드 실패")
        return
    
    # 2. 모델 생성
    print("\n🤖 수정된 SignSpeak 모델 생성 중...")
    models = {
        'GRU': ModifiedSignSpeakModels.create_modified_gru(),
        'LSTM': ModifiedSignSpeakModels.create_modified_lstm(),
        'Transformer': ModifiedSignSpeakModels.create_modified_transformer()
    }
    
    # 3. 모델별 훈련 및 평가
    print("\n" + "=" * 60)
    print("🎯 모델별 훈련 시작")
    
    results = {}
    
    for model_name, model in models.items():
        print(f"\n🔬 {model_name} 모델 훈련 중...")
        
        # 훈련
        trainer = SignGloveTrainer(dataset, batch_size=32, learning_rate=0.001)
        training_result = trainer.train_model(model, epochs=100, early_stopping_patience=15)
        
        # 평가
        accuracy, report = trainer.evaluate_model(training_result['model'])
        
        # 결과 저장
        results[model_name] = {
            'training_result': training_result,
            'final_accuracy': accuracy,
            'classification_report': report
        }
        
        print(f"  ✅ {model_name} 완료: 정확도 {accuracy:.3f}")
    
    # 4. 결과 분석 및 시각화
    print("\n" + "=" * 60)
    print("📊 결과 분석 및 시각화 생성 중...")
    
    create_performance_visualization(results)
    save_results(results)
    
    print("\n🎯 모든 훈련 완료!")
    print("\n📁 생성된 파일들:")
    print("  - best_model.pth (최고 성능 모델)")
    print("  - performance_comparison.png")
    print("  - training_results.pkl")

def create_performance_visualization(results):
    """성능 시각화"""
    print("  📊 성능 시각화 생성 중...")
    
    try:
        # 1. 모델별 성능 비교
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('SignGlove 직접 접근법 모델 성능 비교', fontsize=16, fontweight='bold')
        
        model_names = list(results.keys())
        accuracies = [results[name]['final_accuracy'] for name in model_names]
        
        # 막대 그래프
        bars = ax1.bar(model_names, accuracies, color=['skyblue', 'lightgreen', 'lightcoral'], alpha=0.8)
        ax1.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Accuracy')
        ax1.set_ylim(0, 1)
        
        # 값 표시
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. 훈련 과정 비교
        for i, model_name in enumerate(model_names):
            training_result = results[model_name]['training_result']
            ax2.plot(training_result['val_accuracies'], label=f'{model_name}', alpha=0.8)
        
        ax2.set_title('Training Progress Comparison', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Validation Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig('performance_comparison.png', dpi=300, bbox_inches='tight')
        print("    ✅ 성능 비교 차트 저장: performance_comparison.png")
        
    except Exception as e:
        print(f"    ❌ 시각화 생성 실패: {e}")

def save_results(results):
    """결과 저장"""
    print("  💾 결과 저장 중...")
    
    try:
        # Pickle 파일로 저장
        with open('training_results.pkl', 'wb') as f:
            pickle.dump(results, f)
        
        # CSV 요약 저장
        summary_data = []
        for model_name, result in results.items():
            summary_data.append({
                'Model': model_name,
                'Final_Accuracy': result['final_accuracy'],
                'Best_Val_Accuracy': result['training_result']['best_val_acc'],
                'Total_Epochs': len(result['training_result']['train_losses'])
            })
        
        df_summary = pd.DataFrame(summary_data)
        df_summary.to_csv('training_summary.csv', index=False)
        
        print("    ✅ 결과 저장 완료:")
        print("      - training_results.pkl")
        print("      - training_summary.csv")
        
    except Exception as e:
        print(f"    ❌ 결과 저장 실패: {e}")

def main():
    """메인 실행 함수"""
    run_complete_training()

if __name__ == "__main__":
    main()

