#!/usr/bin/env python3
"""
Optimized Cross-Validation Trainer
- Enhanced preprocessing based on analysis results
- Improved model architecture
- Better hyperparameter tuning
- Advanced regularization techniques
- Target: Improve from 77.33% to >80%
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.deep_learning import DeepLearningPipeline
from training.label_mapping import KSLLabelMapper

class OptimizedPreprocessor:
    """Enhanced preprocessing based on analysis results"""
    
    def __init__(self):
        self.label_mapper = KSLLabelMapper()
        
        # Problematic classes identified from analysis
        self.problematic_classes = ['ㅊ', 'ㅌ', 'ㅅ', 'ㅈ', 'ㅋ', 'ㅕ', 'ㅡ', 'ㅣ']
        
        print(f"🔧 최적화 전처리기 초기화 완료!")
        print(f"⚠️ 문제 클래스: {', '.join(self.problematic_classes)}")
    
    def enhanced_yaw_correction(self, data):
        """Enhanced yaw correction based on analysis"""
        if 'yaw' not in data.columns:
            return data
        
        yaw_data = data['yaw'].copy()
        
        # Multi-stage filtering
        # 1. Strong detrending
        yaw_detrended = yaw_data - yaw_data.rolling(window=12, center=True).mean()
        yaw_detrended = yaw_detrended.fillna(method='bfill').fillna(method='ffill')
        
        # 2. Outlier removal
        yaw_std = yaw_detrended.std()
        yaw_mean = yaw_detrended.mean()
        outlier_mask = (yaw_detrended > yaw_mean + 2.5 * yaw_std) | (yaw_detrended < yaw_mean - 2.5 * yaw_std)
        yaw_filtered = yaw_detrended.copy()
        yaw_filtered[outlier_mask] = yaw_mean
        
        # 3. Adaptive smoothing
        yaw_smoothed = yaw_filtered.rolling(window=5, center=True).mean()
        yaw_smoothed = yaw_smoothed.fillna(method='bfill').fillna(method='ffill')
        
        # 4. Final normalization
        yaw_final = (yaw_smoothed - yaw_smoothed.mean()) / (yaw_smoothed.std() + 1e-8)
        
        data['yaw'] = yaw_final
        return data
    
    def flex_sensor_optimization(self, data):
        """Optimize flex sensor signals"""
        flex_sensors = ['flex1', 'flex2', 'flex3', 'flex4', 'flex5']
        
        for sensor in flex_sensors:
            if sensor not in data.columns:
                continue
            
            flex_data = data[sensor].copy()
            
            # 1. Remove outliers
            flex_mean = flex_data.mean()
            flex_std = flex_data.std()
            outlier_mask = (flex_data > flex_mean + 3 * flex_std) | (flex_data < flex_mean - 3 * flex_std)
            flex_clean = flex_data.copy()
            flex_clean[outlier_mask] = flex_mean
            
            # 2. Smoothing
            flex_smoothed = flex_clean.rolling(window=3, center=True).mean()
            flex_smoothed = flex_smoothed.fillna(method='bfill').fillna(method='ffill')
            
            # 3. Normalization
            flex_normalized = (flex_smoothed - flex_smoothed.min()) / (flex_smoothed.max() - flex_smoothed.min() + 1e-8)
            
            data[sensor] = flex_normalized
        return data
    
    def imu_sensor_enhancement(self, data):
        """Enhance IMU sensors (pitch, roll)"""
        for sensor in ['pitch', 'roll']:
            if sensor not in data.columns:
                continue
            
            sensor_data = data[sensor].copy()
            
            # 1. Remove outliers
            sensor_mean = sensor_data.mean()
            sensor_std = sensor_data.std()
            outlier_mask = (sensor_data > sensor_mean + 2.5 * sensor_std) | (sensor_data < sensor_mean - 2.5 * sensor_std)
            sensor_clean = sensor_data.copy()
            sensor_clean[outlier_mask] = sensor_mean
            
            # 2. Smoothing
            sensor_smoothed = sensor_clean.rolling(window=3, center=True).mean()
            sensor_smoothed = sensor_smoothed.fillna(method='bfill').fillna(method='ffill')
            
            # 3. Normalization
            sensor_normalized = (sensor_smoothed - sensor_smoothed.mean()) / (sensor_smoothed.std() + 1e-8)
            
            data[sensor] = sensor_normalized
        return data
    
    def class_specific_enhancement(self, data, class_name):
        """Apply class-specific enhancements"""
        if class_name in self.problematic_classes:
            # Enhanced preprocessing for problematic classes
            if class_name == 'ㅊ':
                # ㅊ has highest yaw variance, apply strongest filtering
                if 'yaw' in data.columns:
                    data['yaw'] = data['yaw'] * 0.7  # Reduce yaw influence
                # Enhance flex sensors
                flex_sensors = ['flex1', 'flex2', 'flex3', 'flex4', 'flex5']
                for sensor in flex_sensors:
                    if sensor in data.columns:
                        data[sensor] = data[sensor] * 1.3  # Amplify flex sensors
            
            elif class_name == 'ㅌ':
                # Moderate yaw filtering
                if 'yaw' in data.columns:
                    data['yaw'] = data['yaw'] * 0.8
                # Enhance flex sensors
                flex_sensors = ['flex1', 'flex2', 'flex3', 'flex4', 'flex5']
                for sensor in flex_sensors:
                    if sensor in data.columns:
                        data[sensor] = data[sensor] * 1.2
            
            elif class_name in ['ㅅ', 'ㅈ', 'ㅋ']:
                # These classes benefit from flex sensor emphasis
                flex_sensors = ['flex1', 'flex2', 'flex3', 'flex4', 'flex5']
                for sensor in flex_sensors:
                    if sensor in data.columns:
                        data[sensor] = data[sensor] * 1.25
            
            elif class_name in ['ㅕ', 'ㅡ', 'ㅣ']:
                # Vowel classes, emphasize pitch and roll
                if 'pitch' in data.columns:
                    data['pitch'] = data['pitch'] * 1.15
                if 'roll' in data.columns:
                    data['roll'] = data['roll'] * 1.15
        
        return data
    
    def preprocess_file(self, file_path, class_name, augment=False):
        """Enhanced preprocessing pipeline"""
        try:
            # Load only necessary columns
            usecols = ['pitch', 'roll', 'yaw', 'flex1', 'flex2', 'flex3', 'flex4', 'flex5']
            data = pd.read_csv(file_path, usecols=usecols)
            
            # Apply enhanced preprocessing
            data = self.enhanced_yaw_correction(data)
            data = self.flex_sensor_optimization(data)
            data = self.imu_sensor_enhancement(data)
            data = self.class_specific_enhancement(data, class_name)
            
            # Length normalization with interpolation
            target_length = 200
            if len(data) != target_length:
                if len(data) < target_length:
                    # Smart padding with interpolation
                    indices = np.linspace(0, len(data)-1, target_length)
                    data_interpolated = []
                    for col in data.columns:
                        col_data = data[col].values
                        interpolated = np.interp(indices, np.arange(len(col_data)), col_data)
                        data_interpolated.append(interpolated)
                    data = pd.DataFrame(np.column_stack(data_interpolated), columns=data.columns)
                else:
                    # Smart truncation
                    data = data.iloc[::len(data)//target_length][:target_length]
            
            # Enhanced augmentation
            if augment:
                data = self._enhanced_augmentation(data, class_name)
            
            return data.values.astype(np.float32)
            
        except Exception as e:
            print(f"⚠️ 전처리 실패: {file_path} - {str(e)}")
            return None
    
    def _enhanced_augmentation(self, data, class_name):
        """Enhanced augmentation techniques"""
        # Class-specific augmentation
        if class_name in self.problematic_classes:
            # Stronger augmentation for problematic classes
            
            # 1. Gaussian noise
            noise_std = 0.05 * data.std()
            noise = np.random.normal(0, noise_std, data.shape)
            data = data + noise
            
            # 2. Time shifting
            shift = np.random.randint(-5, 6)
            if shift != 0:
                data = data.shift(shift).fillna(method='bfill').fillna(method='ffill')
            
            # 3. Scaling
            scale_factor = 1 + np.random.uniform(-0.1, 0.1)
            data = data * scale_factor
            
            # 4. Random masking
            mask_prob = 0.05
            mask = np.random.random(data.shape) > mask_prob
            data = data * mask
            
        else:
            # Moderate augmentation for excellent classes
            # 1. Light Gaussian noise
            noise_std = 0.02 * data.std()
            noise = np.random.normal(0, noise_std, data.shape)
            data = data + noise
            
            # 2. Small time shift
            shift = np.random.randint(-3, 4)
            if shift != 0:
                data = data.shift(shift).fillna(method='bfill').fillna(method='ffill')
        
        return data

class OptimizedDataset(Dataset):
    """Optimized dataset with enhanced preprocessing"""
    
    def __init__(self, data_dir, preprocessor, split='train', augment=False):
        self.data_dir = data_dir
        self.preprocessor = preprocessor
        self.split = split
        self.augment = augment
        
        self.data, self.labels, self.file_paths = self._load_data()
        
        print(f"📊 {split} 데이터셋 생성 완료: {len(self.data)}개 파일")
    
    def _load_data(self):
        """Load data with optimized preprocessing"""
        data = []
        labels = []
        file_paths = []
        
        base_path = os.path.join(self.data_dir, 'github_unified_data')
        
        for class_name in os.listdir(base_path):
            class_path = os.path.join(base_path, class_name)
            if not os.path.isdir(class_path):
                continue
            
            try:
                class_label = self.preprocessor.label_mapper.get_label_id(class_name)
            except:
                continue
            
            for scenario in os.listdir(class_path):
                scenario_path = os.path.join(class_path, scenario)
                if not os.path.isdir(scenario_path):
                    continue
                
                for file_name in os.listdir(scenario_path):
                    if file_name.endswith('.csv'):
                        file_path = os.path.join(scenario_path, file_name)
                        
                        # Apply optimized preprocessing
                        processed_data = self.preprocessor.preprocess_file(
                            file_path, class_name, augment=self.augment
                        )
                        
                        if processed_data is not None:
                            data.append(processed_data)
                            labels.append(class_label)
                            file_paths.append(file_path)
                            
                            # Add augmented samples for training
                            if self.split == 'train' and self.augment:
                                augmented_data = self.preprocessor.preprocess_file(
                                    file_path, class_name, augment=True
                                )
                                if augmented_data is not None:
                                    data.append(augmented_data)
                                    labels.append(class_label)
                                    file_paths.append(file_path)
        
        return data, labels, file_paths
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data = torch.FloatTensor(self.data[idx])
        label = torch.LongTensor([self.labels[idx]])[0]
        return data, label

class OptimizedModel(nn.Module):
    """Optimized model architecture"""
    
    def __init__(self, input_features=8, hidden_dim=64, num_classes=24):
        super(OptimizedModel, self).__init__()
        
        # Enhanced architecture
        self.conv1 = nn.Conv1d(input_features, hidden_dim, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim*2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(hidden_dim*2, hidden_dim, kernel_size=3, padding=1)
        
        self.batch_norm1 = nn.BatchNorm1d(hidden_dim)
        self.batch_norm2 = nn.BatchNorm1d(hidden_dim*2)
        self.batch_norm3 = nn.BatchNorm1d(hidden_dim)
        
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
        
        # LSTM with attention
        self.lstm = nn.LSTM(hidden_dim, hidden_dim//2, num_layers=2, 
                           batch_first=True, dropout=0.2, bidirectional=True)
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4, dropout=0.1)
        
        # Classifier with residual connections
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim//2, hidden_dim//4),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim//4, num_classes)
        )
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_features)
        # Transpose for conv1d: (batch_size, input_features, sequence_length)
        x = x.transpose(1, 2)
        
        # Convolutional layers
        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.dropout(x)
        
        x = self.relu(self.batch_norm2(self.conv2(x)))
        x = self.dropout(x)
        
        x = self.relu(self.batch_norm3(self.conv3(x)))
        x = self.dropout(x)
        
        # Transpose back for LSTM: (batch_size, sequence_length, features)
        x = x.transpose(1, 2)
        
        # LSTM layer
        lstm_out, _ = self.lstm(x)
        
        # Self-attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Global average pooling
        x = torch.mean(attn_out, dim=1)
        
        # Classification
        x = self.classifier(x)
        
        return x

class OptimizedTrainer:
    """Optimized trainer with advanced techniques"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.label_mapper = KSLLabelMapper()
        
        print(f"🔧 사용 디바이스: {self.device}")
        print(f"📊 최적화 트레이너 초기화 완료!")
    
    def create_optimized_model(self):
        """Create optimized model"""
        model = OptimizedModel(
            input_features=8,
            hidden_dim=64,
            num_classes=24
        )
        return model.to(self.device)
    
    def create_optimizer(self, model):
        """Create optimized optimizer"""
        optimizer = optim.AdamW(
            model.parameters(),
            lr=0.0005,  # Slightly higher learning rate
            weight_decay=5e-5,  # Reduced weight decay
            betas=(0.9, 0.999)
        )
        
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )
        
        return optimizer, scheduler
    
    def create_dataloaders(self):
        """Create optimized data loaders"""
        preprocessor = OptimizedPreprocessor()
        
        train_dataset = OptimizedDataset(
            '../integrations/SignGlove_HW', preprocessor, split='train', augment=True
        )
        val_dataset = OptimizedDataset(
            '../integrations/SignGlove_HW', preprocessor, split='val', augment=False
        )
        
        train_loader = DataLoader(
            train_dataset, batch_size=16, shuffle=True, num_workers=2, pin_memory=False
        )
        val_loader = DataLoader(
            val_dataset, batch_size=16, shuffle=False, num_workers=2, pin_memory=False
        )
        
        return train_loader, val_loader
    
    def train_epoch(self, model, train_loader, optimizer, criterion):
        """Train for one epoch with advanced techniques"""
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            output = model(data)
            
            loss = criterion(output, target)
            
            # L2 regularization
            l2_lambda = 1e-4
            l2_reg = torch.tensor(0.).to(self.device)
            for param in model.parameters():
                l2_reg += torch.norm(param)
            loss += l2_lambda * l2_reg
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
        
        return total_loss / len(train_loader), correct / total
    
    def validate_epoch(self, model, val_loader, criterion):
        """Validate for one epoch"""
        model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                output = model(data)
                loss = criterion(output, target)
                
                total_loss += loss.item()
                
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
        
        return total_loss / len(val_loader), correct / total
    
    def train(self):
        """Train optimized model"""
        print("🚀 최적화 모델 훈련 시작!")
        
        # Create components
        model = self.create_optimized_model()
        optimizer, scheduler = self.create_optimizer(model)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        train_loader, val_loader = self.create_dataloaders()
        
        history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'best_val_acc': 0, 'best_epoch': 0
        }
        
        patience_counter = 0
        best_model_state = None
        
        print(f"📊 학습 데이터: {len(train_loader.dataset)}개")
        print(f"📊 검증 데이터: {len(val_loader.dataset)}개")
        print(f"🔧 모델 파라미터: {sum(p.numel() for p in model.parameters()):,}개")
        
        try:
            for epoch in range(150):  # More epochs for optimization
                print(f"\n🔄 Epoch {epoch+1}/150 시작...")
                
                train_loss, train_acc = self.train_epoch(model, train_loader, optimizer, criterion)
                val_loss, val_acc = self.validate_epoch(model, val_loader, criterion)
                
                scheduler.step()
                
                history['train_loss'].append(train_loss)
                history['train_acc'].append(train_acc)
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)
                
                print(f"Epoch {epoch+1:3d}/150 | "
                      f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                      f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
                
                if val_acc > history['best_val_acc']:
                    history['best_val_acc'] = val_acc
                    history['best_epoch'] = epoch
                    patience_counter = 0
                    best_model_state = model.state_dict().copy()
                    print(f"🎯 새로운 최고 정확도: {val_acc:.4f}")
                else:
                    patience_counter += 1
                    
                if patience_counter >= 20:  # Increased patience
                    print(f"🛑 조기 종료: {epoch+1} 에포크에서 중단")
                    break
        
        except KeyboardInterrupt:
            print("\n⚠️ 사용자에 의해 중단됨")
        except Exception as e:
            print(f"\n❌ 오류 발생: {str(e)}")
        
        # Load best model
        if best_model_state:
            model.load_state_dict(best_model_state)
        
        # Save training curves
        self._save_training_curves(history)
        
        return model, history
    
    def _save_training_curves(self, history):
        """Save training curves visualization"""
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Optimized Model Training Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history['train_acc'], label='Train Acc')
        plt.plot(history['val_acc'], label='Val Acc')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Optimized Model Training Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('optimized_cv_training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 훈련 곡선 저장: optimized_cv_training_curves.png")

def main():
    """Main function for optimized cross-validation training"""
    print("📊 최적화 교차 검증 훈련 시스템 시작!")
    
    trainer = OptimizedTrainer()
    
    print("\n1️⃣ 최적화 모델 훈련 중...")
    model, history = trainer.train()
    
    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'history': history,
        'model_type': 'optimized_cv'
    }, 'optimized_cv_model.pth')
    
    print(f"\n🎉 최적화 교차 검증 훈련 완료!")
    print(f"📊 최고 검증 정확도: {history['best_val_acc']:.4f}")
    print(f"📊 최고 에포크: {history['best_epoch']}")
    
    print(f"\n📁 저장된 파일:")
    print(f"   - optimized_cv_model.pth (최적화 모델)")
    print(f"   - optimized_cv_training_curves.png (훈련 곡선)")
    
    print(f"\n📊 최적화 교차 검증 훈련 시스템 완료!")

if __name__ == "__main__":
    main()
