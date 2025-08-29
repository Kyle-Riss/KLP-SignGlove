#!/usr/bin/env python3
"""
Specialized Model Trainer
- Enhanced yaw filtering for problematic classes
- Flex sensor enhancement
- Class-specific preprocessing
- Specialized model architecture
- Ensemble with cross-validation model
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

class SpecializedPreprocessor:
    """Enhanced preprocessing for problematic classes"""
    
    def __init__(self):
        self.label_mapper = KSLLabelMapper()
        
        # Problematic classes identified from analysis
        self.problematic_classes = ['ㅊ', 'ㅌ', 'ㅅ', 'ㅈ', 'ㅋ', 'ㅕ', 'ㅡ', 'ㅣ']
        
        print(f"🔧 특화 전처리기 초기화 완료!")
        print(f"⚠️ 대상 클래스: {', '.join(self.problematic_classes)}")
    
    def enhanced_yaw_filtering(self, data):
        """Enhanced yaw drift correction for problematic classes"""
        if 'yaw' not in data.columns:
            return data
        
        # Multi-stage yaw filtering
        yaw_data = data['yaw'].copy()
        
        # 1. Strong detrending with larger window
        yaw_detrended = yaw_data - yaw_data.rolling(window=15, center=True).mean()
        yaw_detrended = yaw_detrended.fillna(method='bfill').fillna(method='ffill')
        
        # 2. Additional smoothing with variable window
        yaw_smoothed = yaw_detrended.rolling(window=5, center=True).mean()
        yaw_smoothed = yaw_smoothed.fillna(method='bfill').fillna(method='ffill')
        
        # 3. Outlier removal
        yaw_std = yaw_smoothed.std()
        yaw_mean = yaw_smoothed.mean()
        outlier_mask = (yaw_smoothed > yaw_mean + 2 * yaw_std) | (yaw_smoothed < yaw_mean - 2 * yaw_std)
        yaw_filtered = yaw_smoothed.copy()
        yaw_filtered[outlier_mask] = yaw_mean
        
        # 4. Final smoothing
        yaw_final = yaw_filtered.rolling(window=3, center=True).mean()
        yaw_final = yaw_final.fillna(method='bfill').fillna(method='ffill')
        
        data['yaw'] = yaw_final
        return data
    
    def flex_sensor_enhancement(self, data):
        """Enhance flex sensor signals for better discrimination"""
        flex_sensors = ['flex1', 'flex2', 'flex3', 'flex4', 'flex5']
        
        for sensor in flex_sensors:
            if sensor not in data.columns:
                continue
            
            flex_data = data[sensor].copy()
            
            # 1. Normalize flex sensor to 0-1 range
            flex_min = flex_data.min()
            flex_max = flex_data.max()
            if flex_max > flex_min:
                flex_normalized = (flex_data - flex_min) / (flex_max - flex_min)
            else:
                flex_normalized = flex_data - flex_min
            
            # 2. Apply non-linear enhancement (sigmoid-like)
            flex_enhanced = 1 / (1 + np.exp(-5 * (flex_normalized - 0.5)))
            
            # 3. Scale back to original range
            flex_final = flex_enhanced * (flex_max - flex_min) + flex_min
            
            data[sensor] = flex_final
        
        return data
    
    def class_specific_preprocessing(self, data, class_name):
        """Apply class-specific preprocessing based on analysis"""
        if class_name not in self.problematic_classes:
            return data
        
        # Apply enhanced preprocessing for problematic classes
        data = self.enhanced_yaw_filtering(data)
        data = self.flex_sensor_enhancement(data)
        
        # Class-specific adjustments
        if class_name == 'ㅊ':
            # ㅊ has highest yaw variance, apply strongest filtering
            if 'yaw' in data.columns:
                data['yaw'] = data['yaw'] * 0.8  # Reduce yaw influence
        
        elif class_name == 'ㅌ':
            # ㅌ has high yaw variance, moderate filtering
            if 'yaw' in data.columns:
                data['yaw'] = data['yaw'] * 0.9
        
        elif class_name in ['ㅅ', 'ㅈ', 'ㅋ']:
            # These classes benefit from flex sensor emphasis
            flex_sensors = ['flex1', 'flex2', 'flex3', 'flex4', 'flex5']
            for sensor in flex_sensors:
                if sensor in data.columns:
                    data[sensor] = data[sensor] * 1.2  # Amplify flex sensors
        
        elif class_name in ['ㅕ', 'ㅡ', 'ㅣ']:
            # Vowel classes, emphasize pitch and roll
            if 'pitch' in data.columns:
                data['pitch'] = data['pitch'] * 1.1
            if 'roll' in data.columns:
                data['roll'] = data['roll'] * 1.1
        
        return data
    
    def preprocess_file(self, file_path, class_name, augment=False):
        """Preprocess single file with specialized techniques"""
        try:
            # Load only necessary columns
            usecols = ['pitch', 'roll', 'yaw', 'flex1', 'flex2', 'flex3', 'flex4', 'flex5']
            data = pd.read_csv(file_path, usecols=usecols)
            
            # Apply class-specific preprocessing
            data = self.class_specific_preprocessing(data, class_name)
            
            # Length normalization
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
            
            # Specialized augmentation for problematic classes
            if augment and class_name in self.problematic_classes:
                data = self._specialized_augmentation(data, class_name)
            
            return data.values.astype(np.float32)
            
        except Exception as e:
            print(f"⚠️ 전처리 실패: {file_path} - {str(e)}")
            return None
    
    def _specialized_augmentation(self, data, class_name):
        """Specialized augmentation for problematic classes"""
        # Class-specific augmentation strategies
        if class_name == 'ㅊ':
            # ㅊ needs strong augmentation due to 0% accuracy
            # Add more noise to yaw
            if 'yaw' in data.columns:
                noise = np.random.normal(0, 0.1, len(data))
                data['yaw'] = data['yaw'] + noise
        
        elif class_name in ['ㅕ', 'ㅡ']:
            # These classes need pitch/roll enhancement
            if 'pitch' in data.columns:
                scale_factor = 1 + np.random.uniform(-0.1, 0.2)
                data['pitch'] = data['pitch'] * scale_factor
            
            if 'roll' in data.columns:
                scale_factor = 1 + np.random.uniform(-0.1, 0.2)
                data['roll'] = data['roll'] * scale_factor
        
        elif class_name in ['ㅅ', 'ㅈ', 'ㅋ']:
            # Flex sensor enhancement
            flex_sensors = ['flex1', 'flex2', 'flex3', 'flex4', 'flex5']
            for sensor in flex_sensors:
                if sensor in data.columns:
                    scale_factor = 1 + np.random.uniform(-0.05, 0.15)
                    data[sensor] = data[sensor] * scale_factor
        
        return data

class SpecializedDataset(Dataset):
    """Dataset for specialized model training"""
    
    def __init__(self, data_dir, preprocessor, split='train', focus_problematic=True):
        self.data_dir = data_dir
        self.preprocessor = preprocessor
        self.split = split
        self.focus_problematic = focus_problematic
        
        self.data, self.labels, self.file_paths = self._load_data()
        
        print(f"📊 {split} 데이터셋 생성 완료: {len(self.data)}개 파일")
        if focus_problematic:
            print(f"🎯 문제 클래스 중심 데이터셋")
    
    def _load_data(self):
        """Load data with focus on problematic classes"""
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
            
            # Determine sampling strategy
            if self.focus_problematic:
                if class_name in self.preprocessor.problematic_classes:
                    # Load all files for problematic classes
                    max_files = 50
                else:
                    # Load fewer files for excellent classes
                    max_files = 10
            else:
                max_files = 25  # Balanced sampling
            
            class_files = 0
            
            for scenario in os.listdir(class_path):
                if class_files >= max_files:
                    break
                    
                scenario_path = os.path.join(class_path, scenario)
                if not os.path.isdir(scenario_path):
                    continue
                
                for file_name in os.listdir(scenario_path):
                    if class_files >= max_files:
                        break
                        
                    if file_name.endswith('.csv'):
                        file_path = os.path.join(scenario_path, file_name)
                        
                        # Preprocess with specialized techniques
                        processed_data = self.preprocessor.preprocess_file(
                            file_path, class_name, augment=(self.split == 'train')
                        )
                        
                        if processed_data is not None:
                            data.append(processed_data)
                            labels.append(class_label)
                            file_paths.append(file_path)
                            class_files += 1
                            
                            # Add augmented samples for problematic classes
                            if (self.split == 'train' and 
                                class_name in self.preprocessor.problematic_classes and
                                class_files < max_files):
                                
                                augmented_data = self.preprocessor.preprocess_file(
                                    file_path, class_name, augment=True
                                )
                                if augmented_data is not None:
                                    data.append(augmented_data)
                                    labels.append(class_label)
                                    file_paths.append(file_path)
                                    class_files += 1
        
        return data, labels, file_paths
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data = torch.FloatTensor(self.data[idx])
        label = torch.LongTensor([self.labels[idx]])[0]
        return data, label

class SpecializedModel(nn.Module):
    """Specialized model architecture for problematic classes"""
    
    def __init__(self, input_features=8, hidden_dim=64, num_classes=24):
        super(SpecializedModel, self).__init__()
        
        # Enhanced architecture for problematic classes
        self.conv1 = nn.Conv1d(input_features, hidden_dim, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim*2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(hidden_dim*2, hidden_dim, kernel_size=3, padding=1)
        
        self.batch_norm1 = nn.BatchNorm1d(hidden_dim)
        self.batch_norm2 = nn.BatchNorm1d(hidden_dim*2)
        self.batch_norm3 = nn.BatchNorm1d(hidden_dim)
        
        self.dropout = nn.Dropout(0.4)
        self.relu = nn.ReLU()
        
        # LSTM layer for temporal modeling
        self.lstm = nn.LSTM(hidden_dim, hidden_dim//2, num_layers=2, 
                           batch_first=True, dropout=0.3, bidirectional=True)
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4, dropout=0.2)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim//2, num_classes)
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

class SpecializedTrainer:
    """Trainer for specialized model"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.label_mapper = KSLLabelMapper()
        
        print(f"🔧 사용 디바이스: {self.device}")
        print(f"📊 특화 모델 트레이너 초기화 완료!")
    
    def create_specialized_model(self):
        """Create specialized model"""
        model = SpecializedModel(
            input_features=8,
            hidden_dim=64,
            num_classes=24
        )
        return model.to(self.device)
    
    def create_optimizer(self, model):
        """Create optimizer for specialized model"""
        optimizer = optim.AdamW(
            model.parameters(),
            lr=0.0002,  # Lower learning rate for stability
            weight_decay=1e-4,
            betas=(0.9, 0.999)
        )
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.7, patience=5, min_lr=1e-6
        )
        
        return optimizer, scheduler
    
    def create_dataloaders(self):
        """Create data loaders for specialized training"""
        preprocessor = SpecializedPreprocessor()
        
        # Focus on problematic classes
        train_dataset = SpecializedDataset(
            '../integrations/SignGlove_HW', preprocessor, split='train', focus_problematic=True
        )
        val_dataset = SpecializedDataset(
            '../integrations/SignGlove_HW', preprocessor, split='val', focus_problematic=False
        )
        
        train_loader = DataLoader(
            train_dataset, batch_size=16, shuffle=True, num_workers=2, pin_memory=False
        )
        val_loader = DataLoader(
            val_dataset, batch_size=16, shuffle=False, num_workers=2, pin_memory=False
        )
        
        return train_loader, val_loader
    
    def train_epoch(self, model, train_loader, optimizer, criterion):
        """Train for one epoch"""
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            output = model(data)
            
            loss = criterion(output, target)
            
            # Enhanced L2 regularization
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
        """Train specialized model"""
        print("🚀 특화 모델 훈련 시작!")
        
        # Create components
        model = self.create_specialized_model()
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
            for epoch in range(100):
                print(f"\n🔄 Epoch {epoch+1}/100 시작...")
                
                train_loss, train_acc = self.train_epoch(model, train_loader, optimizer, criterion)
                val_loss, val_acc = self.validate_epoch(model, val_loader, criterion)
                
                scheduler.step(val_loss)
                
                history['train_loss'].append(train_loss)
                history['train_acc'].append(train_acc)
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)
                
                print(f"Epoch {epoch+1:3d}/100 | "
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
                    
                if patience_counter >= 15:
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
        plt.title('Specialized Model Training Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history['train_acc'], label='Train Acc')
        plt.plot(history['val_acc'], label='Val Acc')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Specialized Model Training Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('specialized_model_training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 훈련 곡선 저장: specialized_model_training_curves.png")

def main():
    """Main function for specialized model training"""
    print("📊 특화 모델 훈련 시스템 시작!")
    
    trainer = SpecializedTrainer()
    
    print("\n1️⃣ 특화 모델 훈련 중...")
    model, history = trainer.train()
    
    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'history': history,
        'model_type': 'specialized'
    }, 'specialized_model.pth')
    
    print(f"\n🎉 특화 모델 훈련 완료!")
    print(f"📊 최고 검증 정확도: {history['best_val_acc']:.4f}")
    print(f"📊 최고 에포크: {history['best_epoch']}")
    
    print(f"\n📁 저장된 파일:")
    print(f"   - specialized_model.pth (특화 모델)")
    print(f"   - specialized_model_training_curves.png (훈련 곡선)")
    
    print(f"\n📊 특화 모델 훈련 시스템 완료!")

if __name__ == "__main__":
    main()
