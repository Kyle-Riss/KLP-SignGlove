#!/usr/bin/env python3
"""
Cross-Validation Training Strategy for Sign Language Recognition
- Addresses overfitting issues from stratified training
- K-fold cross-validation for robust evaluation
- Enhanced regularization and data augmentation
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.deep_learning import DeepLearningPipeline
from training.label_mapping import KSLLabelMapper

class CrossValidationPreprocessor:
    """Enhanced preprocessing with stronger regularization"""
    
    def __init__(self):
        self.label_mapper = KSLLabelMapper()
    
    def preprocess_file(self, file_path, class_name, augment=False, augment_strength=0.3):
        """Enhanced preprocessing with stronger augmentation"""
        try:
            # Load only necessary columns
            usecols = ['pitch', 'roll', 'yaw', 'flex1', 'flex2', 'flex3', 'flex4', 'flex5']
            data = pd.read_csv(file_path, usecols=usecols)
            
            # Enhanced yaw correction with stronger filtering
            if 'yaw' in data.columns:
                # Stronger detrending
                yaw_detrended = data['yaw'] - data['yaw'].rolling(window=10, center=True).mean()
                yaw_detrended = yaw_detrended.fillna(method='bfill').fillna(method='ffill')
                # Additional smoothing
                yaw_detrended = yaw_detrended.rolling(window=3, center=True).mean().fillna(method='bfill').fillna(method='ffill')
                data['yaw'] = yaw_detrended
            
            # Optimal length normalization
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
            
            # Enhanced augmentation for cross-validation
            if augment:
                # Multiple augmentation techniques with stronger effects
                augmentations = []
                
                # 1. Stronger Gaussian noise
                noise_std = augment_strength * 1.5 * data.std()
                noise = np.random.normal(0, noise_std, data.shape)
                augmentations.append(data + noise)
                
                # 2. Variable time shift
                shift = np.random.randint(-8, 9)
                if shift != 0:
                    shifted_data = data.shift(shift).fillna(method='bfill').fillna(method='ffill')
                    augmentations.append(shifted_data)
                
                # 3. Variable scaling
                scale_factor = 1 + np.random.uniform(-augment_strength*2, augment_strength*2)
                augmentations.append(data * scale_factor)
                
                # 4. Random masking with variable intensity
                mask_prob = np.random.uniform(0.05, 0.15)
                mask = np.random.random(data.shape) > mask_prob
                masked_data = data * mask
                augmentations.append(masked_data)
                
                # 5. Variable smoothing
                window_size = np.random.randint(3, 8)
                smoothed_data = data.rolling(window=window_size, center=True).mean().fillna(method='bfill').fillna(method='ffill')
                augmentations.append(smoothed_data)
                
                # 6. Enhanced jittering
                jitter_std = augment_strength * 0.2 * data.std()
                jitter = np.random.normal(0, jitter_std, data.shape)
                augmentations.append(data + jitter)
                
                # 7. Time warping with variable factor
                warp_factor = 1 + np.random.uniform(-0.15, 0.15)
                warped_indices = np.arange(len(data)) * warp_factor
                warped_indices = np.clip(warped_indices, 0, len(data)-1)
                warped_data = data.iloc[warped_indices.astype(int)]
                augmentations.append(warped_data)
                
                # 8. Random rotation (new technique)
                rotation_angle = np.random.uniform(-0.1, 0.1)
                rotation_matrix = np.array([
                    [np.cos(rotation_angle), -np.sin(rotation_angle)],
                    [np.sin(rotation_angle), np.cos(rotation_angle)]
                ])
                # Apply to pitch and roll
                if 'pitch' in data.columns and 'roll' in data.columns:
                    pitch_roll = data[['pitch', 'roll']].values
                    rotated = np.dot(pitch_roll, rotation_matrix.T)
                    rotated_data = data.copy()
                    rotated_data['pitch'] = rotated[:, 0]
                    rotated_data['roll'] = rotated[:, 1]
                    augmentations.append(rotated_data)
                
                # 9. Frequency domain augmentation (new technique)
                # Apply FFT, add noise in frequency domain, then inverse FFT
                fft_data = np.fft.fft(data.values, axis=0)
                freq_noise = np.random.normal(0, augment_strength * 0.1, fft_data.shape)
                fft_data += freq_noise
                freq_augmented = np.real(np.fft.ifft(fft_data, axis=0))
                augmentations.append(pd.DataFrame(freq_augmented, columns=data.columns))
                
                # Randomly select one augmentation
                data = augmentations[np.random.randint(0, len(augmentations))]
            
            # Convert to float32 for memory efficiency
            processed_data = data.values.astype(np.float32)
            
            return processed_data
            
        except Exception as e:
            print(f"⚠️ 전처리 실패: {file_path} - {str(e)}")
            return None

class CrossValidationDataset(Dataset):
    """Cross-validation dataset with fold-based splitting"""
    
    def __init__(self, data_dir, preprocessor, fold_idx=None, n_folds=5, split='train', augment=False):
        self.data_dir = data_dir
        self.preprocessor = preprocessor
        self.fold_idx = fold_idx
        self.n_folds = n_folds
        self.split = split
        self.augment = augment
        
        self.data, self.labels, self.file_paths = self._load_all_files()
        self._cross_validation_split()
        
        print(f"📊 Fold {fold_idx} {self.split} 데이터: {len(self.data)}개 파일")
    
    def _load_all_files(self):
        """Load all files with their labels"""
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
                        
                        # Load original data
                        processed_data = self.preprocessor.preprocess_file(file_path, class_name, augment=False)
                        if processed_data is not None:
                            data.append(processed_data)
                            labels.append(class_label)
                            file_paths.append(file_path)
        
        print(f"📊 로드된 총 파일: {len(data)}개")
        return data, labels, file_paths
    
    def _cross_validation_split(self):
        """Cross-validation split"""
        if len(self.data) == 0:
            print("⚠️ 데이터가 없습니다!")
            return
        
        # Convert to numpy arrays
        data_array = np.array(self.data)
        labels_array = np.array(self.labels)
        file_paths_array = np.array(self.file_paths)
        
        # Create stratified K-fold split
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        
        # Get fold indices
        fold_indices = list(skf.split(data_array, labels_array))
        
        if self.fold_idx is None:
            # Use all data for final evaluation
            self.data = [data_array[i] for i in range(len(data_array))]
            self.labels = [labels_array[i] for i in range(len(labels_array))]
            self.file_paths = [file_paths_array[i] for i in range(len(file_paths_array))]
        else:
            # Get train and validation indices for this fold
            train_idx, val_idx = fold_indices[self.fold_idx]
            
            if self.split == 'train':
                selected_idx = train_idx
            elif self.split == 'val':
                selected_idx = val_idx
            
            # Filter data
            self.data = [data_array[i] for i in selected_idx]
            self.labels = [labels_array[i] for i in selected_idx]
            self.file_paths = [file_paths_array[i] for i in selected_idx]
        
        print(f"📊 {self.split} 세트 분할 완료: {len(self.data)}개 파일")
        
        # Analyze class distribution
        class_counts = defaultdict(int)
        for label in self.labels:
            class_name = self.preprocessor.label_mapper.get_class_name(label)
            class_counts[class_name] += 1
        
        print(f"📊 {self.split} 세트 클래스 분포:")
        for class_name in sorted(class_counts.keys()):
            print(f"  {class_name}: {class_counts[class_name]}개 파일")
        
        # Add cross-validation augmentation for training
        if self.split == 'train' and self.augment:
            self._add_cv_augmentation()
    
    def _add_cv_augmentation(self):
        """Add cross-validation specific augmentation"""
        print("🔄 교차 검증 데이터 증강 중...")
        
        # Count current samples per class
        class_counts = defaultdict(int)
        for label in self.labels:
            class_counts[label] += 1
        
        # Find target count (median of class counts)
        target_count = int(np.median(list(class_counts.values())))
        print(f"📊 목표 샘플 수/클래스: {target_count}개")
        
        # Add augmented samples for classes with fewer samples
        original_data = self.data.copy()
        original_labels = self.labels.copy()
        original_files = self.file_paths.copy()
        
        for label in range(24):  # 24 classes
            current_count = class_counts[label]
            if current_count < target_count:
                needed = target_count - current_count
                print(f"  클래스 {self.preprocessor.label_mapper.get_class_name(label)}: {current_count} → {target_count} (+{needed})")
                
                # Find files for this class
                class_files = [f for i, f in enumerate(original_files) if original_labels[i] == label]
                
                # Add augmented samples with stronger augmentation
                for _ in range(needed):
                    # Randomly select a file from this class
                    selected_file = np.random.choice(class_files)
                    class_name = self.preprocessor.label_mapper.get_class_name(label)
                    
                    # Create augmented version with stronger augmentation
                    augmented_data = self.preprocessor.preprocess_file(
                        selected_file, class_name, augment=True, augment_strength=0.4
                    )
                    
                    if augmented_data is not None:
                        self.data.append(augmented_data)
                        self.labels.append(label)
                        self.file_paths.append(selected_file)
        
        print(f"📊 증강 후 총 파일 수: {len(self.data)}개")
        
        # Verify balanced distribution
        final_counts = defaultdict(int)
        for label in self.labels:
            final_counts[label] += 1
        
        print(f"📊 최종 클래스 분포:")
        for label in range(24):
            class_name = self.preprocessor.label_mapper.get_class_name(label)
            count = final_counts[label]
            print(f"  {class_name}: {count}개 파일")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data = torch.FloatTensor(self.data[idx])
        label = torch.LongTensor([self.labels[idx]])[0]
        return data, label

class CrossValidationTrainer:
    """Cross-validation trainer with enhanced regularization"""
    
    def __init__(self, n_folds=5):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.label_mapper = KSLLabelMapper()
        self.n_folds = n_folds
        
        print(f"🔧 사용 디바이스: {self.device}")
        print(f"📊 교차 검증 폴드 수: {n_folds}")
    
    def create_regularized_model(self):
        """Create a model with enhanced regularization"""
        model = DeepLearningPipeline(
            input_features=8,
            hidden_dim=48,  # Reduced from 64
            num_layers=1,   # Reduced from 2
            num_classes=24,
            dropout=0.5     # Increased from 0.3
        )
        return model.to(self.device)
    
    def create_optimizer(self, model):
        """Create optimizer with enhanced regularization"""
        optimizer = optim.AdamW(
            model.parameters(),
            lr=0.0003,      # Reduced from 0.0005
            weight_decay=1e-3,  # Increased from 1e-4
            betas=(0.9, 0.999)
        )
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.6, patience=3, min_lr=1e-6
        )
        
        return optimizer, scheduler
    
    def create_dataloaders(self, fold_idx):
        """Create data loaders for cross-validation"""
        preprocessor = CrossValidationPreprocessor()
        
        train_dataset = CrossValidationDataset('../integrations/SignGlove_HW', preprocessor, fold_idx, self.n_folds, split='train', augment=True)
        val_dataset = CrossValidationDataset('../integrations/SignGlove_HW', preprocessor, fold_idx, self.n_folds, split='val', augment=False)
        
        train_loader = DataLoader(
            train_dataset, batch_size=12, shuffle=True, num_workers=2, pin_memory=False  # Reduced batch size
        )
        val_loader = DataLoader(
            val_dataset, batch_size=12, shuffle=False, num_workers=2, pin_memory=False
        )
        
        return train_loader, val_loader
    
    def train_epoch(self, model, train_loader, optimizer, criterion):
        """Train for one epoch with enhanced regularization"""
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            output = model(data)
            
            if isinstance(output, dict):
                loss = criterion(output['class_logits'], target)
            else:
                loss = criterion(output, target)
            
            # Enhanced L2 regularization
            l2_lambda = 2e-3  # Increased from 1e-4
            l2_reg = torch.tensor(0.).to(self.device)
            for param in model.parameters():
                l2_reg += torch.norm(param)
            loss += l2_lambda * l2_reg
            
            loss.backward()
            
            # Enhanced gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)  # Reduced from 1.0
            
            optimizer.step()
            
            total_loss += loss.item()
            
            if isinstance(output, dict):
                pred = output['class_logits'].argmax(dim=1)
            else:
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
                
                if isinstance(output, dict):
                    loss = criterion(output['class_logits'], target)
                    pred = output['class_logits'].argmax(dim=1)
                else:
                    loss = criterion(output, target)
                    pred = output.argmax(dim=1)
                
                total_loss += loss.item()
                correct += pred.eq(target).sum().item()
                total += target.size(0)
        
        return total_loss / len(val_loader), correct / total
    
    def train_fold(self, fold_idx):
        """Train one fold"""
        print(f"\n🔄 Fold {fold_idx + 1}/{self.n_folds} 학습 시작!")
        
        # Create components
        model = self.create_regularized_model()
        optimizer, scheduler = self.create_optimizer(model)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.2)  # Increased from 0.1
        train_loader, val_loader = self.create_dataloaders(fold_idx)
        
        history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'best_val_acc': 0, 'best_epoch': 0,
            'underfitting_detected': False,
            'overfitting_detected': False
        }
        
        patience_counter = 0
        best_model_state = None
        
        print(f"📊 학습 데이터: {len(train_loader.dataset)}개")
        print(f"📊 검증 데이터: {len(val_loader.dataset)}개")
        print(f"🔧 모델 파라미터: {sum(p.numel() for p in model.parameters()):,}개")
        
        try:
            for epoch in range(50):  # Reduced from 100
                print(f"\n🔄 Epoch {epoch+1}/50 시작...")
                
                train_loss, train_acc = self.train_epoch(model, train_loader, optimizer, criterion)
                val_loss, val_acc = self.validate_epoch(model, val_loader, criterion)
                
                scheduler.step(val_loss)
                
                history['train_loss'].append(train_loss)
                history['train_acc'].append(train_acc)
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)
                
                print(f"Epoch {epoch+1:3d}/50 | "
                      f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                      f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
                
                # Enhanced underfitting detection
                if epoch > 15 and train_acc < 0.4:  # Increased threshold
                    print(f"⚠️ Underfitting 감지: Train Acc = {train_acc:.4f}")
                    history['underfitting_detected'] = True
                
                # Enhanced overfitting detection
                if epoch > 15:
                    train_val_gap = train_acc - val_acc
                    if train_val_gap > 0.12:  # Reduced threshold
                        print(f"⚠️ 과적합 감지: Train-Val Gap = {train_val_gap:.4f}")
                        history['overfitting_detected'] = True
                
                if val_acc > history['best_val_acc']:
                    history['best_val_acc'] = val_acc
                    history['best_epoch'] = epoch
                    patience_counter = 0
                    best_model_state = model.state_dict().copy()
                    print(f"🎯 새로운 최고 정확도: {val_acc:.4f}")
                else:
                    patience_counter += 1
                    
                if patience_counter >= 10:  # Reduced from 15
                    print(f"🛑 조기 종료: {epoch+1} 에포크에서 중단")
                    break
        
        except KeyboardInterrupt:
            print("\n⚠️ 사용자에 의해 중단됨")
        except Exception as e:
            print(f"\n❌ 오류 발생: {str(e)}")
        
        # Load best model
        if best_model_state:
            model.load_state_dict(best_model_state)
        
        return {
            'model': model,
            'history': history,
            'val_accuracy': history['best_val_acc']
        }
    
    def cross_validate(self):
        """Perform cross-validation"""
        print("📊 교차 검증 시스템 시작!")
        
        fold_results = []
        
        for fold_idx in range(self.n_folds):
            fold_result = self.train_fold(fold_idx)
            fold_results.append(fold_result)
            
            print(f"\n🎯 Fold {fold_idx + 1} 결과:")
            print(f"   검증 정확도: {fold_result['val_accuracy']:.4f}")
            print(f"   최고 에포크: {fold_result['history']['best_epoch']}")
            print(f"   Underfitting: {fold_result['history']['underfitting_detected']}")
            print(f"   Overfitting: {fold_result['history']['overfitting_detected']}")
        
        # Calculate cross-validation statistics
        val_accuracies = [result['val_accuracy'] for result in fold_results]
        mean_cv_acc = np.mean(val_accuracies)
        std_cv_acc = np.std(val_accuracies)
        
        print(f"\n📊 교차 검증 결과:")
        print(f"   평균 검증 정확도: {mean_cv_acc:.4f} ± {std_cv_acc:.4f}")
        print(f"   최고 검증 정확도: {max(val_accuracies):.4f}")
        print(f"   최저 검증 정확도: {min(val_accuracies):.4f}")
        
        # Save cross-validation results
        cv_results = {
            'n_folds': self.n_folds,
            'mean_validation_accuracy': mean_cv_acc,
            'std_validation_accuracy': std_cv_acc,
            'max_validation_accuracy': max(val_accuracies),
            'min_validation_accuracy': min(val_accuracies),
            'fold_results': [
                {
                    'fold_idx': i,
                    'val_accuracy': result['val_accuracy'],
                    'best_epoch': result['history']['best_epoch'],
                    'underfitting_detected': result['history']['underfitting_detected'],
                    'overfitting_detected': result['history']['overfitting_detected']
                }
                for i, result in enumerate(fold_results)
            ]
        }
        
        with open('cross_validation_results.json', 'w', encoding='utf-8') as f:
            json.dump(cv_results, f, ensure_ascii=False, indent=2)
        
        # Save best model from best fold
        best_fold_idx = np.argmax(val_accuracies)
        best_model = fold_results[best_fold_idx]['model']
        
        torch.save({
            'model_state_dict': best_model.state_dict(),
            'cv_results': cv_results,
            'best_fold_idx': best_fold_idx
        }, 'cross_validation_model.pth')
        
        # Create cross-validation visualization
        self._save_cv_visualization(fold_results, cv_results)
        
        return cv_results, best_model
    
    def _save_cv_visualization(self, fold_results, cv_results):
        """Save cross-validation visualization"""
        plt.figure(figsize=(15, 10))
        
        # 1. Validation accuracy by fold
        plt.subplot(2, 3, 1)
        fold_indices = range(1, self.n_folds + 1)
        val_accuracies = [result['val_accuracy'] for result in fold_results]
        plt.bar(fold_indices, val_accuracies, alpha=0.7, color='skyblue')
        plt.axhline(y=cv_results['mean_validation_accuracy'], color='red', linestyle='--', label='Mean')
        plt.xlabel('Fold')
        plt.ylabel('Validation Accuracy')
        plt.title('Validation Accuracy by Fold')
        plt.legend()
        
        # 2. Training curves for best fold
        plt.subplot(2, 3, 2)
        best_fold_idx = cv_results['best_fold_idx']
        best_history = fold_results[best_fold_idx]['history']
        epochs = range(1, len(best_history['train_loss']) + 1)
        plt.plot(epochs, best_history['train_loss'], label='Train Loss')
        plt.plot(epochs, best_history['val_loss'], label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Training Curves (Best Fold {best_fold_idx + 1})')
        plt.legend()
        
        # 3. Accuracy curves for best fold
        plt.subplot(2, 3, 3)
        plt.plot(epochs, best_history['train_acc'], label='Train Acc')
        plt.plot(epochs, best_history['val_acc'], label='Val Acc')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title(f'Accuracy Curves (Best Fold {best_fold_idx + 1})')
        plt.legend()
        
        # 4. Train-Val Gap analysis
        plt.subplot(2, 3, 4)
        gaps = [t - v for t, v in zip(best_history['train_acc'], best_history['val_acc'])]
        plt.plot(epochs, gaps, color='red', label='Train-Val Gap')
        plt.axhline(y=0.12, color='orange', linestyle='--', label='Overfitting Threshold')
        plt.axhline(y=-0.05, color='green', linestyle='--', label='Underfitting Threshold')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy Gap')
        plt.title('Fitting Detection')
        plt.legend()
        
        # 5. Cross-validation summary
        plt.subplot(2, 3, 5)
        summary_text = f"""
Cross-Validation Summary:
• Folds: {self.n_folds}
• Mean Accuracy: {cv_results['mean_validation_accuracy']:.4f}
• Std Accuracy: {cv_results['std_validation_accuracy']:.4f}
• Best Fold: {best_fold_idx + 1}
• Best Accuracy: {cv_results['max_validation_accuracy']:.4f}
        """
        plt.text(0.1, 0.5, summary_text, fontsize=10, verticalalignment='center')
        plt.axis('off')
        plt.title('CV Summary')
        
        # 6. Overfitting/Underfitting by fold
        plt.subplot(2, 3, 6)
        overfitting_count = sum(1 for result in fold_results if result['history']['overfitting_detected'])
        underfitting_count = sum(1 for result in fold_results if result['history']['underfitting_detected'])
        normal_count = self.n_folds - overfitting_count - underfitting_count
        
        labels = ['Normal', 'Overfitting', 'Underfitting']
        sizes = [normal_count, overfitting_count, underfitting_count]
        colors = ['lightgreen', 'orange', 'red']
        
        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        plt.title('Fitting Issues by Fold')
        
        plt.tight_layout()
        plt.savefig('cross_validation_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\n📊 시각화 저장: cross_validation_analysis.png")

def main():
    """Main function for cross-validation training"""
    print("📊 교차 검증 학습 시스템 시작!")
    
    trainer = CrossValidationTrainer(n_folds=5)
    
    print("\n1️⃣ 교차 검증 수행 중...")
    cv_results, best_model = trainer.cross_validate()
    
    print(f"\n🎉 교차 검증 완료!")
    print(f"📊 평균 검증 정확도: {cv_results['mean_validation_accuracy']:.4f} ± {cv_results['std_validation_accuracy']:.4f}")
    print(f"📊 최고 검증 정확도: {cv_results['max_validation_accuracy']:.4f}")
    print(f"📊 최저 검증 정확도: {cv_results['min_validation_accuracy']:.4f}")
    print(f"🏆 최고 폴드: {cv_results['best_fold_idx'] + 1}")
    
    # Analyze overfitting/underfitting across folds
    overfitting_folds = sum(1 for result in cv_results['fold_results'] if result['overfitting_detected'])
    underfitting_folds = sum(1 for result in cv_results['fold_results'] if result['underfitting_detected'])
    
    print(f"\n⚠️ 과적합 폴드: {overfitting_folds}/{cv_results['n_folds']}")
    print(f"⚠️ 과소적합 폴드: {underfitting_folds}/{cv_results['n_folds']}")
    
    if overfitting_folds == 0 and underfitting_folds == 0:
        print(f"✅ 모든 폴드에서 적절한 학습 완료!")
    elif overfitting_folds > underfitting_folds:
        print(f"⚠️ 과적합 경향 - 더 강한 정규화 필요")
    else:
        print(f"⚠️ 과소적합 경향 - 모델 복잡도 증가 필요")
    
    print(f"\n📁 저장된 파일:")
    print(f"   - cross_validation_model.pth (최고 모델)")
    print(f"   - cross_validation_results.json (CV 결과)")
    print(f"   - cross_validation_analysis.png (CV 분석)")
    
    print(f"\n📊 교차 검증 학습 시스템 완료!")

if __name__ == "__main__":
    main()
