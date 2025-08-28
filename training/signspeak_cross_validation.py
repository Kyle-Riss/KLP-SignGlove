#!/usr/bin/env python3
"""
SignSpeak Style Cross-Validation Training
- SignSpeakLSTM, SignSpeakGRU, SignSpeakTransformer 개별 훈련
- K-Fold 교차 검증으로 성능 측정
- SignSpeak의 92% 정확도에 도전
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
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

sys.path.append('.')
from models.signspeak_style_models import SignSpeakLSTM, SignSpeakGRU, SignSpeakTransformer
from training.label_mapping import KSLLabelMapper


class SignSpeakPreprocessor:
    """SignSpeak 스타일 전처리"""
    
    def __init__(self):
        self.label_mapper = KSLLabelMapper()
    
    def preprocess_file(self, file_path, class_name, augment=False, augment_strength=0.3):
        """SignSpeak 스타일 전처리"""
        try:
            # Load only necessary columns (IMU 3개 + Flex 5개)
            usecols = ['pitch', 'roll', 'yaw', 'flex1', 'flex2', 'flex3', 'flex4', 'flex5']
            data = pd.read_csv(file_path, usecols=usecols)
            
            # SignSpeak 스타일 정규화
            # 1. Yaw correction (SignSpeak 스타일)
            if 'yaw' in data.columns:
                yaw_detrended = data['yaw'] - data['yaw'].rolling(window=15, center=True).mean()
                yaw_detrended = yaw_detrended.fillna(method='bfill').fillna(method='ffill')
                data['yaw'] = yaw_detrended
            
            # 2. Length normalization to 200 (SignSpeak 스타일)
            target_length = 200
            if len(data) != target_length:
                if len(data) < target_length:
                    # Interpolation padding
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
            
            # 3. SignSpeak 스타일 증강
            if augment:
                augmentations = []
                
                # Gaussian noise (SignSpeak 스타일)
                noise_std = augment_strength * data.std()
                noise = np.random.normal(0, noise_std, data.shape)
                augmentations.append(data + noise)
                
                # Time shift (SignSpeak 스타일)
                shift = np.random.randint(-10, 11)
                if shift != 0:
                    shifted_data = data.shift(shift).fillna(method='bfill').fillna(method='ffill')
                    augmentations.append(shifted_data)
                
                # Scaling (SignSpeak 스타일)
                scale_factor = 1 + np.random.uniform(-augment_strength, augment_strength)
                augmentations.append(data * scale_factor)
                
                # Random selection
                data = augmentations[np.random.randint(0, len(augmentations))]
            
            # Convert to float32
            processed_data = data.values.astype(np.float32)
            
            return processed_data
            
        except Exception as e:
            print(f"⚠️ 전처리 실패: {file_path} - {str(e)}")
            return None


class SignSpeakDataset(Dataset):
    """SignSpeak 스타일 데이터셋"""
    
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
        """모든 파일 로드"""
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
                        
                        processed_data = self.preprocessor.preprocess_file(file_path, class_name, augment=False)
                        if processed_data is not None:
                            data.append(processed_data)
                            labels.append(class_label)
                            file_paths.append(file_path)
        
        print(f"📊 로드된 총 파일: {len(data)}개")
        return data, labels, file_paths
    
    def _cross_validation_split(self):
        """교차 검증 분할"""
        if len(self.data) == 0:
            print("⚠️ 데이터가 없습니다!")
            return
        
        data_array = np.array(self.data)
        labels_array = np.array(self.labels)
        file_paths_array = np.array(self.file_paths)
        
        # Stratified K-fold split
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        fold_indices = list(skf.split(data_array, labels_array))
        
        if self.fold_idx is None:
            self.data = [data_array[i] for i in range(len(data_array))]
            self.labels = [labels_array[i] for i in range(len(labels_array))]
            self.file_paths = [file_paths_array[i] for i in range(len(file_paths_array))]
        else:
            train_idx, val_idx = fold_indices[self.fold_idx]
            
            if self.split == 'train':
                selected_idx = train_idx
            elif self.split == 'val':
                selected_idx = val_idx
            
            self.data = [data_array[i] for i in selected_idx]
            self.labels = [labels_array[i] for i in selected_idx]
            self.file_paths = [file_paths_array[i] for i in selected_idx]
        
        print(f"📊 {self.split} 세트 분할 완료: {len(self.data)}개 파일")
        
        # 클래스 분포 분석
        class_counts = defaultdict(int)
        for label in self.labels:
            class_name = self.preprocessor.label_mapper.get_class_name(label)
            class_counts[class_name] += 1
        
        print(f"📊 {self.split} 세트 클래스 분포:")
        for class_name in sorted(class_counts.keys()):
            print(f"  {class_name}: {class_counts[class_name]}개 파일")
        
        # 훈련 시 증강
        if self.split == 'train' and self.augment:
            self._add_signspeak_augmentation()
    
    def _add_signspeak_augmentation(self):
        """SignSpeak 스타일 증강"""
        print("🔄 SignSpeak 스타일 데이터 증강 중...")
        
        class_counts = defaultdict(int)
        for label in self.labels:
            class_counts[label] += 1
        
        target_count = int(np.median(list(class_counts.values())))
        print(f"📊 목표 샘플 수/클래스: {target_count}개")
        
        original_data = self.data.copy()
        original_labels = self.labels.copy()
        original_files = self.file_paths.copy()
        
        for label in range(24):
            current_count = class_counts[label]
            if current_count < target_count:
                needed = target_count - current_count
                print(f"  클래스 {self.preprocessor.label_mapper.get_class_name(label)}: {current_count} → {target_count} (+{needed})")
                
                class_files = [f for i, f in enumerate(original_files) if original_labels[i] == label]
                
                for _ in range(needed):
                    selected_file = np.random.choice(class_files)
                    class_name = self.preprocessor.label_mapper.get_class_name(label)
                    
                    augmented_data = self.preprocessor.preprocess_file(
                        selected_file, class_name, augment=True, augment_strength=0.3
                    )
                    
                    if augmented_data is not None:
                        self.data.append(augmented_data)
                        self.labels.append(label)
                        self.file_paths.append(selected_file)
        
        print(f"📊 증강 후 총 파일 수: {len(self.data)}개")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data = torch.FloatTensor(self.data[idx])
        label = torch.LongTensor([self.labels[idx]])[0]
        return data, label


class SignSpeakTrainer:
    """SignSpeak 스타일 모델 훈련기"""
    
    def __init__(self, model_type='lstm', n_folds=5):
        self.model_type = model_type
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.label_mapper = KSLLabelMapper()
        self.n_folds = n_folds
        
        print(f"🚀 SignSpeak {model_type.upper()} 훈련기 초기화")
        print(f"🔧 사용 디바이스: {self.device}")
        print(f"📊 교차 검증 폴드 수: {n_folds}")
    
    def create_model(self):
        """SignSpeak 스타일 모델 생성"""
        if self.model_type == 'lstm':
            model = SignSpeakLSTM(
                input_size=8,
                hidden_size=64,
                classes=24,
                num_layers=3,
                dropout=0.2,
                bidirectional=True
            )
        elif self.model_type == 'gru':
            model = SignSpeakGRU(
                input_size=8,
                hidden_size=64,
                classes=24,
                num_layers=3,
                dropout=0.2,
                bidirectional=True
            )
        elif self.model_type == 'transformer':
            model = SignSpeakTransformer(
                input_size=8,
                hidden_size=64,
                classes=24,
                num_layers=6,
                num_heads=8,
                dropout=0.2,
                dim_feedforward=256
            )
        else:
            raise ValueError(f"지원하지 않는 모델 타입: {self.model_type}")
        
        return model.to(self.device)
    
    def create_optimizer(self, model):
        """SignSpeak 스타일 옵티마이저"""
        optimizer = optim.AdamW(
            model.parameters(),
            lr=0.0005,
            weight_decay=1e-4,
            betas=(0.9, 0.999)
        )
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.7, patience=5, min_lr=1e-6
        )
        
        return optimizer, scheduler
    
    def create_dataloaders(self, fold_idx):
        """데이터 로더 생성"""
        preprocessor = SignSpeakPreprocessor()
        
        train_dataset = SignSpeakDataset(
            '../integrations/SignGlove_HW', 
            preprocessor, 
            fold_idx, 
            self.n_folds, 
            split='train', 
            augment=True
        )
        val_dataset = SignSpeakDataset(
            '../integrations/SignGlove_HW', 
            preprocessor, 
            fold_idx, 
            self.n_folds, 
            split='val', 
            augment=False
        )
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=16, 
            shuffle=True, 
            num_workers=2, 
            pin_memory=False
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=16, 
            shuffle=False, 
            num_workers=2, 
            pin_memory=False
        )
        
        return train_loader, val_loader
    
    def train_epoch(self, model, train_loader, optimizer, criterion):
        """한 에포크 훈련"""
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            logits, loss = model(data, target)
            
            if loss is not None:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()
            
            pred = logits.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
        
        return total_loss / len(train_loader), correct / total
    
    def validate_epoch(self, model, val_loader, criterion):
        """한 에포크 검증"""
        model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                logits, loss = model(data, target)
                
                if loss is not None:
                    total_loss += loss.item()
                
                pred = logits.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
        
        return total_loss / len(val_loader), correct / total
    
    def train_fold(self, fold_idx):
        """한 폴드 훈련"""
        print(f"\n🔄 Fold {fold_idx + 1}/{self.n_folds} 학습 시작!")
        
        model = self.create_model()
        optimizer, scheduler = self.create_optimizer(model)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        train_loader, val_loader = self.create_dataloaders(fold_idx)
        
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
                if (epoch + 1) % 10 == 0:
                    print(f"\n🔄 Epoch {epoch+1}/100 시작...")
                
                train_loss, train_acc = self.train_epoch(model, train_loader, optimizer, criterion)
                val_loss, val_acc = self.validate_epoch(model, val_loader, criterion)
                
                scheduler.step(val_acc)
                
                history['train_loss'].append(train_loss)
                history['train_acc'].append(train_acc)
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)
                
                if (epoch + 1) % 10 == 0:
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
        
        if best_model_state:
            model.load_state_dict(best_model_state)
        
        return {
            'model': model,
            'history': history,
            'val_accuracy': history['best_val_acc']
        }
    
    def cross_validate(self):
        """교차 검증 수행"""
        print(f"📊 SignSpeak {self.model_type.upper()} 교차 검증 시작!")
        
        fold_results = []
        
        for fold_idx in range(self.n_folds):
            fold_result = self.train_fold(fold_idx)
            fold_results.append(fold_result)
            
            print(f"\n🎯 Fold {fold_idx + 1} 결과:")
            print(f"   검증 정확도: {fold_result['val_accuracy']:.4f}")
            print(f"   최고 에포크: {fold_result['history']['best_epoch']}")
        
        # 교차 검증 통계 계산
        val_accuracies = [result['val_accuracy'] for result in fold_results]
        mean_cv_acc = np.mean(val_accuracies)
        std_cv_acc = np.std(val_accuracies)
        
        print(f"\n📊 SignSpeak {self.model_type.upper()} 교차 검증 결과:")
        print(f"   평균 검증 정확도: {mean_cv_acc:.4f} ± {std_cv_acc:.4f}")
        print(f"   최고 검증 정확도: {max(val_accuracies):.4f}")
        print(f"   최저 검증 정확도: {min(val_accuracies):.4f}")
        
        # SignSpeak 92% 정확도와 비교
        if max(val_accuracies) >= 0.92:
            print(f"🎉 SignSpeak 92% 정확도 달성! 최고 정확도: {max(val_accuracies):.4f}")
        else:
            print(f"📈 SignSpeak 92% 정확도까지: {0.92 - max(val_accuracies):.4f} 부족")
        
        # 결과 저장
        cv_results = {
            'model_type': self.model_type,
            'n_folds': self.n_folds,
            'mean_validation_accuracy': mean_cv_acc,
            'std_validation_accuracy': std_cv_acc,
            'max_validation_accuracy': max(val_accuracies),
            'min_validation_accuracy': min(val_accuracies),
            'fold_results': [
                {
                    'fold_idx': i,
                    'val_accuracy': result['val_accuracy'],
                    'best_epoch': result['history']['best_epoch']
                }
                for i, result in enumerate(fold_results)
            ]
        }
        
        # 파일 저장
        model_save_path = f'signspeak_{self.model_type}_cv_model.pth'
        results_save_path = f'signspeak_{self.model_type}_cv_results.json'
        
        best_fold_idx = np.argmax(val_accuracies)
        best_model = fold_results[best_fold_idx]['model']
        
        torch.save({
            'model_state_dict': best_model.state_dict(),
            'cv_results': cv_results,
            'best_fold_idx': best_fold_idx,
            'model_type': self.model_type
        }, model_save_path)
        
        with open(results_save_path, 'w', encoding='utf-8') as f:
            json.dump(cv_results, f, ensure_ascii=False, indent=2)
        
        # 시각화 저장
        self._save_visualization(fold_results, cv_results)
        
        return cv_results, best_model
    
    def _save_visualization(self, fold_results, cv_results):
        """시각화 저장"""
        plt.figure(figsize=(15, 10))
        
        # 1. Validation accuracy by fold
        plt.subplot(2, 3, 1)
        fold_indices = range(1, self.n_folds + 1)
        val_accuracies = [result['val_accuracy'] for result in fold_results]
        plt.bar(fold_indices, val_accuracies, alpha=0.7, color='skyblue')
        plt.axhline(y=cv_results['mean_validation_accuracy'], color='red', linestyle='--', label='Mean')
        plt.axhline(y=0.92, color='green', linestyle='--', label='SignSpeak Target')
        plt.xlabel('Fold')
        plt.ylabel('Validation Accuracy')
        plt.title(f'SignSpeak {self.model_type.upper()} Validation Accuracy')
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
        plt.axhline(y=0.92, color='green', linestyle='--', label='SignSpeak Target')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title(f'Accuracy Curves (Best Fold {best_fold_idx + 1})')
        plt.legend()
        
        # 4. Cross-validation summary
        plt.subplot(2, 3, 4)
        summary_text = f"""
SignSpeak {self.model_type.upper()} Summary:
• Folds: {self.n_folds}
• Mean Accuracy: {cv_results['mean_validation_accuracy']:.4f}
• Std Accuracy: {cv_results['std_validation_accuracy']:.4f}
• Best Fold: {best_fold_idx + 1}
• Best Accuracy: {cv_results['max_validation_accuracy']:.4f}
• SignSpeak Target: 0.9200
• Gap to Target: {0.92 - cv_results['max_validation_accuracy']:.4f}
        """
        plt.text(0.1, 0.5, summary_text, fontsize=10, verticalalignment='center')
        plt.axis('off')
        plt.title('CV Summary')
        
        # 5. Accuracy distribution
        plt.subplot(2, 3, 5)
        plt.hist(val_accuracies, bins=10, alpha=0.7, color='lightblue', edgecolor='black')
        plt.axvline(x=cv_results['mean_validation_accuracy'], color='red', linestyle='--', label='Mean')
        plt.axvline(x=0.92, color='green', linestyle='--', label='SignSpeak Target')
        plt.xlabel('Validation Accuracy')
        plt.ylabel('Frequency')
        plt.title('Accuracy Distribution')
        plt.legend()
        
        # 6. Model comparison placeholder
        plt.subplot(2, 3, 6)
        plt.text(0.5, 0.5, f'SignSpeak {self.model_type.upper()}\nCV Analysis Complete', 
                ha='center', va='center', fontsize=14, fontweight='bold')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(f'signspeak_{self.model_type}_cv_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\n📊 시각화 저장: signspeak_{self.model_type}_cv_analysis.png")


def main():
    """메인 함수"""
    print("🚀 SignSpeak 스타일 모델 교차 검증 시스템 시작!")
    
    # 3가지 모델 타입
    model_types = ['lstm', 'gru', 'transformer']
    
    all_results = {}
    
    for model_type in model_types:
        print(f"\n{'='*60}")
        print(f"🎯 SignSpeak {model_type.upper()} 훈련 시작")
        print(f"{'='*60}")
        
        trainer = SignSpeakTrainer(model_type=model_type, n_folds=5)
        cv_results, best_model = trainer.cross_validate()
        
        all_results[model_type] = cv_results
        
        print(f"\n🎉 SignSpeak {model_type.upper()} 훈련 완료!")
        print(f"📊 평균 검증 정확도: {cv_results['mean_validation_accuracy']:.4f} ± {cv_results['std_validation_accuracy']:.4f}")
        print(f"📊 최고 검증 정확도: {cv_results['max_validation_accuracy']:.4f}")
        
        if cv_results['max_validation_accuracy'] >= 0.92:
            print(f"🎉 SignSpeak 92% 정확도 달성!")
        else:
            gap = 0.92 - cv_results['max_validation_accuracy']
            print(f"📈 SignSpeak 92% 정확도까지 {gap:.4f} 부족")
    
    # 전체 결과 요약
    print(f"\n{'='*60}")
    print(f"📊 SignSpeak 스타일 모델 전체 결과 요약")
    print(f"{'='*60}")
    
    for model_type, results in all_results.items():
        print(f"\n{model_type.upper()}:")
        print(f"  평균 정확도: {results['mean_validation_accuracy']:.4f} ± {results['std_validation_accuracy']:.4f}")
        print(f"  최고 정확도: {results['max_validation_accuracy']:.4f}")
        print(f"  SignSpeak 92% 대비: {results['max_validation_accuracy'] - 0.92:+.4f}")
    
    # 최고 성능 모델 찾기
    best_model_type = max(all_results.keys(), key=lambda x: all_results[x]['max_validation_accuracy'])
    best_accuracy = all_results[best_model_type]['max_validation_accuracy']
    
    print(f"\n🏆 최고 성능 모델: {best_model_type.upper()}")
    print(f"🏆 최고 정확도: {best_accuracy:.4f}")
    
    if best_accuracy >= 0.92:
        print(f"🎉 SignSpeak 92% 정확도 달성 성공!")
    else:
        print(f"📈 SignSpeak 92% 정확도까지 {0.92 - best_accuracy:.4f} 부족")
    
    # 전체 결과 저장
    with open('signspeak_all_results.json', 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    print(f"\n📁 저장된 파일:")
    for model_type in model_types:
        print(f"   - signspeak_{model_type}_cv_model.pth")
        print(f"   - signspeak_{model_type}_cv_results.json")
        print(f"   - signspeak_{model_type}_cv_analysis.png")
    print(f"   - signspeak_all_results.json")
    
    print(f"\n🚀 SignSpeak 스타일 모델 교차 검증 시스템 완료!")


if __name__ == "__main__":
    main()
