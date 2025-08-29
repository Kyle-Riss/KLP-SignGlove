#!/usr/bin/env python3
"""
GRU 모델 클래스별 성능 테스트
각 클래스별 정확도와 혼동 행렬 분석
"""

import torch
import torch.nn as nn
import numpy as np
import h5py
import os
from pathlib import Path
import sys
from collections import defaultdict
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset, DataLoader

# GRU 모델 클래스 import
sys.path.append('.')
from gru_model import GRUSignGloveModel, SignGloveDataset

class ClassSpecificTestDataset(Dataset):
    """클래스별 테스트용 데이터셋"""
    
    def __init__(self, data_path, sequence_length=20, test_classes=None):
        self.data_path = Path(data_path)
        self.sequence_length = sequence_length
        self.scaler = StandardScaler()
        
        # 클래스 목록
        self.class_names = ['ㄱ', 'ㄴ', 'ㄷ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅅ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ',
                           'ㅏ', 'ㅑ', 'ㅓ', 'ㅕ', 'ㅗ', 'ㅛ', 'ㅜ', 'ㅠ', 'ㅡ', 'ㅣ']
        
        # 테스트할 클래스들 (None이면 모든 클래스)
        if test_classes is None:
            self.test_classes = self.class_names
        else:
            self.test_classes = test_classes
        
        # 데이터 로드
        self.data, self.labels, self.class_indices = self.load_test_data()
        
        # 데이터 정규화
        self.normalize_data()
        
    def load_test_data(self):
        """테스트 데이터 로드"""
        print("📊 클래스별 테스트 데이터 로딩 중...")
        
        data = []
        labels = []
        class_indices = defaultdict(list)
        
        for class_idx, class_name in enumerate(self.class_names):
            if class_name not in self.test_classes:
                continue
                
            print(f"  {class_name} 클래스 테스트 데이터 로딩 중...")
            
            class_dir = self.data_path / class_name
            if not class_dir.exists():
                continue
            
            class_data_count = 0
            
            for sub_dir in class_dir.iterdir():
                if sub_dir.is_dir():
                    for h5_file in sub_dir.glob("*.h5"):
                        try:
                            with h5py.File(h5_file, 'r') as f:
                                sensor_data = f['sensor_data'][:]
                                
                                # 시퀀스로 분할 (테스트용으로는 50% 오버랩 없이)
                                for i in range(0, len(sensor_data), self.sequence_length):
                                    if i + self.sequence_length <= len(sensor_data):
                                        sequence = sensor_data[i:i+self.sequence_length]
                                        data.append(sequence)
                                        labels.append(class_idx)
                                        class_indices[class_name].append(len(data) - 1)
                                        class_data_count += 1
                        except Exception as e:
                            print(f"⚠️ 파일 로드 오류 {h5_file}: {e}")
                            continue
            
            print(f"    {class_name}: {class_data_count}개 시퀀스")
        
        return np.array(data), np.array(labels), class_indices
    
    def normalize_data(self):
        """데이터 정규화"""
        print("🔧 데이터 정규화 중...")
        
        # 데이터 형태 변경: (samples, sequence, features) -> (samples, features)
        original_shape = self.data.shape
        data_reshaped = self.data.reshape(-1, self.data.shape[-1])
        
        # 정규화
        data_normalized = self.scaler.fit_transform(data_reshaped)
        
        # 원래 형태로 복원
        self.data = data_normalized.reshape(original_shape)
        
        print(f"✅ 정규화 완료: 범위 [{self.data.min():.3f}, {self.data.max():.3f}]")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sequence = self.data[idx]
        label = self.labels[idx]
        return torch.FloatTensor(sequence), torch.LongTensor([label])

class GRUClassTester:
    """GRU 모델 클래스별 테스터"""
    
    def __init__(self, model_path='best_gru_model.pth', device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.class_names = ['ㄱ', 'ㄴ', 'ㄷ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅅ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ',
                           'ㅏ', 'ㅑ', 'ㅓ', 'ㅕ', 'ㅗ', 'ㅛ', 'ㅜ', 'ㅠ', 'ㅡ', 'ㅣ']
        
        # 모델 로드
        self.model = self.load_model(model_path)
        print(f"✅ GRU 모델 로드 완료 (디바이스: {device})")
    
    def load_model(self, model_path):
        """모델 로드"""
        model = GRUSignGloveModel(
            input_size=8,
            hidden_size=64,
            num_layers=2,
            num_classes=24,
            dropout=0.0  # 테스트 시에는 드롭아웃 비활성화
        )
        
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"📁 모델 로드: {model_path}")
        else:
            print(f"⚠️ 모델 파일을 찾을 수 없습니다: {model_path}")
            return None
        
        model.to(self.device)
        model.eval()
        return model
    
    def test_class_performance(self, test_loader):
        """클래스별 성능 테스트"""
        if self.model is None:
            print("❌ 모델이 로드되지 않았습니다.")
            return None
        
        print("🔍 클래스별 성능 테스트 시작...")
        
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_loader):
                data, target = data.to(self.device), target.squeeze().to(self.device)
                
                # 예측
                output = self.model(data)
                probabilities = torch.softmax(output, dim=1)
                predictions = output.argmax(dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(target.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                
                if batch_idx % 10 == 0:
                    print(f"  배치 {batch_idx}/{len(test_loader)} 처리 완료")
        
        return np.array(all_predictions), np.array(all_labels), np.array(all_probabilities)
    
    def analyze_class_performance(self, predictions, labels, probabilities, class_indices):
        """클래스별 성능 분석"""
        print("\n📊 클래스별 성능 분석")
        print("=" * 60)
        
        # 전체 정확도
        overall_accuracy = np.mean(predictions == labels)
        print(f"🎯 전체 정확도: {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)")
        
        # 클래스별 정확도
        class_accuracies = {}
        class_counts = {}
        
        print("\n📈 클래스별 정확도:")
        print("-" * 40)
        
        for class_name in self.class_names:
            if class_name in class_indices:
                class_idx = self.class_names.index(class_name)
                class_mask = labels == class_idx
                
                if np.sum(class_mask) > 0:
                    class_pred = predictions[class_mask]
                    class_true = labels[class_mask]
                    class_acc = np.mean(class_pred == class_true)
                    class_count = np.sum(class_mask)
                    
                    class_accuracies[class_name] = class_acc
                    class_counts[class_name] = class_count
                    
                    print(f"{class_name:2s}: {class_acc:.4f} ({class_acc*100:.2f}%) - {class_count}개 샘플")
        
        # 가장 어려운 클래스들
        print("\n⚠️ 가장 어려운 클래스들 (정확도 낮음):")
        print("-" * 40)
        sorted_classes = sorted(class_accuracies.items(), key=lambda x: x[1])
        for class_name, acc in sorted_classes[:5]:
            print(f"{class_name:2s}: {acc:.4f} ({acc*100:.2f}%)")
        
        # 가장 쉬운 클래스들
        print("\n✅ 가장 쉬운 클래스들 (정확도 높음):")
        print("-" * 40)
        for class_name, acc in sorted_classes[-5:]:
            print(f"{class_name:2s}: {acc:.4f} ({acc*100:.2f}%)")
        
        return class_accuracies, class_counts
    
    def create_confusion_matrix(self, predictions, labels):
        """혼동 행렬 생성"""
        print("\n🔍 혼동 행렬 생성 중...")
        
        cm = confusion_matrix(labels, predictions)
        
        # 혼동 행렬 시각화
        plt.figure(figsize=(20, 16))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title('GRU 모델 혼동 행렬', fontsize=16, fontweight='bold')
        plt.xlabel('예측 클래스', fontsize=14)
        plt.ylabel('실제 클래스', fontsize=14)
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig('gru_confusion_matrix.png', dpi=300, bbox_inches='tight')
        print("📊 혼동 행렬이 'gru_confusion_matrix.png'에 저장되었습니다.")
        
        return cm
    
    def analyze_misclassifications(self, predictions, labels, probabilities, class_indices):
        """오분류 분석"""
        print("\n🔍 오분류 분석")
        print("=" * 60)
        
        # 가장 많이 혼동되는 클래스 쌍
        misclassifications = defaultdict(int)
        
        for pred, true in zip(predictions, labels):
            if pred != true:
                true_class = self.class_names[true]
                pred_class = self.class_names[pred]
                misclassifications[(true_class, pred_class)] += 1
        
        print("❌ 가장 많이 혼동되는 클래스 쌍:")
        print("-" * 40)
        sorted_misclass = sorted(misclassifications.items(), key=lambda x: x[1], reverse=True)
        for (true_class, pred_class), count in sorted_misclass[:10]:
            print(f"{true_class} → {pred_class}: {count}회")
        
        # 각 클래스별 주요 오분류
        print("\n📋 클래스별 주요 오분류:")
        print("-" * 40)
        for class_name in self.class_names:
            if class_name in class_indices:
                class_idx = self.class_names.index(class_name)
                class_mask = labels == class_idx
                
                if np.sum(class_mask) > 0:
                    class_pred = predictions[class_mask]
                    class_true = labels[class_mask]
                    
                    # 오분류된 것들만
                    misclassified = class_pred != class_true
                    if np.sum(misclassified) > 0:
                        misclassified_preds = class_pred[misclassified]
                        
                        # 가장 많이 잘못 분류된 클래스
                        unique, counts = np.unique(misclassified_preds, return_counts=True)
                        most_common_idx = unique[np.argmax(counts)]
                        most_common_class = self.class_names[most_common_idx]
                        most_common_count = np.max(counts)
                        
                        print(f"{class_name}: 주로 {most_common_class}로 잘못 분류됨 ({most_common_count}회)")
    
    def test_confidence_analysis(self, predictions, labels, probabilities):
        """신뢰도 분석"""
        print("\n🎯 신뢰도 분석")
        print("=" * 60)
        
        # 예측 확률의 최대값 (신뢰도)
        confidence_scores = np.max(probabilities, axis=1)
        
        # 정확도별 신뢰도
        correct_mask = predictions == labels
        correct_confidence = confidence_scores[correct_mask]
        incorrect_confidence = confidence_scores[~correct_mask]
        
        print(f"✅ 정확한 예측의 평균 신뢰도: {np.mean(correct_confidence):.4f}")
        print(f"❌ 잘못된 예측의 평균 신뢰도: {np.mean(incorrect_confidence):.4f}")
        print(f"📊 전체 평균 신뢰도: {np.mean(confidence_scores):.4f}")
        
        # 신뢰도 분포 시각화
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.hist(correct_confidence, bins=20, alpha=0.7, label='정확한 예측', color='green')
        plt.hist(incorrect_confidence, bins=20, alpha=0.7, label='잘못된 예측', color='red')
        plt.xlabel('신뢰도')
        plt.ylabel('빈도')
        plt.title('신뢰도 분포')
        plt.legend()
        
        plt.subplot(2, 2, 2)
        plt.scatter(confidence_scores, (predictions == labels).astype(int), alpha=0.5)
        plt.xlabel('신뢰도')
        plt.ylabel('정확도 (0/1)')
        plt.title('신뢰도 vs 정확도')
        
        plt.subplot(2, 2, 3)
        # 클래스별 평균 신뢰도
        class_confidence = {}
        for i, class_name in enumerate(self.class_names):
            class_mask = labels == i
            if np.sum(class_mask) > 0:
                class_confidence[class_name] = np.mean(confidence_scores[class_mask])
        
        classes = list(class_confidence.keys())
        confidences = list(class_confidence.values())
        plt.bar(range(len(classes)), confidences)
        plt.xlabel('클래스')
        plt.ylabel('평균 신뢰도')
        plt.title('클래스별 평균 신뢰도')
        plt.xticks(range(len(classes)), classes, rotation=45)
        
        plt.subplot(2, 2, 4)
        # 정확도별 신뢰도 박스플롯
        plt.boxplot([correct_confidence, incorrect_confidence], 
                   labels=['정확한 예측', '잘못된 예측'])
        plt.ylabel('신뢰도')
        plt.title('정확도별 신뢰도 분포')
        
        plt.tight_layout()
        plt.savefig('gru_confidence_analysis.png', dpi=300, bbox_inches='tight')
        print("📊 신뢰도 분석이 'gru_confidence_analysis.png'에 저장되었습니다.")

def main():
    """메인 함수"""
    print("🚀 GRU 모델 클래스별 성능 테스트 시작")
    print("=" * 60)
    
    # 테스트 데이터 로드
    test_dataset = ClassSpecificTestDataset(
        data_path="../SignGlove_HW/datasets/unified",
        sequence_length=20
    )
    
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)
    
    print(f"📊 테스트 데이터: {len(test_dataset)}개 시퀀스")
    
    # 테스터 생성
    tester = GRUClassTester()
    
    # 성능 테스트
    predictions, labels, probabilities = tester.test_class_performance(test_loader)
    
    if predictions is not None:
        # 클래스별 성능 분석
        class_accuracies, class_counts = tester.analyze_class_performance(
            predictions, labels, probabilities, test_dataset.class_indices
        )
        
        # 혼동 행렬 생성
        cm = tester.create_confusion_matrix(predictions, labels)
        
        # 오분류 분석
        tester.analyze_misclassifications(
            predictions, labels, probabilities, test_dataset.class_indices
        )
        
        # 신뢰도 분석
        tester.test_confidence_analysis(predictions, labels, probabilities)
        
        # 상세 분류 리포트
        print("\n📋 상세 분류 리포트:")
        print("=" * 60)
        report = classification_report(labels, predictions, 
                                     target_names=tester.class_names, 
                                     digits=4)
        print(report)
        
        # 결과 요약
        print("\n🎉 클래스별 테스트 완료!")
        print("=" * 60)
        print(f"📊 전체 정확도: {np.mean(predictions == labels):.4f}")
        print(f"📈 최고 정확도 클래스: {max(class_accuracies.items(), key=lambda x: x[1])}")
        print(f"📉 최저 정확도 클래스: {min(class_accuracies.items(), key=lambda x: x[1])}")
        
    else:
        print("❌ 테스트를 완료할 수 없습니다.")

if __name__ == "__main__":
    main()



