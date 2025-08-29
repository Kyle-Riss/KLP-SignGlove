#!/usr/bin/env python3
"""
GRU 모델 과적합 분석
높은 정확도가 과적합인지 실제 성능인지 분석
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
import random

# GRU 모델 클래스 import
sys.path.append('.')
from gru_model import GRUSignGloveModel

class OverfittingTestDataset(Dataset):
    """과적합 테스트용 데이터셋 - 완전히 새로운 데이터 사용"""
    
    def __init__(self, data_path, sequence_length=20, test_ratio=0.3, random_seed=42):
        self.data_path = Path(data_path)
        self.sequence_length = sequence_length
        self.scaler = StandardScaler()
        
        # 클래스 목록
        self.class_names = ['ㄱ', 'ㄴ', 'ㄷ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅅ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ',
                           'ㅏ', 'ㅑ', 'ㅓ', 'ㅕ', 'ㅗ', 'ㅛ', 'ㅜ', 'ㅠ', 'ㅡ', 'ㅣ']
        
        # 랜덤 시드 설정
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        # 데이터 로드 (완전히 새로운 파일들만 사용)
        self.data, self.labels, self.class_indices = self.load_holdout_data(test_ratio)
        
        # 데이터 정규화
        self.normalize_data()
        
    def load_holdout_data(self, test_ratio):
        """홀드아웃 데이터 로드 - 훈련에 사용되지 않은 파일들만"""
        print("📊 과적합 테스트용 홀드아웃 데이터 로딩 중...")
        
        data = []
        labels = []
        class_indices = defaultdict(list)
        
        for class_idx, class_name in enumerate(self.class_names):
            print(f"  {class_name} 클래스 홀드아웃 데이터 로딩 중...")
            
            class_dir = self.data_path / class_name
            if not class_dir.exists():
                continue
            
            class_data_count = 0
            
            for sub_dir in class_dir.iterdir():
                if sub_dir.is_dir():
                    h5_files = list(sub_dir.glob("*.h5"))
                    
                    # 파일들을 랜덤하게 섞고 일부만 테스트용으로 사용
                    random.shuffle(h5_files)
                    test_files = h5_files[:int(len(h5_files) * test_ratio)]
                    
                    for h5_file in test_files:
                        try:
                            with h5py.File(h5_file, 'r') as f:
                                sensor_data = f['sensor_data'][:]
                                
                                # 시퀀스로 분할 (50% 오버랩 없이)
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
            
            print(f"    {class_name}: {class_data_count}개 시퀀스 (홀드아웃)")
        
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

class OverfittingAnalyzer:
    """과적합 분석기"""
    
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
    
    def test_holdout_performance(self, test_loader):
        """홀드아웃 데이터 성능 테스트"""
        if self.model is None:
            print("❌ 모델이 로드되지 않았습니다.")
            return None
        
        print("🔍 홀드아웃 데이터 성능 테스트 시작...")
        
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
    
    def analyze_overfitting_indicators(self, predictions, labels, probabilities, class_indices):
        """과적합 지표 분석"""
        print("\n🔍 과적합 지표 분석")
        print("=" * 60)
        
        # 1. 전체 정확도
        overall_accuracy = np.mean(predictions == labels)
        print(f"🎯 홀드아웃 데이터 전체 정확도: {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)")
        
        # 2. 클래스별 정확도 분석
        class_accuracies = {}
        class_counts = {}
        
        print("\n📈 클래스별 정확도 (홀드아웃):")
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
        
        # 3. 과적합 의심 지표들
        print("\n⚠️ 과적합 의심 지표들:")
        print("-" * 40)
        
        # 3-1. 정확도가 매우 높은지 확인
        if overall_accuracy > 0.99:
            print(f"🔴 매우 높은 정확도: {overall_accuracy:.4f} (과적합 의심)")
        elif overall_accuracy > 0.95:
            print(f"🟡 높은 정확도: {overall_accuracy:.4f} (주의 필요)")
        else:
            print(f"🟢 적절한 정확도: {overall_accuracy:.4f}")
        
        # 3-2. 클래스별 정확도 편차 분석
        accuracies = list(class_accuracies.values())
        acc_std = np.std(accuracies)
        acc_range = max(accuracies) - min(accuracies)
        
        print(f"📊 클래스별 정확도 표준편차: {acc_std:.4f}")
        print(f"📊 클래스별 정확도 범위: {acc_range:.4f}")
        
        if acc_std > 0.05:
            print("🔴 클래스별 성능 편차가 큼 (과적합 의심)")
        elif acc_std > 0.02:
            print("🟡 클래스별 성능 편차가 있음 (주의 필요)")
        else:
            print("🟢 클래스별 성능이 균등함")
        
        # 3-3. 신뢰도 분석
        confidence_scores = np.max(probabilities, axis=1)
        correct_mask = predictions == labels
        correct_confidence = confidence_scores[correct_mask]
        incorrect_confidence = confidence_scores[~correct_mask]
        
        print(f"✅ 정확한 예측의 평균 신뢰도: {np.mean(correct_confidence):.4f}")
        print(f"❌ 잘못된 예측의 평균 신뢰도: {np.mean(incorrect_confidence):.4f}")
        
        # 신뢰도가 너무 높으면 과적합 의심
        if np.mean(correct_confidence) > 0.999:
            print("🔴 매우 높은 신뢰도 (과적합 의심)")
        elif np.mean(correct_confidence) > 0.99:
            print("🟡 높은 신뢰도 (주의 필요)")
        else:
            print("🟢 적절한 신뢰도")
        
        # 3-4. 오분류 패턴 분석
        misclassifications = defaultdict(int)
        for pred, true in zip(predictions, labels):
            if pred != true:
                true_class = self.class_names[true]
                pred_class = self.class_names[pred]
                misclassifications[(true_class, pred_class)] += 1
        
        total_misclass = sum(misclassifications.values())
        print(f"📊 총 오분류 수: {total_misclass}")
        
        if total_misclass < len(predictions) * 0.01:  # 1% 미만
            print("🔴 매우 적은 오분류 (과적합 의심)")
        elif total_misclass < len(predictions) * 0.05:  # 5% 미만
            print("🟡 적은 오분류 (주의 필요)")
        else:
            print("🟢 적절한 오분류 수")
        
        return class_accuracies, class_counts
    
    def create_overfitting_visualization(self, predictions, labels, probabilities):
        """과적합 시각화"""
        print("\n📊 과적합 시각화 생성 중...")
        
        # 1. 정확도 vs 신뢰도 산점도
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 3, 1)
        confidence_scores = np.max(probabilities, axis=1)
        accuracy_binary = (predictions == labels).astype(int)
        
        plt.scatter(confidence_scores, accuracy_binary, alpha=0.6, s=20)
        plt.xlabel('신뢰도')
        plt.ylabel('정확도 (0/1)')
        plt.title('신뢰도 vs 정확도')
        plt.grid(True, alpha=0.3)
        
        # 2. 신뢰도 분포
        plt.subplot(2, 3, 2)
        correct_mask = predictions == labels
        correct_confidence = confidence_scores[correct_mask]
        incorrect_confidence = confidence_scores[~correct_mask]
        
        plt.hist(correct_confidence, bins=30, alpha=0.7, label='정확한 예측', color='green', density=True)
        plt.hist(incorrect_confidence, bins=30, alpha=0.7, label='잘못된 예측', color='red', density=True)
        plt.xlabel('신뢰도')
        plt.ylabel('밀도')
        plt.title('신뢰도 분포')
        plt.legend()
        
        # 3. 클래스별 정확도
        plt.subplot(2, 3, 3)
        class_accuracies = []
        for i in range(len(self.class_names)):
            class_mask = labels == i
            if np.sum(class_mask) > 0:
                class_acc = np.mean(predictions[class_mask] == labels[class_mask])
                class_accuracies.append(class_acc)
            else:
                class_accuracies.append(0)
        
        plt.bar(range(len(self.class_names)), class_accuracies)
        plt.xlabel('클래스')
        plt.ylabel('정확도')
        plt.title('클래스별 정확도')
        plt.xticks(range(len(self.class_names)), self.class_names, rotation=45)
        plt.ylim(0, 1)
        
        # 4. 예측 확률 분포
        plt.subplot(2, 3, 4)
        plt.hist(confidence_scores, bins=50, alpha=0.7, color='blue')
        plt.xlabel('신뢰도')
        plt.ylabel('빈도')
        plt.title('전체 신뢰도 분포')
        
        # 5. 오분류 분석
        plt.subplot(2, 3, 5)
        misclassifications = defaultdict(int)
        for pred, true in zip(predictions, labels):
            if pred != true:
                true_class = self.class_names[true]
                pred_class = self.class_names[pred]
                misclassifications[(true_class, pred_class)] += 1
        
        if misclassifications:
            top_misclass = sorted(misclassifications.items(), key=lambda x: x[1], reverse=True)[:10]
            pairs = [f"{true}→{pred}" for (true, pred), _ in top_misclass]
            counts = [count for _, count in top_misclass]
            
            plt.barh(range(len(pairs)), counts)
            plt.yticks(range(len(pairs)), pairs)
            plt.xlabel('오분류 횟수')
            plt.title('주요 오분류 패턴')
        
        # 6. 과적합 지표 요약
        plt.subplot(2, 3, 6)
        overall_accuracy = np.mean(predictions == labels)
        avg_confidence = np.mean(confidence_scores)
        
        indicators = ['전체 정확도', '평균 신뢰도', '오분류 비율']
        values = [overall_accuracy, avg_confidence, 1-overall_accuracy]
        colors = ['green' if v > 0.95 else 'orange' if v > 0.9 else 'red' for v in values]
        
        bars = plt.bar(indicators, values, color=colors)
        plt.ylabel('값')
        plt.title('과적합 지표 요약')
        plt.ylim(0, 1)
        
        # 값 표시
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('overfitting_analysis.png', dpi=300, bbox_inches='tight')
        print("📊 과적합 분석이 'overfitting_analysis.png'에 저장되었습니다.")
    
    def generate_overfitting_report(self, predictions, labels, probabilities):
        """과적합 리포트 생성"""
        print("\n📋 과적합 분석 리포트")
        print("=" * 60)
        
        overall_accuracy = np.mean(predictions == labels)
        confidence_scores = np.max(probabilities, axis=1)
        avg_confidence = np.mean(confidence_scores)
        
        print(f"📊 전체 정확도: {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)")
        print(f"🎯 평균 신뢰도: {avg_confidence:.4f} ({avg_confidence*100:.2f}%)")
        print(f"❌ 오분류 비율: {1-overall_accuracy:.4f} ({(1-overall_accuracy)*100:.2f}%)")
        
        # 과적합 판단
        print("\n🔍 과적합 판단:")
        print("-" * 40)
        
        overfitting_score = 0
        reasons = []
        
        # 1. 정확도가 너무 높은지
        if overall_accuracy > 0.99:
            overfitting_score += 2
            reasons.append("정확도가 99%를 초과함")
        elif overall_accuracy > 0.95:
            overfitting_score += 1
            reasons.append("정확도가 95%를 초과함")
        
        # 2. 신뢰도가 너무 높은지
        if avg_confidence > 0.999:
            overfitting_score += 2
            reasons.append("신뢰도가 99.9%를 초과함")
        elif avg_confidence > 0.99:
            overfitting_score += 1
            reasons.append("신뢰도가 99%를 초과함")
        
        # 3. 오분류가 너무 적은지
        if (1-overall_accuracy) < 0.01:
            overfitting_score += 2
            reasons.append("오분류 비율이 1% 미만임")
        elif (1-overall_accuracy) < 0.05:
            overfitting_score += 1
            reasons.append("오분류 비율이 5% 미만임")
        
        # 4. 클래스별 성능 편차
        class_accuracies = []
        for i in range(len(self.class_names)):
            class_mask = labels == i
            if np.sum(class_mask) > 0:
                class_acc = np.mean(predictions[class_mask] == labels[class_mask])
                class_accuracies.append(class_acc)
        
        if class_accuracies:
            acc_std = np.std(class_accuracies)
            if acc_std > 0.05:
                overfitting_score += 1
                reasons.append("클래스별 성능 편차가 큼")
        
        # 최종 판단
        print(f"🏆 과적합 점수: {overfitting_score}/6")
        
        if overfitting_score >= 4:
            print("🔴 높은 과적합 의심")
            print("   - 모델이 훈련 데이터를 외웠을 가능성이 높음")
            print("   - 더 많은 데이터나 정규화가 필요할 수 있음")
        elif overfitting_score >= 2:
            print("🟡 중간 과적합 의심")
            print("   - 일부 과적합 징후가 있음")
            print("   - 주의 깊게 모니터링 필요")
        else:
            print("🟢 과적합 징후 없음")
            print("   - 모델이 일반화를 잘 하고 있음")
        
        if reasons:
            print("\n📝 과적합 의심 이유:")
            for i, reason in enumerate(reasons, 1):
                print(f"   {i}. {reason}")

def main():
    """메인 함수"""
    print("🚀 GRU 모델 과적합 분석 시작")
    print("=" * 60)
    
    # 홀드아웃 테스트 데이터 로드
    test_dataset = OverfittingTestDataset(
        data_path="../SignGlove_HW/datasets/unified",
        sequence_length=20,
        test_ratio=0.3  # 30%를 테스트용으로 사용
    )
    
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)
    
    print(f"📊 홀드아웃 테스트 데이터: {len(test_dataset)}개 시퀀스")
    
    # 분석기 생성
    analyzer = OverfittingAnalyzer()
    
    # 성능 테스트
    predictions, labels, probabilities = analyzer.test_holdout_performance(test_loader)
    
    if predictions is not None:
        # 과적합 지표 분석
        class_accuracies, class_counts = analyzer.analyze_overfitting_indicators(
            predictions, labels, probabilities, test_dataset.class_indices
        )
        
        # 과적합 시각화
        analyzer.create_overfitting_visualization(predictions, labels, probabilities)
        
        # 과적합 리포트
        analyzer.generate_overfitting_report(predictions, labels, probabilities)
        
        # 상세 분류 리포트
        print("\n📋 상세 분류 리포트 (홀드아웃):")
        print("=" * 60)
        report = classification_report(labels, predictions, 
                                     target_names=analyzer.class_names, 
                                     digits=4)
        print(report)
        
        # 결과 요약
        print("\n🎉 과적합 분석 완료!")
        print("=" * 60)
        print(f"📊 홀드아웃 정확도: {np.mean(predictions == labels):.4f}")
        print(f"🎯 평균 신뢰도: {np.mean(np.max(probabilities, axis=1)):.4f}")
        
    else:
        print("❌ 분석을 완료할 수 없습니다.")

if __name__ == "__main__":
    main()



