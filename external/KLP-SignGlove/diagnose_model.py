#!/usr/bin/env python3
"""
모델 진단 시스템
학습과 추론 문제를 분석하여 원인을 파악
"""

import torch
import torch.nn as nn
import numpy as np
import h5py
import os
from pathlib import Path
import sys
from collections import defaultdict, Counter
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset, DataLoader
import random
import json

# 과적합 방지 모델 클래스 import
sys.path.append('.')
from anti_overfitting_gru import AntiOverfittingGRUModel

class ModelDiagnosticDataset(Dataset):
    """모델 진단용 데이터셋"""
    
    def __init__(self, data_path, sequence_length=20, validation_type='user_split'):
        self.data_path = Path(data_path)
        self.sequence_length = sequence_length
        self.scaler = StandardScaler()
        
        # 클래스 목록
        self.class_names = ['ㄱ', 'ㄴ', 'ㄷ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅅ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ',
                           'ㅏ', 'ㅑ', 'ㅓ', 'ㅕ', 'ㅗ', 'ㅛ', 'ㅜ', 'ㅠ', 'ㅡ', 'ㅣ']
        
        # 데이터 로드
        self.data, self.labels, self.class_indices = self.load_data(validation_type)
        
        # 데이터 정규화
        self.normalize_data()
        
    def load_data(self, validation_type):
        """데이터 로드"""
        print(f"📊 {validation_type} 진단 데이터 로딩 중...")
        
        data = []
        labels = []
        class_indices = defaultdict(list)
        
        for class_idx, class_name in enumerate(self.class_names):
            print(f"  {class_name} 클래스 데이터 로딩 중...")
            
            class_dir = self.data_path / class_name
            if not class_dir.exists():
                continue
            
            class_data_count = 0
            
            for sub_dir in class_dir.iterdir():
                if sub_dir.is_dir():
                    h5_files = list(sub_dir.glob("*.h5"))
                    
                    if validation_type == 'user_split':
                        # 사용자별 분할 - 사용자 5만 테스트용
                        user_id = sub_dir.name
                        if user_id != '5':
                            continue
                    elif validation_type == 'time_split':
                        # 시간별 분할 - 마지막 20%만 테스트용
                        h5_files.sort()
                        h5_files = h5_files[-int(len(h5_files) * 0.2):]
                    elif validation_type == 'no_augmentation':
                        # 증강 없는 데이터
                        random.shuffle(h5_files)
                        h5_files = h5_files[int(len(h5_files) * 0.7):]
                    
                    for h5_file in h5_files:
                        try:
                            with h5py.File(h5_file, 'r') as f:
                                sensor_data = f['sensor_data'][:]
                                
                                # 시퀀스로 분할
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
        
        original_shape = self.data.shape
        data_reshaped = self.data.reshape(-1, self.data.shape[-1])
        
        data_normalized = self.scaler.fit_transform(data_reshaped)
        self.data = data_normalized.reshape(original_shape)
        
        print(f"✅ 정규화 완료: 범위 [{self.data.min():.3f}, {self.data.max():.3f}]")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sequence = self.data[idx]
        label = self.labels[idx]
        return torch.FloatTensor(sequence), torch.LongTensor([label])

class ModelDiagnostic:
    """모델 진단 시스템"""
    
    def __init__(self, model_path='best_anti_overfitting_model.pth', device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.class_names = ['ㄱ', 'ㄴ', 'ㄷ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅅ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ',
                           'ㅏ', 'ㅑ', 'ㅓ', 'ㅕ', 'ㅗ', 'ㅛ', 'ㅜ', 'ㅠ', 'ㅡ', 'ㅣ']
        
        # 모델 로드
        self.model = self.load_model(model_path)
        print(f"🔍 모델 진단 시스템 초기화 완료 (디바이스: {device})")
    
    def load_model(self, model_path):
        """모델 로드"""
        model = AntiOverfittingGRUModel(
            input_size=8,
            hidden_size=32,
            num_layers=1,
            num_classes=24,
            dropout=0.0
        )
        
        if Path(model_path).exists():
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"📁 모델 로드: {model_path}")
        else:
            print(f"⚠️ 모델 파일을 찾을 수 없습니다: {model_path}")
            return None
        
        model.to(self.device)
        model.eval()
        return model
    
    def analyze_model_weights(self):
        """모델 가중치 분석"""
        print("\n🔍 모델 가중치 분석")
        print("=" * 50)
        
        if self.model is None:
            print("❌ 모델이 로드되지 않았습니다.")
            return
        
        # 마지막 분류 레이어의 가중치 분석
        classifier = self.model.classifier
        last_layer = classifier[-1]  # 마지막 Linear 레이어
        
        if isinstance(last_layer, nn.Linear):
            weights = last_layer.weight.data.cpu().numpy()
            biases = last_layer.bias.data.cpu().numpy()
            
            print(f"📊 마지막 레이어 가중치 분석:")
            print(f"  가중치 형태: {weights.shape}")
            print(f"  편향 형태: {biases.shape}")
            
            # 각 클래스별 가중치 통계
            print(f"\n📈 클래스별 가중치 통계:")
            print("-" * 40)
            
            for i, class_name in enumerate(self.class_names):
                class_weights = weights[i]
                weight_mean = np.mean(class_weights)
                weight_std = np.std(class_weights)
                weight_max = np.max(class_weights)
                weight_min = np.min(class_weights)
                bias = biases[i]
                
                print(f"{class_name:2s}: 평균={weight_mean:6.3f}, 표준편차={weight_std:6.3f}, "
                      f"최대={weight_max:6.3f}, 최소={weight_min:6.3f}, 편향={bias:6.3f}")
            
            # 가중치 분포 시각화
            plt.figure(figsize=(15, 10))
            
            plt.subplot(2, 2, 1)
            plt.hist(weights.flatten(), bins=50, alpha=0.7)
            plt.title('전체 가중치 분포')
            plt.xlabel('가중치 값')
            plt.ylabel('빈도')
            
            plt.subplot(2, 2, 2)
            plt.hist(biases, bins=20, alpha=0.7)
            plt.title('편향 분포')
            plt.xlabel('편향 값')
            plt.ylabel('빈도')
            
            plt.subplot(2, 2, 3)
            weight_norms = np.linalg.norm(weights, axis=1)
            plt.bar(range(24), weight_norms)
            plt.title('클래스별 가중치 노름')
            plt.xlabel('클래스 인덱스')
            plt.ylabel('가중치 노름')
            
            plt.subplot(2, 2, 4)
            plt.bar(range(24), biases)
            plt.title('클래스별 편향')
            plt.xlabel('클래스 인덱스')
            plt.ylabel('편향 값')
            
            plt.tight_layout()
            plt.savefig('model_weights_analysis.png', dpi=300, bbox_inches='tight')
            print("📊 가중치 분석 차트가 'model_weights_analysis.png'에 저장되었습니다.")
    
    def test_prediction_distribution(self, test_loader, num_samples=1000):
        """예측 분포 테스트"""
        print(f"\n🎯 예측 분포 테스트 ({num_samples}개 샘플)")
        print("=" * 50)
        
        if self.model is None:
            print("❌ 모델이 로드되지 않았습니다.")
            return
        
        self.model.eval()
        predictions = []
        confidences = []
        all_probabilities = []
        
        with torch.no_grad():
            sample_count = 0
            for data, target in test_loader:
                if sample_count >= num_samples:
                    break
                
                data, target = data.to(self.device), target.squeeze().to(self.device)
                output = self.model(data)
                probabilities = torch.softmax(output, dim=1)
                
                # 예측과 신뢰도
                max_prob, predicted_class = torch.max(probabilities, 1)
                
                predictions.extend(predicted_class.cpu().numpy())
                confidences.extend(max_prob.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                
                sample_count += data.size(0)
        
        # 예측 분포 분석
        prediction_counts = Counter(predictions)
        total_predictions = len(predictions)
        
        print(f"📊 예측 분포 분석:")
        print("-" * 40)
        
        for i, class_name in enumerate(self.class_names):
            count = prediction_counts.get(i, 0)
            frequency = count / total_predictions
            print(f"{class_name:2s}: {count:3d}회 ({frequency*100:5.1f}%)")
        
        # 신뢰도 분석
        avg_confidence = np.mean(confidences)
        confidence_std = np.std(confidences)
        
        print(f"\n📈 신뢰도 분석:")
        print(f"  평균 신뢰도: {avg_confidence:.4f}")
        print(f"  신뢰도 표준편차: {confidence_std:.4f}")
        print(f"  최대 신뢰도: {np.max(confidences):.4f}")
        print(f"  최소 신뢰도: {np.min(confidences):.4f}")
        
        # 확률 분포 시각화
        all_probabilities = np.array(all_probabilities)
        
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 2, 1)
        plt.hist(confidences, bins=30, alpha=0.7)
        plt.title('신뢰도 분포')
        plt.xlabel('신뢰도')
        plt.ylabel('빈도')
        
        plt.subplot(2, 2, 2)
        prediction_freqs = [prediction_counts.get(i, 0) / total_predictions for i in range(24)]
        plt.bar(range(24), prediction_freqs)
        plt.title('클래스별 예측 빈도')
        plt.xlabel('클래스 인덱스')
        plt.ylabel('예측 빈도')
        
        plt.subplot(2, 2, 3)
        plt.hist(all_probabilities.flatten(), bins=50, alpha=0.7)
        plt.title('전체 확률 분포')
        plt.xlabel('확률 값')
        plt.ylabel('빈도')
        
        plt.subplot(2, 2, 4)
        class_avg_probs = np.mean(all_probabilities, axis=0)
        plt.bar(range(24), class_avg_probs)
        plt.title('클래스별 평균 확률')
        plt.xlabel('클래스 인덱스')
        plt.ylabel('평균 확률')
        
        plt.tight_layout()
        plt.savefig('prediction_distribution_analysis.png', dpi=300, bbox_inches='tight')
        print("📊 예측 분포 분석 차트가 'prediction_distribution_analysis.png'에 저장되었습니다.")
        
        return prediction_counts, confidences
    
    def analyze_class_imbalance(self, test_loader):
        """클래스 불균형 분석"""
        print(f"\n⚖️ 클래스 불균형 분석")
        print("=" * 50)
        
        # 실제 레이블 분포
        all_labels = []
        for data, target in test_loader:
            all_labels.extend(target.squeeze().numpy())
        
        label_counts = Counter(all_labels)
        total_samples = len(all_labels)
        
        print(f"📊 실제 데이터 분포:")
        print("-" * 40)
        
        for i, class_name in enumerate(self.class_names):
            count = label_counts.get(i, 0)
            frequency = count / total_samples
            print(f"{class_name:2s}: {count:3d}개 ({frequency*100:5.1f}%)")
        
        # 불균형 지수 계산
        frequencies = [label_counts.get(i, 0) / total_samples for i in range(24)]
        imbalance_ratio = max(frequencies) / min(frequencies) if min(frequencies) > 0 else float('inf')
        
        print(f"\n📈 불균형 분석:")
        print(f"  최대 빈도: {max(frequencies)*100:.1f}%")
        print(f"  최소 빈도: {min(frequencies)*100:.1f}%")
        print(f"  불균형 비율: {imbalance_ratio:.2f}")
        
        if imbalance_ratio > 10:
            print("  ⚠️ 심각한 클래스 불균형이 감지되었습니다!")
        elif imbalance_ratio > 5:
            print("  ⚠️ 중간 정도의 클래스 불균형이 감지되었습니다.")
        else:
            print("  ✅ 클래스 분포가 비교적 균형잡혀 있습니다.")
        
        return label_counts, imbalance_ratio
    
    def diagnose_learning_issues(self):
        """학습 문제 진단"""
        print(f"\n🔍 학습 문제 진단")
        print("=" * 50)
        
        # 1. 모델 가중치 분석
        self.analyze_model_weights()
        
        # 2. 테스트 데이터로 예측 분포 분석
        test_dataset = ModelDiagnosticDataset(
            data_path="../SignGlove_HW/datasets/unified",
            sequence_length=20,
            validation_type='user_split'
        )
        
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        # 3. 예측 분포 테스트
        prediction_counts, confidences = self.test_prediction_distribution(test_loader)
        
        # 4. 클래스 불균형 분석
        label_counts, imbalance_ratio = self.analyze_class_imbalance(test_loader)
        
        # 5. 문제 진단 결과
        print(f"\n🎯 진단 결과")
        print("=" * 50)
        
        # 가장 많이 예측되는 클래스
        most_predicted = max(prediction_counts.items(), key=lambda x: x[1])
        most_predicted_class = self.class_names[most_predicted[0]]
        most_predicted_freq = most_predicted[1] / sum(prediction_counts.values())
        
        print(f"📊 가장 많이 예측되는 클래스: {most_predicted_class} ({most_predicted_freq*100:.1f}%)")
        
        # 문제 유형 판단
        if most_predicted_freq > 0.5:
            print("🔴 심각한 문제: 모델이 한 클래스에만 집중하고 있습니다.")
            print("   원인: 학습 과정에서 모델이 특정 클래스에 과적합되었을 가능성")
        elif most_predicted_freq > 0.3:
            print("🟡 중간 문제: 모델이 특정 클래스에 편향되어 있습니다.")
            print("   원인: 클래스 불균형이나 학습 데이터 문제")
        else:
            print("🟢 양호: 모델이 비교적 균형잡힌 예측을 하고 있습니다.")
        
        # 신뢰도 분석
        avg_confidence = np.mean(confidences)
        if avg_confidence > 0.8:
            print("🔴 높은 신뢰도: 모델이 과도하게 확신하고 있습니다.")
        elif avg_confidence < 0.3:
            print("🟡 낮은 신뢰도: 모델이 불확실해하고 있습니다.")
        else:
            print("🟢 적절한 신뢰도: 모델이 적절한 불확실성을 보여줍니다.")
        
        # 해결 방안 제시
        print(f"\n💡 해결 방안")
        print("=" * 50)
        
        if most_predicted_freq > 0.3:
            print("1. 🔄 모델 재학습:")
            print("   - 클래스 가중치를 사용한 손실 함수")
            print("   - 데이터 증강으로 클래스 불균형 해결")
            print("   - 더 강한 정규화 적용")
            
            print("2. 📊 데이터 품질 개선:")
            print("   - 각 클래스별 데이터 품질 검증")
            print("   - 노이즈가 있는 데이터 제거")
            print("   - 더 다양한 센서 데이터 수집")
            
            print("3. ⚙️ 모델 구조 개선:")
            print("   - 더 복잡한 모델 구조 사용")
            print("   - 어텐션 메커니즘 강화")
            print("   - 앙상블 모델 고려")
        
        return {
            'most_predicted_class': most_predicted_class,
            'most_predicted_frequency': most_predicted_freq,
            'average_confidence': avg_confidence,
            'imbalance_ratio': imbalance_ratio,
            'issue_severity': 'high' if most_predicted_freq > 0.5 else 'medium' if most_predicted_freq > 0.3 else 'low'
        }

def main():
    """메인 함수"""
    print("🔍 SignGlove 모델 진단 시스템")
    print("=" * 50)
    
    # 모델 진단 시스템 초기화
    diagnostic = ModelDiagnostic()
    
    # 학습 문제 진단
    results = diagnostic.diagnose_learning_issues()
    
    # 결과 저장
    with open('model_diagnosis_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n📁 진단 결과가 'model_diagnosis_results.json'에 저장되었습니다.")
    print("🎯 진단 완료! 위의 결과를 참고하여 모델을 개선하세요.")

if __name__ == "__main__":
    main()



