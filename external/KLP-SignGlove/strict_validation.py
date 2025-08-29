#!/usr/bin/env python3
"""
엄격한 검증 스크립트
과적합을 더 정확히 진단하기 위한 다양한 검증 방법
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
import json

# GRU 모델 클래스 import
sys.path.append('.')
from gru_model import GRUSignGloveModel

class StrictValidationDataset(Dataset):
    """엄격한 검증용 데이터셋"""
    
    def __init__(self, data_path, sequence_length=20, validation_type='user_split', random_seed=42):
        self.data_path = Path(data_path)
        self.sequence_length = sequence_length
        self.validation_type = validation_type
        self.scaler = StandardScaler()
        
        # 클래스 목록
        self.class_names = ['ㄱ', 'ㄴ', 'ㄷ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅅ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ',
                           'ㅏ', 'ㅑ', 'ㅓ', 'ㅕ', 'ㅗ', 'ㅛ', 'ㅜ', 'ㅠ', 'ㅡ', 'ㅣ']
        
        # 랜덤 시드 설정
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        # 검증 타입에 따른 데이터 로드
        if validation_type == 'user_split':
            self.data, self.labels, self.class_indices = self.load_user_split_data()
        elif validation_type == 'time_split':
            self.data, self.labels, self.class_indices = self.load_time_split_data()
        elif validation_type == 'no_augmentation':
            self.data, self.labels, self.class_indices = self.load_no_augmentation_data()
        else:
            raise ValueError(f"Unknown validation type: {validation_type}")
        
        # 데이터 정규화
        self.normalize_data()
        
    def load_user_split_data(self):
        """사용자별 분할 - 특정 사용자만 테스트용으로 사용"""
        print("📊 사용자별 분할 검증 데이터 로딩 중...")
        
        data = []
        labels = []
        class_indices = defaultdict(list)
        
        # 테스트용 사용자 (마지막 사용자만)
        test_users = ['5']  # 사용자 5만 테스트용
        
        for class_idx, class_name in enumerate(self.class_names):
            print(f"  {class_name} 클래스 사용자별 분할 데이터 로딩 중...")
            
            class_dir = self.data_path / class_name
            if not class_dir.exists():
                continue
            
            class_data_count = 0
            
            for sub_dir in class_dir.iterdir():
                if sub_dir.is_dir():
                    # 사용자 번호 확인
                    user_id = sub_dir.name
                    if user_id not in test_users:
                        continue  # 테스트 사용자가 아니면 건너뛰기
                    
                    for h5_file in sub_dir.glob("*.h5"):
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
            
            print(f"    {class_name}: {class_data_count}개 시퀀스 (사용자 {test_users})")
        
        return np.array(data), np.array(labels), class_indices
    
    def load_time_split_data(self):
        """시간별 분할 - 특정 파일들만 테스트용으로 사용"""
        print("📊 시간별 분할 검증 데이터 로딩 중...")
        
        data = []
        labels = []
        class_indices = defaultdict(list)
        
        for class_idx, class_name in enumerate(self.class_names):
            print(f"  {class_name} 클래스 시간별 분할 데이터 로딩 중...")
            
            class_dir = self.data_path / class_name
            if not class_dir.exists():
                continue
            
            class_data_count = 0
            
            for sub_dir in class_dir.iterdir():
                if sub_dir.is_dir():
                    h5_files = list(sub_dir.glob("*.h5"))
                    
                    # 파일들을 정렬하고 마지막 20%만 테스트용으로 사용
                    h5_files.sort()
                    test_files = h5_files[-int(len(h5_files) * 0.2):]
                    
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
            
            print(f"    {class_name}: {class_data_count}개 시퀀스 (시간별 분할)")
        
        return np.array(data), np.array(labels), class_indices
    
    def load_no_augmentation_data(self):
        """증강 없는 데이터 - 원본 시퀀스만 사용"""
        print("📊 증강 없는 검증 데이터 로딩 중...")
        
        data = []
        labels = []
        class_indices = defaultdict(list)
        
        for class_idx, class_name in enumerate(self.class_names):
            print(f"  {class_name} 클래스 증강 없는 데이터 로딩 중...")
            
            class_dir = self.data_path / class_name
            if not class_dir.exists():
                continue
            
            class_data_count = 0
            
            for sub_dir in class_dir.iterdir():
                if sub_dir.is_dir():
                    h5_files = list(sub_dir.glob("*.h5"))
                    
                    # 랜덤하게 일부 파일만 선택
                    random.shuffle(h5_files)
                    selected_files = h5_files[:int(len(h5_files) * 0.3)]
                    
                    for h5_file in selected_files:
                        try:
                            with h5py.File(h5_file, 'r') as f:
                                sensor_data = f['sensor_data'][:]
                                
                                # 원본 시퀀스 그대로 사용 (증강 없이)
                                if len(sensor_data) >= self.sequence_length:
                                    # 첫 번째 시퀀스만 사용
                                    sequence = sensor_data[:self.sequence_length]
                                    data.append(sequence)
                                    labels.append(class_idx)
                                    class_indices[class_name].append(len(data) - 1)
                                    class_data_count += 1
                        except Exception as e:
                            print(f"⚠️ 파일 로드 오류 {h5_file}: {e}")
                            continue
            
            print(f"    {class_name}: {class_data_count}개 시퀀스 (증강 없음)")
        
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

class StrictValidator:
    """엄격한 검증기"""
    
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
    
    def test_performance(self, test_loader):
        """성능 테스트"""
        if self.model is None:
            print("❌ 모델이 로드되지 않았습니다.")
            return None
        
        print("🔍 엄격한 검증 성능 테스트 시작...")
        
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
    
    def analyze_results(self, predictions, labels, probabilities, validation_type):
        """결과 분석"""
        print(f"\n📊 {validation_type} 검증 결과 분석")
        print("=" * 60)
        
        # 기본 지표
        overall_accuracy = np.mean(predictions == labels)
        confidence_scores = np.max(probabilities, axis=1)
        avg_confidence = np.mean(confidence_scores)
        
        print(f"🎯 전체 정확도: {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)")
        print(f"🎯 평균 신뢰도: {avg_confidence:.4f} ({avg_confidence*100:.2f}%)")
        print(f"❌ 오분류 비율: {1-overall_accuracy:.4f} ({(1-overall_accuracy)*100:.2f}%)")
        
        # 클래스별 정확도
        print("\n📈 클래스별 정확도:")
        print("-" * 40)
        
        class_accuracies = {}
        for i, class_name in enumerate(self.class_names):
            class_mask = labels == i
            if np.sum(class_mask) > 0:
                class_acc = np.mean(predictions[class_mask] == labels[class_mask])
                class_count = np.sum(class_mask)
                class_accuracies[class_name] = class_acc
                print(f"{class_name:2s}: {class_acc:.4f} ({class_acc*100:.2f}%) - {class_count}개 샘플")
        
        # 과적합 점수 계산
        overfitting_score = 0
        reasons = []
        
        if overall_accuracy > 0.99:
            overfitting_score += 2
            reasons.append("정확도가 99%를 초과함")
        elif overall_accuracy > 0.95:
            overfitting_score += 1
            reasons.append("정확도가 95%를 초과함")
        
        if avg_confidence > 0.999:
            overfitting_score += 2
            reasons.append("신뢰도가 99.9%를 초과함")
        elif avg_confidence > 0.99:
            overfitting_score += 1
            reasons.append("신뢰도가 99%를 초과함")
        
        if (1-overall_accuracy) < 0.01:
            overfitting_score += 2
            reasons.append("오분류 비율이 1% 미만임")
        elif (1-overall_accuracy) < 0.05:
            overfitting_score += 1
            reasons.append("오분류 비율이 5% 미만임")
        
        # 클래스별 성능 편차
        if class_accuracies:
            acc_std = np.std(list(class_accuracies.values()))
            if acc_std > 0.05:
                overfitting_score += 1
                reasons.append("클래스별 성능 편차가 큼")
        
        print(f"\n🏆 과적합 점수: {overfitting_score}/6")
        
        if overfitting_score >= 4:
            print("🔴 높은 과적합 의심")
        elif overfitting_score >= 2:
            print("🟡 중간 과적합 의심")
        else:
            print("🟢 과적합 징후 없음")
        
        if reasons:
            print("\n📝 과적합 의심 이유:")
            for i, reason in enumerate(reasons, 1):
                print(f"   {i}. {reason}")
        
        return {
            'validation_type': validation_type,
            'overall_accuracy': float(overall_accuracy),
            'avg_confidence': float(avg_confidence),
            'overfitting_score': overfitting_score,
            'reasons': reasons,
            'class_accuracies': {k: float(v) for k, v in class_accuracies.items()}
        }

def main():
    """메인 함수"""
    print("🚀 엄격한 검증 시작")
    print("=" * 60)
    
    # 검증 타입들
    validation_types = ['user_split', 'time_split', 'no_augmentation']
    results = []
    
    validator = StrictValidator()
    
    for validation_type in validation_types:
        print(f"\n🔍 {validation_type} 검증 시작")
        print("-" * 40)
        
        # 검증 데이터 로드
        test_dataset = StrictValidationDataset(
            data_path="../SignGlove_HW/datasets/unified",
            sequence_length=20,
            validation_type=validation_type
        )
        
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)
        
        print(f"📊 {validation_type} 테스트 데이터: {len(test_dataset)}개 시퀀스")
        
        # 성능 테스트
        predictions, labels, probabilities = validator.test_performance(test_loader)
        
        if predictions is not None:
            # 결과 분석
            result = validator.analyze_results(predictions, labels, probabilities, validation_type)
            results.append(result)
            
            # 상세 분류 리포트
            print(f"\n📋 상세 분류 리포트 ({validation_type}):")
            print("-" * 40)
            report = classification_report(labels, predictions, 
                                         target_names=validator.class_names, 
                                         digits=4)
            print(report)
        else:
            print(f"❌ {validation_type} 검증을 완료할 수 없습니다.")
    
    # 종합 결과
    print("\n🎉 엄격한 검증 완료!")
    print("=" * 60)
    
    print("\n📊 종합 검증 결과:")
    print("-" * 40)
    
    for result in results:
        print(f"\n{result['validation_type']}:")
        print(f"  정확도: {result['overall_accuracy']:.4f}")
        print(f"  신뢰도: {result['avg_confidence']:.4f}")
        print(f"  과적합 점수: {result['overfitting_score']}/6")
        
        if result['overfitting_score'] >= 4:
            print("  상태: 🔴 높은 과적합 의심")
        elif result['overfitting_score'] >= 2:
            print("  상태: 🟡 중간 과적합 의심")
        else:
            print("  상태: 🟢 과적합 징후 없음")
    
    # 결과 저장
    with open('strict_validation_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n📁 결과가 'strict_validation_results.json'에 저장되었습니다.")

if __name__ == "__main__":
    main()
