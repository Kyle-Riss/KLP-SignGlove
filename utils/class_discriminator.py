#!/usr/bin/env python3
"""
ㄹ과 ㅕ 클래스 차별화 모델
다른 센서들에 영향 없이 두 클래스를 구분하는 방법들
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.unicode_minus'] = False

class ClassDiscriminator:
    """ㄹ과 ㅕ 클래스 차별화기"""
    
    def __init__(self, method='rule_based'):
        self.method = method
        self.thresholds = {}
        self.model = None
        self.is_trained = False
    
    def extract_discriminative_features(self, data):
        """차별화 특징 추출"""
        features = {}
        
        for i, sample_data in enumerate(data):
            # 1. 핵심 차이점 특징
            flex3_mean = np.mean(sample_data[:, 2])  # Flex3 평균
            flex5_mean = np.mean(sample_data[:, 4])  # Flex5 평균
            flex1_std = np.std(sample_data[:, 0])    # Flex1 표준편차
            
            # 2. 상관관계 특징
            flex5_imu_x_corr = np.corrcoef(sample_data[:, 4], sample_data[:, 5])[0, 1]
            flex1_imu_y_corr = np.corrcoef(sample_data[:, 0], sample_data[:, 6])[0, 1]
            
            # 3. 변화율 특징
            diff_flex1 = np.mean(np.diff(sample_data[:, 0]))
            
            # 4. 피크 특징
            peak_flex3 = np.max(sample_data[:, 2]) - np.min(sample_data[:, 2])
            peak_flex5 = np.max(sample_data[:, 4]) - np.min(sample_data[:, 4])
            
            features[i] = {
                'flex3_mean': flex3_mean,
                'flex5_mean': flex5_mean,
                'flex1_std': flex1_std,
                'flex5_imu_x_corr': flex5_imu_x_corr,
                'flex1_imu_y_corr': flex1_imu_y_corr,
                'diff_flex1': diff_flex1,
                'peak_flex3': peak_flex3,
                'peak_flex5': peak_flex5
            }
        
        return features
    
    def train_rule_based(self, class1_data, class2_data, class1_name='ㄹ', class2_name='ㅕ'):
        """규칙 기반 학습"""
        print(f'🔧 규칙 기반 차별화기 학습 중...')
        
        # 특징 추출
        class1_features = self.extract_discriminative_features(class1_data)
        class2_features = self.extract_discriminative_features(class2_data)
        
        # 각 특징별 임계값 계산
        feature_names = ['flex3_mean', 'flex5_mean', 'flex1_std', 'flex5_imu_x_corr', 
                        'flex1_imu_y_corr', 'diff_flex1', 'peak_flex3', 'peak_flex5']
        
        for feature_name in feature_names:
            class1_vals = [f[feature_name] for f in class1_features.values()]
            class2_vals = [f[feature_name] for f in class2_features.values()]
            
            # 분리 가능한 임계값 찾기
            threshold = self.find_optimal_threshold(class1_vals, class2_vals)
            self.thresholds[feature_name] = threshold
            
            print(f'  {feature_name}: 임계값 = {threshold:.4f}')
        
        self.is_trained = True
        print('✅ 규칙 기반 차별화기 학습 완료')
    
    def find_optimal_threshold(self, class1_vals, class2_vals):
        """최적 임계값 찾기"""
        all_vals = class1_vals + class2_vals
        min_val, max_val = min(all_vals), max(all_vals)
        
        best_threshold = (min_val + max_val) / 2
        best_separation = 0
        
        for threshold in np.linspace(min_val, max_val, 100):
            # 임계값 기준으로 분리
            class1_below = sum(1 for v in class1_vals if v < threshold)
            class2_above = sum(1 for v in class2_vals if v >= threshold)
            
            separation = class1_below + class2_above
            if separation > best_separation:
                best_separation = separation
                best_threshold = threshold
        
        return best_threshold
    
    def train_ml_model(self, class1_data, class2_data, class1_name='ㄹ', class2_name='ㅕ'):
        """머신러닝 모델 학습"""
        print(f'🤖 머신러닝 차별화기 학습 중...')
        
        # 특징 추출
        class1_features = self.extract_discriminative_features(class1_data)
        class2_features = self.extract_discriminative_features(class2_data)
        
        # 데이터 준비
        X = []
        y = []
        
        for features in class1_features.values():
            feature_vector = list(features.values())
            X.append(feature_vector)
            y.append(0)  # class1 (ㄹ)
        
        for features in class2_features.values():
            feature_vector = list(features.values())
            X.append(feature_vector)
            y.append(1)  # class2 (ㅕ)
        
        X = np.array(X)
        y = np.array(y)
        
        # 모델 선택 및 학습
        if self.method == 'random_forest':
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif self.method == 'svm':
            self.model = SVC(kernel='rbf', probability=True, random_state=42)
        
        self.model.fit(X, y)
        self.is_trained = True
        print('✅ 머신러닝 차별화기 학습 완료')
    
    def predict_rule_based(self, data):
        """규칙 기반 예측"""
        if not self.is_trained:
            raise ValueError("모델을 먼저 학습해야 합니다.")
        
        features = self.extract_discriminative_features([data])
        feature = features[0]
        
        # 각 특징별 점수 계산
        scores = []
        
        # Flex5 평균 (가장 큰 차이)
        if feature['flex5_mean'] < self.thresholds['flex5_mean']:
            scores.append(1)  # ㄹ에 가까움
        else:
            scores.append(0)  # ㅕ에 가까움
        
        # Flex3 평균
        if feature['flex3_mean'] < self.thresholds['flex3_mean']:
            scores.append(1)
        else:
            scores.append(0)
        
        # Flex1 표준편차
        if feature['flex1_std'] > self.thresholds['flex1_std']:
            scores.append(1)
        else:
            scores.append(0)
        
        # 상관관계 특징
        if abs(feature['flex5_imu_x_corr']) < abs(self.thresholds['flex5_imu_x_corr']):
            scores.append(1)
        else:
            scores.append(0)
        
        # 최종 결정 (다수결)
        final_score = np.mean(scores)
        prediction = 0 if final_score > 0.5 else 1  # 0: ㄹ, 1: ㅕ
        
        confidence = abs(final_score - 0.5) * 2  # 0~1 범위로 정규화
        
        return prediction, confidence, feature
    
    def predict_ml(self, data):
        """머신러닝 모델 예측"""
        if not self.is_trained:
            raise ValueError("모델을 먼저 학습해야 합니다.")
        
        features = self.extract_discriminative_features([data])
        feature_vector = list(features[0].values())
        
        prediction = self.model.predict([feature_vector])[0]
        confidence = np.max(self.model.predict_proba([feature_vector])[0])
        
        return prediction, confidence, features[0]
    
    def predict(self, data):
        """통합 예측"""
        if self.method == 'rule_based':
            return self.predict_rule_based(data)
        else:
            return self.predict_ml(data)

class EnhancedSGRU(nn.Module):
    """향상된 S-GRU (ㄹ/ㅕ 차별화 기능 포함)"""
    
    def __init__(self, input_size=8, hidden_size=32, num_classes=24, dropout=0.5):
        super(EnhancedSGRU, self).__init__()
        self.hidden_size = hidden_size
        
        # 기본 GRU 레이어
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True, dropout=dropout)
        
        # 차별화 특징 추출 레이어
        self.discriminative_fc = nn.Linear(hidden_size + 8, hidden_size)  # +8 for discriminative features
        
        # 분류 레이어
        self.fc = nn.Linear(hidden_size, num_classes)
        
        # 드롭아웃
        self.dropout = nn.Dropout(dropout)
        
        # ㄹ/ㅕ 차별화기
        self.discriminator = ClassDiscriminator(method='rule_based')
    
    def extract_discriminative_features(self, x):
        """차별화 특징 추출"""
        batch_size = x.size(0)
        features = torch.zeros(batch_size, 8).to(x.device)
        
        for i in range(batch_size):
            sample = x[i].cpu().numpy()
            
            # 핵심 차별화 특징
            flex3_mean = np.mean(sample[:, 2])
            flex5_mean = np.mean(sample[:, 4])
            flex1_std = np.std(sample[:, 0])
            
            # 상관관계
            flex5_imu_x_corr = np.corrcoef(sample[:, 4], sample[:, 5])[0, 1]
            if np.isnan(flex5_imu_x_corr):
                flex5_imu_x_corr = 0
            
            flex1_imu_y_corr = np.corrcoef(sample[:, 0], sample[:, 6])[0, 1]
            if np.isnan(flex1_imu_y_corr):
                flex1_imu_y_corr = 0
            
            # 변화율
            diff_flex1 = np.mean(np.diff(sample[:, 0]))
            
            # 피크
            peak_flex3 = np.max(sample[:, 2]) - np.min(sample[:, 2])
            peak_flex5 = np.max(sample[:, 4]) - np.min(sample[:, 4])
            
            features[i] = torch.tensor([
                flex3_mean, flex5_mean, flex1_std, flex5_imu_x_corr,
                flex1_imu_y_corr, diff_flex1, peak_flex3, peak_flex5
            ], dtype=torch.float32)
        
        return features
    
    def forward(self, x):
        # GRU 처리
        gru_out, _ = self.gru(x)
        last_output = gru_out[:, -1, :]
        
        # 차별화 특징 추출
        disc_features = self.extract_discriminative_features(x)
        
        # 특징 결합
        combined = torch.cat([last_output, disc_features], dim=1)
        
        # 차별화 특징 처리
        enhanced = self.discriminative_fc(combined)
        dropped = self.dropout(enhanced)
        
        # 분류
        output = self.fc(dropped)
        
        return output

def create_discrimination_strategy():
    """차별화 전략 생성"""
    print('🎯 ㄹ과 ㅕ 차별화 전략')
    print('=' * 50)
    
    strategies = {
        '1. 특징 기반 필터링': {
            'description': '핵심 차이점 특징만 사용하여 분류',
            'features': ['Flex5_mean', 'Flex3_mean', 'Flex1_std', 'Flex5-IMU_X_corr'],
            'advantage': '다른 센서에 영향 없음, 해석 가능',
            'method': 'rule_based'
        },
        '2. 앙상블 분류': {
            'description': '여러 특징 조합으로 투표 기반 분류',
            'features': '모든 차별화 특징',
            'advantage': '높은 정확도, 안정성',
            'method': 'ensemble'
        },
        '3. 하이브리드 모델': {
            'description': '기본 모델 + 차별화 모델 결합',
            'features': 'GRU + 차별화 특징',
            'advantage': '전체 성능 향상, 특정 클래스 정확도 개선',
            'method': 'hybrid'
        },
        '4. 후처리 필터': {
            'description': '추론 후 ㄹ/ㅕ 결과만 재검증',
            'features': '차별화 특징',
            'advantage': '기존 모델 수정 없음, 선택적 적용',
            'method': 'post_processing'
        }
    }
    
    for strategy_name, details in strategies.items():
        print(f'\n{strategy_name}:')
        print(f'  📝 {details["description"]}')
        print(f'  🎯 특징: {details["features"]}')
        print(f'  ✅ 장점: {details["advantage"]}')
        print(f'  🔧 방법: {details["method"]}')
    
    return strategies

def test_discrimination_methods(data_dir):
    """차별화 방법 테스트"""
    print('\n🧪 차별화 방법 테스트')
    print('=' * 50)
    
    # 데이터 로드
    class1_data = load_class_data(data_dir, 'ㄹ')
    class2_data = load_class_data(data_dir, 'ㅕ')
    
    if not class1_data or not class2_data:
        print('⚠️ 데이터 로드 실패')
        return
    
    # 1. 규칙 기반 차별화기 테스트
    print('\n🔧 규칙 기반 차별화기 테스트:')
    rule_discriminator = ClassDiscriminator(method='rule_based')
    rule_discriminator.train_rule_based(class1_data, class2_data)
    
    # 테스트
    correct = 0
    total = 0
    
    for data in class1_data:
        pred, conf, _ = rule_discriminator.predict(data)
        if pred == 0:  # ㄹ로 예측
            correct += 1
        total += 1
    
    for data in class2_data:
        pred, conf, _ = rule_discriminator.predict(data)
        if pred == 1:  # ㅕ로 예측
            correct += 1
        total += 1
    
    accuracy = correct / total
    print(f'  📊 정확도: {accuracy:.4f} ({correct}/{total})')
    
    # 2. 머신러닝 차별화기 테스트
    print('\n🤖 머신러닝 차별화기 테스트:')
    ml_discriminator = ClassDiscriminator(method='random_forest')
    ml_discriminator.train_ml_model(class1_data, class2_data)
    
    # 테스트
    correct = 0
    total = 0
    
    for data in class1_data:
        pred, conf, _ = ml_discriminator.predict(data)
        if pred == 0:
            correct += 1
        total += 1
    
    for data in class2_data:
        pred, conf, _ = ml_discriminator.predict(data)
        if pred == 1:
            correct += 1
        total += 1
    
    accuracy = correct / total
    print(f'  📊 정확도: {accuracy:.4f} ({correct}/{total})')

def load_class_data(data_dir, class_name):
    """클래스 데이터 로드"""
    class_dir = Path(data_dir) / class_name
    if not class_dir.exists():
        return []
    
    all_data = []
    for angle in range(1, 6):
        angle_dir = class_dir / str(angle)
        if angle_dir.exists():
            csv_files = list(angle_dir.glob("*.csv"))
            for file_path in csv_files:
                try:
                    df = pd.read_csv(file_path)
                    data = df.iloc[:, :8].values.astype(np.float32)
                    all_data.append(data)
                except:
                    pass
    
    return all_data

def main():
    """메인 실행 함수"""
    print('🎯 ㄹ과 ㅕ 차별화 방법')
    print('=' * 60)
    
    # 차별화 전략 제시
    strategies = create_discrimination_strategy()
    
    # 실제 테스트
    data_dir = 'real_data_filtered'
    test_discrimination_methods(data_dir)
    
    print('\n📋 구현 권장사항:')
    print('=' * 30)
    print('🎯 1. 후처리 필터 방식 (가장 안전)')
    print('   - 기존 모델 수정 없음')
    print('   - ㄹ/ㅕ 예측 결과만 재검증')
    print('   - 다른 클래스에 영향 없음')
    print('')
    print('🎯 2. 특징 기반 필터링 (가장 해석 가능)')
    print('   - Flex5_mean, Flex3_mean 등 핵심 특징만 사용')
    print('   - 규칙 기반으로 명확한 기준 제공')
    print('   - 실시간 적용 가능')
    print('')
    print('🎯 3. 하이브리드 모델 (가장 정확)')
    print('   - 기존 모델에 차별화 특징 추가')
    print('   - 전체 성능 향상 기대')
    print('   - 모델 재훈련 필요')

if __name__ == "__main__":
    main()

