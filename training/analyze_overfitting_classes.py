#!/usr/bin/env python3
"""
과적합 클래스 분석 스크립트
ㄱ, ㄴ, ㅂ, ㅇ, ㅎ, ㅏ, ㅣ 클래스의 1.000 성능 원인 분석
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import glob
import json
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# 프로젝트 루트 추가
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.deep_learning import DeepLearningPipeline
from training.label_mapping import KSLLabelMapper

class OverfittingAnalyzer:
    """과적합 클래스 분석기"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.label_mapper = KSLLabelMapper()
        
        # 과적합 의심 클래스들
        self.overfitting_classes = ['ㄱ', 'ㄴ', 'ㅂ', 'ㅇ', 'ㅎ', 'ㅏ', 'ㅣ']
        
        print(f"🔍 과적합 클래스 분석기 초기화")
        print(f"  장치: {self.device}")
        print(f"  분석 대상: {self.overfitting_classes}")
    
    def load_model_and_data(self):
        """모델과 데이터 로드"""
        print("🔄 모델과 데이터 로드 중...")
        
        # 모델 로드
        model = DeepLearningPipeline(
            input_features=8,
            hidden_dim=48,
            num_classes=self.label_mapper.get_num_classes(),
            sequence_length=20,
            num_layers=2,
            dropout=0.4
        ).to(self.device)
        
        # 최신 모델 가중치 로드
        model_paths = [
            'best_class_specific_model.pth',
            'best_anti_overfitting_model.pth',
            'best_balanced_model.pth'
        ]
        
        model_loaded = False
        for path in model_paths:
            if os.path.exists(path):
                model.load_state_dict(torch.load(path, map_location=self.device))
                print(f"✅ 모델 로드: {path}")
                model_loaded = True
                break
        
        if not model_loaded:
            print("⚠️  모델 파일을 찾을 수 없습니다. 분석을 중단합니다.")
            return None, None
        
        model.eval()
        
        # 데이터 로드
        data_dir = 'integrations/SignGlove_HW/github_unified_data'
        data_files = glob.glob(os.path.join(data_dir, "**/episode_*.csv"), recursive=True)
        
        print(f"📁 발견된 Episode 파일: {len(data_files)}개")
        
        return model, data_files
    
    def analyze_class_data_patterns(self, data_files):
        """클래스별 데이터 패턴 분석"""
        print("\n🔍 클래스별 데이터 패턴 분석")
        print("=" * 60)
        
        class_data = {class_name: [] for class_name in self.overfitting_classes}
        
        for file_path in data_files:
            try:
                # 클래스 이름과 시나리오 추출
                class_name, scenario_id = self._extract_class_and_scenario(file_path)
                if class_name not in self.overfitting_classes:
                    continue
                
                # CSV 파일 읽기
                df = pd.read_csv(file_path)
                sensor_data = df[['pitch', 'roll', 'yaw', 'flex1', 'flex2', 'flex3', 'flex4', 'flex5']].values
                
                # 통계 정보 계산
                stats = {
                    'mean': np.mean(sensor_data, axis=0),
                    'std': np.std(sensor_data, axis=0),
                    'min': np.min(sensor_data, axis=0),
                    'max': np.max(sensor_data, axis=0),
                    'scenario': scenario_id,
                    'file_path': file_path
                }
                
                class_data[class_name].append(stats)
                
            except Exception as e:
                print(f"⚠️  파일 분석 실패: {file_path} - {e}")
        
        # 클래스별 통계 분석
        for class_name in self.overfitting_classes:
            if class_data[class_name]:
                print(f"\n📊 {class_name} 클래스 분석:")
                print(f"  샘플 수: {len(class_data[class_name])}개")
                
                # 센서별 평균 통계
                all_means = np.array([stats['mean'] for stats in class_data[class_name]])
                all_stds = np.array([stats['std'] for stats in class_data[class_name]])
                
                print(f"  센서별 평균 (pitch, roll, yaw, flex1-5):")
                print(f"    평균: {np.mean(all_means, axis=0)}")
                print(f"    표준편차: {np.mean(all_stds, axis=0)}")
                
                # 시나리오별 분포
                scenarios = [stats['scenario'] for stats in class_data[class_name]]
                scenario_counts = Counter(scenarios)
                print(f"  시나리오별 분포: {dict(scenario_counts)}")
        
        return class_data
    
    def analyze_cross_scenario_performance(self, model, data_files):
        """시나리오 간 성능 분석"""
        print("\n🔍 시나리오 간 성능 분석")
        print("=" * 60)
        
        # 시나리오별 데이터 수집
        scenario_data = {i: [] for i in range(1, 6)}
        
        for file_path in data_files:
            try:
                class_name, scenario_id = self._extract_class_and_scenario(file_path)
                if class_name not in self.overfitting_classes:
                    continue
                
                scenario_num = int(scenario_id)
                df = pd.read_csv(file_path)
                sensor_data = df[['pitch', 'roll', 'yaw', 'flex1', 'flex2', 'flex3', 'flex4', 'flex5']].values
                
                # 윈도우 생성
                windows = self._create_windows(sensor_data, window_size=20, stride=5)
                
                for window in windows:
                    scenario_data[scenario_num].append({
                        'data': window,
                        'class_name': class_name,
                        'class_id': self.label_mapper.class_to_id[class_name]
                    })
                
            except Exception as e:
                print(f"⚠️  파일 분석 실패: {file_path} - {e}")
        
        # 시나리오별 성능 테스트
        for class_name in self.overfitting_classes:
            print(f"\n🎯 {class_name} 클래스 시나리오별 성능:")
            
            class_id = self.label_mapper.class_to_id[class_name]
            
            for train_scenario in [1, 2, 3]:
                for test_scenario in [4, 5]:
                    # 훈련 시나리오에서 데이터 추출
                    train_data = []
                    train_labels = []
                    
                    for scenario in [train_scenario]:
                        for item in scenario_data[scenario]:
                            if item['class_name'] == class_name:
                                train_data.append(item['data'])
                                train_labels.append(item['class_id'])
                    
                    # 테스트 시나리오에서 데이터 추출
                    test_data = []
                    test_labels = []
                    
                    for item in scenario_data[test_scenario]:
                        if item['class_name'] == class_name:
                            test_data.append(item['data'])
                            test_labels.append(item['class_id'])
                    
                    if len(train_data) > 0 and len(test_data) > 0:
                        # 간단한 유사도 계산 (코사인 유사도)
                        train_mean = np.mean(train_data, axis=0)
                        test_mean = np.mean(test_data, axis=0)
                        
                        similarity = self._cosine_similarity(train_mean.flatten(), test_mean.flatten())
                        
                        print(f"    훈련 시나리오 {train_scenario} → 테스트 시나리오 {test_scenario}: 유사도 {similarity:.3f}")
    
    def analyze_sensor_patterns(self, data_files):
        """센서 패턴 분석"""
        print("\n🔍 센서 패턴 분석")
        print("=" * 60)
        
        for class_name in self.overfitting_classes:
            print(f"\n📊 {class_name} 클래스 센서 패턴:")
            
            class_files = [f for f in data_files if class_name in f]
            
            if not class_files:
                continue
            
            # 첫 번째 파일로 패턴 분석
            df = pd.read_csv(class_files[0])
            sensor_data = df[['pitch', 'roll', 'yaw', 'flex1', 'flex2', 'flex3', 'flex4', 'flex5']].values
            
            # 센서별 특성 분석
            sensor_names = ['pitch', 'roll', 'yaw', 'flex1', 'flex2', 'flex3', 'flex4', 'flex5']
            
            print(f"  센서별 통계:")
            for i, sensor_name in enumerate(sensor_names):
                sensor_values = sensor_data[:, i]
                print(f"    {sensor_name}: 평균={np.mean(sensor_values):.3f}, 표준편차={np.std(sensor_values):.3f}, 범위=[{np.min(sensor_values):.3f}, {np.max(sensor_values):.3f}]")
            
            # 센서 간 상관관계 분석
            correlation_matrix = np.corrcoef(sensor_data.T)
            
            print(f"  센서 간 상관관계 (상위 3개):")
            for i in range(len(sensor_names)):
                for j in range(i+1, len(sensor_names)):
                    corr = correlation_matrix[i, j]
                    if abs(corr) > 0.7:  # 높은 상관관계만 표시
                        print(f"    {sensor_names[i]} ↔ {sensor_names[j]}: {corr:.3f}")
    
    def analyze_class_distinctiveness(self, data_files):
        """클래스 간 구별성 분석"""
        print("\n🔍 클래스 간 구별성 분석")
        print("=" * 60)
        
        # 모든 클래스의 평균 패턴 수집
        class_patterns = {}
        
        for file_path in data_files:
            try:
                class_name, scenario_id = self._extract_class_and_scenario(file_path)
                if class_name not in self.label_mapper.class_to_id:
                    continue
                
                df = pd.read_csv(file_path)
                sensor_data = df[['pitch', 'roll', 'yaw', 'flex1', 'flex2', 'flex3', 'flex4', 'flex5']].values
                
                if class_name not in class_patterns:
                    class_patterns[class_name] = []
                
                class_patterns[class_name].append(np.mean(sensor_data, axis=0))
                
            except Exception as e:
                continue
        
        # 과적합 클래스들 간의 유사도 분석
        print("📊 과적합 클래스들 간의 유사도:")
        
        for i, class1 in enumerate(self.overfitting_classes):
            for j, class2 in enumerate(self.overfitting_classes[i+1:], i+1):
                if class1 in class_patterns and class2 in class_patterns:
                    # 평균 패턴 간 유사도 계산
                    pattern1 = np.mean(class_patterns[class1], axis=0)
                    pattern2 = np.mean(class_patterns[class2], axis=0)
                    
                    similarity = self._cosine_similarity(pattern1, pattern2)
                    print(f"  {class1} ↔ {class2}: 유사도 {similarity:.3f}")
        
        # 과적합 클래스 vs 다른 클래스들 간의 유사도
        print("\n📊 과적합 클래스 vs 다른 클래스들 간의 유사도:")
        
        other_classes = [name for name in self.label_mapper.class_to_id.keys() 
                        if name not in self.overfitting_classes]
        
        for overfitting_class in self.overfitting_classes:
            if overfitting_class not in class_patterns:
                continue
            
            pattern1 = np.mean(class_patterns[overfitting_class], axis=0)
            
            similarities = []
            for other_class in other_classes:
                if other_class in class_patterns:
                    pattern2 = np.mean(class_patterns[other_class], axis=0)
                    similarity = self._cosine_similarity(pattern1, pattern2)
                    similarities.append((other_class, similarity))
            
            # 가장 유사한 클래스 3개 표시
            similarities.sort(key=lambda x: x[1], reverse=True)
            print(f"  {overfitting_class}와 가장 유사한 클래스들:")
            for other_class, similarity in similarities[:3]:
                print(f"    {other_class}: {similarity:.3f}")
    
    def _extract_class_and_scenario(self, file_path):
        """파일 경로에서 클래스 이름과 시나리오 추출"""
        parts = file_path.split(os.sep)
        
        # 클래스 이름 찾기
        class_name = None
        for part in parts:
            if part in self.label_mapper.class_to_id:
                class_name = part
                break
        
        # 시나리오 ID 찾기 (숫자 폴더)
        scenario_id = None
        for part in parts:
            if part.isdigit() and 1 <= int(part) <= 5:
                scenario_id = part
                break
        
        return class_name, scenario_id
    
    def _create_windows(self, sensor_data, window_size=20, stride=5):
        """윈도우 생성"""
        windows = []
        for i in range(0, len(sensor_data) - window_size + 1, stride):
            window = sensor_data[i:i + window_size]
            windows.append(window)
        return windows
    
    def _cosine_similarity(self, a, b):
        """코사인 유사도 계산"""
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        return dot_product / (norm_a * norm_b) if norm_a * norm_b != 0 else 0
    
    def save_analysis_report(self):
        """분석 리포트 저장"""
        report = {
            'overfitting_classes': self.overfitting_classes,
            'analysis_summary': {
                'total_classes': len(self.label_mapper.class_to_id),
                'overfitting_ratio': len(self.overfitting_classes) / len(self.label_mapper.class_to_id),
                'analysis_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        }
        
        with open('overfitting_analysis_report.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print("\n📄 분석 리포트 저장: overfitting_analysis_report.json")

def main():
    """메인 함수"""
    print("🔍 과적합 클래스 분석 시작")
    print("=" * 60)
    
    # 분석기 초기화
    analyzer = OverfittingAnalyzer()
    
    # 모델과 데이터 로드
    model, data_files = analyzer.load_model_and_data()
    
    if model is None or data_files is None:
        print("❌ 모델 또는 데이터 로드 실패")
        return
    
    # 1. 클래스별 데이터 패턴 분석
    analyzer.analyze_class_data_patterns(data_files)
    
    # 2. 시나리오 간 성능 분석
    analyzer.analyze_cross_scenario_performance(model, data_files)
    
    # 3. 센서 패턴 분석
    analyzer.analyze_sensor_patterns(data_files)
    
    # 4. 클래스 간 구별성 분석
    analyzer.analyze_class_distinctiveness(data_files)
    
    # 5. 분석 리포트 저장
    analyzer.save_analysis_report()
    
    print("\n🎉 과적합 클래스 분석 완료!")
    print("\n📋 분석 결과 요약:")
    print("  1. 클래스별 데이터 패턴 분석")
    print("  2. 시나리오 간 성능 분석")
    print("  3. 센서 패턴 분석")
    print("  4. 클래스 간 구별성 분석")
    print("  5. 분석 리포트 저장")

if __name__ == "__main__":
    main()
