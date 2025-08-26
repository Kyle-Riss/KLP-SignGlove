#!/usr/bin/env python3
"""
실패한 클래스 분석 스크립트
ㅊ, ㅕ 클래스의 0.000 성능 원인 분석
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

class FailedClassAnalyzer:
    """실패한 클래스 분석기"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.label_mapper = KSLLabelMapper()
        
        # 실패한 클래스들
        self.failed_classes = ['ㅊ', 'ㅕ']
        
        print(f"🔍 실패한 클래스 분석기 초기화")
        print(f"  장치: {self.device}")
        print(f"  분석 대상: {self.failed_classes}")
    
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
    
    def analyze_failed_class_data(self, data_files):
        """실패한 클래스 데이터 분석"""
        print("\n🔍 실패한 클래스 데이터 분석")
        print("=" * 60)
        
        for class_name in self.failed_classes:
            print(f"\n📊 {class_name} 클래스 상세 분석:")
            
            # 해당 클래스의 모든 파일 찾기
            class_files = [f for f in data_files if class_name in f]
            
            if not class_files:
                print(f"  ❌ {class_name} 클래스 파일을 찾을 수 없습니다.")
                continue
            
            print(f"  📁 파일 수: {len(class_files)}개")
            
            # 각 파일별 분석
            all_sensor_data = []
            scenario_data = {i: [] for i in range(1, 6)}
            
            for file_path in class_files:
                try:
                    # 시나리오 추출
                    _, scenario_id = self._extract_class_and_scenario(file_path)
                    scenario_num = int(scenario_id)
                    
                    # CSV 파일 읽기
                    df = pd.read_csv(file_path)
                    sensor_data = df[['pitch', 'roll', 'yaw', 'flex1', 'flex2', 'flex3', 'flex4', 'flex5']].values
                    
                    all_sensor_data.append(sensor_data)
                    scenario_data[scenario_num].append(sensor_data)
                    
                    # 파일별 통계
                    file_stats = {
                        'mean': np.mean(sensor_data, axis=0),
                        'std': np.std(sensor_data, axis=0),
                        'min': np.min(sensor_data, axis=0),
                        'max': np.max(sensor_data, axis=0),
                        'scenario': scenario_num
                    }
                    
                    print(f"    📄 {os.path.basename(file_path)} (시나리오 {scenario_num}):")
                    print(f"      센서별 평균: {file_stats['mean']}")
                    print(f"      센서별 표준편차: {file_stats['std']}")
                    
                except Exception as e:
                    print(f"    ⚠️  파일 분석 실패: {file_path} - {e}")
            
            if all_sensor_data:
                # 전체 통계
                all_data = np.vstack(all_sensor_data)
                print(f"\n  📊 {class_name} 전체 통계:")
                print(f"    전체 평균: {np.mean(all_data, axis=0)}")
                print(f"    전체 표준편차: {np.std(all_data, axis=0)}")
                print(f"    전체 범위: [{np.min(all_data, axis=0)}, {np.max(all_data, axis=0)}]")
                
                # 시나리오별 분석
                print(f"\n  📊 {class_name} 시나리오별 분석:")
                for scenario in range(1, 6):
                    if scenario_data[scenario]:
                        scenario_all = np.vstack(scenario_data[scenario])
                        print(f"    시나리오 {scenario}: {len(scenario_data[scenario])}개 파일")
                        print(f"      평균: {np.mean(scenario_all, axis=0)}")
                        print(f"      표준편차: {np.std(scenario_all, axis=0)}")
    
    def analyze_class_confusion(self, model, data_files):
        """클래스 혼동 분석"""
        print("\n🔍 클래스 혼동 분석")
        print("=" * 60)
        
        # 실패한 클래스들의 데이터 수집
        failed_data = {class_name: [] for class_name in self.failed_classes}
        
        for file_path in data_files:
            try:
                class_name, scenario_id = self._extract_class_and_scenario(file_path)
                if class_name not in self.failed_classes:
                    continue
                
                df = pd.read_csv(file_path)
                sensor_data = df[['pitch', 'roll', 'yaw', 'flex1', 'flex2', 'flex3', 'flex4', 'flex5']].values
                
                # 윈도우 생성
                windows = self._create_windows(sensor_data, window_size=20, stride=5)
                
                for window in windows:
                    failed_data[class_name].append({
                        'data': window,
                        'scenario': int(scenario_id)
                    })
                
            except Exception as e:
                continue
        
        # 모델 예측 테스트
        for class_name in self.failed_classes:
            if not failed_data[class_name]:
                continue
            
            print(f"\n🎯 {class_name} 클래스 모델 예측 분석:")
            
            predictions = []
            true_label = self.label_mapper.class_to_id[class_name]
            
            for item in failed_data[class_name]:
                data = torch.FloatTensor(item['data']).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    outputs = model(data)
                    _, predicted = outputs['class_logits'].max(1)
                    predictions.append(predicted.item())
            
            # 예측 결과 분석
            prediction_counts = Counter(predictions)
            print(f"  예측 결과 분포:")
            for pred_id, count in prediction_counts.most_common():
                pred_class = list(self.label_mapper.class_to_id.keys())[pred_id]
                percentage = (count / len(predictions)) * 100
                print(f"    {pred_class}: {count}개 ({percentage:.1f}%)")
            
            # 시나리오별 예측 분석
            print(f"  시나리오별 예측 분석:")
            for scenario in range(1, 6):
                scenario_predictions = [p for i, p in enumerate(predictions) 
                                      if failed_data[class_name][i]['scenario'] == scenario]
                
                if scenario_predictions:
                    scenario_counts = Counter(scenario_predictions)
                    most_common = scenario_counts.most_common(1)[0]
                    pred_class = list(self.label_mapper.class_to_id.keys())[most_common[0]]
                    percentage = (most_common[1] / len(scenario_predictions)) * 100
                    print(f"    시나리오 {scenario}: 주로 {pred_class}로 예측 ({percentage:.1f}%)")
    
    def analyze_sensor_patterns(self, data_files):
        """센서 패턴 분석"""
        print("\n🔍 센서 패턴 분석")
        print("=" * 60)
        
        for class_name in self.failed_classes:
            print(f"\n📊 {class_name} 클래스 센서 패턴:")
            
            class_files = [f for f in data_files if class_name in f]
            
            if not class_files:
                continue
            
            # 모든 파일의 데이터 통합
            all_data = []
            for file_path in class_files:
                try:
                    df = pd.read_csv(file_path)
                    sensor_data = df[['pitch', 'roll', 'yaw', 'flex1', 'flex2', 'flex3', 'flex4', 'flex5']].values
                    all_data.append(sensor_data)
                except:
                    continue
            
            if not all_data:
                continue
            
            all_data = np.vstack(all_data)
            
            # 센서별 특성 분석
            sensor_names = ['pitch', 'roll', 'yaw', 'flex1', 'flex2', 'flex3', 'flex4', 'flex5']
            
            print(f"  센서별 통계:")
            for i, sensor_name in enumerate(sensor_names):
                sensor_values = all_data[:, i]
                print(f"    {sensor_name}: 평균={np.mean(sensor_values):.3f}, 표준편차={np.std(sensor_values):.3f}, 범위=[{np.min(sensor_values):.3f}, {np.max(sensor_values):.3f}]")
            
            # 센서 간 상관관계 분석
            correlation_matrix = np.corrcoef(all_data.T)
            
            print(f"  센서 간 상관관계 (상위 5개):")
            correlations = []
            for i in range(len(sensor_names)):
                for j in range(i+1, len(sensor_names)):
                    corr = correlation_matrix[i, j]
                    correlations.append((sensor_names[i], sensor_names[j], abs(corr)))
            
            # 상관관계 높은 순으로 정렬
            correlations.sort(key=lambda x: x[2], reverse=True)
            for i, (sensor1, sensor2, corr) in enumerate(correlations[:5]):
                print(f"    {sensor1} ↔ {sensor2}: {corr:.3f}")
    
    def compare_with_successful_classes(self, data_files):
        """성공한 클래스와 비교 분석"""
        print("\n🔍 성공한 클래스와 비교 분석")
        print("=" * 60)
        
        # 성공한 클래스들 (1.000 성능)
        successful_classes = ['ㄱ', 'ㄴ', 'ㅂ', 'ㅇ', 'ㅎ', 'ㅏ', 'ㅣ']
        
        # 모든 클래스의 평균 패턴 수집
        class_patterns = {}
        
        for file_path in data_files:
            try:
                class_name, scenario_id = self._extract_class_and_scenario(file_path)
                if class_name not in self.label_mapper.class_to_id:
                    continue
                
                if class_name not in self.failed_classes + successful_classes:
                    continue
                
                df = pd.read_csv(file_path)
                sensor_data = df[['pitch', 'roll', 'yaw', 'flex1', 'flex2', 'flex3', 'flex4', 'flex5']].values
                
                if class_name not in class_patterns:
                    class_patterns[class_name] = []
                
                class_patterns[class_name].append(np.mean(sensor_data, axis=0))
                
            except Exception as e:
                continue
        
        # 실패한 클래스 vs 성공한 클래스 비교
        for failed_class in self.failed_classes:
            if failed_class not in class_patterns:
                continue
            
            print(f"\n📊 {failed_class} vs 성공한 클래스들 비교:")
            
            failed_pattern = np.mean(class_patterns[failed_class], axis=0)
            
            similarities = []
            for successful_class in successful_classes:
                if successful_class in class_patterns:
                    successful_pattern = np.mean(class_patterns[successful_class], axis=0)
                    similarity = self._cosine_similarity(failed_pattern, successful_pattern)
                    similarities.append((successful_class, similarity))
            
            # 유사도 순으로 정렬
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            print(f"  {failed_class}와 가장 유사한 성공한 클래스들:")
            for successful_class, similarity in similarities[:3]:
                print(f"    {successful_class}: 유사도 {similarity:.3f}")
            
            # 센서별 차이점 분석
            print(f"  센서별 차이점 (가장 유사한 클래스와 비교):")
            if similarities:
                most_similar_class = similarities[0][0]
                most_similar_pattern = np.mean(class_patterns[most_similar_class], axis=0)
                
                sensor_names = ['pitch', 'roll', 'yaw', 'flex1', 'flex2', 'flex3', 'flex4', 'flex5']
                for i, sensor_name in enumerate(sensor_names):
                    diff = abs(failed_pattern[i] - most_similar_pattern[i])
                    print(f"    {sensor_name}: 차이 {diff:.3f}")
    
    def analyze_data_quality(self, data_files):
        """데이터 품질 분석"""
        print("\n🔍 데이터 품질 분석")
        print("=" * 60)
        
        for class_name in self.failed_classes:
            print(f"\n📊 {class_name} 데이터 품질 분석:")
            
            class_files = [f for f in data_files if class_name in f]
            
            if not class_files:
                continue
            
            # 데이터 길이 분석
            lengths = []
            for file_path in class_files:
                try:
                    df = pd.read_csv(file_path)
                    lengths.append(len(df))
                except:
                    continue
            
            if lengths:
                print(f"  데이터 길이 통계:")
                print(f"    평균: {np.mean(lengths):.1f}")
                print(f"    표준편차: {np.std(lengths):.1f}")
                print(f"    최소: {np.min(lengths)}")
                print(f"    최대: {np.max(lengths)}")
            
            # 결측값 분석
            missing_data = []
            for file_path in class_files:
                try:
                    df = pd.read_csv(file_path)
                    missing_count = df[['pitch', 'roll', 'yaw', 'flex1', 'flex2', 'flex3', 'flex4', 'flex5']].isnull().sum().sum()
                    missing_data.append(missing_count)
                except:
                    continue
            
            if missing_data:
                print(f"  결측값 통계:")
                print(f"    총 결측값: {sum(missing_data)}")
                print(f"    평균 결측값: {np.mean(missing_data):.1f}")
            
            # 이상값 분석
            outlier_data = []
            for file_path in class_files:
                try:
                    df = pd.read_csv(file_path)
                    sensor_data = df[['pitch', 'roll', 'yaw', 'flex1', 'flex2', 'flex3', 'flex4', 'flex5']].values
                    
                    # Z-score 기반 이상값 탐지
                    z_scores = np.abs((sensor_data - np.mean(sensor_data, axis=0)) / np.std(sensor_data, axis=0))
                    outliers = np.sum(z_scores > 3)  # 3 표준편차 이상
                    outlier_data.append(outliers)
                except:
                    continue
            
            if outlier_data:
                print(f"  이상값 통계 (Z-score > 3):")
                print(f"    총 이상값: {sum(outlier_data)}")
                print(f"    평균 이상값: {np.mean(outlier_data):.1f}")
    
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
            'failed_classes': self.failed_classes,
            'analysis_summary': {
                'total_classes': len(self.label_mapper.class_to_id),
                'failed_ratio': len(self.failed_classes) / len(self.label_mapper.class_to_id),
                'analysis_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        }
        
        with open('failed_class_analysis_report.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print("\n📄 분석 리포트 저장: failed_class_analysis_report.json")

def main():
    """메인 함수"""
    print("🔍 실패한 클래스 분석 시작")
    print("=" * 60)
    
    # 분석기 초기화
    analyzer = FailedClassAnalyzer()
    
    # 모델과 데이터 로드
    model, data_files = analyzer.load_model_and_data()
    
    if model is None or data_files is None:
        print("❌ 모델 또는 데이터 로드 실패")
        return
    
    # 1. 실패한 클래스 데이터 분석
    analyzer.analyze_failed_class_data(data_files)
    
    # 2. 클래스 혼동 분석
    analyzer.analyze_class_confusion(model, data_files)
    
    # 3. 센서 패턴 분석
    analyzer.analyze_sensor_patterns(data_files)
    
    # 4. 성공한 클래스와 비교 분석
    analyzer.compare_with_successful_classes(data_files)
    
    # 5. 데이터 품질 분석
    analyzer.analyze_data_quality(data_files)
    
    # 6. 분석 리포트 저장
    analyzer.save_analysis_report()
    
    print("\n🎉 실패한 클래스 분석 완료!")
    print("\n📋 분석 결과 요약:")
    print("  1. 실패한 클래스 데이터 분석")
    print("  2. 클래스 혼동 분석")
    print("  3. 센서 패턴 분석")
    print("  4. 성공한 클래스와 비교 분석")
    print("  5. 데이터 품질 분석")
    print("  6. 분석 리포트 저장")

if __name__ == "__main__":
    main()
