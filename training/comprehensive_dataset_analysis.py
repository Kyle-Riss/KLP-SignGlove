#!/usr/bin/env python3
"""
전체 24개 자음/모음 데이터셋 종합 분석 스크립트
각 클래스별 특성 분석 및 맞춤형 솔루션 제안
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import glob
import json
from collections import defaultdict, Counter
import warnings
from scipy import stats
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
warnings.filterwarnings('ignore')

# 프로젝트 루트 추가
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from training.label_mapping import KSLLabelMapper

class ComprehensiveDatasetAnalyzer:
    """전체 데이터셋 종합 분석기"""
    
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.label_mapper = KSLLabelMapper()
        self.all_classes = self.label_mapper.get_all_classes()
        
        # 분석 결과 저장
        self.analysis_results = {}
        self.class_data = {}
        self.similarity_matrix = None
        
        print(f"🔍 전체 24개 자음/모음 데이터셋 종합 분석 시작")
        print(f"📁 데이터 디렉토리: {data_dir}")
        print(f"🎯 분석 대상: {len(self.all_classes)}개 클래스")
    
    def analyze_all_classes(self):
        """전체 클래스 분석 실행"""
        print(f"\n📊 1단계: 기본 데이터 로딩 및 통계 분석")
        self._load_all_data()
        
        print(f"\n📊 2단계: 센서별 특성 분석")
        self._analyze_sensor_characteristics()
        
        print(f"\n📊 3단계: 클래스별 패턴 분석")
        self._analyze_class_patterns()
        
        print(f"\n📊 4단계: 클래스 간 유사도 분석")
        self._analyze_class_similarities()
        
        print(f"\n📊 5단계: 데이터 품질 분석")
        self._analyze_data_quality()
        
        print(f"\n📊 6단계: 시각화 및 보고서 생성")
        self._create_visualizations()
        self._generate_comprehensive_report()
        
        print(f"\n✅ 종합 분석 완료!")
    
    def _load_all_data(self):
        """전체 데이터 로딩"""
        print("  📥 데이터 로딩 중...")
        
        for class_name in self.all_classes:
            print(f"    {class_name} 클래스 로딩 중...")
            
            # 파일 패턴으로 데이터 찾기
            pattern = os.path.join(self.data_dir, f"**/{class_name}/**/episode_*.csv")
            files = glob.glob(pattern, recursive=True)
            
            if not files:
                print(f"    ⚠️  {class_name}: 파일을 찾을 수 없음")
                continue
            
            # 데이터 로딩
            class_data = []
            for file_path in files:
                try:
                    df = pd.read_csv(file_path)
                    sensor_data = df[['pitch', 'roll', 'yaw', 'flex1', 'flex2', 'flex3', 'flex4', 'flex5']].values
                    class_data.append(sensor_data)
                except Exception as e:
                    print(f"    ⚠️  {file_path} 로드 실패: {e}")
            
            if class_data:
                self.class_data[class_name] = class_data
                print(f"    ✅ {class_name}: {len(class_data)}개 파일, {sum(len(d) for d in class_data)}개 샘플")
            else:
                print(f"    ❌ {class_name}: 유효한 데이터 없음")
    
    def _analyze_sensor_characteristics(self):
        """센서별 특성 분석"""
        print("  🔍 센서별 특성 분석 중...")
        
        sensor_names = ['pitch', 'roll', 'yaw', 'flex1', 'flex2', 'flex3', 'flex4', 'flex5']
        
        for class_name, data_list in self.class_data.items():
            print(f"    📊 {class_name} 센서 분석...")
            
            # 모든 데이터 합치기
            all_data = np.vstack(data_list)
            
            class_stats = {}
            for i, sensor_name in enumerate(sensor_names):
                sensor_values = all_data[:, i]
                
                # 기본 통계
                stats_dict = {
                    'mean': np.mean(sensor_values),
                    'std': np.std(sensor_values),
                    'min': np.min(sensor_values),
                    'max': np.max(sensor_values),
                    'range': np.max(sensor_values) - np.min(sensor_values),
                    'median': np.median(sensor_values),
                    'q25': np.percentile(sensor_values, 25),
                    'q75': np.percentile(sensor_values, 75),
                    'iqr': np.percentile(sensor_values, 75) - np.percentile(sensor_values, 25),
                    'skewness': stats.skew(sensor_values),
                    'kurtosis': stats.kurtosis(sensor_values),
                    'zero_count': np.sum(sensor_values == 0),
                    'zero_ratio': np.sum(sensor_values == 0) / len(sensor_values),
                    'outlier_count': np.sum(np.abs(sensor_values - np.mean(sensor_values)) > 3 * np.std(sensor_values)),
                    'outlier_ratio': np.sum(np.abs(sensor_values - np.mean(sensor_values)) > 3 * np.std(sensor_values)) / len(sensor_values)
                }
                
                class_stats[sensor_name] = stats_dict
            
            self.analysis_results[class_name] = {
                'sensor_stats': class_stats,
                'data_count': len(data_list),
                'total_samples': sum(len(d) for d in data_list),
                'avg_sequence_length': np.mean([len(d) for d in data_list])
            }
    
    def _analyze_class_patterns(self):
        """클래스별 패턴 분석"""
        print("  🔍 클래스별 패턴 분석 중...")
        
        for class_name, data_list in self.class_data.items():
            print(f"    📊 {class_name} 패턴 분석...")
            
            # 시계열 패턴 분석
            pattern_analysis = self._analyze_temporal_patterns(data_list)
            
            # 센서 간 상관관계 분석
            correlation_analysis = self._analyze_sensor_correlations(data_list)
            
            # 클래스별 특징 추출
            feature_analysis = self._extract_class_features(data_list)
            
            self.analysis_results[class_name].update({
                'temporal_patterns': pattern_analysis,
                'sensor_correlations': correlation_analysis,
                'class_features': feature_analysis
            })
    
    def _analyze_temporal_patterns(self, data_list):
        """시계열 패턴 분석"""
        patterns = {}
        
        # 시퀀스 길이 분석
        lengths = [len(data) for data in data_list]
        patterns['length_stats'] = {
            'mean': np.mean(lengths),
            'std': np.std(lengths),
            'min': np.min(lengths),
            'max': np.max(lengths),
            'consistency': 1 - (np.std(lengths) / np.mean(lengths)) if np.mean(lengths) > 0 else 0
        }
        
        # 시계열 안정성 분석
        stability_scores = []
        for data in data_list:
            # 각 센서별 변동성 계산
            sensor_variances = np.var(data, axis=0)
            stability_score = 1 / (1 + np.mean(sensor_variances))
            stability_scores.append(stability_score)
        
        patterns['stability'] = {
            'mean': np.mean(stability_scores),
            'std': np.std(stability_scores),
            'min': np.min(stability_scores),
            'max': np.max(stability_scores)
        }
        
        return patterns
    
    def _analyze_sensor_correlations(self, data_list):
        """센서 간 상관관계 분석"""
        # 모든 데이터 합치기
        all_data = np.vstack(data_list)
        
        # 상관관계 행렬 계산
        correlation_matrix = np.corrcoef(all_data.T)
        
        # 높은 상관관계 쌍 찾기
        high_correlations = []
        sensor_names = ['pitch', 'roll', 'yaw', 'flex1', 'flex2', 'flex3', 'flex4', 'flex5']
        
        for i in range(len(sensor_names)):
            for j in range(i+1, len(sensor_names)):
                corr_value = correlation_matrix[i, j]
                if abs(corr_value) > 0.7:  # 높은 상관관계 임계값
                    high_correlations.append({
                        'sensor1': sensor_names[i],
                        'sensor2': sensor_names[j],
                        'correlation': corr_value
                    })
        
        return {
            'correlation_matrix': correlation_matrix.tolist(),
            'high_correlations': high_correlations
        }
    
    def _extract_class_features(self, data_list):
        """클래스별 특징 추출"""
        features = {}
        
        # 모든 데이터 합치기
        all_data = np.vstack(data_list)
        
        # 센서별 주요 특징
        sensor_names = ['pitch', 'roll', 'yaw', 'flex1', 'flex2', 'flex3', 'flex4', 'flex5']
        
        for i, sensor_name in enumerate(sensor_names):
            sensor_values = all_data[:, i]
            
            # 분포 특징
            features[sensor_name] = {
                'distribution_type': self._classify_distribution(sensor_values),
                'peak_count': self._count_peaks(sensor_values),
                'trend': self._analyze_trend(sensor_values),
                'seasonality': self._detect_seasonality(sensor_values)
            }
        
        return features
    
    def _classify_distribution(self, values):
        """분포 유형 분류"""
        # 정규성 검정
        _, p_value = stats.normaltest(values)
        
        if p_value > 0.05:
            return "normal"
        elif np.abs(stats.skew(values)) > 1:
            return "skewed"
        else:
            return "non_normal"
    
    def _count_peaks(self, values):
        """피크 개수 계산"""
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(values, height=np.mean(values) + np.std(values))
        return len(peaks)
    
    def _analyze_trend(self, values):
        """트렌드 분석"""
        x = np.arange(len(values))
        slope, _, r_value, _, _ = stats.linregress(x, values)
        
        if abs(r_value) > 0.7:
            if slope > 0:
                return "increasing"
            else:
                return "decreasing"
        else:
            return "stable"
    
    def _detect_seasonality(self, values):
        """계절성 검출"""
        # 간단한 계절성 검출 (FFT 기반)
        fft = np.fft.fft(values)
        freqs = np.fft.fftfreq(len(values))
        
        # 주요 주파수 찾기
        main_freq_idx = np.argmax(np.abs(fft[1:len(fft)//2])) + 1
        main_freq = freqs[main_freq_idx]
        
        if abs(main_freq) > 0.1:  # 주기성이 있는 경우
            return f"seasonal_{1/abs(main_freq):.1f}"
        else:
            return "no_seasonality"
    
    def _analyze_class_similarities(self):
        """클래스 간 유사도 분석"""
        print("  🔍 클래스 간 유사도 분석 중...")
        
        # 각 클래스의 평균 특성 벡터 계산
        class_features = {}
        sensor_names = ['pitch', 'roll', 'yaw', 'flex1', 'flex2', 'flex3', 'flex4', 'flex5']
        
        for class_name, data_list in self.class_data.items():
            all_data = np.vstack(data_list)
            feature_vector = []
            
            for i, sensor_name in enumerate(sensor_names):
                sensor_values = all_data[:, i]
                feature_vector.extend([
                    np.mean(sensor_values),
                    np.std(sensor_values),
                    np.median(sensor_values),
                    stats.skew(sensor_values),
                    stats.kurtosis(sensor_values)
                ])
            
            class_features[class_name] = np.array(feature_vector)
        
        # 유사도 행렬 계산
        class_names = list(class_features.keys())
        similarity_matrix = np.zeros((len(class_names), len(class_names)))
        
        for i, class1 in enumerate(class_names):
            for j, class2 in enumerate(class_names):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                else:
                    # 코사인 유사도 계산
                    similarity = cosine_similarity(
                        class_features[class1].reshape(1, -1),
                        class_features[class2].reshape(1, -1)
                    )[0, 0]
                    similarity_matrix[i, j] = similarity
        
        self.similarity_matrix = similarity_matrix
        
        # 높은 유사도 쌍 찾기
        high_similarity_pairs = []
        for i in range(len(class_names)):
            for j in range(i+1, len(class_names)):
                similarity = similarity_matrix[i, j]
                if similarity > 0.9:  # 높은 유사도 임계값
                    high_similarity_pairs.append({
                        'class1': class_names[i],
                        'class2': class_names[j],
                        'similarity': similarity
                    })
        
        # 유사도 분석 결과 저장
        self.analysis_results['similarity_analysis'] = {
            'similarity_matrix': similarity_matrix.tolist(),
            'class_names': class_names,
            'high_similarity_pairs': high_similarity_pairs
        }
    
    def _analyze_data_quality(self):
        """데이터 품질 분석"""
        print("  🔍 데이터 품질 분석 중...")
        
        quality_report = {}
        
        for class_name, data_list in self.class_data.items():
            print(f"    📊 {class_name} 품질 분석...")
            
            quality_metrics = {
                'data_completeness': self._check_data_completeness(data_list),
                'data_consistency': self._check_data_consistency(data_list),
                'sensor_health': self._check_sensor_health(data_list),
                'outlier_analysis': self._analyze_outliers(data_list),
                'data_anomalies': self._detect_anomalies(data_list)
            }
            
            quality_report[class_name] = quality_metrics
        
        self.analysis_results['data_quality'] = quality_report
    
    def _check_data_completeness(self, data_list):
        """데이터 완성도 검사"""
        total_samples = sum(len(data) for data in data_list)
        expected_samples = len(data_list) * 300  # 예상 시퀀스 길이
        
        completeness = total_samples / expected_samples if expected_samples > 0 else 0
        
        return {
            'completeness_ratio': completeness,
            'total_samples': total_samples,
            'expected_samples': expected_samples,
            'missing_ratio': 1 - completeness
        }
    
    def _check_data_consistency(self, data_list):
        """데이터 일관성 검사"""
        lengths = [len(data) for data in data_list]
        length_std = np.std(lengths)
        length_mean = np.mean(lengths)
        
        consistency = 1 - (length_std / length_mean) if length_mean > 0 else 0
        
        return {
            'consistency_score': consistency,
            'length_variation': length_std / length_mean if length_mean > 0 else 0,
            'min_length': np.min(lengths),
            'max_length': np.max(lengths),
            'avg_length': length_mean
        }
    
    def _check_sensor_health(self, data_list):
        """센서 상태 검사"""
        all_data = np.vstack(data_list)
        sensor_names = ['pitch', 'roll', 'yaw', 'flex1', 'flex2', 'flex3', 'flex4', 'flex5']
        
        sensor_health = {}
        for i, sensor_name in enumerate(sensor_names):
            sensor_values = all_data[:, i]
            
            # 센서 고장 지표
            zero_ratio = np.sum(sensor_values == 0) / len(sensor_values)
            constant_ratio = np.sum(np.diff(sensor_values) == 0) / (len(sensor_values) - 1)
            outlier_ratio = np.sum(np.abs(sensor_values - np.mean(sensor_values)) > 3 * np.std(sensor_values)) / len(sensor_values)
            
            # 건강도 점수 (0-1, 높을수록 건강)
            health_score = 1 - (zero_ratio + constant_ratio + outlier_ratio) / 3
            
            sensor_health[sensor_name] = {
                'health_score': health_score,
                'zero_ratio': zero_ratio,
                'constant_ratio': constant_ratio,
                'outlier_ratio': outlier_ratio,
                'status': 'healthy' if health_score > 0.7 else 'warning' if health_score > 0.4 else 'faulty'
            }
        
        return sensor_health
    
    def _analyze_outliers(self, data_list):
        """이상치 분석"""
        all_data = np.vstack(data_list)
        sensor_names = ['pitch', 'roll', 'yaw', 'flex1', 'flex2', 'flex3', 'flex4', 'flex5']
        
        outlier_analysis = {}
        for i, sensor_name in enumerate(sensor_names):
            sensor_values = all_data[:, i]
            
            # IQR 기반 이상치 검출
            Q1 = np.percentile(sensor_values, 25)
            Q3 = np.percentile(sensor_values, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = sensor_values[(sensor_values < lower_bound) | (sensor_values > upper_bound)]
            
            outlier_analysis[sensor_name] = {
                'outlier_count': len(outliers),
                'outlier_ratio': len(outliers) / len(sensor_values),
                'outlier_values': outliers.tolist() if len(outliers) <= 10 else outliers[:10].tolist(),
                'bounds': {'lower': lower_bound, 'upper': upper_bound}
            }
        
        return outlier_analysis
    
    def _detect_anomalies(self, data_list):
        """이상 패턴 검출"""
        anomalies = []
        
        for i, data in enumerate(data_list):
            # 급격한 변화 검출
            for j in range(len(data) - 1):
                diff = np.abs(data[j+1] - data[j])
                if np.any(diff > 100):  # 임계값
                    anomalies.append({
                        'file_index': i,
                        'position': j,
                        'anomaly_type': 'sudden_change',
                        'magnitude': np.max(diff)
                    })
            
            # 센서 고장 패턴 검출
            for sensor_idx in range(data.shape[1]):
                sensor_values = data[:, sensor_idx]
                if np.all(sensor_values == sensor_values[0]):  # 모든 값이 동일
                    anomalies.append({
                        'file_index': i,
                        'sensor_index': sensor_idx,
                        'anomaly_type': 'sensor_frozen',
                        'value': sensor_values[0]
                    })
        
        return anomalies
    
    def _create_visualizations(self):
        """시각화 생성"""
        print("  📊 시각화 생성 중...")
        
        # 1. 클래스별 센서 분포 히스토그램
        self._create_sensor_distribution_plots()
        
        # 2. 클래스 간 유사도 히트맵
        self._create_similarity_heatmap()
        
        # 3. 데이터 품질 대시보드
        self._create_quality_dashboard()
        
        # 4. 센서 상관관계 분석
        self._create_correlation_analysis()
        
        # 5. 클래스별 특징 비교
        self._create_feature_comparison()
    
    def _create_sensor_distribution_plots(self):
        """센서 분포 히스토그램 생성"""
        sensor_names = ['pitch', 'roll', 'yaw', 'flex1', 'flex2', 'flex3', 'flex4', 'flex5']
        
        fig, axes = plt.subplots(4, 2, figsize=(15, 20))
        axes = axes.flatten()
        
        for i, sensor_name in enumerate(sensor_names):
            ax = axes[i]
            
            # 모든 클래스의 해당 센서 데이터 수집
            all_values = []
            class_labels = []
            
            for class_name, data_list in self.class_data.items():
                all_data = np.vstack(data_list)
                sensor_values = all_data[:, i]
                all_values.extend(sensor_values)
                class_labels.extend([class_name] * len(sensor_values))
            
            # 히스토그램 그리기
            ax.hist(all_values, bins=50, alpha=0.7, edgecolor='black')
            ax.set_title(f'{sensor_name} Distribution (All Classes)')
            ax.set_xlabel(sensor_name)
            ax.set_ylabel('Frequency')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('sensor_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("    ✅ 센서 분포 히스토그램 저장: sensor_distributions.png")
    
    def _create_similarity_heatmap(self):
        """유사도 히트맵 생성"""
        if self.similarity_matrix is None:
            return
        
        class_names = self.analysis_results['similarity_analysis']['class_names']
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(self.similarity_matrix, 
                   xticklabels=class_names, 
                   yticklabels=class_names,
                   annot=True, 
                   fmt='.2f', 
                   cmap='RdYlBu_r',
                   center=0.5)
        plt.title('Class Similarity Matrix')
        plt.tight_layout()
        plt.savefig('class_similarity_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("    ✅ 클래스 유사도 히트맵 저장: class_similarity_heatmap.png")
    
    def _create_quality_dashboard(self):
        """데이터 품질 대시보드 생성"""
        quality_data = self.analysis_results['data_quality']
        
        # 품질 지표 추출
        class_names = list(quality_data.keys())
        completeness_scores = [quality_data[cls]['data_completeness']['completeness_ratio'] for cls in class_names]
        consistency_scores = [quality_data[cls]['data_consistency']['consistency_score'] for cls in class_names]
        
        # 평균 센서 건강도
        avg_health_scores = []
        for cls in class_names:
            sensor_health = quality_data[cls]['sensor_health']
            avg_health = np.mean([sensor_health[sensor]['health_score'] for sensor in sensor_health])
            avg_health_scores.append(avg_health)
        
        # 대시보드 생성
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 완성도 점수
        axes[0, 0].bar(class_names, completeness_scores, color='skyblue')
        axes[0, 0].set_title('Data Completeness Scores')
        axes[0, 0].set_ylabel('Completeness Ratio')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. 일관성 점수
        axes[0, 1].bar(class_names, consistency_scores, color='lightgreen')
        axes[0, 1].set_title('Data Consistency Scores')
        axes[0, 1].set_ylabel('Consistency Score')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. 센서 건강도
        axes[1, 0].bar(class_names, avg_health_scores, color='lightcoral')
        axes[1, 0].set_title('Average Sensor Health Scores')
        axes[1, 0].set_ylabel('Health Score')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 4. 종합 품질 점수
        overall_scores = [(c + con + h) / 3 for c, con, h in zip(completeness_scores, consistency_scores, avg_health_scores)]
        axes[1, 1].bar(class_names, overall_scores, color='gold')
        axes[1, 1].set_title('Overall Quality Scores')
        axes[1, 1].set_ylabel('Overall Score')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('data_quality_dashboard.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("    ✅ 데이터 품질 대시보드 저장: data_quality_dashboard.png")
    
    def _create_correlation_analysis(self):
        """센서 상관관계 분석 시각화"""
        # 모든 클래스의 평균 상관관계 행렬 계산
        all_correlations = []
        sensor_names = ['pitch', 'roll', 'yaw', 'flex1', 'flex2', 'flex3', 'flex4', 'flex5']
        
        for class_name, data_list in self.class_data.items():
            all_data = np.vstack(data_list)
            corr_matrix = np.corrcoef(all_data.T)
            all_correlations.append(corr_matrix)
        
        avg_correlation = np.mean(all_correlations, axis=0)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(avg_correlation, 
                   xticklabels=sensor_names, 
                   yticklabels=sensor_names,
                   annot=True, 
                   fmt='.2f', 
                   cmap='coolwarm',
                   center=0)
        plt.title('Average Sensor Correlation Matrix (All Classes)')
        plt.tight_layout()
        plt.savefig('sensor_correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("    ✅ 센서 상관관계 행렬 저장: sensor_correlation_matrix.png")
    
    def _create_feature_comparison(self):
        """클래스별 특징 비교 시각화"""
        # 주요 특징 추출
        features = []
        class_names = []
        
        for class_name in self.all_classes:
            if class_name in self.analysis_results:
                stats = self.analysis_results[class_name]['sensor_stats']
                
                # 대표 특징 선택
                feature_vector = [
                    stats['pitch']['mean'],
                    stats['roll']['mean'],
                    stats['yaw']['mean'],
                    stats['flex1']['mean'],
                    stats['flex2']['mean'],
                    stats['flex3']['mean'],
                    stats['flex4']['mean'],
                    stats['flex5']['mean']
                ]
                
                features.append(feature_vector)
                class_names.append(class_name)
        
        if not features:
            return
        
        features = np.array(features)
        
        # PCA로 차원 축소
        pca = PCA(n_components=2)
        features_2d = pca.fit_transform(features)
        
        plt.figure(figsize=(12, 10))
        plt.scatter(features_2d[:, 0], features_2d[:, 1], s=100, alpha=0.7)
        
        for i, class_name in enumerate(class_names):
            plt.annotate(class_name, (features_2d[i, 0], features_2d[i, 1]), 
                        fontsize=12, ha='center', va='center')
        
        plt.title('Class Feature Comparison (PCA)')
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('class_feature_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("    ✅ 클래스 특징 비교 저장: class_feature_comparison.png")
    
    def _generate_comprehensive_report(self):
        """종합 분석 보고서 생성"""
        print("  📝 종합 분석 보고서 생성 중...")
        
        report = {
            'analysis_summary': self._generate_summary(),
            'class_analysis': self._generate_class_analysis(),
            'similarity_analysis': self._generate_similarity_analysis(),
            'quality_analysis': self._generate_quality_analysis(),
            'recommendations': self._generate_recommendations()
        }
        
        # JSON 파일로 저장
        with open('comprehensive_analysis_report.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        # 텍스트 보고서 생성
        self._generate_text_report(report)
        
        print("    ✅ 종합 분석 보고서 저장: comprehensive_analysis_report.json")
        print("    ✅ 텍스트 보고서 저장: comprehensive_analysis_report.txt")
    
    def _generate_summary(self):
        """분석 요약 생성"""
        total_classes = len(self.all_classes)
        analyzed_classes = len(self.class_data)
        
        total_samples = sum(self.analysis_results[cls]['total_samples'] 
                          for cls in self.class_data.keys())
        
        # 데이터 품질 요약
        quality_scores = []
        for cls in self.class_data.keys():
            if cls in self.analysis_results.get('data_quality', {}):
                quality = self.analysis_results['data_quality'][cls]
                completeness = quality['data_completeness']['completeness_ratio']
                consistency = quality['data_consistency']['consistency_score']
                avg_health = np.mean([quality['sensor_health'][sensor]['health_score'] 
                                    for sensor in quality['sensor_health']])
                overall_score = (completeness + consistency + avg_health) / 3
                quality_scores.append(overall_score)
        
        avg_quality = np.mean(quality_scores) if quality_scores else 0
        
        return {
            'total_classes': total_classes,
            'analyzed_classes': analyzed_classes,
            'total_samples': total_samples,
            'average_quality_score': avg_quality,
            'analysis_timestamp': pd.Timestamp.now().isoformat()
        }
    
    def _generate_class_analysis(self):
        """클래스별 분석 결과 생성"""
        class_analysis = {}
        
        for class_name in self.class_data.keys():
            if class_name in self.analysis_results:
                analysis = self.analysis_results[class_name]
                
                # 주요 특징 추출
                sensor_stats = analysis['sensor_stats']
                key_features = {}
                
                for sensor_name, stats in sensor_stats.items():
                    key_features[sensor_name] = {
                        'mean': stats['mean'],
                        'std': stats['std'],
                        'range': stats['range'],
                        'zero_ratio': stats['zero_ratio'],
                        'outlier_ratio': stats['outlier_ratio']
                    }
                
                class_analysis[class_name] = {
                    'data_count': analysis['data_count'],
                    'total_samples': analysis['total_samples'],
                    'avg_sequence_length': analysis['avg_sequence_length'],
                    'key_features': key_features,
                    'temporal_patterns': analysis['temporal_patterns'],
                    'stability_score': analysis['temporal_patterns']['stability']['mean']
                }
        
        return class_analysis
    
    def _generate_similarity_analysis(self):
        """유사도 분석 결과 생성"""
        if 'similarity_analysis' not in self.analysis_results:
            return {}
        
        similarity_data = self.analysis_results['similarity_analysis']
        
        # 가장 유사한 클래스 쌍 찾기
        high_similarity_pairs = similarity_data['high_similarity_pairs']
        
        # 클래스별 평균 유사도
        class_names = similarity_data['class_names']
        similarity_matrix = np.array(similarity_data['similarity_matrix'])
        
        avg_similarities = {}
        for i, class_name in enumerate(class_names):
            # 자기 자신 제외한 평균 유사도
            similarities = similarity_matrix[i, :]
            similarities = similarities[similarities != 1.0]  # 자기 자신 제외
            avg_similarities[class_name] = np.mean(similarities)
        
        return {
            'high_similarity_pairs': high_similarity_pairs,
            'average_similarities': avg_similarities,
            'most_similar_classes': sorted(avg_similarities.items(), key=lambda x: x[1], reverse=True)
        }
    
    def _generate_quality_analysis(self):
        """품질 분석 결과 생성"""
        if 'data_quality' not in self.analysis_results:
            return {}
        
        quality_data = self.analysis_results['data_quality']
        
        # 클래스별 품질 점수
        quality_scores = {}
        for class_name, quality in quality_data.items():
            completeness = quality['data_completeness']['completeness_ratio']
            consistency = quality['data_consistency']['consistency_score']
            avg_health = np.mean([quality['sensor_health'][sensor]['health_score'] 
                                for sensor in quality['sensor_health']])
            
            overall_score = (completeness + consistency + avg_health) / 3
            quality_scores[class_name] = {
                'overall_score': overall_score,
                'completeness': completeness,
                'consistency': consistency,
                'sensor_health': avg_health
            }
        
        # 문제가 있는 클래스 식별
        problematic_classes = []
        for class_name, scores in quality_scores.items():
            if scores['overall_score'] < 0.7:
                problematic_classes.append({
                    'class': class_name,
                    'score': scores['overall_score'],
                    'issues': []
                })
                
                if scores['completeness'] < 0.8:
                    problematic_classes[-1]['issues'].append('low_completeness')
                if scores['consistency'] < 0.8:
                    problematic_classes[-1]['issues'].append('low_consistency')
                if scores['sensor_health'] < 0.8:
                    problematic_classes[-1]['issues'].append('sensor_issues')
        
        return {
            'quality_scores': quality_scores,
            'problematic_classes': problematic_classes,
            'best_quality_classes': sorted(quality_scores.items(), key=lambda x: x[1]['overall_score'], reverse=True)[:5]
        }
    
    def _generate_recommendations(self):
        """권장사항 생성"""
        recommendations = {
            'model_architecture': [],
            'data_preprocessing': [],
            'training_strategy': [],
            'class_specific_approaches': []
        }
        
        # 유사도 기반 권장사항
        if 'similarity_analysis' in self.analysis_results:
            high_similarity_pairs = self.analysis_results['similarity_analysis']['high_similarity_pairs']
            
            if high_similarity_pairs:
                recommendations['model_architecture'].append({
                    'type': 'attention_mechanism',
                    'reason': f'{len(high_similarity_pairs)}개의 높은 유사도 클래스 쌍 발견',
                    'details': [f"{pair['class1']}-{pair['class2']} (유사도: {pair['similarity']:.3f})" 
                               for pair in high_similarity_pairs[:5]]
                })
        
        # 품질 기반 권장사항
        if 'data_quality' in self.analysis_results:
            quality_data = self.analysis_results['data_quality']
            
            # 센서 문제가 있는 클래스들
            sensor_issues = []
            for class_name, quality in quality_data.items():
                sensor_health = quality['sensor_health']
                for sensor_name, health in sensor_health.items():
                    if health['status'] == 'faulty':
                        sensor_issues.append(f"{class_name}.{sensor_name}")
            
            if sensor_issues:
                recommendations['data_preprocessing'].append({
                    'type': 'sensor_data_cleaning',
                    'reason': f'{len(sensor_issues)}개의 센서 문제 발견',
                    'details': sensor_issues[:10]  # 상위 10개만
                })
        
        # 클래스별 특화 접근법
        for class_name in self.class_data.keys():
            if class_name in self.analysis_results:
                analysis = self.analysis_results[class_name]
                
                # 안정성 기반 권장사항
                stability = analysis['temporal_patterns']['stability']['mean']
                if stability < 0.5:
                    recommendations['class_specific_approaches'].append({
                        'class': class_name,
                        'approach': 'robust_training',
                        'reason': f'낮은 안정성 점수 ({stability:.3f})',
                        'suggestions': ['데이터 증강', '노이즈 제거', '강건한 손실 함수']
                    })
        
        return recommendations
    
    def _generate_text_report(self, report):
        """텍스트 보고서 생성"""
        with open('comprehensive_analysis_report.txt', 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("전체 24개 자음/모음 데이터셋 종합 분석 보고서\n")
            f.write("=" * 80 + "\n\n")
            
            # 요약
            summary = report['analysis_summary']
            f.write("📊 분석 요약\n")
            f.write("-" * 40 + "\n")
            f.write(f"총 클래스 수: {summary['total_classes']}\n")
            f.write(f"분석된 클래스 수: {summary['analyzed_classes']}\n")
            f.write(f"총 샘플 수: {summary['total_samples']:,}\n")
            f.write(f"평균 품질 점수: {summary['average_quality_score']:.3f}\n")
            f.write(f"분석 시간: {summary['analysis_timestamp']}\n\n")
            
            # 클래스별 분석
            f.write("📈 클래스별 분석\n")
            f.write("-" * 40 + "\n")
            for class_name, analysis in report['class_analysis'].items():
                f.write(f"\n{class_name} 클래스:\n")
                f.write(f"  데이터 파일 수: {analysis['data_count']}\n")
                f.write(f"  총 샘플 수: {analysis['total_samples']}\n")
                f.write(f"  평균 시퀀스 길이: {analysis['avg_sequence_length']:.1f}\n")
                f.write(f"  안정성 점수: {analysis['stability_score']:.3f}\n")
            
            # 유사도 분석
            if report['similarity_analysis']:
                f.write("\n🔗 유사도 분석\n")
                f.write("-" * 40 + "\n")
                similarity_data = report['similarity_analysis']
                
                f.write("높은 유사도 클래스 쌍:\n")
                for pair in similarity_data['high_similarity_pairs'][:10]:
                    f.write(f"  {pair['class1']} - {pair['class2']}: {pair['similarity']:.3f}\n")
                
                f.write("\n클래스별 평균 유사도 (높은 순):\n")
                for class_name, similarity in similarity_data['most_similar_classes'][:10]:
                    f.write(f"  {class_name}: {similarity:.3f}\n")
            
            # 품질 분석
            if report['quality_analysis']:
                f.write("\n🔍 데이터 품질 분석\n")
                f.write("-" * 40 + "\n")
                quality_data = report['quality_analysis']
                
                f.write("문제가 있는 클래스들:\n")
                for problem in quality_data['problematic_classes']:
                    f.write(f"  {problem['class']}: 점수 {problem['score']:.3f}, 문제: {', '.join(problem['issues'])}\n")
                
                f.write("\n품질이 좋은 클래스들 (상위 5개):\n")
                for class_name, scores in quality_data['best_quality_classes']:
                    f.write(f"  {class_name}: {scores['overall_score']:.3f}\n")
            
            # 권장사항
            f.write("\n💡 권장사항\n")
            f.write("-" * 40 + "\n")
            
            for category, recs in report['recommendations'].items():
                f.write(f"\n{category.upper()}:\n")
                for rec in recs:
                    if isinstance(rec, dict):
                        f.write(f"  - {rec.get('type', 'N/A')}: {rec.get('reason', 'N/A')}\n")
                        if 'details' in rec:
                            for detail in rec['details'][:3]:  # 상위 3개만
                                f.write(f"    * {detail}\n")
                    else:
                        f.write(f"  - {rec}\n")

def main():
    """메인 함수"""
    data_dir = 'integrations/SignGlove_HW/github_unified_data'
    
    # 분석기 생성 및 실행
    analyzer = ComprehensiveDatasetAnalyzer(data_dir)
    analyzer.analyze_all_classes()
    
    print(f"\n🎉 종합 분석 완료!")
    print(f"📁 생성된 파일들:")
    print(f"  - comprehensive_analysis_report.json: 상세 분석 결과")
    print(f"  - comprehensive_analysis_report.txt: 텍스트 보고서")
    print(f"  - sensor_distributions.png: 센서 분포 히스토그램")
    print(f"  - class_similarity_heatmap.png: 클래스 유사도 히트맵")
    print(f"  - data_quality_dashboard.png: 데이터 품질 대시보드")
    print(f"  - sensor_correlation_matrix.png: 센서 상관관계 행렬")
    print(f"  - class_feature_comparison.png: 클래스 특징 비교")

if __name__ == "__main__":
    main()
