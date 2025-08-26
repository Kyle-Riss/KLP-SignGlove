#!/usr/bin/env python3
"""
데이터 품질 개선 분석 및 전략
현재 데이터셋의 품질 문제를 분석하고 개선 방안 제시
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import json
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from training.label_mapping import KSLLabelMapper

class DataQualityAnalyzer:
    """데이터 품질 분석기"""
    
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.label_mapper = KSLLabelMapper()
        
        print(f"🔍 데이터 품질 분석 시작")
        print(f"📁 데이터 디렉토리: {data_dir}")
    
    def analyze_data_quality(self):
        """데이터 품질 종합 분석"""
        print(f"\n📊 1단계: 기본 데이터 품질 분석")
        basic_quality = self._analyze_basic_quality()
        
        print(f"\n📊 2단계: 센서별 품질 분석")
        sensor_quality = self._analyze_sensor_quality()
        
        print(f"\n📊 3단계: 클래스별 데이터 일관성 분석")
        class_consistency = self._analyze_class_consistency()
        
        print(f"\n📊 4단계: 노이즈 및 이상치 분석")
        noise_analysis = self._analyze_noise_and_outliers()
        
        print(f"\n📊 5단계: 데이터 불균형 분석")
        imbalance_analysis = self._analyze_data_imbalance()
        
        print(f"\n📊 6단계: 시각화 및 보고서 생성")
        self._create_visualizations(basic_quality, sensor_quality, class_consistency, noise_analysis, imbalance_analysis)
        self._generate_improvement_strategies(basic_quality, sensor_quality, class_consistency, noise_analysis, imbalance_analysis)
        
        print(f"\n✅ 데이터 품질 분석 완료!")
    
    def _analyze_basic_quality(self):
        """기본 데이터 품질 분석"""
        print("  📥 기본 데이터 품질 분석 중...")
        
        quality_metrics = {
            'total_files': 0,
            'total_samples': 0,
            'missing_files': 0,
            'corrupted_files': 0,
            'inconsistent_lengths': 0,
            'zero_data_files': 0
        }
        
        class_data_info = {}
        
        for class_name in self.label_mapper.get_all_classes():
            pattern = os.path.join(self.data_dir, f"**/{class_name}/**/episode_*.csv")
            files = glob.glob(pattern, recursive=True)
            
            if not files:
                quality_metrics['missing_files'] += 1
                continue
            
            class_info = {
                'file_count': len(files),
                'total_samples': 0,
                'lengths': [],
                'corrupted_files': 0,
                'zero_data_files': 0
            }
            
            for file_path in files:
                try:
                    df = pd.read_csv(file_path)
                    
                    # 기본 품질 체크
                    if df.empty:
                        class_info['zero_data_files'] += 1
                        quality_metrics['zero_data_files'] += 1
                        continue
                    
                    # 필수 컬럼 체크
                    required_columns = ['pitch', 'roll', 'yaw', 'flex1', 'flex2', 'flex3', 'flex4', 'flex5']
                    if not all(col in df.columns for col in required_columns):
                        class_info['corrupted_files'] += 1
                        quality_metrics['corrupted_files'] += 1
                        continue
                    
                    # 데이터 길이 체크
                    data_length = len(df)
                    class_info['lengths'].append(data_length)
                    class_info['total_samples'] += data_length
                    
                except Exception as e:
                    class_info['corrupted_files'] += 1
                    quality_metrics['corrupted_files'] += 1
            
            # 길이 일관성 체크
            if len(set(class_info['lengths'])) > 1:
                quality_metrics['inconsistent_lengths'] += 1
            
            quality_metrics['total_files'] += class_info['file_count']
            quality_metrics['total_samples'] += class_info['total_samples']
            
            class_data_info[class_name] = class_info
        
        return {
            'metrics': quality_metrics,
            'class_info': class_data_info
        }
    
    def _analyze_sensor_quality(self):
        """센서별 품질 분석"""
        print("  📊 센서별 품질 분석 중...")
        
        sensor_names = ['pitch', 'roll', 'yaw', 'flex1', 'flex2', 'flex3', 'flex4', 'flex5']
        sensor_quality = {}
        
        for sensor in sensor_names:
            sensor_quality[sensor] = {
                'zero_ratio': [],
                'constant_ratio': [],
                'outlier_ratio': [],
                'noise_level': [],
                'range_stats': [],
                'class_issues': {}
            }
        
        for class_name in self.label_mapper.get_all_classes():
            pattern = os.path.join(self.data_dir, f"**/{class_name}/**/episode_*.csv")
            files = glob.glob(pattern, recursive=True)
            
            if not files:
                continue
            
            class_sensor_data = {sensor: [] for sensor in sensor_names}
            
            for file_path in files:
                try:
                    df = pd.read_csv(file_path)
                    
                    for sensor in sensor_names:
                        if sensor in df.columns:
                            class_sensor_data[sensor].extend(df[sensor].values)
                
                except Exception as e:
                    continue
            
            # 센서별 품질 메트릭 계산
            for sensor in sensor_names:
                if class_sensor_data[sensor]:
                    values = np.array(class_sensor_data[sensor])
                    
                    # 제로 비율
                    zero_ratio = np.sum(values == 0) / len(values)
                    sensor_quality[sensor]['zero_ratio'].append(zero_ratio)
                    
                    # 상수 비율 (변화가 없는 데이터)
                    constant_ratio = np.sum(np.diff(values) == 0) / (len(values) - 1) if len(values) > 1 else 0
                    sensor_quality[sensor]['constant_ratio'].append(constant_ratio)
                    
                    # 이상치 비율
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    outlier_ratio = np.sum(np.abs(values - mean_val) > 3 * std_val) / len(values)
                    sensor_quality[sensor]['outlier_ratio'].append(outlier_ratio)
                    
                    # 노이즈 레벨
                    noise_level = std_val / (np.mean(np.abs(values)) + 1e-8)
                    sensor_quality[sensor]['noise_level'].append(noise_level)
                    
                    # 범위 통계
                    range_val = np.max(values) - np.min(values)
                    sensor_quality[sensor]['range_stats'].append(range_val)
                    
                    # 클래스별 문제점 식별
                    issues = []
                    if zero_ratio > 0.5:
                        issues.append('high_zero_ratio')
                    if constant_ratio > 0.8:
                        issues.append('high_constant_ratio')
                    if outlier_ratio > 0.1:
                        issues.append('high_outlier_ratio')
                    if noise_level > 1.0:
                        issues.append('high_noise')
                    if range_val < 0.1:
                        issues.append('low_range')
                    
                    if issues:
                        sensor_quality[sensor]['class_issues'][class_name] = issues
        
        return sensor_quality
    
    def _analyze_class_consistency(self):
        """클래스별 데이터 일관성 분석"""
        print("  🔍 클래스별 일관성 분석 중...")
        
        consistency_analysis = {}
        
        for class_name in self.label_mapper.get_all_classes():
            pattern = os.path.join(self.data_dir, f"**/{class_name}/**/episode_*.csv")
            files = glob.glob(pattern, recursive=True)
            
            if not files:
                continue
            
            # 각 에피소드의 특징 벡터 추출
            episode_features = []
            
            for file_path in files:
                try:
                    df = pd.read_csv(file_path)
                    
                    if df.empty:
                        continue
                    
                    # 특징 벡터 생성 (각 센서의 통계)
                    features = []
                    for sensor in ['pitch', 'roll', 'yaw', 'flex1', 'flex2', 'flex3', 'flex4', 'flex5']:
                        if sensor in df.columns:
                            values = df[sensor].values
                            features.extend([
                                np.mean(values),
                                np.std(values),
                                np.max(values) - np.min(values),
                                np.median(values)
                            ])
                        else:
                            features.extend([0, 0, 0, 0])
                    
                    episode_features.append(features)
                
                except Exception as e:
                    continue
            
            if len(episode_features) < 2:
                continue
            
            episode_features = np.array(episode_features)
            
            # 일관성 메트릭 계산
            consistency_metrics = {
                'feature_std': np.std(episode_features, axis=0),
                'feature_cv': np.std(episode_features, axis=0) / (np.mean(episode_features, axis=0) + 1e-8),
                'avg_feature_cv': np.mean(np.std(episode_features, axis=0) / (np.mean(episode_features, axis=0) + 1e-8)),
                'episode_count': len(episode_features)
            }
            
            consistency_analysis[class_name] = consistency_metrics
        
        return consistency_analysis
    
    def _analyze_noise_and_outliers(self):
        """노이즈 및 이상치 분석"""
        print("  📈 노이즈 및 이상치 분석 중...")
        
        noise_analysis = {
            'sensor_noise_levels': {},
            'outlier_patterns': {},
            'noise_correlation': {}
        }
        
        sensor_names = ['pitch', 'roll', 'yaw', 'flex1', 'flex2', 'flex3', 'flex4', 'flex5']
        
        for sensor in sensor_names:
            noise_analysis['sensor_noise_levels'][sensor] = []
            noise_analysis['outlier_patterns'][sensor] = []
        
        for class_name in self.label_mapper.get_all_classes():
            pattern = os.path.join(self.data_dir, f"**/{class_name}/**/episode_*.csv")
            files = glob.glob(pattern, recursive=True)
            
            if not files:
                continue
            
            class_sensor_data = {sensor: [] for sensor in sensor_names}
            
            for file_path in files:
                try:
                    df = pd.read_csv(file_path)
                    
                    for sensor in sensor_names:
                        if sensor in df.columns:
                            class_sensor_data[sensor].extend(df[sensor].values)
                
                except Exception as e:
                    continue
            
            # 센서별 노이즈 분석
            for sensor in sensor_names:
                if class_sensor_data[sensor]:
                    values = np.array(class_sensor_data[sensor])
                    
                    # 노이즈 레벨
                    noise_level = np.std(values) / (np.mean(np.abs(values)) + 1e-8)
                    noise_analysis['sensor_noise_levels'][sensor].append(noise_level)
                    
                    # 이상치 패턴
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    outliers = values[np.abs(values - mean_val) > 2 * std_val]
                    
                    if len(outliers) > 0:
                        outlier_pattern = {
                            'class': class_name,
                            'outlier_count': len(outliers),
                            'outlier_ratio': len(outliers) / len(values),
                            'outlier_range': [np.min(outliers), np.max(outliers)]
                        }
                        noise_analysis['outlier_patterns'][sensor].append(outlier_pattern)
        
        return noise_analysis
    
    def _analyze_data_imbalance(self):
        """데이터 불균형 분석"""
        print("  ⚖️ 데이터 불균형 분석 중...")
        
        imbalance_analysis = {
            'class_distribution': {},
            'episode_distribution': {},
            'sample_distribution': {}
        }
        
        for class_name in self.label_mapper.get_all_classes():
            pattern = os.path.join(self.data_dir, f"**/{class_name}/**/episode_*.csv")
            files = glob.glob(pattern, recursive=True)
            
            if not files:
                imbalance_analysis['class_distribution'][class_name] = 0
                continue
            
            total_samples = 0
            episode_lengths = []
            
            for file_path in files:
                try:
                    df = pd.read_csv(file_path)
                    episode_length = len(df)
                    episode_lengths.append(episode_length)
                    total_samples += episode_length
                
                except Exception as e:
                    continue
            
            imbalance_analysis['class_distribution'][class_name] = len(files)
            imbalance_analysis['episode_distribution'][class_name] = episode_lengths
            imbalance_analysis['sample_distribution'][class_name] = total_samples
        
        return imbalance_analysis
    
    def _create_visualizations(self, basic_quality, sensor_quality, class_consistency, noise_analysis, imbalance_analysis):
        """시각화 생성"""
        print("  📊 시각화 생성 중...")
        
        # 1. 기본 품질 메트릭
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 3, 1)
        metrics = basic_quality['metrics']
        labels = ['Total Files', 'Total Samples', 'Missing Files', 'Corrupted Files', 'Inconsistent Lengths', 'Zero Data Files']
        values = [metrics['total_files'], metrics['total_samples'], metrics['missing_files'], 
                 metrics['corrupted_files'], metrics['inconsistent_lengths'], metrics['zero_data_files']]
        
        plt.bar(labels, values, color=['blue', 'green', 'red', 'orange', 'purple', 'brown'])
        plt.title('Basic Data Quality Metrics')
        plt.xticks(rotation=45)
        plt.ylabel('Count')
        
        # 2. 센서별 품질 점수
        plt.subplot(2, 3, 2)
        sensor_names = list(sensor_quality.keys())
        avg_noise_levels = [np.mean(sensor_quality[sensor]['noise_level']) if sensor_quality[sensor]['noise_level'] else 0 
                           for sensor in sensor_names]
        
        plt.bar(sensor_names, avg_noise_levels, color='skyblue')
        plt.title('Average Noise Levels by Sensor')
        plt.ylabel('Noise Level')
        plt.xticks(rotation=45)
        
        # 3. 클래스별 일관성
        plt.subplot(2, 3, 3)
        class_names = list(class_consistency.keys())
        consistency_scores = [1 / (1 + class_consistency[cls]['avg_feature_cv']) for cls in class_names]
        
        plt.bar(class_names, consistency_scores, color='lightgreen')
        plt.title('Class Consistency Scores')
        plt.ylabel('Consistency Score')
        plt.xticks(rotation=45)
        
        # 4. 데이터 불균형
        plt.subplot(2, 3, 4)
        class_dist = imbalance_analysis['class_distribution']
        plt.bar(class_dist.keys(), class_dist.values(), color='lightcoral')
        plt.title('Class Distribution (Episode Count)')
        plt.ylabel('Episode Count')
        plt.xticks(rotation=45)
        
        # 5. 센서별 문제점 분포
        plt.subplot(2, 3, 5)
        sensor_issues = {}
        for sensor in sensor_names:
            issue_count = len(sensor_quality[sensor]['class_issues'])
            sensor_issues[sensor] = issue_count
        
        plt.bar(sensor_issues.keys(), sensor_issues.values(), color='gold')
        plt.title('Sensor Issues by Class Count')
        plt.ylabel('Number of Classes with Issues')
        plt.xticks(rotation=45)
        
        # 6. 샘플 분포
        plt.subplot(2, 3, 6)
        sample_dist = imbalance_analysis['sample_distribution']
        plt.bar(sample_dist.keys(), sample_dist.values(), color='lightblue')
        plt.title('Sample Distribution by Class')
        plt.ylabel('Total Samples')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig('data_quality_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("    ✅ 시각화 파일 저장 완료")
    
    def _generate_improvement_strategies(self, basic_quality, sensor_quality, class_consistency, noise_analysis, imbalance_analysis):
        """개선 전략 생성"""
        print("  💡 개선 전략 생성 중...")
        
        strategies = {
            'data_cleaning': [],
            'sensor_optimization': [],
            'class_balance': [],
            'noise_reduction': [],
            'consistency_improvement': [],
            'preprocessing_recommendations': []
        }
        
        # 1. 데이터 정리 전략
        if basic_quality['metrics']['corrupted_files'] > 0:
            strategies['data_cleaning'].append(f"손상된 파일 {basic_quality['metrics']['corrupted_files']}개 재수집 필요")
        
        if basic_quality['metrics']['zero_data_files'] > 0:
            strategies['data_cleaning'].append(f"빈 데이터 파일 {basic_quality['metrics']['zero_data_files']}개 제거 또는 재수집")
        
        if basic_quality['metrics']['inconsistent_lengths'] > 0:
            strategies['data_cleaning'].append(f"일관성 없는 길이의 데이터 {basic_quality['metrics']['inconsistent_lengths']}개 클래스 정규화 필요")
        
        # 2. 센서 최적화 전략
        for sensor, quality in sensor_quality.items():
            if quality['class_issues']:
                problematic_classes = list(quality['class_issues'].keys())
                strategies['sensor_optimization'].append(f"{sensor} 센서: {len(problematic_classes)}개 클래스 문제 - {', '.join(problematic_classes[:3])}")
        
        # 3. 클래스 균형 전략
        class_dist = imbalance_analysis['class_distribution']
        min_episodes = min(class_dist.values())
        max_episodes = max(class_dist.values())
        
        if max_episodes / min_episodes > 2:
            strategies['class_balance'].append(f"클래스 간 에피소드 수 불균형 (최소: {min_episodes}, 최대: {max_episodes})")
        
        # 4. 노이즈 감소 전략
        for sensor, noise_levels in noise_analysis['sensor_noise_levels'].items():
            if noise_levels and np.mean(noise_levels) > 0.5:
                strategies['noise_reduction'].append(f"{sensor} 센서 평균 노이즈 레벨 {np.mean(noise_levels):.3f} - 필터링 필요")
        
        # 5. 일관성 개선 전략
        low_consistency_classes = []
        for class_name, consistency in class_consistency.items():
            if consistency['avg_feature_cv'] > 0.5:
                low_consistency_classes.append(class_name)
        
        if low_consistency_classes:
            strategies['consistency_improvement'].append(f"낮은 일관성 클래스들: {', '.join(low_consistency_classes[:5])}")
        
        # 6. 전처리 권장사항
        strategies['preprocessing_recommendations'].extend([
            "상보 필터 적용으로 IMU 센서 노이즈 감소",
            "이상치 제거 및 정규화",
            "데이터 길이 통일 (패딩/자르기)",
            "클래스별 가중치 적용",
            "센서별 특화 전처리"
        ])
        
        # 전략 저장
        with open('data_quality_improvement_strategies.json', 'w', encoding='utf-8') as f:
            json.dump(strategies, f, ensure_ascii=False, indent=2)
        
        # 텍스트 보고서 생성
        self._generate_text_report(strategies, basic_quality, sensor_quality, class_consistency, noise_analysis, imbalance_analysis)
        
        print("    ✅ 개선 전략 저장 완료")
    
    def _generate_text_report(self, strategies, basic_quality, sensor_quality, class_consistency, noise_analysis, imbalance_analysis):
        """텍스트 보고서 생성"""
        with open('data_quality_improvement_report.txt', 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("데이터 품질 개선 분석 보고서\n")
            f.write("=" * 80 + "\n\n")
            
            # 기본 품질 현황
            f.write("📊 기본 데이터 품질 현황\n")
            f.write("-" * 40 + "\n")
            metrics = basic_quality['metrics']
            f.write(f"총 파일 수: {metrics['total_files']}\n")
            f.write(f"총 샘플 수: {metrics['total_samples']}\n")
            f.write(f"손상된 파일: {metrics['corrupted_files']}\n")
            f.write(f"빈 데이터 파일: {metrics['zero_data_files']}\n")
            f.write(f"일관성 없는 길이: {metrics['inconsistent_lengths']}개 클래스\n\n")
            
            # 센서별 문제점
            f.write("🔍 센서별 주요 문제점\n")
            f.write("-" * 40 + "\n")
            for sensor, quality in sensor_quality.items():
                if quality['class_issues']:
                    f.write(f"{sensor}:\n")
                    for class_name, issues in quality['class_issues'].items():
                        f.write(f"  {class_name}: {', '.join(issues)}\n")
                    f.write("\n")
            
            # 클래스별 일관성
            f.write("📈 클래스별 일관성 점수\n")
            f.write("-" * 40 + "\n")
            for class_name, consistency in class_consistency.items():
                score = 1 / (1 + consistency['avg_feature_cv'])
                f.write(f"{class_name}: {score:.3f}\n")
            f.write("\n")
            
            # 개선 전략
            f.write("💡 데이터 품질 개선 전략\n")
            f.write("-" * 40 + "\n")
            
            for category, recs in strategies.items():
                if recs:
                    f.write(f"{category}:\n")
                    for rec in recs:
                        f.write(f"  - {rec}\n")
                    f.write("\n")

def main():
    """메인 함수"""
    data_dir = 'integrations/SignGlove_HW/github_unified_data'
    
    analyzer = DataQualityAnalyzer(data_dir)
    analyzer.analyze_data_quality()
    
    print(f"\n🎉 데이터 품질 분석 완료!")
    print(f"📁 생성된 파일들:")
    print(f"  - data_quality_analysis.png: 품질 분석 시각화")
    print(f"  - data_quality_improvement_strategies.json: 개선 전략")
    print(f"  - data_quality_improvement_report.txt: 상세 보고서")

if __name__ == "__main__":
    main()
