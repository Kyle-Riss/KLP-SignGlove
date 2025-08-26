#!/usr/bin/env python3
"""
센서 데이터 분리 가능성 분석
pitch, roll, yaw, flex1-5 센서로 24개 자음/모음을 구분할 수 있는지 분석
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
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from training.label_mapping import KSLLabelMapper

class SensorSeparabilityAnalyzer:
    """센서 데이터 분리 가능성 분석기"""
    
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.label_mapper = KSLLabelMapper()
        self.all_classes = self.label_mapper.get_all_classes()
        
        print(f"🔍 센서 데이터 분리 가능성 분석 시작")
        print(f"📁 데이터 디렉토리: {data_dir}")
        print(f"🎯 분석 대상: {len(self.all_classes)}개 클래스")
    
    def analyze_sensor_separability(self):
        """센서 데이터 분리 가능성 종합 분석"""
        print(f"\n📊 1단계: 센서 데이터 로딩 및 기본 통계")
        sensor_data = self._load_sensor_data()
        
        print(f"\n📊 2단계: 센서별 분별력 분석")
        self._analyze_sensor_discriminative_power(sensor_data)
        
        print(f"\n📊 3단계: 클래스 간 센서 패턴 비교")
        self._analyze_class_sensor_patterns(sensor_data)
        
        print(f"\n📊 4단계: 차원 축소 및 클러스터링 분석")
        self._analyze_dimensionality_and_clustering(sensor_data)
        
        print(f"\n📊 5단계: 센서 조합별 분별력 테스트")
        self._test_sensor_combinations(sensor_data)
        
        print(f"\n📊 6단계: 시각화 및 보고서 생성")
        self._create_visualizations(sensor_data)
        self._generate_separability_report()
        
        print(f"\n✅ 센서 분리 가능성 분석 완료!")
    
    def _load_sensor_data(self):
        """센서 데이터 로딩"""
        print("  📥 센서 데이터 로딩 중...")
        
        sensor_data = {}
        sensor_names = ['pitch', 'roll', 'yaw', 'flex1', 'flex2', 'flex3', 'flex4', 'flex5']
        
        for class_name in self.all_classes:
            print(f"    📊 {class_name} 클래스 데이터 로딩...")
            
            pattern = os.path.join(self.data_dir, f"**/{class_name}/**/episode_*.csv")
            files = glob.glob(pattern, recursive=True)
            
            if not files:
                print(f"    ⚠️  {class_name}: 파일 없음")
                continue
            
            class_sensor_data = {sensor: [] for sensor in sensor_names}
            
            for file_path in files:
                try:
                    df = pd.read_csv(file_path)
                    for sensor in sensor_names:
                        if sensor in df.columns:
                            class_sensor_data[sensor].extend(df[sensor].values)
                except Exception as e:
                    print(f"    ⚠️  {file_path} 로드 실패: {e}")
            
            # 각 센서별 통계 계산
            sensor_stats = {}
            for sensor in sensor_names:
                if class_sensor_data[sensor]:
                    values = np.array(class_sensor_data[sensor])
                    sensor_stats[sensor] = {
                        'mean': float(np.mean(values)),
                        'std': float(np.std(values)),
                        'min': float(np.min(values)),
                        'max': float(np.max(values)),
                        'range': float(np.max(values) - np.min(values)),
                        'median': float(np.median(values)),
                        'q25': float(np.percentile(values, 25)),
                        'q75': float(np.percentile(values, 75)),
                        'iqr': float(np.percentile(values, 75) - np.percentile(values, 25)),
                        'zero_ratio': float(np.sum(values == 0) / len(values)),
                        'constant_ratio': float(np.sum(np.diff(values) == 0) / (len(values) - 1)) if len(values) > 1 else 0
                    }
                else:
                    sensor_stats[sensor] = None
            
            sensor_data[class_name] = {
                'raw_data': class_sensor_data,
                'stats': sensor_stats,
                'sample_count': len(class_sensor_data[sensor_names[0]]) if class_sensor_data[sensor_names[0]] else 0
            }
            
            print(f"    ✅ {class_name}: {sensor_data[class_name]['sample_count']}개 샘플")
        
        return sensor_data
    
    def _analyze_sensor_discriminative_power(self, sensor_data):
        """센서별 분별력 분석"""
        print("  🔍 센서별 분별력 분석 중...")
        
        sensor_names = ['pitch', 'roll', 'yaw', 'flex1', 'flex2', 'flex3', 'flex4', 'flex5']
        
        # 각 센서별 클래스 간 분별력 계산
        sensor_discriminative_power = {}
        
        for sensor in sensor_names:
            print(f"    📊 {sensor} 센서 분석...")
            
            # 모든 클래스의 해당 센서 평균값 수집
            class_means = []
            class_names = []
            
            for class_name, data in sensor_data.items():
                if data['stats'][sensor] is not None:
                    class_means.append(data['stats'][sensor]['mean'])
                    class_names.append(class_name)
            
            if len(class_means) < 2:
                sensor_discriminative_power[sensor] = {
                    'discriminative_score': 0.0,
                    'class_means': class_means,
                    'class_names': class_names,
                    'analysis': 'insufficient_data'
                }
                continue
            
            class_means = np.array(class_means)
            
            # 분별력 지표 계산
            # 1. 클래스 간 평균 차이의 표준편차 (클수록 분별력 높음)
            mean_std = np.std(class_means)
            
            # 2. 클래스 간 평균 차이의 범위
            mean_range = np.max(class_means) - np.min(class_means)
            
            # 3. 클래스 간 평균 차이의 변동계수
            mean_cv = mean_std / np.mean(np.abs(class_means)) if np.mean(np.abs(class_means)) > 0 else 0
            
            # 4. 클래스 간 유사도 (낮을수록 분별력 높음)
            # 각 클래스의 평균값을 정규화하여 유사도 계산
            normalized_means = (class_means - np.mean(class_means)) / (np.std(class_means) + 1e-8)
            similarity_matrix = cosine_similarity(normalized_means.reshape(-1, 1), normalized_means.reshape(-1, 1))
            avg_similarity = np.mean(similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)])
            
            # 종합 분별력 점수 (0-1, 높을수록 분별력 높음)
            discriminative_score = (mean_cv * (1 - avg_similarity)) / 2
            
            sensor_discriminative_power[sensor] = {
                'discriminative_score': float(discriminative_score),
                'mean_std': float(mean_std),
                'mean_range': float(mean_range),
                'mean_cv': float(mean_cv),
                'avg_similarity': float(avg_similarity),
                'class_means': class_means.tolist(),
                'class_names': class_names,
                'analysis': 'normal'
            }
        
        self.sensor_discriminative_power = sensor_discriminative_power
    
    def _analyze_class_sensor_patterns(self, sensor_data):
        """클래스 간 센서 패턴 비교"""
        print("  🔍 클래스 간 센서 패턴 분석 중...")
        
        sensor_names = ['pitch', 'roll', 'yaw', 'flex1', 'flex2', 'flex3', 'flex4', 'flex5']
        
        # 각 클래스의 센서 패턴 벡터 생성
        class_patterns = {}
        
        for class_name, data in sensor_data.items():
            pattern_vector = []
            
            for sensor in sensor_names:
                if data['stats'][sensor] is not None:
                    stats = data['stats'][sensor]
                    # 주요 통계값들을 패턴 벡터에 포함
                    pattern_vector.extend([
                        stats['mean'],
                        stats['std'],
                        stats['range'],
                        stats['median'],
                        stats['iqr']
                    ])
                else:
                    pattern_vector.extend([0, 0, 0, 0, 0])
            
            class_patterns[class_name] = np.array(pattern_vector)
        
        # 클래스 간 패턴 유사도 계산
        class_names = list(class_patterns.keys())
        pattern_similarity_matrix = np.zeros((len(class_names), len(class_names)))
        
        for i, class1 in enumerate(class_names):
            for j, class2 in enumerate(class_names):
                if i == j:
                    pattern_similarity_matrix[i, j] = 1.0
                else:
                    similarity = cosine_similarity(
                        class_patterns[class1].reshape(1, -1),
                        class_patterns[class2].reshape(1, -1)
                    )[0, 0]
                    pattern_similarity_matrix[i, j] = similarity
        
        # 높은 유사도 클래스 쌍 찾기
        high_similarity_pairs = []
        for i in range(len(class_names)):
            for j in range(i+1, len(class_names)):
                similarity = pattern_similarity_matrix[i, j]
                if similarity > 0.95:  # 매우 높은 유사도
                    high_similarity_pairs.append({
                        'class1': class_names[i],
                        'class2': class_names[j],
                        'similarity': float(similarity)
                    })
        
        self.class_pattern_analysis = {
            'pattern_similarity_matrix': pattern_similarity_matrix.tolist(),
            'class_names': class_names,
            'high_similarity_pairs': high_similarity_pairs,
            'avg_similarity': float(np.mean(pattern_similarity_matrix[np.triu_indices_from(pattern_similarity_matrix, k=1)]))
        }
    
    def _analyze_dimensionality_and_clustering(self, sensor_data):
        """차원 축소 및 클러스터링 분석"""
        print("  🔍 차원 축소 및 클러스터링 분석 중...")
        
        sensor_names = ['pitch', 'roll', 'yaw', 'flex1', 'flex2', 'flex3', 'flex4', 'flex5']
        
        # 모든 클래스의 센서 데이터를 하나의 행렬로 결합
        all_data = []
        class_labels = []
        
        for class_name, data in sensor_data.items():
            for sensor in sensor_names:
                if data['raw_data'][sensor]:
                    # 각 센서의 통계값들을 특징으로 사용
                    stats = data['stats'][sensor]
                    if stats is not None:
                        feature_vector = [
                            stats['mean'],
                            stats['std'],
                            stats['range'],
                            stats['median'],
                            stats['iqr']
                        ]
                        all_data.append(feature_vector)
                        class_labels.append(class_name)
        
        if not all_data:
            print("    ⚠️  분석할 데이터가 없습니다.")
            return
        
        all_data = np.array(all_data)
        
        # PCA 차원 축소
        pca = PCA(n_components=2)
        data_2d = pca.fit_transform(all_data)
        
        # 클러스터링 분석
        kmeans = KMeans(n_clusters=24, random_state=42)
        cluster_labels = kmeans.fit_predict(all_data)
        
        # 클러스터링 품질 평가
        silhouette_avg = silhouette_score(all_data, cluster_labels)
        
        # 클래스별 클러스터 분포 분석
        class_cluster_distribution = defaultdict(lambda: defaultdict(int))
        for i, class_name in enumerate(class_labels):
            class_cluster_distribution[class_name][cluster_labels[i]] += 1
        
        # 클러스터 순도 계산 (각 클러스터에서 가장 많은 클래스의 비율)
        cluster_purity = []
        for cluster_id in range(24):
            cluster_classes = [class_labels[i] for i in range(len(class_labels)) if cluster_labels[i] == cluster_id]
            if cluster_classes:
                most_common_class = max(set(cluster_classes), key=cluster_classes.count)
                purity = cluster_classes.count(most_common_class) / len(cluster_classes)
                cluster_purity.append(purity)
            else:
                cluster_purity.append(0.0)
        
        self.clustering_analysis = {
            'pca_explained_variance': pca.explained_variance_ratio_.tolist(),
            'silhouette_score': float(silhouette_avg),
            'cluster_purity': cluster_purity,
            'avg_cluster_purity': float(np.mean(cluster_purity)),
            'class_cluster_distribution': dict(class_cluster_distribution),
            'data_2d': data_2d.tolist(),
            'cluster_labels': cluster_labels.tolist(),
            'class_labels': class_labels
        }
    
    def _test_sensor_combinations(self, sensor_data):
        """센서 조합별 분별력 테스트"""
        print("  🔍 센서 조합별 분별력 테스트 중...")
        
        sensor_names = ['pitch', 'roll', 'yaw', 'flex1', 'flex2', 'flex3', 'flex4', 'flex5']
        
        # 다양한 센서 조합 테스트
        sensor_combinations = [
            ['pitch', 'roll'],  # IMU 센서만
            ['flex1', 'flex2', 'flex3', 'flex4', 'flex5'],  # Flex 센서만
            ['pitch', 'roll', 'yaw'],  # 모든 IMU 센서
            ['flex1', 'flex2'],  # 일부 Flex 센서
            ['pitch', 'roll', 'flex1', 'flex2'],  # IMU + 일부 Flex
            sensor_names  # 모든 센서
        ]
        
        combination_results = {}
        
        for i, combination in enumerate(sensor_combinations):
            print(f"    📊 조합 {i+1}: {combination}")
            
            # 해당 조합의 센서들만 사용하여 패턴 벡터 생성
            combination_patterns = {}
            
            for class_name, data in sensor_data.items():
                pattern_vector = []
                
                for sensor in combination:
                    if data['stats'][sensor] is not None:
                        stats = data['stats'][sensor]
                        pattern_vector.extend([
                            stats['mean'],
                            stats['std'],
                            stats['range']
                        ])
                    else:
                        pattern_vector.extend([0, 0, 0])
                
                combination_patterns[class_name] = np.array(pattern_vector)
            
            # 분별력 계산
            class_names = list(combination_patterns.keys())
            if len(class_names) < 2:
                combination_results[f"combination_{i+1}"] = {
                    'sensors': combination,
                    'discriminative_score': 0.0,
                    'analysis': 'insufficient_data'
                }
                continue
            
            # 클래스 간 유사도 계산
            similarity_matrix = np.zeros((len(class_names), len(class_names)))
            for j, class1 in enumerate(class_names):
                for k, class2 in enumerate(class_names):
                    if j == k:
                        similarity_matrix[j, k] = 1.0
                    else:
                        similarity = cosine_similarity(
                            combination_patterns[class1].reshape(1, -1),
                            combination_patterns[class2].reshape(1, -1)
                        )[0, 0]
                        similarity_matrix[j, k] = similarity
            
            # 평균 유사도 (낮을수록 분별력 높음)
            avg_similarity = np.mean(similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)])
            discriminative_score = 1 - avg_similarity
            
            combination_results[f"combination_{i+1}"] = {
                'sensors': combination,
                'discriminative_score': float(discriminative_score),
                'avg_similarity': float(avg_similarity),
                'analysis': 'normal'
            }
        
        self.sensor_combination_analysis = combination_results
    
    def _create_visualizations(self, sensor_data):
        """시각화 생성"""
        print("  📊 시각화 생성 중...")
        
        # 1. 센서별 분별력 비교
        if hasattr(self, 'sensor_discriminative_power'):
            sensor_names = list(self.sensor_discriminative_power.keys())
            discriminative_scores = [self.sensor_discriminative_power[sensor]['discriminative_score'] 
                                   for sensor in sensor_names]
            
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.bar(sensor_names, discriminative_scores, color='skyblue')
            plt.title('센서별 분별력 점수')
            plt.xlabel('센서')
            plt.ylabel('분별력 점수')
            plt.xticks(rotation=45)
            plt.ylim(0, 1)
            
            # 2. 클래스별 센서 패턴 히트맵
            plt.subplot(1, 2, 2)
            if hasattr(self, 'class_pattern_analysis'):
                similarity_matrix = np.array(self.class_pattern_analysis['pattern_similarity_matrix'])
                class_names = self.class_pattern_analysis['class_names']
                
                sns.heatmap(similarity_matrix, 
                           xticklabels=class_names, 
                           yticklabels=class_names,
                           cmap='RdYlBu_r', 
                           center=0.5,
                           annot=False)
                plt.title('클래스 간 센서 패턴 유사도')
                plt.xticks(rotation=45)
                plt.yticks(rotation=0)
            
            plt.tight_layout()
            plt.savefig('sensor_separability_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 3. 클러스터링 결과 시각화
        if hasattr(self, 'clustering_analysis'):
            plt.figure(figsize=(10, 8))
            data_2d = np.array(self.clustering_analysis['data_2d'])
            cluster_labels = np.array(self.clustering_analysis['cluster_labels'])
            
            scatter = plt.scatter(data_2d[:, 0], data_2d[:, 1], c=cluster_labels, 
                                cmap='tab20', alpha=0.7, s=50)
            plt.colorbar(scatter)
            plt.title('센서 데이터 클러스터링 결과 (PCA)')
            plt.xlabel(f'PC1 ({self.clustering_analysis["pca_explained_variance"][0]:.2%} variance)')
            plt.ylabel(f'PC2 ({self.clustering_analysis["pca_explained_variance"][1]:.2%} variance)')
            plt.grid(True, alpha=0.3)
            plt.savefig('sensor_clustering_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 4. 센서 조합별 분별력 비교
        if hasattr(self, 'sensor_combination_analysis'):
            combinations = list(self.sensor_combination_analysis.keys())
            scores = [self.sensor_combination_analysis[combo]['discriminative_score'] 
                     for combo in combinations]
            
            plt.figure(figsize=(12, 6))
            plt.bar(range(len(combinations)), scores, color='lightgreen')
            plt.title('센서 조합별 분별력 점수')
            plt.xlabel('센서 조합')
            plt.ylabel('분별력 점수')
            plt.xticks(range(len(combinations)), 
                      [f"조합{i+1}" for i in range(len(combinations))], 
                      rotation=45)
            plt.ylim(0, 1)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('sensor_combination_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        print("    ✅ 시각화 파일 저장 완료")
    
    def _generate_separability_report(self):
        """분리 가능성 보고서 생성"""
        print("  📝 분리 가능성 보고서 생성 중...")
        
        report = {
            'summary': self._generate_separability_summary(),
            'sensor_analysis': self._generate_sensor_analysis(),
            'pattern_analysis': self._generate_pattern_analysis(),
            'clustering_analysis': self._generate_clustering_analysis(),
            'combination_analysis': self._generate_combination_analysis(),
            'conclusions': self._generate_conclusions()
        }
        
        # JSON 저장
        with open('sensor_separability_report.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        # 텍스트 보고서
        self._generate_text_report(report)
        
        print("    ✅ 분리 가능성 보고서 저장 완료")
    
    def _generate_separability_summary(self):
        """분리 가능성 요약"""
        summary = {
            'total_classes': len(self.all_classes),
            'analysis_timestamp': pd.Timestamp.now().isoformat()
        }
        
        if hasattr(self, 'sensor_discriminative_power'):
            scores = [self.sensor_discriminative_power[sensor]['discriminative_score'] 
                     for sensor in self.sensor_discriminative_power]
            summary['avg_sensor_discriminative_score'] = float(np.mean(scores))
            summary['best_sensor'] = max(self.sensor_discriminative_power.keys(), 
                                       key=lambda x: self.sensor_discriminative_power[x]['discriminative_score'])
        
        if hasattr(self, 'class_pattern_analysis'):
            summary['avg_class_similarity'] = self.class_pattern_analysis['avg_similarity']
            summary['high_similarity_pairs_count'] = len(self.class_pattern_analysis['high_similarity_pairs'])
        
        if hasattr(self, 'clustering_analysis'):
            summary['clustering_silhouette_score'] = self.clustering_analysis['silhouette_score']
            summary['avg_cluster_purity'] = self.clustering_analysis['avg_cluster_purity']
        
        return summary
    
    def _generate_sensor_analysis(self):
        """센서 분석 결과"""
        if not hasattr(self, 'sensor_discriminative_power'):
            return {}
        
        return {
            'sensor_rankings': sorted(self.sensor_discriminative_power.items(), 
                                    key=lambda x: x[1]['discriminative_score'], reverse=True),
            'best_sensors': [sensor for sensor, data in self.sensor_discriminative_power.items() 
                           if data['discriminative_score'] > 0.5],
            'poor_sensors': [sensor for sensor, data in self.sensor_discriminative_power.items() 
                           if data['discriminative_score'] < 0.1]
        }
    
    def _generate_pattern_analysis(self):
        """패턴 분석 결과"""
        if not hasattr(self, 'class_pattern_analysis'):
            return {}
        
        return {
            'high_similarity_pairs': self.class_pattern_analysis['high_similarity_pairs'],
            'avg_similarity': self.class_pattern_analysis['avg_similarity'],
            'most_similar_classes': sorted(self.class_pattern_analysis['high_similarity_pairs'], 
                                         key=lambda x: x['similarity'], reverse=True)[:10]
        }
    
    def _generate_clustering_analysis(self):
        """클러스터링 분석 결과"""
        if not hasattr(self, 'clustering_analysis'):
            return {}
        
        return {
            'silhouette_score': self.clustering_analysis['silhouette_score'],
            'avg_cluster_purity': self.clustering_analysis['avg_cluster_purity'],
            'cluster_purity_distribution': {
                'high_purity': len([p for p in self.clustering_analysis['cluster_purity'] if p > 0.8]),
                'medium_purity': len([p for p in self.clustering_analysis['cluster_purity'] if 0.5 <= p <= 0.8]),
                'low_purity': len([p for p in self.clustering_analysis['cluster_purity'] if p < 0.5])
            }
        }
    
    def _generate_combination_analysis(self):
        """센서 조합 분석 결과"""
        if not hasattr(self, 'sensor_combination_analysis'):
            return {}
        
        return {
            'best_combination': max(self.sensor_combination_analysis.items(), 
                                  key=lambda x: x[1]['discriminative_score']),
            'combination_rankings': sorted(self.sensor_combination_analysis.items(), 
                                         key=lambda x: x[1]['discriminative_score'], reverse=True)
        }
    
    def _generate_conclusions(self):
        """결론 및 권장사항"""
        conclusions = {
            'separability_assessment': '',
            'key_findings': [],
            'recommendations': []
        }
        
        # 분리 가능성 평가
        if hasattr(self, 'sensor_discriminative_power'):
            avg_score = np.mean([self.sensor_discriminative_power[sensor]['discriminative_score'] 
                               for sensor in self.sensor_discriminative_power])
            
            if avg_score > 0.7:
                conclusions['separability_assessment'] = 'HIGH'
            elif avg_score > 0.4:
                conclusions['separability_assessment'] = 'MEDIUM'
            else:
                conclusions['separability_assessment'] = 'LOW'
        
        # 주요 발견사항
        if hasattr(self, 'class_pattern_analysis'):
            if self.class_pattern_analysis['avg_similarity'] > 0.9:
                conclusions['key_findings'].append('클래스 간 센서 패턴이 매우 유사함')
            elif self.class_pattern_analysis['avg_similarity'] > 0.7:
                conclusions['key_findings'].append('클래스 간 센서 패턴이 상당히 유사함')
        
        if hasattr(self, 'clustering_analysis'):
            if self.clustering_analysis['avg_cluster_purity'] < 0.5:
                conclusions['key_findings'].append('클러스터링 품질이 낮음 - 센서만으로는 구분이 어려움')
        
        # 권장사항
        if conclusions['separability_assessment'] == 'LOW':
            conclusions['recommendations'].append('센서 데이터만으로는 24개 클래스 구분이 어려움')
            conclusions['recommendations'].append('추가적인 특징 추출 또는 다른 접근법 필요')
        elif conclusions['separability_assessment'] == 'MEDIUM':
            conclusions['recommendations'].append('일부 센서 조합으로는 구분 가능하나 제한적')
            conclusions['recommendations'].append('고급 머신러닝 기법 적용 필요')
        else:
            conclusions['recommendations'].append('센서 데이터로 충분한 구분 가능')
            conclusions['recommendations'].append('기존 접근법으로도 성공 가능')
        
        return conclusions
    
    def _generate_text_report(self, report):
        """텍스트 보고서 생성"""
        with open('sensor_separability_report.txt', 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("센서 데이터 분리 가능성 분석 보고서\n")
            f.write("=" * 80 + "\n\n")
            
            # 요약
            summary = report['summary']
            f.write("📊 분석 요약\n")
            f.write("-" * 40 + "\n")
            f.write(f"총 클래스 수: {summary['total_classes']}\n")
            if 'avg_sensor_discriminative_score' in summary:
                f.write(f"평균 센서 분별력 점수: {summary['avg_sensor_discriminative_score']:.3f}\n")
            if 'best_sensor' in summary:
                f.write(f"최고 분별력 센서: {summary['best_sensor']}\n")
            if 'avg_class_similarity' in summary:
                f.write(f"평균 클래스 유사도: {summary['avg_class_similarity']:.3f}\n")
            if 'clustering_silhouette_score' in summary:
                f.write(f"클러스터링 실루엣 점수: {summary['clustering_silhouette_score']:.3f}\n")
            f.write(f"분석 시간: {summary['analysis_timestamp']}\n\n")
            
            # 센서 분석
            if report['sensor_analysis']:
                f.write("🔍 센서별 분별력 분석\n")
                f.write("-" * 40 + "\n")
                for sensor, data in report['sensor_analysis']['sensor_rankings']:
                    f.write(f"{sensor}: {data['discriminative_score']:.3f}\n")
                f.write("\n")
            
            # 패턴 분석
            if report['pattern_analysis']:
                f.write("🔗 클래스 패턴 유사도 분석\n")
                f.write("-" * 40 + "\n")
                f.write(f"평균 유사도: {report['pattern_analysis']['avg_similarity']:.3f}\n")
                f.write(f"높은 유사도 쌍 수: {report['pattern_analysis']['high_similarity_pairs_count']}\n")
                f.write("가장 유사한 클래스 쌍:\n")
                for pair in report['pattern_analysis']['most_similar_classes'][:5]:
                    f.write(f"  {pair['class1']} - {pair['class2']}: {pair['similarity']:.3f}\n")
                f.write("\n")
            
            # 클러스터링 분석
            if report['clustering_analysis']:
                f.write("📊 클러스터링 분석\n")
                f.write("-" * 40 + "\n")
                f.write(f"실루엣 점수: {report['clustering_analysis']['silhouette_score']:.3f}\n")
                f.write(f"평균 클러스터 순도: {report['clustering_analysis']['avg_cluster_purity']:.3f}\n")
                f.write("클러스터 순도 분포:\n")
                purity_dist = report['clustering_analysis']['cluster_purity_distribution']
                f.write(f"  높은 순도 (>0.8): {purity_dist['high_purity']}개\n")
                f.write(f"  중간 순도 (0.5-0.8): {purity_dist['medium_purity']}개\n")
                f.write(f"  낮은 순도 (<0.5): {purity_dist['low_purity']}개\n")
                f.write("\n")
            
            # 센서 조합 분석
            if report['combination_analysis']:
                f.write("🔧 센서 조합 분석\n")
                f.write("-" * 40 + "\n")
                best_combo = report['combination_analysis']['best_combination']
                f.write(f"최고 분별력 조합: {best_combo[1]['sensors']}\n")
                f.write(f"분별력 점수: {best_combo[1]['discriminative_score']:.3f}\n")
                f.write("\n")
            
            # 결론
            if report['conclusions']:
                f.write("💡 결론 및 권장사항\n")
                f.write("-" * 40 + "\n")
                f.write(f"분리 가능성 평가: {report['conclusions']['separability_assessment']}\n\n")
                
                if report['conclusions']['key_findings']:
                    f.write("주요 발견사항:\n")
                    for finding in report['conclusions']['key_findings']:
                        f.write(f"  - {finding}\n")
                    f.write("\n")
                
                if report['conclusions']['recommendations']:
                    f.write("권장사항:\n")
                    for rec in report['conclusions']['recommendations']:
                        f.write(f"  - {rec}\n")

def main():
    """메인 함수"""
    data_dir = 'integrations/SignGlove_HW/github_unified_data'
    
    analyzer = SensorSeparabilityAnalyzer(data_dir)
    analyzer.analyze_sensor_separability()
    
    print(f"\n🎉 센서 분리 가능성 분석 완료!")
    print(f"📁 생성된 파일들:")
    print(f"  - sensor_separability_report.json: 상세 분석 결과")
    print(f"  - sensor_separability_report.txt: 텍스트 보고서")
    print(f"  - sensor_separability_analysis.png: 센서별 분별력 차트")
    print(f"  - sensor_clustering_analysis.png: 클러스터링 결과")
    print(f"  - sensor_combination_analysis.png: 센서 조합 분석")

if __name__ == "__main__":
    main()
