#!/usr/bin/env python3
"""
상보 필터 효과 분석
원본 센서 데이터 vs 상보 필터 적용 데이터 비교 분석
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
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from training.label_mapping import KSLLabelMapper

class ComplementaryFilterAnalyzer:
    """상보 필터 효과 분석기"""
    
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.label_mapper = KSLLabelMapper()
        self.all_classes = self.label_mapper.get_all_classes()
        
        print(f"🔍 상보 필터 효과 분석 시작")
        print(f"📁 데이터 디렉토리: {data_dir}")
    
    def analyze_complementary_filter_effect(self):
        """상보 필터 효과 종합 분석"""
        print(f"\n📊 1단계: 원본 센서 데이터 분석")
        original_data = self._load_original_sensor_data()
        
        print(f"\n📊 2단계: 상보 필터 적용 데이터 분석")
        complementary_data = self._load_complementary_filter_data()
        
        print(f"\n📊 3단계: 데이터 품질 비교")
        self._compare_data_quality(original_data, complementary_data)
        
        print(f"\n📊 4단계: 분별력 비교 분석")
        self._compare_discriminative_power(original_data, complementary_data)
        
        print(f"\n📊 5단계: 클래스 구분 가능성 테스트")
        self._test_class_separability(original_data, complementary_data)
        
        print(f"\n📊 6단계: 시각화 및 보고서 생성")
        self._create_visualizations(original_data, complementary_data)
        self._generate_analysis_report()
        
        print(f"\n✅ 상보 필터 효과 분석 완료!")
    
    def _load_original_sensor_data(self):
        """원본 센서 데이터 로딩"""
        print("  📥 원본 센서 데이터 로딩 중...")
        
        original_data = {}
        sensor_names = ['pitch', 'roll', 'yaw', 'flex1', 'flex2', 'flex3', 'flex4', 'flex5']
        
        for class_name in self.all_classes:
            print(f"    📊 {class_name} 원본 데이터 로딩...")
            
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
            
            # 센서별 통계 계산
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
                        'zero_ratio': float(np.sum(values == 0) / len(values)),
                        'noise_level': float(np.std(values) / (np.mean(np.abs(values)) + 1e-8))
                    }
                else:
                    sensor_stats[sensor] = None
            
            original_data[class_name] = {
                'raw_data': class_sensor_data,
                'stats': sensor_stats,
                'sample_count': len(class_sensor_data[sensor_names[0]]) if class_sensor_data[sensor_names[0]] else 0
            }
            
            print(f"    ✅ {class_name}: {original_data[class_name]['sample_count']}개 샘플")
        
        return original_data
    
    def _load_complementary_filter_data(self):
        """상보 필터 적용 데이터 로딩"""
        print("  📥 상보 필터 적용 데이터 로딩 중...")
        
        # 상보 필터 적용 함수
        def apply_complementary_filter(pitch, roll, yaw, alpha=0.96):
            """상보 필터 적용"""
            filtered_pitch = []
            filtered_roll = []
            filtered_yaw = []
            
            # 초기값 설정
            prev_pitch = pitch[0] if len(pitch) > 0 else 0
            prev_roll = roll[0] if len(roll) > 0 else 0
            prev_yaw = yaw[0] if len(yaw) > 0 else 0
            
            for i in range(len(pitch)):
                # 상보 필터 공식: filtered = alpha * (prev + gyro) + (1-alpha) * accel
                filtered_pitch.append(alpha * (prev_pitch + pitch[i]) + (1-alpha) * pitch[i])
                filtered_roll.append(alpha * (prev_roll + roll[i]) + (1-alpha) * roll[i])
                filtered_yaw.append(alpha * (prev_yaw + yaw[i]) + (1-alpha) * yaw[i])
                
                prev_pitch = filtered_pitch[-1]
                prev_roll = filtered_roll[-1]
                prev_yaw = filtered_yaw[-1]
            
            return np.array(filtered_pitch), np.array(filtered_roll), np.array(filtered_yaw)
        
        complementary_data = {}
        sensor_names = ['pitch', 'roll', 'yaw', 'flex1', 'flex2', 'flex3', 'flex4', 'flex5']
        
        for class_name in self.all_classes:
            print(f"    📊 {class_name} 상보 필터 적용...")
            
            pattern = os.path.join(self.data_dir, f"**/{class_name}/**/episode_*.csv")
            files = glob.glob(pattern, recursive=True)
            
            if not files:
                print(f"    ⚠️  {class_name}: 파일 없음")
                continue
            
            class_sensor_data = {sensor: [] for sensor in sensor_names}
            
            for file_path in files:
                try:
                    df = pd.read_csv(file_path)
                    
                    # IMU 센서에 상보 필터 적용
                    if 'pitch' in df.columns and 'roll' in df.columns and 'yaw' in df.columns:
                        pitch = df['pitch'].values
                        roll = df['roll'].values
                        yaw = df['yaw'].values
                        
                        filtered_pitch, filtered_roll, filtered_yaw = apply_complementary_filter(pitch, roll, yaw)
                        
                        class_sensor_data['pitch'].extend(filtered_pitch)
                        class_sensor_data['roll'].extend(filtered_roll)
                        class_sensor_data['yaw'].extend(filtered_yaw)
                    
                    # Flex 센서는 그대로 사용
                    for sensor in ['flex1', 'flex2', 'flex3', 'flex4', 'flex5']:
                        if sensor in df.columns:
                            class_sensor_data[sensor].extend(df[sensor].values)
                            
                except Exception as e:
                    print(f"    ⚠️  {file_path} 처리 실패: {e}")
            
            # 센서별 통계 계산
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
                        'zero_ratio': float(np.sum(values == 0) / len(values)),
                        'noise_level': float(np.std(values) / (np.mean(np.abs(values)) + 1e-8))
                    }
                else:
                    sensor_stats[sensor] = None
            
            complementary_data[class_name] = {
                'raw_data': class_sensor_data,
                'stats': sensor_stats,
                'sample_count': len(class_sensor_data[sensor_names[0]]) if class_sensor_data[sensor_names[0]] else 0
            }
            
            print(f"    ✅ {class_name}: {complementary_data[class_name]['sample_count']}개 샘플")
        
        return complementary_data
    
    def _compare_data_quality(self, original_data, complementary_data):
        """데이터 품질 비교"""
        print("  🔍 데이터 품질 비교 중...")
        
        sensor_names = ['pitch', 'roll', 'yaw', 'flex1', 'flex2', 'flex3', 'flex4', 'flex5']
        
        quality_comparison = {}
        
        for sensor in sensor_names:
            print(f"    📊 {sensor} 센서 품질 비교...")
            
            original_noise = []
            complementary_noise = []
            
            for class_name in self.all_classes:
                if class_name in original_data and original_data[class_name]['stats'][sensor]:
                    original_noise.append(original_data[class_name]['stats'][sensor]['noise_level'])
                
                if class_name in complementary_data and complementary_data[class_name]['stats'][sensor]:
                    complementary_noise.append(complementary_data[class_name]['stats'][sensor]['noise_level'])
            
            if original_noise and complementary_noise:
                quality_comparison[sensor] = {
                    'original_avg_noise': float(np.mean(original_noise)),
                    'complementary_avg_noise': float(np.mean(complementary_noise)),
                    'noise_reduction': float(np.mean(original_noise) - np.mean(complementary_noise)),
                    'noise_reduction_ratio': float((np.mean(original_noise) - np.mean(complementary_noise)) / np.mean(original_noise)) if np.mean(original_noise) > 0 else 0
                }
        
        self.quality_comparison = quality_comparison
    
    def _compare_discriminative_power(self, original_data, complementary_data):
        """분별력 비교 분석"""
        print("  🔍 분별력 비교 분석 중...")
        
        sensor_names = ['pitch', 'roll', 'yaw', 'flex1', 'flex2', 'flex3', 'flex4', 'flex5']
        
        discriminative_comparison = {}
        
        for sensor in sensor_names:
            print(f"    📊 {sensor} 센서 분별력 비교...")
            
            # 원본 데이터 분별력 계산
            original_means = []
            original_class_names = []
            
            for class_name, data in original_data.items():
                if data['stats'][sensor] is not None:
                    original_means.append(data['stats'][sensor]['mean'])
                    original_class_names.append(class_name)
            
            # 상보 필터 데이터 분별력 계산
            complementary_means = []
            complementary_class_names = []
            
            for class_name, data in complementary_data.items():
                if data['stats'][sensor] is not None:
                    complementary_means.append(data['stats'][sensor]['mean'])
                    complementary_class_names.append(class_name)
            
            if len(original_means) >= 2 and len(complementary_means) >= 2:
                # 원본 데이터 분별력
                original_std = np.std(original_means)
                original_cv = original_std / np.mean(np.abs(original_means)) if np.mean(np.abs(original_means)) > 0 else 0
                
                # 상보 필터 데이터 분별력
                complementary_std = np.std(complementary_means)
                complementary_cv = complementary_std / np.mean(np.abs(complementary_means)) if np.mean(np.abs(complementary_means)) > 0 else 0
                
                discriminative_comparison[sensor] = {
                    'original_discriminative_score': float(original_cv),
                    'complementary_discriminative_score': float(complementary_cv),
                    'improvement': float(complementary_cv - original_cv),
                    'improvement_ratio': float((complementary_cv - original_cv) / original_cv) if original_cv > 0 else 0
                }
        
        self.discriminative_comparison = discriminative_comparison
    
    def _test_class_separability(self, original_data, complementary_data):
        """클래스 구분 가능성 테스트"""
        print("  🔍 클래스 구분 가능성 테스트 중...")
        
        # 원본 데이터로 클래스 구분 가능성 테스트
        original_separability = self._test_data_separability(original_data, "원본")
        
        # 상보 필터 데이터로 클래스 구분 가능성 테스트
        complementary_separability = self._test_data_separability(complementary_data, "상보 필터")
        
        self.separability_test = {
            'original': original_separability,
            'complementary': complementary_separability
        }
    
    def _test_data_separability(self, data, data_type):
        """데이터 분리 가능성 테스트"""
        print(f"    📊 {data_type} 데이터 분리 가능성 테스트...")
        
        sensor_names = ['pitch', 'roll', 'yaw', 'flex1', 'flex2', 'flex3', 'flex4', 'flex5']
        
        # 모든 클래스의 특징 벡터 생성
        feature_vectors = []
        class_labels = []
        
        for class_name, class_data in data.items():
            feature_vector = []
            
            for sensor in sensor_names:
                if class_data['stats'][sensor] is not None:
                    stats = class_data['stats'][sensor]
                    feature_vector.extend([
                        stats['mean'],
                        stats['std'],
                        stats['range'],
                        stats['median']
                    ])
                else:
                    feature_vector.extend([0, 0, 0, 0])
            
            feature_vectors.append(feature_vector)
            class_labels.append(class_name)
        
        if len(feature_vectors) < 2:
            return {'separability_score': 0.0, 'analysis': 'insufficient_data'}
        
        feature_vectors = np.array(feature_vectors)
        
        # 클래스 간 유사도 계산
        similarity_matrix = cosine_similarity(feature_vectors)
        avg_similarity = np.mean(similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)])
        
        # 분리 가능성 점수 (낮은 유사도 = 높은 분리 가능성)
        separability_score = 1 - avg_similarity
        
        # 클러스터링 테스트 (클러스터 수를 데이터 크기에 맞게 조정)
        n_clusters = min(24, len(feature_vectors) - 1)
        if n_clusters < 2:
            silhouette_avg = 0.0
            cluster_labels = np.zeros(len(feature_vectors))
        else:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(feature_vectors)
            
            # 클러스터링 품질 평가
            silhouette_avg = silhouette_score(feature_vectors, cluster_labels)
        
        # 클래스별 클러스터 분포
        class_cluster_distribution = defaultdict(lambda: defaultdict(int))
        for i, class_name in enumerate(class_labels):
            class_cluster_distribution[class_name][cluster_labels[i]] += 1
        
        # 클러스터 순도 계산
        cluster_purity = []
        unique_clusters = np.unique(cluster_labels)
        for cluster_id in unique_clusters:
            cluster_classes = [class_labels[i] for i in range(len(class_labels)) if cluster_labels[i] == cluster_id]
            if cluster_classes:
                most_common_class = max(set(cluster_classes), key=cluster_classes.count)
                purity = cluster_classes.count(most_common_class) / len(cluster_classes)
                cluster_purity.append(purity)
            else:
                cluster_purity.append(0.0)
        
        return {
            'separability_score': float(separability_score),
            'avg_similarity': float(avg_similarity),
            'silhouette_score': float(silhouette_avg),
            'avg_cluster_purity': float(np.mean(cluster_purity)),
            'cluster_purity_distribution': {
                'high_purity': len([p for p in cluster_purity if p > 0.8]),
                'medium_purity': len([p for p in cluster_purity if 0.5 <= p <= 0.8]),
                'low_purity': len([p for p in cluster_purity if p < 0.5])
            },
            'feature_vectors': feature_vectors.tolist(),
            'class_labels': class_labels,
            'cluster_labels': cluster_labels.tolist()
        }
    
    def _create_visualizations(self, original_data, complementary_data):
        """시각화 생성"""
        print("  📊 시각화 생성 중...")
        
        # 1. 노이즈 레벨 비교
        if hasattr(self, 'quality_comparison'):
            sensor_names = list(self.quality_comparison.keys())
            original_noise = [self.quality_comparison[sensor]['original_avg_noise'] for sensor in sensor_names]
            complementary_noise = [self.quality_comparison[sensor]['complementary_avg_noise'] for sensor in sensor_names]
            
            plt.figure(figsize=(15, 10))
            
            plt.subplot(2, 2, 1)
            x = np.arange(len(sensor_names))
            width = 0.35
            plt.bar(x - width/2, original_noise, width, label='원본 데이터', color='skyblue')
            plt.bar(x + width/2, complementary_noise, width, label='상보 필터 데이터', color='lightcoral')
            plt.xlabel('센서')
            plt.ylabel('평균 노이즈 레벨')
            plt.title('센서별 노이즈 레벨 비교')
            plt.xticks(x, sensor_names, rotation=45)
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # 2. 분별력 비교
            if hasattr(self, 'discriminative_comparison'):
                sensor_names_disc = list(self.discriminative_comparison.keys())
                original_disc = [self.discriminative_comparison[sensor]['original_discriminative_score'] for sensor in sensor_names_disc]
                complementary_disc = [self.discriminative_comparison[sensor]['complementary_discriminative_score'] for sensor in sensor_names_disc]
                
                plt.subplot(2, 2, 2)
                x = np.arange(len(sensor_names_disc))
                plt.bar(x - width/2, original_disc, width, label='원본 데이터', color='skyblue')
                plt.bar(x + width/2, complementary_disc, width, label='상보 필터 데이터', color='lightcoral')
                plt.xlabel('센서')
                plt.ylabel('분별력 점수')
                plt.title('센서별 분별력 비교')
                plt.xticks(x, sensor_names_disc, rotation=45)
                plt.legend()
                plt.grid(True, alpha=0.3)
            
            # 3. 분리 가능성 비교
            if hasattr(self, 'separability_test'):
                plt.subplot(2, 2, 3)
                methods = ['원본 데이터', '상보 필터 데이터']
                separability_scores = [
                    self.separability_test['original']['separability_score'],
                    self.separability_test['complementary']['separability_score']
                ]
                colors = ['skyblue', 'lightcoral']
                
                plt.bar(methods, separability_scores, color=colors)
                plt.ylabel('분리 가능성 점수')
                plt.title('클래스 분리 가능성 비교')
                plt.ylim(0, 1)
                plt.grid(True, alpha=0.3)
                
                # 4. 클러스터링 품질 비교
                plt.subplot(2, 2, 4)
                silhouette_scores = [
                    self.separability_test['original']['silhouette_score'],
                    self.separability_test['complementary']['silhouette_score']
                ]
                
                plt.bar(methods, silhouette_scores, color=colors)
                plt.ylabel('실루엣 점수')
                plt.title('클러스터링 품질 비교')
                plt.ylim(-1, 1)
                plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('complementary_filter_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        print("    ✅ 시각화 파일 저장 완료")
    
    def _generate_analysis_report(self):
        """분석 보고서 생성"""
        print("  📝 분석 보고서 생성 중...")
        
        report = {
            'summary': self._generate_summary(),
            'quality_comparison': self._generate_quality_analysis(),
            'discriminative_comparison': self._generate_discriminative_analysis(),
            'separability_analysis': self._generate_separability_analysis(),
            'conclusions': self._generate_conclusions()
        }
        
        # JSON 저장
        with open('complementary_filter_analysis_report.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        # 텍스트 보고서
        self._generate_text_report(report)
        
        print("    ✅ 분석 보고서 저장 완료")
    
    def _generate_summary(self):
        """요약 생성"""
        summary = {
            'total_classes': len(self.all_classes),
            'analysis_timestamp': pd.Timestamp.now().isoformat()
        }
        
        if hasattr(self, 'quality_comparison'):
            avg_noise_reduction = np.mean([self.quality_comparison[sensor]['noise_reduction_ratio'] 
                                         for sensor in self.quality_comparison])
            summary['avg_noise_reduction_ratio'] = float(avg_noise_reduction)
        
        if hasattr(self, 'discriminative_comparison'):
            avg_improvement = np.mean([self.discriminative_comparison[sensor]['improvement_ratio'] 
                                     for sensor in self.discriminative_comparison])
            summary['avg_discriminative_improvement'] = float(avg_improvement)
        
        if hasattr(self, 'separability_test'):
            original_score = self.separability_test['original']['separability_score']
            complementary_score = self.separability_test['complementary']['separability_score']
            summary['original_separability_score'] = float(original_score)
            summary['complementary_separability_score'] = float(complementary_score)
            summary['separability_improvement'] = float(complementary_score - original_score)
        
        return summary
    
    def _generate_quality_analysis(self):
        """품질 분석 결과"""
        if not hasattr(self, 'quality_comparison'):
            return {}
        
        return {
            'sensor_quality_rankings': sorted(self.quality_comparison.items(), 
                                            key=lambda x: x[1]['noise_reduction_ratio'], reverse=True),
            'best_noise_reduction': max(self.quality_comparison.items(), 
                                      key=lambda x: x[1]['noise_reduction_ratio']),
            'worst_noise_reduction': min(self.quality_comparison.items(), 
                                       key=lambda x: x[1]['noise_reduction_ratio'])
        }
    
    def _generate_discriminative_analysis(self):
        """분별력 분석 결과"""
        if not hasattr(self, 'discriminative_comparison'):
            return {}
        
        return {
            'sensor_discriminative_rankings': sorted(self.discriminative_comparison.items(), 
                                                   key=lambda x: x[1]['improvement_ratio'], reverse=True),
            'best_improvement': max(self.discriminative_comparison.items(), 
                                  key=lambda x: x[1]['improvement_ratio']),
            'worst_improvement': min(self.discriminative_comparison.items(), 
                                   key=lambda x: x[1]['improvement_ratio'])
        }
    
    def _generate_separability_analysis(self):
        """분리 가능성 분석 결과"""
        if not hasattr(self, 'separability_test'):
            return {}
        
        return {
            'original_analysis': self.separability_test['original'],
            'complementary_analysis': self.separability_test['complementary'],
            'improvement_summary': {
                'separability_improvement': self.separability_test['complementary']['separability_score'] - 
                                          self.separability_test['original']['separability_score'],
                'silhouette_improvement': self.separability_test['complementary']['silhouette_score'] - 
                                        self.separability_test['original']['silhouette_score'],
                'cluster_purity_improvement': self.separability_test['complementary']['avg_cluster_purity'] - 
                                            self.separability_test['original']['avg_cluster_purity']
            }
        }
    
    def _generate_conclusions(self):
        """결론 및 권장사항"""
        conclusions = {
            'complementary_filter_effectiveness': '',
            'key_findings': [],
            'recommendations': []
        }
        
        # 상보 필터 효과성 평가
        if hasattr(self, 'separability_test'):
            original_score = self.separability_test['original']['separability_score']
            complementary_score = self.separability_test['complementary']['separability_score']
            
            if complementary_score > original_score * 1.2:
                conclusions['complementary_filter_effectiveness'] = 'HIGH'
            elif complementary_score > original_score * 1.1:
                conclusions['complementary_filter_effectiveness'] = 'MEDIUM'
            else:
                conclusions['complementary_filter_effectiveness'] = 'LOW'
        
        # 주요 발견사항
        if hasattr(self, 'quality_comparison'):
            avg_noise_reduction = np.mean([self.quality_comparison[sensor]['noise_reduction_ratio'] 
                                         for sensor in self.quality_comparison])
            if avg_noise_reduction > 0.1:
                conclusions['key_findings'].append(f'상보 필터로 평균 {avg_noise_reduction:.1%} 노이즈 감소')
        
        if hasattr(self, 'separability_test'):
            if self.separability_test['complementary']['separability_score'] < 0.3:
                conclusions['key_findings'].append('상보 필터 적용 후에도 클래스 분리가 어려움')
            elif self.separability_test['complementary']['separability_score'] > 0.7:
                conclusions['key_findings'].append('상보 필터로 클래스 분리가 크게 개선됨')
        
        # 권장사항
        if conclusions['complementary_filter_effectiveness'] == 'HIGH':
            conclusions['recommendations'].append('상보 필터 사용을 강력히 권장')
            conclusions['recommendations'].append('기존 모델에 상보 필터 적용 고려')
        elif conclusions['complementary_filter_effectiveness'] == 'MEDIUM':
            conclusions['recommendations'].append('상보 필터 사용을 권장')
            conclusions['recommendations'].append('추가적인 특징 추출과 함께 사용')
        else:
            conclusions['recommendations'].append('상보 필터만으로는 한계가 있음')
            conclusions['recommendations'].append('다른 접근법과 함께 고려')
        
        return conclusions
    
    def _generate_text_report(self, report):
        """텍스트 보고서 생성"""
        with open('complementary_filter_analysis_report.txt', 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("상보 필터 효과 분석 보고서\n")
            f.write("=" * 80 + "\n\n")
            
            # 요약
            summary = report['summary']
            f.write("📊 분석 요약\n")
            f.write("-" * 40 + "\n")
            f.write(f"총 클래스 수: {summary['total_classes']}\n")
            if 'avg_noise_reduction_ratio' in summary:
                f.write(f"평균 노이즈 감소율: {summary['avg_noise_reduction_ratio']:.3f}\n")
            if 'avg_discriminative_improvement' in summary:
                f.write(f"평균 분별력 개선율: {summary['avg_discriminative_improvement']:.3f}\n")
            if 'original_separability_score' in summary:
                f.write(f"원본 데이터 분리 가능성: {summary['original_separability_score']:.3f}\n")
            if 'complementary_separability_score' in summary:
                f.write(f"상보 필터 데이터 분리 가능성: {summary['complementary_separability_score']:.3f}\n")
            if 'separability_improvement' in summary:
                f.write(f"분리 가능성 개선: {summary['separability_improvement']:.3f}\n")
            f.write(f"분석 시간: {summary['analysis_timestamp']}\n\n")
            
            # 품질 분석
            if report['quality_comparison']:
                f.write("🔍 데이터 품질 분석\n")
                f.write("-" * 40 + "\n")
                f.write("센서별 노이즈 감소율:\n")
                for sensor, data in report['quality_comparison']['sensor_quality_rankings']:
                    f.write(f"  {sensor}: {data['noise_reduction_ratio']:.3f}\n")
                f.write("\n")
            
            # 분별력 분석
            if report['discriminative_comparison']:
                f.write("🎯 분별력 분석\n")
                f.write("-" * 40 + "\n")
                f.write("센서별 분별력 개선율:\n")
                for sensor, data in report['discriminative_comparison']['sensor_discriminative_rankings']:
                    f.write(f"  {sensor}: {data['improvement_ratio']:.3f}\n")
                f.write("\n")
            
            # 분리 가능성 분석
            if report['separability_analysis']:
                f.write("📊 분리 가능성 분석\n")
                f.write("-" * 40 + "\n")
                sep_analysis = report['separability_analysis']
                f.write(f"원본 데이터 분리 가능성: {sep_analysis['original_analysis']['separability_score']:.3f}\n")
                f.write(f"상보 필터 데이터 분리 가능성: {sep_analysis['complementary_analysis']['separability_score']:.3f}\n")
                f.write(f"실루엣 점수 개선: {sep_analysis['improvement_summary']['silhouette_improvement']:.3f}\n")
                f.write(f"클러스터 순도 개선: {sep_analysis['improvement_summary']['cluster_purity_improvement']:.3f}\n")
                f.write("\n")
            
            # 결론
            if report['conclusions']:
                f.write("💡 결론 및 권장사항\n")
                f.write("-" * 40 + "\n")
                f.write(f"상보 필터 효과성: {report['conclusions']['complementary_filter_effectiveness']}\n\n")
                
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
    
    analyzer = ComplementaryFilterAnalyzer(data_dir)
    analyzer.analyze_complementary_filter_effect()
    
    print(f"\n🎉 상보 필터 효과 분석 완료!")
    print(f"📁 생성된 파일들:")
    print(f"  - complementary_filter_analysis_report.json: 상세 분석 결과")
    print(f"  - complementary_filter_analysis_report.txt: 텍스트 보고서")
    print(f"  - complementary_filter_comparison.png: 비교 시각화")

if __name__ == "__main__":
    main()
