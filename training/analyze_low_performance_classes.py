#!/usr/bin/env python3
"""
낮은 성능 클래스 분석
상보 필터 모델에서 성능이 낮은 ㅈ, ㅍ, ㅕ 클래스 분석
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
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from training.label_mapping import KSLLabelMapper

class LowPerformanceAnalyzer:
    """낮은 성능 클래스 분석기"""
    
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.label_mapper = KSLLabelMapper()
        self.low_performance_classes = ['ㅈ', 'ㅍ', 'ㅕ']
        
        print(f"🔍 낮은 성능 클래스 분석 시작")
        print(f"📁 분석 대상: {', '.join(self.low_performance_classes)}")
    
    def analyze_low_performance_classes(self):
        """낮은 성능 클래스 종합 분석"""
        print(f"\n📊 1단계: 낮은 성능 클래스 데이터 로딩")
        class_data = self._load_class_data()
        
        print(f"\n📊 2단계: 센서 패턴 분석")
        self._analyze_sensor_patterns(class_data)
        
        print(f"\n📊 3단계: 클래스 간 유사도 분석")
        self._analyze_class_similarities(class_data)
        
        print(f"\n📊 4단계: 데이터 품질 분석")
        self._analyze_data_quality(class_data)
        
        print(f"\n📊 5단계: 오분류 패턴 분석")
        self._analyze_misclassification_patterns(class_data)
        
        print(f"\n📊 6단계: 시각화 및 보고서 생성")
        self._create_visualizations(class_data)
        self._generate_analysis_report()
        
        print(f"\n✅ 낮은 성능 클래스 분석 완료!")
    
    def _load_class_data(self):
        """클래스별 데이터 로딩"""
        print("  📥 낮은 성능 클래스 데이터 로딩 중...")
        
        class_data = {}
        sensor_names = ['pitch', 'roll', 'yaw', 'flex1', 'flex2', 'flex3', 'flex4', 'flex5']
        
        for class_name in self.low_performance_classes:
            print(f"    📊 {class_name} 데이터 로딩...")
            
            pattern = os.path.join(self.data_dir, f"**/{class_name}/**/episode_*.csv")
            files = glob.glob(pattern, recursive=True)
            
            if not files:
                print(f"    ⚠️  {class_name}: 파일 없음")
                continue
            
            class_sensor_data = {sensor: [] for sensor in sensor_names}
            episode_data = []
            
            for file_path in files:
                try:
                    df = pd.read_csv(file_path)
                    episode_sensors = {}
                    
                    for sensor in sensor_names:
                        if sensor in df.columns:
                            values = df[sensor].values
                            class_sensor_data[sensor].extend(values)
                            episode_sensors[sensor] = values
                    
                    episode_data.append(episode_sensors)
                    
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
                        'noise_level': float(np.std(values) / (np.mean(np.abs(values)) + 1e-8)),
                        'outlier_ratio': float(np.sum(np.abs(values - np.mean(values)) > 2 * np.std(values)) / len(values))
                    }
                else:
                    sensor_stats[sensor] = None
            
            class_data[class_name] = {
                'raw_data': class_sensor_data,
                'episode_data': episode_data,
                'stats': sensor_stats,
                'sample_count': len(class_sensor_data[sensor_names[0]]) if class_sensor_data[sensor_names[0]] else 0
            }
            
            print(f"    ✅ {class_name}: {class_data[class_name]['sample_count']}개 샘플")
        
        return class_data
    
    def _analyze_sensor_patterns(self, class_data):
        """센서 패턴 분석"""
        print("  🔍 센서 패턴 분석 중...")
        
        sensor_names = ['pitch', 'roll', 'yaw', 'flex1', 'flex2', 'flex3', 'flex4', 'flex5']
        
        pattern_analysis = {}
        
        for class_name, data in class_data.items():
            print(f"    📊 {class_name} 센서 패턴 분석...")
            
            pattern_analysis[class_name] = {}
            
            for sensor in sensor_names:
                if data['stats'][sensor] is not None:
                    stats = data['stats'][sensor]
                    
                    # 패턴 특성 분석
                    pattern_analysis[class_name][sensor] = {
                        'dominance': 'high' if stats['range'] > np.mean([data['stats'][s]['range'] for s in sensor_names if data['stats'][s] is not None]) else 'low',
                        'stability': 'high' if stats['std'] < np.mean([data['stats'][s]['std'] for s in sensor_names if data['stats'][s] is not None]) else 'low',
                        'noise_level': 'high' if stats['noise_level'] > 0.5 else 'low',
                        'outlier_ratio': stats['outlier_ratio'],
                        'zero_ratio': stats['zero_ratio']
                    }
        
        self.pattern_analysis = pattern_analysis
    
    def _analyze_class_similarities(self, class_data):
        """클래스 간 유사도 분석"""
        print("  🔍 클래스 간 유사도 분석 중...")
        
        sensor_names = ['pitch', 'roll', 'yaw', 'flex1', 'flex2', 'flex3', 'flex4', 'flex5']
        
        # 모든 클래스의 특징 벡터 생성
        feature_vectors = {}
        
        for class_name, data in class_data.items():
            feature_vector = []
            
            for sensor in sensor_names:
                if data['stats'][sensor] is not None:
                    stats = data['stats'][sensor]
                    feature_vector.extend([
                        stats['mean'],
                        stats['std'],
                        stats['range'],
                        stats['median']
                    ])
                else:
                    feature_vector.extend([0, 0, 0, 0])
            
            feature_vectors[class_name] = feature_vector
        
        # 유사도 계산
        similarity_matrix = {}
        for class1 in self.low_performance_classes:
            similarity_matrix[class1] = {}
            for class2 in self.low_performance_classes:
                if class1 != class2:
                    vec1 = np.array(feature_vectors[class1])
                    vec2 = np.array(feature_vectors[class2])
                    similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
                    similarity_matrix[class1][class2] = float(similarity)
        
        self.similarity_matrix = similarity_matrix
    
    def _analyze_data_quality(self, class_data):
        """데이터 품질 분석"""
        print("  🔍 데이터 품질 분석 중...")
        
        quality_analysis = {}
        
        for class_name, data in class_data.items():
            print(f"    📊 {class_name} 데이터 품질 분석...")
            
            quality_scores = {}
            
            # 센서별 품질 점수 계산
            for sensor, stats in data['stats'].items():
                if stats is not None:
                    # 데이터 일관성 점수
                    consistency_score = 1 - stats['outlier_ratio']
                    
                    # 노이즈 점수
                    noise_score = 1 - min(stats['noise_level'], 1.0)
                    
                    # 범위 점수 (너무 작거나 큰 범위는 문제)
                    range_score = 1.0
                    if stats['range'] < 0.1:  # 너무 작은 범위
                        range_score = 0.5
                    elif stats['range'] > 1000:  # 너무 큰 범위
                        range_score = 0.5
                    
                    # 종합 품질 점수
                    quality_score = (consistency_score + noise_score + range_score) / 3
                    quality_scores[sensor] = quality_score
            
            quality_analysis[class_name] = quality_scores
        
        self.quality_analysis = quality_analysis
    
    def _analyze_misclassification_patterns(self, class_data):
        """오분류 패턴 분석"""
        print("  🔍 오분류 패턴 분석 중...")
        
        # 상보 필터 모델의 혼동 행렬에서 오분류 패턴 추출
        try:
            with open('complementary_filter_classification_report.json', 'r', encoding='utf-8') as f:
                classification_report = json.load(f)
            
            misclassification_patterns = {}
            
            for class_name in self.low_performance_classes:
                class_idx = self.label_mapper.get_label_id(class_name)
                class_key = str(class_idx)
                
                if class_key in classification_report:
                    class_report = classification_report[class_key]
                    
                    # 오분류된 클래스들 찾기
                    misclassified_classes = []
                    for other_class_name in self.label_mapper.get_all_classes():
                        other_class_idx = self.label_mapper.get_label_id(other_class_name)
                        other_class_key = str(other_class_idx)
                        
                        if other_class_key in classification_report:
                            other_report = classification_report[other_class_key]
                            
                            # 다른 클래스가 이 클래스로 잘못 분류된 경우
                            if 'support' in other_report and other_report['support'] > 0:
                                # 실제로는 other_class이지만 class_name으로 분류된 비율
                                misclassification_rate = 1 - other_report['precision']
                                if misclassification_rate > 0.1:  # 10% 이상 오분류
                                    misclassified_classes.append({
                                        'class': other_class_name,
                                        'rate': misclassification_rate
                                    })
                    
                    misclassification_patterns[class_name] = sorted(
                        misclassified_classes, 
                        key=lambda x: x['rate'], 
                        reverse=True
                    )
            
            self.misclassification_patterns = misclassification_patterns
            
        except FileNotFoundError:
            print("    ⚠️  분류 보고서 파일을 찾을 수 없습니다.")
            self.misclassification_patterns = {}
    
    def _create_visualizations(self, class_data):
        """시각화 생성"""
        print("  📊 시각화 생성 중...")
        
        # 1. 센서별 통계 비교
        sensor_names = ['pitch', 'roll', 'yaw', 'flex1', 'flex2', 'flex3', 'flex4', 'flex5']
        
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()
        
        for i, sensor in enumerate(sensor_names):
            means = []
            stds = []
            ranges = []
            class_names = []
            
            for class_name in self.low_performance_classes:
                if class_name in class_data and class_data[class_name]['stats'][sensor] is not None:
                    stats = class_data[class_name]['stats'][sensor]
                    means.append(stats['mean'])
                    stds.append(stats['std'])
                    ranges.append(stats['range'])
                    class_names.append(class_name)
            
            if means:
                x = np.arange(len(class_names))
                width = 0.25
                
                axes[i].bar(x - width, means, width, label='Mean', alpha=0.8)
                axes[i].bar(x, stds, width, label='Std', alpha=0.8)
                axes[i].bar(x + width, ranges, width, label='Range', alpha=0.8)
                
                axes[i].set_title(f'{sensor} Statistics')
                axes[i].set_xticks(x)
                axes[i].set_xticklabels(class_names)
                axes[i].legend()
                axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('low_performance_sensor_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. 클래스 간 유사도 히트맵
        if hasattr(self, 'similarity_matrix'):
            plt.figure(figsize=(8, 6))
            similarity_data = []
            class_names = list(self.similarity_matrix.keys())
            
            for class1 in class_names:
                row = []
                for class2 in class_names:
                    if class1 == class2:
                        row.append(1.0)
                    else:
                        row.append(self.similarity_matrix[class1].get(class2, 0.0))
                similarity_data.append(row)
            
            sns.heatmap(similarity_data, annot=True, fmt='.3f', 
                       xticklabels=class_names, yticklabels=class_names,
                       cmap='RdYlBu_r', center=0.5)
            plt.title('Low Performance Classes Similarity Matrix')
            plt.tight_layout()
            plt.savefig('low_performance_similarity_matrix.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 3. 데이터 품질 점수
        if hasattr(self, 'quality_analysis'):
            plt.figure(figsize=(12, 6))
            
            quality_data = []
            for class_name in self.low_performance_classes:
                if class_name in self.quality_analysis:
                    class_qualities = list(self.quality_analysis[class_name].values())
                    quality_data.append(class_qualities)
            
            if quality_data:
                quality_data = np.array(quality_data)
                plt.imshow(quality_data, cmap='RdYlGn', aspect='auto')
                plt.colorbar(label='Quality Score')
                plt.xticks(range(len(sensor_names)), sensor_names, rotation=45)
                plt.yticks(range(len(self.low_performance_classes)), self.low_performance_classes)
                plt.title('Data Quality Scores by Sensor and Class')
                plt.tight_layout()
                plt.savefig('low_performance_quality_scores.png', dpi=300, bbox_inches='tight')
                plt.close()
        
        print("    ✅ 시각화 파일 저장 완료")
    
    def _generate_analysis_report(self):
        """분석 보고서 생성"""
        print("  📝 분석 보고서 생성 중...")
        
        report = {
            'summary': self._generate_summary(),
            'sensor_pattern_analysis': self._generate_pattern_analysis(),
            'similarity_analysis': self._generate_similarity_analysis(),
            'quality_analysis': self._generate_quality_analysis(),
            'misclassification_analysis': self._generate_misclassification_analysis(),
            'recommendations': self._generate_recommendations()
        }
        
        # JSON 저장
        with open('low_performance_analysis_report.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        # 텍스트 보고서
        self._generate_text_report(report)
        
        print("    ✅ 분석 보고서 저장 완료")
    
    def _generate_summary(self):
        """요약 생성"""
        return {
            'low_performance_classes': self.low_performance_classes,
            'analysis_timestamp': pd.Timestamp.now().isoformat(),
            'total_classes_analyzed': len(self.low_performance_classes)
        }
    
    def _generate_pattern_analysis(self):
        """패턴 분석 결과"""
        if not hasattr(self, 'pattern_analysis'):
            return {}
        
        return self.pattern_analysis
    
    def _generate_similarity_analysis(self):
        """유사도 분석 결과"""
        if not hasattr(self, 'similarity_matrix'):
            return {}
        
        return self.similarity_matrix
    
    def _generate_quality_analysis(self):
        """품질 분석 결과"""
        if not hasattr(self, 'quality_analysis'):
            return {}
        
        return self.quality_analysis
    
    def _generate_misclassification_analysis(self):
        """오분류 분석 결과"""
        if not hasattr(self, 'misclassification_patterns'):
            return {}
        
        return self.misclassification_patterns
    
    def _generate_recommendations(self):
        """권장사항 생성"""
        recommendations = {
            'data_quality_improvements': [],
            'model_improvements': [],
            'feature_engineering': [],
            'training_strategies': []
        }
        
        # 데이터 품질 개선 권장사항
        if hasattr(self, 'quality_analysis'):
            for class_name, qualities in self.quality_analysis.items():
                low_quality_sensors = [sensor for sensor, score in qualities.items() if score < 0.5]
                if low_quality_sensors:
                    recommendations['data_quality_improvements'].append(
                        f"{class_name}: {', '.join(low_quality_sensors)} 센서 데이터 품질 개선 필요"
                    )
        
        # 유사도 기반 권장사항
        if hasattr(self, 'similarity_matrix'):
            high_similarity_pairs = []
            for class1, similarities in self.similarity_matrix.items():
                for class2, similarity in similarities.items():
                    if similarity > 0.8:
                        high_similarity_pairs.append(f"{class1}-{class2}: {similarity:.3f}")
            
            if high_similarity_pairs:
                recommendations['feature_engineering'].append(
                    f"높은 유사도 클래스 쌍: {', '.join(high_similarity_pairs)}"
                )
                recommendations['model_improvements'].append(
                    "높은 유사도 클래스들을 위한 특화된 특징 추출 필요"
                )
        
        # 일반적인 권장사항
        recommendations['training_strategies'].extend([
            "낮은 성능 클래스들에 대한 클래스 가중치 증가",
            "낮은 성능 클래스들에 대한 데이터 증강 강화",
            "낮은 성능 클래스들을 위한 특화된 손실 함수 고려"
        ])
        
        return recommendations
    
    def _generate_text_report(self, report):
        """텍스트 보고서 생성"""
        with open('low_performance_analysis_report.txt', 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("낮은 성능 클래스 분석 보고서\n")
            f.write("=" * 80 + "\n\n")
            
            # 요약
            summary = report['summary']
            f.write("📊 분석 요약\n")
            f.write("-" * 40 + "\n")
            f.write(f"분석 대상 클래스: {', '.join(summary['low_performance_classes'])}\n")
            f.write(f"분석 시간: {summary['analysis_timestamp']}\n\n")
            
            # 센서 패턴 분석
            if report['sensor_pattern_analysis']:
                f.write("🔍 센서 패턴 분석\n")
                f.write("-" * 40 + "\n")
                for class_name, patterns in report['sensor_pattern_analysis'].items():
                    f.write(f"{class_name}:\n")
                    for sensor, pattern in patterns.items():
                        f.write(f"  {sensor}: dominance={pattern['dominance']}, "
                               f"stability={pattern['stability']}, "
                               f"noise={pattern['noise_level']}\n")
                    f.write("\n")
            
            # 유사도 분석
            if report['similarity_analysis']:
                f.write("🎯 클래스 간 유사도 분석\n")
                f.write("-" * 40 + "\n")
                for class1, similarities in report['similarity_analysis'].items():
                    f.write(f"{class1}와의 유사도:\n")
                    for class2, similarity in similarities.items():
                        f.write(f"  {class2}: {similarity:.3f}\n")
                    f.write("\n")
            
            # 품질 분석
            if report['quality_analysis']:
                f.write("📈 데이터 품질 분석\n")
                f.write("-" * 40 + "\n")
                for class_name, qualities in report['quality_analysis'].items():
                    f.write(f"{class_name} 품질 점수:\n")
                    for sensor, score in qualities.items():
                        f.write(f"  {sensor}: {score:.3f}\n")
                    f.write("\n")
            
            # 오분류 분석
            if report['misclassification_analysis']:
                f.write("❌ 오분류 패턴 분석\n")
                f.write("-" * 40 + "\n")
                for class_name, patterns in report['misclassification_analysis'].items():
                    f.write(f"{class_name} 오분류 패턴:\n")
                    for pattern in patterns[:5]:  # 상위 5개만
                        f.write(f"  {pattern['class']}: {pattern['rate']:.3f}\n")
                    f.write("\n")
            
            # 권장사항
            if report['recommendations']:
                f.write("💡 권장사항\n")
                f.write("-" * 40 + "\n")
                
                for category, recs in report['recommendations'].items():
                    if recs:
                        f.write(f"{category}:\n")
                        for rec in recs:
                            f.write(f"  - {rec}\n")
                        f.write("\n")

def main():
    """메인 함수"""
    data_dir = 'integrations/SignGlove_HW/github_unified_data'
    
    analyzer = LowPerformanceAnalyzer(data_dir)
    analyzer.analyze_low_performance_classes()
    
    print(f"\n🎉 낮은 성능 클래스 분석 완료!")
    print(f"📁 생성된 파일들:")
    print(f"  - low_performance_analysis_report.json: 상세 분석 결과")
    print(f"  - low_performance_analysis_report.txt: 텍스트 보고서")
    print(f"  - low_performance_sensor_analysis.png: 센서 분석 시각화")
    print(f"  - low_performance_similarity_matrix.png: 유사도 행렬")
    print(f"  - low_performance_quality_scores.png: 품질 점수")

if __name__ == "__main__":
    main()
