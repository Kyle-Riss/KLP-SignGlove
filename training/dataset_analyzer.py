#!/usr/bin/env python3
"""
24개 자음/모음 데이터셋 분석 스크립트
각 클래스별 특성 분석 및 맞춤형 솔루션 제안
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

class DatasetAnalyzer:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.label_mapper = KSLLabelMapper()
        self.all_classes = self.label_mapper.get_all_classes()
        self.class_data = {}
        self.analysis_results = {}
        
        print(f"🔍 24개 자음/모음 데이터셋 분석 시작")
        print(f"📁 데이터 디렉토리: {data_dir}")
    
    def analyze_all_classes(self):
        """전체 클래스 분석"""
        print("\n📊 1단계: 데이터 로딩")
        self._load_all_data()
        
        print("\n📊 2단계: 기본 통계 분석")
        self._analyze_basic_stats()
        
        print("\n📊 3단계: 클래스 간 유사도 분석")
        self._analyze_similarities()
        
        print("\n📊 4단계: 데이터 품질 분석")
        self._analyze_data_quality()
        
        print("\n📊 5단계: 시각화 및 보고서 생성")
        self._create_visualizations()
        self._generate_report()
        
        print("\n✅ 분석 완료!")
    
    def _load_all_data(self):
        """전체 데이터 로딩"""
        for class_name in self.all_classes:
            pattern = os.path.join(self.data_dir, f"**/{class_name}/**/episode_*.csv")
            files = glob.glob(pattern, recursive=True)
            
            if not files:
                print(f"  ⚠️  {class_name}: 파일 없음")
                continue
            
            class_data = []
            for file_path in files:
                try:
                    df = pd.read_csv(file_path)
                    sensor_data = df[['pitch', 'roll', 'yaw', 'flex1', 'flex2', 'flex3', 'flex4', 'flex5']].values
                    class_data.append(sensor_data)
                except Exception as e:
                    print(f"  ⚠️  {file_path} 로드 실패: {e}")
            
            if class_data:
                self.class_data[class_name] = class_data
                print(f"  ✅ {class_name}: {len(class_data)}개 파일")
    
    def _analyze_basic_stats(self):
        """기본 통계 분석"""
        sensor_names = ['pitch', 'roll', 'yaw', 'flex1', 'flex2', 'flex3', 'flex4', 'flex5']
        
        for class_name, data_list in self.class_data.items():
            all_data = np.vstack(data_list)
            
            stats = {}
            for i, sensor_name in enumerate(sensor_names):
                values = all_data[:, i]
                stats[sensor_name] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'range': float(np.max(values) - np.min(values)),
                    'zero_ratio': float(np.sum(values == 0) / len(values)),
                    'outlier_ratio': float(np.sum(np.abs(values - np.mean(values)) > 3 * np.std(values)) / len(values))
                }
            
            self.analysis_results[class_name] = {
                'stats': stats,
                'data_count': len(data_list),
                'total_samples': sum(len(d) for d in data_list),
                'avg_length': float(np.mean([len(d) for d in data_list]))
            }
    
    def _analyze_similarities(self):
        """클래스 간 유사도 분석"""
        # 각 클래스의 평균 특성 벡터 계산
        class_features = {}
        sensor_names = ['pitch', 'roll', 'yaw', 'flex1', 'flex2', 'flex3', 'flex4', 'flex5']
        
        for class_name in self.class_data.keys():
            all_data = np.vstack(self.class_data[class_name])
            feature_vector = []
            
            for sensor_name in sensor_names:
                values = all_data[:, sensor_names.index(sensor_name)]
                feature_vector.extend([
                    np.mean(values),
                    np.std(values),
                    np.median(values)
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
                    similarity = cosine_similarity(
                        class_features[class1].reshape(1, -1),
                        class_features[class2].reshape(1, -1)
                    )[0, 0]
                    similarity_matrix[i, j] = similarity
        
        # 높은 유사도 쌍 찾기
        high_similarity_pairs = []
        for i in range(len(class_names)):
            for j in range(i+1, len(class_names)):
                similarity = similarity_matrix[i, j]
                if similarity > 0.9:
                    high_similarity_pairs.append({
                        'class1': class_names[i],
                        'class2': class_names[j],
                        'similarity': float(similarity)
                    })
        
        self.analysis_results['similarity'] = {
            'matrix': similarity_matrix.tolist(),
            'class_names': class_names,
            'high_similarity_pairs': high_similarity_pairs
        }
    
    def _analyze_data_quality(self):
        """데이터 품질 분석"""
        quality_report = {}
        
        for class_name, data_list in self.class_data.items():
            all_data = np.vstack(data_list)
            sensor_names = ['pitch', 'roll', 'yaw', 'flex1', 'flex2', 'flex3', 'flex4', 'flex5']
            
            # 센서별 품질 점수
            sensor_quality = {}
            for i, sensor_name in enumerate(sensor_names):
                values = all_data[:, i]
                
                # 품질 지표
                zero_ratio = np.sum(values == 0) / len(values)
                constant_ratio = np.sum(np.diff(values) == 0) / (len(values) - 1)
                outlier_ratio = np.sum(np.abs(values - np.mean(values)) > 3 * np.std(values)) / len(values)
                
                health_score = 1 - (zero_ratio + constant_ratio + outlier_ratio) / 3
                
                sensor_quality[sensor_name] = {
                    'health_score': float(health_score),
                    'zero_ratio': float(zero_ratio),
                    'constant_ratio': float(constant_ratio),
                    'outlier_ratio': float(outlier_ratio),
                    'status': 'healthy' if health_score > 0.7 else 'warning' if health_score > 0.4 else 'faulty'
                }
            
            # 전체 품질 점수
            avg_health = np.mean([sensor_quality[sensor]['health_score'] for sensor in sensor_quality])
            
            quality_report[class_name] = {
                'sensor_quality': sensor_quality,
                'overall_health': float(avg_health),
                'problematic_sensors': [sensor for sensor, quality in sensor_quality.items() if quality['status'] == 'faulty']
            }
        
        self.analysis_results['quality'] = quality_report
    
    def _create_visualizations(self):
        """시각화 생성"""
        # 1. 클래스별 샘플 수 비교
        plt.figure(figsize=(15, 6))
        class_names = list(self.class_data.keys())
        sample_counts = [self.analysis_results[cls]['total_samples'] for cls in class_names]
        
        plt.subplot(1, 2, 1)
        plt.bar(class_names, sample_counts, color='skyblue')
        plt.title('클래스별 샘플 수')
        plt.xlabel('클래스')
        plt.ylabel('샘플 수')
        plt.xticks(rotation=45)
        
        # 2. 품질 점수 비교
        plt.subplot(1, 2, 2)
        quality_scores = [self.analysis_results['quality'][cls]['overall_health'] for cls in class_names]
        colors = ['red' if score < 0.5 else 'orange' if score < 0.7 else 'green' for score in quality_scores]
        
        plt.bar(class_names, quality_scores, color=colors)
        plt.title('클래스별 데이터 품질 점수')
        plt.xlabel('클래스')
        plt.ylabel('품질 점수')
        plt.xticks(rotation=45)
        plt.ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig('class_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. 유사도 히트맵
        if 'similarity' in self.analysis_results:
            similarity_data = self.analysis_results['similarity']
            plt.figure(figsize=(12, 10))
            sns.heatmap(np.array(similarity_data['matrix']), 
                       xticklabels=similarity_data['class_names'],
                       yticklabels=similarity_data['class_names'],
                       annot=True, fmt='.2f', cmap='RdYlBu_r', center=0.5)
            plt.title('클래스 간 유사도 행렬')
            plt.tight_layout()
            plt.savefig('class_similarity.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        print("  ✅ 시각화 파일 저장 완료")
    
    def _generate_report(self):
        """분석 보고서 생성"""
        report = {
            'summary': self._generate_summary(),
            'class_details': self._generate_class_details(),
            'similarity_analysis': self._generate_similarity_summary(),
            'quality_analysis': self._generate_quality_summary(),
            'recommendations': self._generate_recommendations()
        }
        
        # JSON 저장
        with open('dataset_analysis_report.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        # 텍스트 보고서
        self._generate_text_report(report)
        
        print("  ✅ 분석 보고서 저장 완료")
    
    def _generate_summary(self):
        """요약 생성"""
        total_classes = len(self.all_classes)
        analyzed_classes = len(self.class_data)
        total_samples = sum(self.analysis_results[cls]['total_samples'] for cls in self.class_data.keys())
        
        return {
            'total_classes': total_classes,
            'analyzed_classes': analyzed_classes,
            'total_samples': total_samples,
            'avg_samples_per_class': total_samples / analyzed_classes if analyzed_classes > 0 else 0
        }
    
    def _generate_class_details(self):
        """클래스별 상세 정보"""
        details = {}
        for class_name in self.class_data.keys():
            analysis = self.analysis_results[class_name]
            details[class_name] = {
                'data_count': analysis['data_count'],
                'total_samples': analysis['total_samples'],
                'avg_length': analysis['avg_length'],
                'key_stats': {
                    'pitch_mean': analysis['stats']['pitch']['mean'],
                    'roll_mean': analysis['stats']['roll']['mean'],
                    'yaw_mean': analysis['stats']['yaw']['mean'],
                    'flex1_mean': analysis['stats']['flex1']['mean'],
                    'flex2_mean': analysis['stats']['flex2']['mean']
                }
            }
        return details
    
    def _generate_similarity_summary(self):
        """유사도 분석 요약"""
        if 'similarity' not in self.analysis_results:
            return {}
        
        similarity_data = self.analysis_results['similarity']
        return {
            'high_similarity_pairs': similarity_data['high_similarity_pairs'],
            'total_high_similarity_pairs': len(similarity_data['high_similarity_pairs'])
        }
    
    def _generate_quality_summary(self):
        """품질 분석 요약"""
        if 'quality' not in self.analysis_results:
            return {}
        
        quality_data = self.analysis_results['quality']
        
        # 문제가 있는 클래스들
        problematic_classes = []
        for class_name, quality in quality_data.items():
            if quality['overall_health'] < 0.7:
                problematic_classes.append({
                    'class': class_name,
                    'health_score': quality['overall_health'],
                    'problematic_sensors': quality['problematic_sensors']
                })
        
        return {
            'problematic_classes': problematic_classes,
            'total_problematic': len(problematic_classes)
        }
    
    def _generate_recommendations(self):
        """권장사항 생성"""
        recommendations = []
        
        # 유사도 기반 권장사항
        if 'similarity' in self.analysis_results:
            high_similarity_pairs = self.analysis_results['similarity']['high_similarity_pairs']
            if high_similarity_pairs:
                recommendations.append({
                    'type': 'attention_mechanism',
                    'reason': f'{len(high_similarity_pairs)}개의 높은 유사도 클래스 쌍 발견',
                    'details': [f"{pair['class1']}-{pair['class2']}" for pair in high_similarity_pairs[:5]]
                })
        
        # 품질 기반 권장사항
        if 'quality' in self.analysis_results:
            quality_data = self.analysis_results['quality']
            problematic_classes = [cls for cls, quality in quality_data.items() if quality['overall_health'] < 0.7]
            
            if problematic_classes:
                recommendations.append({
                    'type': 'data_cleaning',
                    'reason': f'{len(problematic_classes)}개의 품질 문제 클래스 발견',
                    'details': problematic_classes
                })
        
        return recommendations
    
    def _generate_text_report(self, report):
        """텍스트 보고서 생성"""
        with open('dataset_analysis_report.txt', 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("24개 자음/모음 데이터셋 분석 보고서\n")
            f.write("=" * 60 + "\n\n")
            
            # 요약
            summary = report['summary']
            f.write("📊 분석 요약\n")
            f.write("-" * 30 + "\n")
            f.write(f"총 클래스 수: {summary['total_classes']}\n")
            f.write(f"분석된 클래스 수: {summary['analyzed_classes']}\n")
            f.write(f"총 샘플 수: {summary['total_samples']:,}\n")
            f.write(f"클래스당 평균 샘플 수: {summary['avg_samples_per_class']:.1f}\n\n")
            
            # 클래스별 상세 정보
            f.write("📈 클래스별 상세 정보\n")
            f.write("-" * 30 + "\n")
            for class_name, details in report['class_details'].items():
                f.write(f"\n{class_name}:\n")
                f.write(f"  파일 수: {details['data_count']}\n")
                f.write(f"  샘플 수: {details['total_samples']}\n")
                f.write(f"  평균 길이: {details['avg_length']:.1f}\n")
                f.write(f"  주요 특성: pitch={details['key_stats']['pitch_mean']:.1f}, roll={details['key_stats']['roll_mean']:.1f}\n")
            
            # 유사도 분석
            if report['similarity_analysis']:
                f.write("\n🔗 유사도 분석\n")
                f.write("-" * 30 + "\n")
                f.write(f"높은 유사도 쌍 수: {report['similarity_analysis']['total_high_similarity_pairs']}\n")
                for pair in report['similarity_analysis']['high_similarity_pairs'][:10]:
                    f.write(f"  {pair['class1']} - {pair['class2']}: {pair['similarity']:.3f}\n")
            
            # 품질 분석
            if report['quality_analysis']:
                f.write("\n🔍 품질 분석\n")
                f.write("-" * 30 + "\n")
                f.write(f"문제 클래스 수: {report['quality_analysis']['total_problematic']}\n")
                for problem in report['quality_analysis']['problematic_classes']:
                    f.write(f"  {problem['class']}: 점수 {problem['health_score']:.3f}, 문제 센서: {', '.join(problem['problematic_sensors'])}\n")
            
            # 권장사항
            f.write("\n💡 권장사항\n")
            f.write("-" * 30 + "\n")
            for rec in report['recommendations']:
                f.write(f"\n{rec['type'].upper()}:\n")
                f.write(f"  이유: {rec['reason']}\n")
                f.write(f"  세부사항: {', '.join(rec['details'])}\n")

def main():
    """메인 함수"""
    data_dir = 'integrations/SignGlove_HW/github_unified_data'
    
    analyzer = DatasetAnalyzer(data_dir)
    analyzer.analyze_all_classes()
    
    print(f"\n🎉 분석 완료!")
    print(f"📁 생성된 파일들:")
    print(f"  - dataset_analysis_report.json: 상세 분석 결과")
    print(f"  - dataset_analysis_report.txt: 텍스트 보고서")
    print(f"  - class_analysis.png: 클래스별 분석 차트")
    print(f"  - class_similarity.png: 클래스 유사도 히트맵")

if __name__ == "__main__":
    main()
