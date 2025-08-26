#!/usr/bin/env python3
"""
ㅕ, ㅌ, ㄹ 클래스 상세 분석 스크립트
데이터 패턴 및 혼동 원인 분석
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
from collections import Counter, defaultdict
import warnings
warnings.filterwarnings('ignore')

# 프로젝트 루트 추가
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from training.label_mapping import KSLLabelMapper

class YeoTeulRieulAnalyzer:
    """ㅕ, ㅌ, ㄹ 클래스 상세 분석기"""
    
    def __init__(self):
        self.data_dir = 'integrations/SignGlove_HW/github_unified_data'
        self.label_mapper = KSLLabelMapper()
        self.target_classes = ['ㅕ', 'ㅌ', 'ㄹ']
        
        print(f"🔍 ㅕ, ㅌ, ㄹ 클래스 상세 분석 시작")
        print(f"📁 데이터 디렉토리: {self.data_dir}")
        print(f"🎯 분석 대상: {self.target_classes}")
    
    def load_class_data(self):
        """ㅕ, ㅌ, ㄹ 클래스 데이터 로딩"""
        print("\n📊 클래스별 데이터 로딩 중...")
        
        class_data = {}
        
        for class_name in self.target_classes:
            class_data[class_name] = {
                'files': [],
                'scenarios': defaultdict(list),
                'sensor_data': [],
                'file_paths': []
            }
            
            # 클래스별 파일 찾기
            pattern = os.path.join(self.data_dir, f"**/{class_name}/**/episode_*.csv")
            files = glob.glob(pattern, recursive=True)
            
            print(f"  {class_name}: {len(files)}개 파일 발견")
            
            for file_path in files:
                try:
                    # 시나리오 추출
                    scenario = self._extract_scenario(file_path)
                    
                    # CSV 파일 읽기
                    df = pd.read_csv(file_path)
                    sensor_data = df[['pitch', 'roll', 'yaw', 'flex1', 'flex2', 'flex3', 'flex4', 'flex5']].values
                    
                    class_data[class_name]['files'].append(file_path)
                    class_data[class_name]['scenarios'][scenario].append(file_path)
                    class_data[class_name]['sensor_data'].append(sensor_data)
                    class_data[class_name]['file_paths'].append(file_path)
                    
                except Exception as e:
                    print(f"⚠️  파일 로드 실패: {file_path} - {e}")
        
        return class_data
    
    def _extract_scenario(self, file_path):
        """파일 경로에서 시나리오 추출"""
        parts = file_path.split(os.sep)
        for part in parts:
            if part.isdigit() and 1 <= int(part) <= 5:
                return int(part)
        return None
    
    def analyze_sensor_patterns(self, class_data):
        """센서 패턴 분석"""
        print("\n📈 센서 패턴 분석")
        
        sensor_names = ['pitch', 'roll', 'yaw', 'flex1', 'flex2', 'flex3', 'flex4', 'flex5']
        
        for class_name in self.target_classes:
            print(f"\n🎯 {class_name} 클래스 센서 패턴:")
            
            all_sensor_data = []
            for sensor_data in class_data[class_name]['sensor_data']:
                all_sensor_data.extend(sensor_data)
            
            all_sensor_data = np.array(all_sensor_data)
            
            # 기본 통계
            print(f"  📊 전체 데이터 포인트: {len(all_sensor_data)}")
            print(f"  📊 파일 수: {len(class_data[class_name]['files'])}")
            
            # 센서별 통계
            for i, sensor_name in enumerate(sensor_names):
                sensor_values = all_sensor_data[:, i]
                
                print(f"  {sensor_name}:")
                print(f"    평균: {np.mean(sensor_values):.3f}")
                print(f"    표준편차: {np.std(sensor_values):.3f}")
                print(f"    최소값: {np.min(sensor_values):.3f}")
                print(f"    최대값: {np.max(sensor_values):.3f}")
                print(f"    범위: [{np.min(sensor_values):.3f}, {np.max(sensor_values):.3f}]")
                
                # 이상값 분석
                q1 = np.percentile(sensor_values, 25)
                q3 = np.percentile(sensor_values, 75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                
                outliers = np.sum((sensor_values < lower_bound) | (sensor_values > upper_bound))
                print(f"    이상값 개수: {outliers} ({outliers/len(sensor_values)*100:.1f}%)")
                
                # 특별한 패턴 체크
                if sensor_name == 'flex2' and class_name == 'ㅕ':
                    high_values = np.sum(sensor_values > 100)
                    if high_values > 0:
                        print(f"    ⚠️  flex2 > 100 값: {high_values}개 ({high_values/len(sensor_values)*100:.1f}%)")
    
    def analyze_scenario_distribution(self, class_data):
        """시나리오별 분포 분석"""
        print("\n📊 시나리오별 분포 분석")
        
        for class_name in self.target_classes:
            print(f"\n🎯 {class_name} 클래스:")
            
            scenarios = class_data[class_name]['scenarios']
            total_files = len(class_data[class_name]['files'])
            
            for scenario in sorted(scenarios.keys()):
                file_count = len(scenarios[scenario])
                percentage = (file_count / total_files) * 100
                print(f"  시나리오 {scenario}: {file_count}개 파일 ({percentage:.1f}%)")
    
    def analyze_sensor_correlations(self, class_data):
        """센서 간 상관관계 분석"""
        print("\n🔗 센서 간 상관관계 분석")
        
        sensor_names = ['pitch', 'roll', 'yaw', 'flex1', 'flex2', 'flex3', 'flex4', 'flex5']
        
        for class_name in self.target_classes:
            print(f"\n🎯 {class_name} 클래스:")
            
            # 모든 센서 데이터 결합
            all_sensor_data = []
            for sensor_data in class_data[class_name]['sensor_data']:
                all_sensor_data.extend(sensor_data)
            
            all_sensor_data = np.array(all_sensor_data)
            
            # 상관관계 행렬 계산
            correlation_matrix = np.corrcoef(all_sensor_data.T)
            
            # 높은 상관관계 쌍 찾기
            high_corr_pairs = []
            for i in range(len(sensor_names)):
                for j in range(i+1, len(sensor_names)):
                    corr = correlation_matrix[i, j]
                    if abs(corr) > 0.7:  # 0.7 이상의 상관관계
                        high_corr_pairs.append((sensor_names[i], sensor_names[j], corr))
            
            # 상관관계 높은 순으로 정렬
            high_corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
            
            print(f"  높은 상관관계 센서 쌍 (|r| > 0.7):")
            for sensor1, sensor2, corr in high_corr_pairs[:5]:
                print(f"    {sensor1} - {sensor2}: {corr:.3f}")
    
    def compare_all_classes(self, class_data):
        """모든 클래스 비교 분석"""
        print("\n🔄 ㅕ, ㅌ, ㄹ 클래스 비교 분석")
        
        sensor_names = ['pitch', 'roll', 'yaw', 'flex1', 'flex2', 'flex3', 'flex4', 'flex5']
        
        # 각 클래스의 평균 센서 값 계산
        class_means = {}
        
        for class_name in self.target_classes:
            all_sensor_data = []
            for sensor_data in class_data[class_name]['sensor_data']:
                all_sensor_data.extend(sensor_data)
            
            all_sensor_data = np.array(all_sensor_data)
            class_means[class_name] = np.mean(all_sensor_data, axis=0)
        
        # 클래스 간 차이 분석
        print("\n📊 클래스 간 센서 값 차이:")
        for i, sensor_name in enumerate(sensor_names):
            print(f"  {sensor_name}:")
            yeo_mean = class_means['ㅕ'][i]
            teul_mean = class_means['ㅌ'][i]
            rieul_mean = class_means['ㄹ'][i]
            
            print(f"    ㅕ 평균: {yeo_mean:.3f}")
            print(f"    ㅌ 평균: {teul_mean:.3f}")
            print(f"    ㄹ 평균: {rieul_mean:.3f}")
            
            # ㅕ vs ㅌ
            diff_yeo_teul = abs(yeo_mean - teul_mean)
            diff_percent_yeo_teul = (diff_yeo_teul / ((yeo_mean + teul_mean) / 2)) * 100
            print(f"    ㅕ-ㅌ 차이: {diff_yeo_teul:.3f} ({diff_percent_yeo_teul:.1f}%)")
            
            # ㅕ vs ㄹ
            diff_yeo_rieul = abs(yeo_mean - rieul_mean)
            diff_percent_yeo_rieul = (diff_yeo_rieul / ((yeo_mean + rieul_mean) / 2)) * 100
            print(f"    ㅕ-ㄹ 차이: {diff_yeo_rieul:.3f} ({diff_percent_yeo_rieul:.1f}%)")
            
            # ㅌ vs ㄹ
            diff_teul_rieul = abs(teul_mean - rieul_mean)
            diff_percent_teul_rieul = (diff_teul_rieul / ((teul_mean + rieul_mean) / 2)) * 100
            print(f"    ㅌ-ㄹ 차이: {diff_teul_rieul:.3f} ({diff_percent_teul_rieul:.1f}%)")
        
        # 코사인 유사도 계산
        print(f"\n🎯 클래스 간 코사인 유사도:")
        
        # ㅕ vs ㅌ
        yeo_vector = class_means['ㅕ']
        teul_vector = class_means['ㅌ']
        cosine_similarity_yeo_teul = np.dot(yeo_vector, teul_vector) / (np.linalg.norm(yeo_vector) * np.linalg.norm(teul_vector))
        print(f"  ㅕ-ㅌ: {cosine_similarity_yeo_teul:.3f}")
        
        # ㅕ vs ㄹ
        rieul_vector = class_means['ㄹ']
        cosine_similarity_yeo_rieul = np.dot(yeo_vector, rieul_vector) / (np.linalg.norm(yeo_vector) * np.linalg.norm(rieul_vector))
        print(f"  ㅕ-ㄹ: {cosine_similarity_yeo_rieul:.3f}")
        
        # ㅌ vs ㄹ
        cosine_similarity_teul_rieul = np.dot(teul_vector, rieul_vector) / (np.linalg.norm(teul_vector) * np.linalg.norm(rieul_vector))
        print(f"  ㅌ-ㄹ: {cosine_similarity_teul_rieul:.3f}")
        
        # 혼동 가능성 평가
        print(f"\n⚠️  혼동 가능성 평가:")
        if cosine_similarity_yeo_teul > 0.95:
            print(f"  ㅕ-ㅌ: 매우 높은 혼동 가능성")
        elif cosine_similarity_yeo_teul > 0.9:
            print(f"  ㅕ-ㅌ: 높은 혼동 가능성")
        else:
            print(f"  ㅕ-ㅌ: 적절한 구분 가능")
            
        if cosine_similarity_yeo_rieul > 0.95:
            print(f"  ㅕ-ㄹ: 매우 높은 혼동 가능성")
        elif cosine_similarity_yeo_rieul > 0.9:
            print(f"  ㅕ-ㄹ: 높은 혼동 가능성")
        else:
            print(f"  ㅕ-ㄹ: 적절한 구분 가능")
            
        if cosine_similarity_teul_rieul > 0.95:
            print(f"  ㅌ-ㄹ: 매우 높은 혼동 가능성")
        elif cosine_similarity_teul_rieul > 0.9:
            print(f"  ㅌ-ㄹ: 높은 혼동 가능성")
        else:
            print(f"  ㅌ-ㄹ: 적절한 구분 가능")
    
    def analyze_file_by_file(self, class_data):
        """파일별 상세 분석"""
        print("\n📁 파일별 상세 분석")
        
        for class_name in self.target_classes:
            print(f"\n🎯 {class_name} 클래스 파일별 분석:")
            
            for i, file_path in enumerate(class_data[class_name]['file_paths'][:3]):  # 처음 3개 파일만
                print(f"\n  파일 {i+1}: {os.path.basename(file_path)}")
                
                try:
                    df = pd.read_csv(file_path)
                    sensor_data = df[['pitch', 'roll', 'yaw', 'flex1', 'flex2', 'flex3', 'flex4', 'flex5']].values
                    
                    print(f"    데이터 길이: {len(sensor_data)}")
                    print(f"    flex1 범위: [{np.min(sensor_data[:, 3]):.1f}, {np.max(sensor_data[:, 3]):.1f}]")
                    print(f"    flex2 범위: [{np.min(sensor_data[:, 4]):.1f}, {np.max(sensor_data[:, 4]):.1f}]")
                    print(f"    flex3 범위: [{np.min(sensor_data[:, 5]):.1f}, {np.max(sensor_data[:, 5]):.1f}]")
                    print(f"    flex4 범위: [{np.min(sensor_data[:, 6]):.1f}, {np.max(sensor_data[:, 6]):.1f}]")
                    print(f"    flex5 범위: [{np.min(sensor_data[:, 7]):.1f}, {np.max(sensor_data[:, 7]):.1f}]")
                    
                    # 특별한 패턴 체크
                    if class_name == 'ㅕ':
                        high_flex2 = np.sum(sensor_data[:, 4] > 100)
                        if high_flex2 > 0:
                            print(f"    ⚠️  flex2 > 100: {high_flex2}개 포인트")
                    
                except Exception as e:
                    print(f"    ⚠️  파일 읽기 실패: {e}")
    
    def create_visualization(self, class_data):
        """시각화 생성"""
        print("\n📊 시각화 생성 중...")
        
        sensor_names = ['pitch', 'roll', 'yaw', 'flex1', 'flex2', 'flex3', 'flex4', 'flex5']
        
        # 각 클래스의 센서 데이터 준비
        class_sensor_data = {}
        for class_name in self.target_classes:
            all_data = []
            for sensor_data in class_data[class_name]['sensor_data']:
                all_data.extend(sensor_data)
            class_sensor_data[class_name] = np.array(all_data)
        
        # 박스플롯 생성
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        fig.suptitle('ㅕ vs ㅌ vs ㄹ 클래스 센서 패턴 비교', fontsize=16)
        
        for i, sensor_name in enumerate(sensor_names):
            row = i // 4
            col = i % 4
            ax = axes[row, col]
            
            yeo_data = class_sensor_data['ㅕ'][:, i]
            teul_data = class_sensor_data['ㅌ'][:, i]
            rieul_data = class_sensor_data['ㄹ'][:, i]
            
            ax.boxplot([yeo_data, teul_data, rieul_data], labels=['ㅕ', 'ㅌ', 'ㄹ'])
            ax.set_title(f'{sensor_name}')
            ax.set_ylabel('값')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('yeo_teul_rieul_sensor_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ 시각화 저장: yeo_teul_rieul_sensor_comparison.png")
    
    def save_analysis_report(self, class_data):
        """분석 보고서 저장"""
        print("\n📄 분석 보고서 저장 중...")
        
        report = {
            'analysis_date': pd.Timestamp.now().isoformat(),
            'target_classes': self.target_classes,
            'summary': {}
        }
        
        for class_name in self.target_classes:
            report['summary'][class_name] = {
                'total_files': len(class_data[class_name]['files']),
                'scenarios': dict(class_data[class_name]['scenarios']),
                'total_data_points': sum(len(data) for data in class_data[class_name]['sensor_data'])
            }
        
        # 센서 통계 추가
        sensor_names = ['pitch', 'roll', 'yaw', 'flex1', 'flex2', 'flex3', 'flex4', 'flex5']
        
        for class_name in self.target_classes:
            all_sensor_data = []
            for sensor_data in class_data[class_name]['sensor_data']:
                all_sensor_data.extend(sensor_data)
            
            all_sensor_data = np.array(all_sensor_data)
            
            report['summary'][class_name]['sensor_stats'] = {}
            for i, sensor_name in enumerate(sensor_names):
                sensor_values = all_sensor_data[:, i]
                report['summary'][class_name]['sensor_stats'][sensor_name] = {
                    'mean': float(np.mean(sensor_values)),
                    'std': float(np.std(sensor_values)),
                    'min': float(np.min(sensor_values)),
                    'max': float(np.max(sensor_values)),
                    'range': [float(np.min(sensor_values)), float(np.max(sensor_values))]
                }
        
        with open('yeo_teul_rieul_analysis_report.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print("✅ 분석 보고서 저장: yeo_teul_rieul_analysis_report.json")
    
    def run_analysis(self):
        """전체 분석 실행"""
        print("🚀 ㅕ, ㅌ, ㄹ 클래스 상세 분석 시작")
        print("=" * 60)
        
        # 데이터 로딩
        class_data = self.load_class_data()
        
        # 각종 분석 실행
        self.analyze_sensor_patterns(class_data)
        self.analyze_scenario_distribution(class_data)
        self.analyze_sensor_correlations(class_data)
        self.compare_all_classes(class_data)
        self.analyze_file_by_file(class_data)
        
        # 시각화 및 보고서 생성
        self.create_visualization(class_data)
        self.save_analysis_report(class_data)
        
        print("\n🎉 ㅕ, ㅌ, ㄹ 클래스 분석 완료!")
        print("📁 생성된 파일들:")
        print("  - yeo_teul_rieul_sensor_comparison.png")
        print("  - yeo_teul_rieul_analysis_report.json")

def main():
    """메인 함수"""
    analyzer = YeoTeulRieulAnalyzer()
    analyzer.run_analysis()

if __name__ == "__main__":
    main()
