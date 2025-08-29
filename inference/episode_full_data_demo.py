#!/usr/bin/env python3
"""
전체 Episode 데이터 기반 실시간 추론 데모
600개 Episode 파일 모두 사용한 포괄적 테스트
"""

import sys
import os
import time
import json
import glob
import numpy as np
from pathlib import Path
import pandas as pd
from typing import Dict, List, Optional

# 프로젝트 루트 추가
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from inference.episode_inference import (
    EpisodeInferencePipeline, 
    EpisodeSensorReading, 
    create_episode_inference_pipeline
)

class EpisodeFullDataDemo:
    """전체 Episode 데이터 기반 실시간 추론 데모"""
    
    def __init__(self):
        self.pipeline = None
        self.results_log = []
        
    def setup_episode_pipeline(self):
        """Episode 데이터용 추론 파이프라인 설정"""
        print("🚀 전체 Episode 데이터 기반 추론 파이프라인 초기화")
        print("=" * 70)
        
        # 균형잡힌 Episode 모델 사용 (99.92% 정확도)
        config = {
            'window_size': 20
        }
        self.pipeline = create_episode_inference_pipeline(
            model_path="best_balanced_episode_model.pth",
            config=config
        )
            
        print("✅ 전체 Episode 추론 파이프라인 초기화 완료")
            
        # 성능 통계 출력
        stats = self.pipeline.get_performance_stats()
        print(f"📊 초기 시스템 상태:")
        print(f"  - 윈도우 크기: {stats['window_size']}")
        print(f"  - 버퍼 크기: {stats['buffer_size']}")
        print(f"  - 장치: {stats['device']}")
        
    def load_all_episode_data(self) -> List[Dict]:
        """전체 Episode 센서 데이터 로드 (600개 파일 모두)"""
        print("\n📁 전체 Episode 센서 데이터 로드 중...")
        
        sensor_data = []
        # 모든 Episode CSV 파일들 찾기
        data_sources = glob.glob(os.path.join("integrations/SignGlove_HW/github_unified_data", "**/episode_*.csv"), recursive=True)
        
        print(f"📁 발견된 Episode 파일: {len(data_sources)}개")
        
        # 클래스별 파일 카운트
        class_file_counts = {}
        
        for data_file in data_sources:
            if os.path.exists(data_file):
                try:
                    df = pd.read_csv(data_file)
                    filename = os.path.basename(data_file)
                    
                    # Ground Truth 라벨 추출
                    ground_truth = None
                    for class_name in ['ㄱ', 'ㄴ', 'ㄷ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅅ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ', 'ㅏ', 'ㅑ', 'ㅓ', 'ㅕ', 'ㅗ', 'ㅛ', 'ㅜ', 'ㅠ', 'ㅡ', 'ㅣ']:
                        if f'_{class_name}_' in filename:
                            ground_truth = class_name
                            break
                    
                    if not ground_truth:
                        continue
                    
                    # 클래스별 파일 카운트 업데이트
                    if ground_truth not in class_file_counts:
                        class_file_counts[ground_truth] = 0
                    class_file_counts[ground_truth] += 1
                    
                    print(f"  📄 로드: {filename} ({len(df)}개 샘플) - {ground_truth}")
                    
                    # 데이터 변환 (Episode 형태)
                    class_samples = 0
                    for idx, row in df.iterrows():
                        try:
                            # Episode 데이터 형태로 변환
                            available_cols = df.columns.tolist()
                            
                            # Flex 센서 컬럼 찾기
                            flex_cols = []
                            for i in range(1, 6):
                                flex_patterns = [f'flex{i}', f'Flex{i}', f'FLEX{i}']
                                for pattern in flex_patterns:
                                    if pattern in available_cols:
                                        flex_cols.append(pattern)
                                        break
                            
                            # Orientation 컬럼 찾기
                            pitch_col = None
                            roll_col = None
                            yaw_col = None
                            
                            for col in available_cols:
                                if 'pitch' in col.lower():
                                    pitch_col = col
                                elif 'roll' in col.lower():
                                    roll_col = col
                                elif 'yaw' in col.lower():
                                    yaw_col = col
                            
                            if not flex_cols or not all([pitch_col, roll_col, yaw_col]):
                                continue
                            
                            # 센서 데이터 추출
                            flex_data = [float(row[col]) for col in flex_cols]
                            orientation_data = [float(row[pitch_col]), float(row[roll_col]), float(row[yaw_col])]
                            
                            sensor_data.append({
                                'flex_data': flex_data,
                                'orientation_data': orientation_data,
                                'ground_truth': ground_truth,
                                'file': filename,
                                'row': idx
                            })
                            
                            class_samples += 1
                            
                            # 파일당 최대 100개 샘플로 제한 (메모리 절약)
                            if class_samples >= 100:
                                break
                                
                        except Exception as e:
                            continue
                    
                    print(f"    📊 변환된 샘플: {class_samples}개")
                    
                except Exception as e:
                    print(f"  ⚠️ 파일 로드 실패: {data_file} - {e}")
                    continue
        
        print(f"\n📊 클래스별 파일 분포:")
        for class_name in sorted(class_file_counts.keys()):
            print(f"  {class_name}: {class_file_counts[class_name]}개 파일")
        
        print(f"\n✅ 총 {len(sensor_data)}개 센서 데이터 로드 완료")
        print(f"📈 데이터 사용률: {len(data_sources)}/600 = {len(data_sources)/600*100:.1f}%")
        
        return sensor_data
    
    def run_comprehensive_test(self, sensor_data: List[Dict]):
        """포괄적 테스트 실행"""
        print(f"\n🎯 전체 데이터 포괄적 테스트 시작")
        print("=" * 60)
        
        total_tests = len(sensor_data)
        correct_predictions = 0
        class_performance = {}
        confidence_scores = []
        
        print(f"📊 총 {total_tests}개 테스트 실행 중...")
        
        for i, data_item in enumerate(sensor_data):
            if i % 1000 == 0:
                progress = (i / total_tests) * 100
                print(f"  진행률: {i}/{total_tests} ({progress:.1f}%)")
            
            try:
                # 파이프라인에 데이터 추가
                success = self.pipeline.add_sensor_data(
                    data_item['flex_data'],
                    data_item['orientation_data']
                )
                
                if not success:
                    continue
                
                # 추론 실행
                result = self.pipeline.predict_single(
                    expected_class=data_item['ground_truth']
                )
                
                if result is None:
                    continue
                
                # 결과 분석
                if result.correct:
                    correct_predictions += 1
                
                # 클래스별 성능 추적
                class_name = data_item['ground_truth']
                if class_name not in class_performance:
                    class_performance[class_name] = {'total': 0, 'correct': 0}
                
                class_performance[class_name]['total'] += 1
                if result.correct:
                    class_performance[class_name]['correct'] += 1
                
                # 신뢰도 기록
                confidence_scores.append(result.confidence)
                
                # 결과 로그 (메모리 절약을 위해 일부만 저장)
                if i % 100 == 0:
                    self.results_log.append({
                        'file': data_item['file'],
                        'row': data_item['row'],
                        'ground_truth': data_item['ground_truth'],
                        'predicted': result.predicted_class,
                        'confidence': result.confidence,
                        'correct': result.correct,
                        'processing_time': result.processing_time
                    })
                
            except Exception as e:
                continue
        
        # 결과 분석
        overall_accuracy = (correct_predictions / total_tests) * 100 if total_tests > 0 else 0
        avg_confidence = np.mean(confidence_scores) if confidence_scores else 0
        min_confidence = np.min(confidence_scores) if confidence_scores else 0
        max_confidence = np.max(confidence_scores) if confidence_scores else 0
        
        print(f"✅ 전체 데이터 포괄적 테스트 완료: {correct_predictions}개 예측")
        
        # 결과 출력
        print(f"\n📊 전체 Episode 데이터 추론 결과 분석")
        print("=" * 60)
        print(f"📈 전체 성능:")
        print(f"  총 테스트: {total_tests:,}개")
        print(f"  정확한 예측: {correct_predictions:,}개")
        print(f"  전체 정확도: {overall_accuracy:.2f}%")
        
        print(f"\n📋 클래스별 상세 성능:")
        print("-" * 60)
        print("클래스  총수     정확     정확도")
        print("-" * 60)
        
        for class_name in sorted(class_performance.keys()):
            stats = class_performance[class_name]
            accuracy = (stats['correct'] / stats['total']) * 100 if stats['total'] > 0 else 0
            print(f"{class_name:<4} {stats['total']:>6} {stats['correct']:>6} {accuracy:>8.1f} %")
        
        print(f"\n🎯 신뢰도 분석:")
        print(f"  전체 평균 신뢰도: {avg_confidence:.3f}")
        print(f"  신뢰도 범위: {min_confidence:.3f} ~ {max_confidence:.3f}")
        
        return overall_accuracy, class_performance
    
    def run_sample_realtime_simulation(self, sensor_data: List[Dict]):
        """샘플 실시간 추론 시뮬레이션 (전체 데이터에서 일부만)"""
        print(f"\n🎮 샘플 실시간 추론 시뮬레이션 (클래스당 5개 파일)")
        print("=" * 60)
        
        # 클래스별로 샘플 선택 (클래스당 5개 파일에서 각각 10개 샘플)
        class_samples = {}
        for data_item in sensor_data:
            class_name = data_item['ground_truth']
            if class_name not in class_samples:
                class_samples[class_name] = []
            if len(class_samples[class_name]) < 50:  # 클래스당 50개 샘플
                class_samples[class_name].append(data_item)
        
        total_tests = 0
        correct_tests = 0
        
        for class_name in sorted(class_samples.keys()):
            samples = class_samples[class_name]
            if not samples:
                continue
                
            print(f"\n🔍 {class_name} 샘플 테스트:")
            class_correct = 0
            
            # 버퍼 초기화
            self.pipeline.data_buffer.clear()
            
            # 연속적으로 데이터 추가하면서 추론
            for i, sample in enumerate(samples):
                try:
                    # 파이프라인에 데이터 추가
                    success = self.pipeline.add_sensor_data(
                        sample['flex_data'],
                        sample['orientation_data']
                    )
                    
                    if not success:
                        continue
                    
                    # 윈도우가 충분히 쌓인 후부터 추론 시작
                    if len(self.pipeline.data_buffer) >= self.pipeline.window_size:
                        # 추론 실행
                        result = self.pipeline.predict_single(
                            expected_class=sample['ground_truth']
                        )
                        
                        if result is None:
                            continue
                        
                        if result.correct:
                            class_correct += 1
                            correct_tests += 1
                        
                        total_tests += 1
                        
                        # 10개 샘플마다 결과 출력
                        if total_tests % 10 == 0:
                            print(f"    ... 진행률: {total_tests}개 테스트 완료")
                    
                except Exception as e:
                    continue
            
            class_accuracy = (class_correct / len(samples)) * 100 if samples else 0
            print(f"  📊 {class_name} 정확도: {class_accuracy:.1f}% ({class_correct}/{len(samples)})")
        
        overall_accuracy = (correct_tests / total_tests) * 100 if total_tests > 0 else 0
        print(f"\n🎯 전체 샘플 실시간 시뮬레이션 결과:")
        print(f"  총 테스트: {total_tests}개")
        print(f"  정확한 예측: {correct_tests}개")
        print(f"  전체 정확도: {overall_accuracy:.2f}%")
        
        return overall_accuracy
    
    def save_results(self):
        """결과 저장"""
        results = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'model': 'best_balanced_episode_model.pth',
            'pipeline': 'EpisodeInferencePipeline Full Data',
            'results': self.results_log
        }
        
        with open('episode_full_data_demo_results.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print("✅ 결과 저장: episode_full_data_demo_results.json")
    
    def run_demo(self):
        """전체 데모 실행"""
        print("🚀 전체 Episode 데이터 기반 실시간 추론 데모 시작")
        print("=" * 70)
        
        # 파이프라인 설정
        self.setup_episode_pipeline()
        
        # 전체 데이터 로드
        sensor_data = self.load_all_episode_data()
        
        if not sensor_data:
            print("❌ 로드된 센서 데이터가 없습니다.")
            return
        
        # 전체 데이터 포괄적 테스트
        overall_accuracy, class_performance = self.run_comprehensive_test(sensor_data)
        
        # 샘플 실시간 시뮬레이션
        realtime_accuracy = self.run_sample_realtime_simulation(sensor_data)
        
        # 결과 저장
        self.save_results()
        
        print(f"\n🎉 전체 Episode 데이터 추론 데모 완료!")
        print(f"📊 최종 성능:")
        print(f"  - 전체 데이터 테스트: {overall_accuracy:.2f}%")
        print(f"  - 샘플 실시간 시뮬레이션: {realtime_accuracy:.2f}%")

def main():
    """메인 함수"""
    demo = EpisodeFullDataDemo()
    demo.run_demo()

if __name__ == "__main__":
    main()
