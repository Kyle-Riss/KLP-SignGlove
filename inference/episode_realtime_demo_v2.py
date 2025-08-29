#!/usr/bin/env python3
"""
Episode 데이터 기반 실시간 추론 데모 V2
새로운 Episode 전용 추론 파이프라인 사용
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

class EpisodeRealtimeDemoV2:
    """Episode 데이터 기반 실시간 추론 데모 V2"""
    
    def __init__(self):
        self.pipeline = None
        self.results_log = []
        
    def setup_episode_pipeline(self):
        """Episode 데이터용 추론 파이프라인 설정"""
        print("🚀 Episode 데이터 기반 추론 파이프라인 V2 초기화")
        print("=" * 70)
        
        # 균형잡힌 Episode 모델 사용 (99.92% 정확도)
        config = {
            'window_size': 20
        }
        self.pipeline = create_episode_inference_pipeline(
            model_path="best_balanced_episode_model.pth",
            config=config
        )
            
        print("✅ Episode 추론 파이프라인 V2 초기화 완료")
            
        # 성능 통계 출력
        stats = self.pipeline.get_performance_stats()
        print(f"📊 초기 시스템 상태:")
        print(f"  - 윈도우 크기: {stats['window_size']}")
        print(f"  - 버퍼 크기: {stats['buffer_size']}")
        print(f"  - 장치: {stats['device']}")
        
    def load_episode_sensor_data(self) -> List[Dict]:
        """Episode 센서 데이터 로드"""
        print("\n📁 Episode 센서 데이터 로드 중...")
        
        sensor_data = []
        # Episode CSV 파일들 찾기
        data_sources = glob.glob(os.path.join("integrations/SignGlove_HW/github_unified_data", "**/episode_*.csv"), recursive=True)
        
        print(f"📁 발견된 Episode 파일: {len(data_sources)}개")
        
        # 다양한 클래스를 테스트하기 위해 각 클래스별로 파일 선택
        class_files = {}
        for data_file in data_sources:
            filename = os.path.basename(data_file)
            
            # 라벨 추출
            ground_truth = None
            for class_name in ['ㄱ', 'ㄴ', 'ㄷ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅅ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ', 'ㅏ', 'ㅑ', 'ㅓ', 'ㅕ', 'ㅗ', 'ㅛ', 'ㅜ', 'ㅠ', 'ㅡ', 'ㅣ']:
                if f'_{class_name}_' in filename:
                    ground_truth = class_name
                    break
            
            if ground_truth and ground_truth not in class_files:
                class_files[ground_truth] = data_file
        
        # 각 클래스별로 파일 처리
        for class_name, data_file in class_files.items():
            if os.path.exists(data_file):
                try:
                    df = pd.read_csv(data_file)
                    print(f"  📄 로드: {os.path.basename(data_file)} ({len(df)}개 샘플)")
                    
                    # 이미 추출된 라벨 사용
                    ground_truth = class_name
                    filename = os.path.basename(data_file)
                    print(f"  ✅ 라벨: {ground_truth}")
                    
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
                            
                            # 클래스당 최대 50개 샘플로 제한
                            if class_samples >= 50:
                                break
                                
                        except Exception as e:
                            print(f"  ⚠️ 데이터 변환 실패: {e}")
                            continue
                    
                    print(f"  📊 변환된 샘플: {class_samples}개")
                    
                except Exception as e:
                    print(f"  ⚠️ 파일 로드 실패: {data_file} - {e}")
                    continue
        
        print(f"✅ 총 {len(sensor_data)}개 센서 데이터 로드 완료")
        return sensor_data
    
    def run_comprehensive_test(self, sensor_data: List[Dict]):
        """포괄적 테스트 실행"""
        print(f"\n🎯 포괄적 테스트 시작")
        print("=" * 60)
        
        total_tests = len(sensor_data)
        correct_predictions = 0
        class_performance = {}
        confidence_scores = []
        
        print(f"📊 총 {total_tests}개 테스트 실행 중...")
        
        for i, data_item in enumerate(sensor_data):
            if i % 100 == 0:
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
                
                # 결과 로그
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
                print(f"  ⚠️ 추론 실패: {e}")
                continue
        
        # 결과 분석
        overall_accuracy = (correct_predictions / total_tests) * 100 if total_tests > 0 else 0
        avg_confidence = np.mean(confidence_scores) if confidence_scores else 0
        min_confidence = np.min(confidence_scores) if confidence_scores else 0
        max_confidence = np.max(confidence_scores) if confidence_scores else 0
        
        print(f"✅ 포괄적 테스트 완료: {correct_predictions}개 예측")
        
        # 결과 출력
        print(f"\n📊 Episode 데이터 추론 결과 분석 V2")
        print("=" * 60)
        print(f"📈 전체 성능:")
        print(f"  총 테스트: {total_tests}개")
        print(f"  정확한 예측: {correct_predictions}개")
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
    
    def run_realtime_simulation(self, sensor_data: List[Dict]):
        """실시간 추론 시뮬레이션 - 연속 데이터 사용"""
        print(f"\n🎮 실시간 추론 시뮬레이션 (연속 Episode 데이터)")
        print("=" * 60)
        
        # 클래스별로 연속 데이터 그룹화
        class_sequences = {}
        for data_item in sensor_data:
            class_name = data_item['ground_truth']
            if class_name not in class_sequences:
                class_sequences[class_name] = []
            class_sequences[class_name].append(data_item)
        
        total_tests = 0
        correct_tests = 0
        
        for class_name in sorted(class_sequences.keys()):
            sequence = class_sequences[class_name]
            if len(sequence) < 20:  # 최소 윈도우 크기
                continue
                
            print(f"\n🔍 {class_name} 연속 테스트 (총 {len(sequence)}개 샘플):")
            class_correct = 0
            
            # 버퍼 초기화
            self.pipeline.data_buffer.clear()
            
            # 연속적으로 데이터 추가하면서 추론
            for i, sample in enumerate(sequence):
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
                        
                        status = "✅" if result.correct else "❌"
                        print(f"  {status} 샘플 {i+1}: {sample['ground_truth']} → {result.predicted_class} (신뢰도: {result.confidence:.3f})")
                        
                        if result.correct:
                            class_correct += 1
                            correct_tests += 1
                        
                        total_tests += 1
                        
                        # 10개 샘플마다 결과 출력 (너무 많은 출력 방지)
                        if total_tests % 10 == 0:
                            print(f"    ... 진행률: {total_tests}개 테스트 완료")
                    
                except Exception as e:
                    print(f"  ⚠️ 샘플 {i+1}: 추론 실패 - {e}")
                    continue
            
            class_accuracy = (class_correct / max(1, total_tests - (total_tests - class_correct))) * 100
            print(f"  📊 {class_name} 정확도: {class_accuracy:.1f}% ({class_correct}/{max(1, total_tests - (total_tests - class_correct))})")
        
        overall_accuracy = (correct_tests / total_tests) * 100 if total_tests > 0 else 0
        print(f"\n🎯 전체 실시간 시뮬레이션 결과:")
        print(f"  총 테스트: {total_tests}개")
        print(f"  정확한 예측: {correct_tests}개")
        print(f"  전체 정확도: {overall_accuracy:.2f}%")
        
        return overall_accuracy
    
    def save_results(self):
        """결과 저장"""
        results = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'model': 'best_balanced_episode_model.pth',
            'pipeline': 'EpisodeInferencePipeline V2',
            'results': self.results_log
        }
        
        with open('episode_inference_demo_v2_results.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print("✅ 결과 저장: episode_inference_demo_v2_results.json")
    
    def run_demo(self):
        """전체 데모 실행"""
        print("🚀 Episode 데이터 기반 실시간 추론 데모 V2 시작")
        print("=" * 70)
        
        # 파이프라인 설정
        self.setup_episode_pipeline()
        
        # 데이터 로드
        sensor_data = self.load_episode_sensor_data()
        
        if not sensor_data:
            print("❌ 로드된 센서 데이터가 없습니다.")
            return
        
        # 포괄적 테스트
        overall_accuracy, class_performance = self.run_comprehensive_test(sensor_data)
        
        # 실시간 시뮬레이션
        realtime_accuracy = self.run_realtime_simulation(sensor_data)
        
        # 결과 저장
        self.save_results()
        
        print(f"\n🎉 Episode 데이터 추론 데모 V2 완료!")
        print(f"📊 최종 성능:")
        print(f"  - 포괄적 테스트: {overall_accuracy:.2f}%")
        print(f"  - 실시간 시뮬레이션: {realtime_accuracy:.2f}%")

def main():
    """메인 함수"""
    demo = EpisodeRealtimeDemoV2()
    demo.run_demo()

if __name__ == "__main__":
    main()
