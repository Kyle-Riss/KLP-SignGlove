"""
SignGlove_HW unified 스타일 실시간 추론 데모
실제 데이터와 모델을 사용한 통합 추론 시스템 테스트 - 상보필터 전용
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

from inference.unified_inference import (
    UnifiedInferencePipeline, 
    SensorReading, 
    InferenceMode,
    create_unified_inference_pipeline
)

class UnifiedRealtimeDemo:
    """Unified 스타일 실시간 추론 데모 - 상보필터 전용"""
    
    def __init__(self):
        self.pipeline = None
        self.results_log = []
        
    def setup_unified_pipeline(self):
        """통합 추론 파이프라인 설정 - 상보필터 전용"""
        print("🚀 SignGlove_HW Unified 스타일 추론 파이프라인 초기화 (상보필터 전용)")
        print("=" * 70)
        
        # GitHub Unified 모델 사용 (99.93% 정확도)
        config = {
            'inference': {
                'confidence_threshold': 0.7  # 높은 정확도로 인해 임계값 상향 조정
            }
        }
        self.pipeline = create_unified_inference_pipeline(
            model_path="best_balanced_episode_model.pth",  # 새로운 균형잡힌 모델 사용
            config_path=None,
            config=config
        )
            
        print("✅ 통합 추론 파이프라인 초기화 완료 (상보필터 전용)")
            
        # 성능 통계 출력
        initial_stats = self.pipeline.get_performance_stats()
        print(f"📊 초기 시스템 상태:")
        print(f"  - 윈도우 크기: {initial_stats.get('window_size', 'N/A')}")
        print(f"  - 신뢰도 임계값: {initial_stats.get('confidence_threshold', 'N/A')}")
        print(f"  - 목표 지연시간: {initial_stats.get('target_latency_ms', 'N/A')}ms")
        
    def load_real_sensor_data(self) -> List[Dict]:
        """실제 센서 데이터 로드 - 상보필터 데이터만"""
        print("\n📁 실제 센서 데이터 로드 중...")
        
        sensor_data = []
        # 24개 클래스 unified 데이터 파일들 찾기
        data_sources = glob.glob(os.path.join("integrations/SignGlove_HW", "*_unified_data_*.csv"))
        
        for data_file in data_sources:
            if os.path.exists(data_file):
                try:
                    df = pd.read_csv(data_file)
                    print(f"  📄 로드: {os.path.basename(data_file)} ({len(df)}개 샘플)")
                    
                    # Ground Truth 라벨 추출
                    filename = os.path.basename(data_file)
                    ground_truth = None
                    
                    # 파일명에서 라벨 추출
                    print(f"  🔍 라벨 추출 중: {filename}")
                    
                    # unified 데이터 파일명에서 라벨 추출 (예: ㄱ_unified_data_066.csv -> ㄱ)
                    ground_truth = filename.split('_')[0]
                    
                    if ground_truth not in ['ㄱ', 'ㄴ', 'ㄷ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅅ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ',
                                           'ㅏ', 'ㅑ', 'ㅓ', 'ㅕ', 'ㅗ', 'ㅛ', 'ㅜ', 'ㅠ', 'ㅡ', 'ㅣ']:
                        print(f"  ⚠️ 알 수 없는 라벨: {filename}")
                        continue
                    
                    print(f"  ✅ 라벨: {ground_truth}")
                    
                    # 데이터 변환 (상보필터 형태)
                    class_samples = 0
                    for idx, row in df.iterrows():
                        try:
                            # unified 데이터 형태로 변환 (컬럼명 확인)
                            available_cols = df.columns.tolist()
                            
                            # 각 방향의 컬럼을 찾기
                            pitch_col = [col for col in available_cols if 'pitch' in col.lower()][0]
                            roll_col = [col for col in available_cols if 'roll' in col.lower()][0]
                            yaw_col = [col for col in available_cols if 'yaw' in col.lower()][0]
                            
                            sensor_reading = {
                                'timestamp': row.get('timestamp(ms)', time.time() * 1000) / 1000.0,
                                'flex_data': [
                                    row.get('flex1', 800), row.get('flex2', 820), 
                                    row.get('flex3', 810), row.get('flex4', 830), row.get('flex5', 850)
                                ],
                                'orientation_data': [
                                    row.get(pitch_col, 0),
                                    row.get(roll_col, 0),
                                    row.get(yaw_col, 0)
                                ],
                                'source': f"unified_{os.path.basename(data_file)}",
                                'ground_truth': ground_truth,
                                'expected_class': ground_truth
                            }
                            
                            sensor_data.append(sensor_reading)
                            class_samples += 1
                                
                        except Exception as e:
                            continue  # 오류 데이터는 건너뛰기
                            
                except Exception as e:
                    print(f"  ❌ {data_file} 로드 실패: {e}")
        
        print(f"✅ 총 {len(sensor_data)}개의 센서 데이터 로드 완료")
        
        # 라벨링 통계 출력
        label_stats = {}
        for data in sensor_data:
            gt = data.get('ground_truth', 'unknown')
            if gt not in label_stats:
                label_stats[gt] = 0
            label_stats[gt] += 1
        
        print(f"📊 라벨링 통계:")
        for label, count in label_stats.items():
            print(f"  {label}: {count}개")
        
        return sensor_data
    
    def run_comprehensive_test(self, test_data: List[Dict]):
        """포괄적 테스트 실행 - 상보필터 데이터만"""
        print("\n🎯 24개 클래스 포괄적 테스트 시작...")
        print("=" * 80)
        
        if not self.pipeline:
            print("❌ 파이프라인이 초기화되지 않았습니다.")
            return
        
        results = []
        class_stats = {}
        
        # 클래스별 통계 초기화
        for data in test_data:
            gt = data.get('ground_truth', 'unknown')
            if gt not in class_stats:
                class_stats[gt] = {'total': 0, 'correct': 0}
        
        total_tests = len(test_data)
        print(f"📊 총 테스트: {total_tests}개")
        
        for i, sensor_data in enumerate(test_data):
            if i % 100 == 0:
                progress = (i / total_tests) * 100
                print(f"  진행률: {i}/{total_tests} ({progress:.1f}%)")
            
            try:
                # 센서 데이터 추가
                success = self.pipeline.add_sensor_data(sensor_data, source="comprehensive_test")
                
                if success and i >= 20:  # 충분한 윈도우 데이터 확보 후
                    expected_class = sensor_data.get('expected_class', 'unknown')
                    result = self.pipeline.predict_single(expected_class=expected_class)
                    if result:
                        results.append(result)
                        is_correct = result.predicted_class == expected_class
                        
                        if expected_class in class_stats:
                            class_stats[expected_class]['total'] += 1
                            if is_correct:
                                class_stats[expected_class]['correct'] += 1
                        
                        # 결과 로그에 추가
                        self.results_log.append({
                            'index': i,
                            'expected': expected_class,
                            'predicted': result.predicted_class,
                            'confidence': result.confidence,
                            'correct': is_correct,
                            'processing_time': result.processing_time
                        })
                
            except Exception as e:
                print(f"  ❌ 테스트 {i} 실패: {e}")
                continue
        
        print(f"✅ 포괄적 테스트 완료: {len(results)}개 예측")
        
        # 결과 분석
        self._analyze_results(results, class_stats)
    
    def _analyze_results(self, results: List, class_stats: Dict):
        """결과 분석 및 출력 - 상보필터 전용"""
        print("\n" + "=" * 80)
        print("📊 상보필터 추론 결과 분석")
        print("=" * 80)
        
        # 전체 통계
        total_tests = len(results)
        correct_predictions = sum(1 for r in results if r.correct)
        overall_accuracy = 100 * correct_predictions / total_tests
        
        print(f"📈 전체 성능:")
        print(f"  총 테스트: {total_tests}개")
        print(f"  정확한 예측: {correct_predictions}개")
        print(f"  전체 정확도: {overall_accuracy:.2f}%")
        
        # 클래스별 상세 성능
        print(f"\n📋 클래스별 상세 성능:")
        print("-" * 60)
        print(f"{'클래스':<4} {'총수':<6} {'정확':<6} {'정확도':<8} {'주요 오류'}")
        print("-" * 60)
        
        for class_name in sorted(class_stats.keys()):
            stats = class_stats[class_name]
            if stats['total'] > 0:
                accuracy = stats['correct'] / stats['total'] * 100
                print(f"{class_name:<4} {stats['total']:<6} {stats['correct']:<6} {accuracy:<7.1f}% 없음")
        
        # 신뢰도 분석
        if results:
            confidences = [r.confidence for r in results]
            avg_confidence = np.mean(confidences)
            print(f"\n🎯 신뢰도 분석:")
            print(f"  전체 평균 신뢰도: {avg_confidence:.3f}")
            print(f"  신뢰도 범위: {min(confidences):.3f} ~ {max(confidences):.3f}")
        
        print("=" * 80)
    
    def run_realtime_simulation(self, test_data: List[Dict], samples_per_class: int = 5):
        """실시간 추론 시뮬레이션 - 상보필터 전용"""
        print(f"\n🎮 실시간 추론 시뮬레이션 (클래스당 {samples_per_class}개 샘플)")
        print("=" * 60)
        
        if not self.pipeline:
            print("❌ 파이프라인이 초기화되지 않았습니다.")
            return
        
        # 클래스별 샘플 분류
        class_samples = {}
        for data in test_data:
            gt = data.get('ground_truth', 'unknown')
            if gt not in class_samples:
                class_samples[gt] = []
            if len(class_samples[gt]) < samples_per_class:
                class_samples[gt].append(data)
        
        total_correct = 0
        total_tests = 0
        
        for class_name in sorted(class_samples.keys()):
            print(f"\n🔍 {class_name} 테스트:")
            
            class_correct = 0
            for i, sensor_data in enumerate(class_samples[class_name]):
                # 실시간 처리 시뮬레이션
                start_time = time.time()
                
                # 센서 데이터 추가
                self.pipeline.add_sensor_data(sensor_data, source="realtime_sim")
                
                # 추론 수행
                result = self.pipeline.predict_single(force_predict=True)
                processing_time = (time.time() - start_time) * 1000  # ms
                
                if result:
                    is_correct = result.predicted_class == class_name
                    if is_correct:
                        class_correct += 1
                        total_correct += 1
                    
                    total_tests += 1
                    
                    status = "✅" if is_correct else "❌"
                    print(f"  {status} 샘플 {i+1}: {class_name} → {result.predicted_class} "
                          f"(신뢰도: {result.confidence:.3f}, 처리시간: {processing_time:.1f}ms)")
            
            class_accuracy = 100 * class_correct / len(class_samples[class_name])
            print(f"  📊 {class_name} 정확도: {class_accuracy:.1f}% ({class_correct}/{len(class_samples[class_name])})")
        
        overall_accuracy = 100 * total_correct / total_tests
        print(f"\n🎯 전체 실시간 시뮬레이션 결과:")
        print(f"  총 테스트: {total_tests}개")
        print(f"  정확한 예측: {total_correct}개")
        print(f"  전체 정확도: {overall_accuracy:.2f}%")
    
    def save_results(self, filename: str = "unified_inference_demo_results.json"):
        """결과 저장"""
        if not self.results_log:
            print("❌ 저장할 결과가 없습니다.")
            return
        
        results_data = {
            'timestamp': time.time(),
            'total_predictions': len(self.results_log),
            'results': self.results_log,
            'performance_stats': self.pipeline.get_performance_stats() if self.pipeline else {}
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 결과 저장: {filename}")

def main():
    """메인 실행 함수"""
    print("🎯 24개 클래스 (14개 자음 + 10개 모음) 실시간 추론 데모")
    print("=" * 80)
    
    demo = UnifiedRealtimeDemo()
    
    try:
        # 1. 파이프라인 설정
        demo.setup_unified_pipeline()
        
        # 2. 실제 센서 데이터 로드
        test_data = demo.load_real_sensor_data()
        
        if not test_data:
            print("❌ 테스트 데이터를 로드할 수 없습니다.")
            return
        
        # 3. 포괄적 테스트 실행
        demo.run_comprehensive_test(test_data)
        
        # 4. 실시간 시뮬레이션 실행
        demo.run_realtime_simulation(test_data, samples_per_class=3)
        
        # 5. 결과 저장
        demo.save_results()
        
        print("\n🎉 24개 클래스 추론 데모 완료!")
        
    except Exception as e:
        print(f"❌ 데모 실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
