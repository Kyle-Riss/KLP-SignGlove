"""
Unified Inference Pipeline 데모
SignGlove_HW/unified 저장소 스타일의 통합 추론 시스템 시연
"""

import sys
import os
import time
import json
from pathlib import Path

# 프로젝트 루트 추가
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from inference.unified_inference import (
    UnifiedInferencePipeline, 
    SensorReading, 
    InferenceMode,
    create_unified_inference_pipeline,
    # create_madgwick_compatible_pipeline  # 제거됨
)
# from integrations.madgwick_adapter import MadgwickDataAdapter  # 제거됨

class UnifiedInferenceDemo:
    """통합 추론 시스템 데모"""
    
    def __init__(self):
        self.pipeline = None
        self.demo_results = []
        
    def demo_basic_inference(self):
        """기본 추론 데모"""
        print("🚀 Unified Inference Pipeline - 기본 추론 데모")
        print("=" * 60)
        
        # 파이프라인 생성
        self.pipeline = create_unified_inference_pipeline()
        
        # 테스트 데이터 생성
        test_data = self._generate_test_sensor_data(50)
        
        print(f"📊 테스트 데이터 생성: {len(test_data)}개 샘플")
        
        # 추론 실행
        results = []
        for i, sensor_data in enumerate(test_data):
            # 센서 데이터 추가
            success = self.pipeline.add_sensor_data(sensor_data, source="demo")
            
            if success and i >= 20:  # 충분한 윈도우 데이터 확보 후
                result = self.pipeline.predict_single()
                if result:
                    results.append(result)
                    print(f"⏰ 샘플 {i:2d} | 예측: {result.predicted_class} | "
                          f"신뢰도: {result.confidence:.3f} | "
                          f"안정성: {result.stability_score:.3f}")
        
        # 성능 통계
        stats = self.pipeline.get_performance_stats()
        print(f"\n📈 성능 통계:")
        print(f"  🔥 FPS: {stats['fps']:.1f}")
        print(f"  ⚡ 평균 지연시간: {stats['avg_latency_ms']:.2f}ms")
        print(f"  📊 총 프레임: {stats['total_frames']}")
        
        self.demo_results.extend(results)
        return results
    
    def demo_madgwick_integration(self):
        """Madgwick 데이터 통합 데모"""
        print("\n🔬 Madgwick 데이터 통합 데모")
        print("=" * 60)
        
        # Madgwick 호환 파이프라인 생성
        self.pipeline = create_madgwick_compatible_pipeline()
        
        # Madgwick 데이터 로드
        madgwick_file = "integrations/SignGlove_HW/madgwick_demo_converted.csv"
        if not Path(madgwick_file).exists():
            print(f"❌ Madgwick 데이터 파일이 없습니다: {madgwick_file}")
            return []
        
        import pandas as pd
        df = pd.read_csv(madgwick_file)
        print(f"📁 Madgwick 데이터 로드: {len(df)}행")
        
        results = []
        for idx, row in df.iterrows():
            # Madgwick 데이터를 SensorReading으로 변환
            sensor_reading = SensorReading(
                timestamp=row['timestamp(ms)'] / 1000.0,
                flex_data=[row[f'flex{i}'] for i in range(1, 6)],
                orientation_data=[row['pitch(°)'], row['roll(°)'], row['yaw(°)']],
                source="madgwick_demo"
            )
            
            # 추론 파이프라인에 추가
            success = self.pipeline.add_sensor_data(sensor_reading)
            
            if success and idx >= 20:
                result = self.pipeline.predict_single()
                if result:
                    results.append(result)
                    if len(results) % 5 == 0:  # 5개마다 출력
                        print(f"🎯 Madgwick {idx:2d} | 예측: {result.predicted_class} | "
                              f"신뢰도: {result.confidence:.3f} | "
                              f"처리시간: {result.processing_time*1000:.1f}ms")
        
        print(f"\n✅ Madgwick 통합 완료: {len(results)}개 예측")
        
        # 안정적인 예측 확인
        stable_result = self.pipeline.get_stable_prediction()
        if stable_result:
            print(f"🏆 안정적인 예측: {stable_result.predicted_class} "
                  f"(안정성: {stable_result.stability_score:.3f})")
        
        return results
    
    def demo_realtime_simulation(self):
        """실시간 추론 시뮬레이션 데모"""
        print("\n⚡ 실시간 추론 시뮬레이션 데모")
        print("=" * 60)
        
        # 실시간 파이프라인 생성
        self.pipeline = create_unified_inference_pipeline()
        
        # 콜백 함수들
        prediction_count = 0
        stable_prediction_count = 0
        
        def prediction_callback(result):
            nonlocal prediction_count
            prediction_count += 1
            if prediction_count % 10 == 0:
                print(f"📊 실시간 예측 #{prediction_count}: {result.predicted_class} "
                      f"({result.confidence:.3f})")
        
        def stable_prediction_callback(result):
            nonlocal stable_prediction_count
            stable_prediction_count += 1
            print(f"🎯 안정적 예측 #{stable_prediction_count}: {result.predicted_class} "
                  f"(안정성: {result.stability_score:.3f})")
        
        # 실시간 추론 시작
        self.pipeline.start_realtime_inference(
            prediction_callback=prediction_callback,
            stable_prediction_callback=stable_prediction_callback
        )
        
        print("🚀 실시간 추론 시작 - 30초간 시뮬레이션")
        
        # 시뮬레이션 데이터 스트리밍
        test_data = self._generate_test_sensor_data(1000)
        
        start_time = time.time()
        for i, sensor_data in enumerate(test_data):
            if time.time() - start_time > 30:  # 30초 제한
                break
                
            # 센서 데이터 추가 (실시간 시뮬레이션)
            self.pipeline.add_sensor_data(sensor_data, source="realtime_sim")
            
            # 실제 센서 주기 시뮬레이션 (50Hz = 20ms)
            time.sleep(0.02)
        
        # 실시간 추론 중지
        self.pipeline.stop_realtime_inference()
        
        # 최종 성능 통계
        final_stats = self.pipeline.get_performance_stats()
        print(f"\n📈 실시간 시뮬레이션 결과:")
        print(f"  🎯 총 예측: {prediction_count}개")
        print(f"  🏆 안정적 예측: {stable_prediction_count}개")
        print(f"  🔥 평균 FPS: {final_stats['fps']:.1f}")
        print(f"  ⚡ 평균 지연시간: {final_stats['avg_latency_ms']:.2f}ms")
        print(f"  ⏱️ 실행 시간: {final_stats['uptime_seconds']:.1f}초")
        
        return final_stats
    
    def demo_performance_comparison(self):
        """성능 비교 데모"""
        print("\n⚖️ 성능 비교 데모 (기존 vs Unified)")
        print("=" * 60)
        
        # 기존 파이프라인 테스트
        print("🔵 기존 RealTimeInferencePipeline 테스트")
        from inference.realtime import RealTimeInferencePipeline
        
        old_pipeline = RealTimeInferencePipeline(window_size=20, stride=5)
        
        test_data = self._generate_test_sensor_data(100)
        old_results = []
        
        old_start_time = time.time()
        for sensor_data in test_data:
            old_pipeline.add_sensor_data(sensor_data)
            result = old_pipeline.predict_single()
            if result:
                old_results.append(result)
        old_total_time = time.time() - old_start_time
        
        old_stats = old_pipeline.get_performance_stats()
        
        # 새로운 Unified 파이프라인 테스트
        print("🟢 새로운 UnifiedInferencePipeline 테스트")
        self.pipeline = create_unified_inference_pipeline()
        
        new_results = []
        new_start_time = time.time()
        for sensor_data in test_data:
            self.pipeline.add_sensor_data(sensor_data)
            result = self.pipeline.predict_single()
            if result:
                new_results.append(result)
        new_total_time = time.time() - new_start_time
        
        new_stats = self.pipeline.get_performance_stats()
        
        # 비교 결과 출력
        print(f"\n📊 성능 비교 결과:")
        print(f"{'항목':<20} {'기존':<15} {'Unified':<15} {'개선율':<10}")
        print("-" * 65)
        
        # FPS 비교
        fps_improvement = (new_stats['fps'] / old_stats['fps'] - 1) * 100 if old_stats['fps'] > 0 else 0
        print(f"{'FPS':<20} {old_stats['fps']:<15.1f} {new_stats['fps']:<15.1f} {fps_improvement:+.1f}%")
        
        # 지연시간 비교
        latency_improvement = (1 - new_stats['avg_latency_ms'] / old_stats.get('avg_inference_time', 1) / 1000) * 100
        print(f"{'지연시간(ms)':<20} {old_stats.get('avg_inference_time', 0)*1000:<15.2f} {new_stats['avg_latency_ms']:<15.2f} {latency_improvement:+.1f}%")
        
        # 총 처리 시간 비교
        time_improvement = (1 - new_total_time / old_total_time) * 100
        print(f"{'총 처리시간(s)':<20} {old_total_time:<15.2f} {new_total_time:<15.2f} {time_improvement:+.1f}%")
        
        # 예측 수 비교
        prediction_improvement = (len(new_results) / len(old_results) - 1) * 100 if len(old_results) > 0 else 0
        print(f"{'예측 수':<20} {len(old_results):<15} {len(new_results):<15} {prediction_improvement:+.1f}%")
        
        return {
            'old_stats': old_stats,
            'new_stats': new_stats,
            'improvements': {
                'fps': fps_improvement,
                'latency': latency_improvement,
                'total_time': time_improvement,
                'predictions': prediction_improvement
            }
        }
    
    def demo_adaptive_thresholding(self):
        """적응형 임계값 데모"""
        print("\n🎛️ 적응형 임계값 조정 데모")
        print("=" * 60)
        
        self.pipeline = create_unified_inference_pipeline()
        
        # 다양한 임계값으로 테스트
        thresholds = [0.5, 0.7, 0.9]
        test_data = self._generate_test_sensor_data(50)
        
        # 데이터 추가
        for sensor_data in test_data:
            self.pipeline.add_sensor_data(sensor_data)
        
        for threshold in thresholds:
            print(f"\n🎯 신뢰도 임계값: {threshold}")
            self.pipeline.config['confidence_threshold'] = threshold
            
            prediction_count = 0
            for i in range(30):  # 30번 예측 시도
                result = self.pipeline.predict_single()
                if result:
                    prediction_count += 1
            
            acceptance_rate = prediction_count / 30 * 100
            print(f"  📊 예측 수용률: {acceptance_rate:.1f}% ({prediction_count}/30)")
            
            # 안정적인 예측 확인
            stable_result = self.pipeline.get_stable_prediction()
            if stable_result:
                print(f"  🏆 안정적 예측: {stable_result.predicted_class} "
                      f"(안정성: {stable_result.stability_score:.3f})")
    
    def _generate_test_sensor_data(self, count: int) -> list:
        """테스트용 센서 데이터 생성"""
        import numpy as np
        
        test_data = []
        for i in range(count):
            # 현실적인 센서 값 생성
            flex_data = [
                np.random.normal(800 + i * 2, 20),  # 시간에 따른 변화
                np.random.normal(820 + i * 1.5, 25),
                np.random.normal(810 + i * 1, 15),
                np.random.normal(830 + i * 0.5, 10),
                np.random.normal(850 - i * 0.8, 18)
            ]
            
            # IMU 데이터 (오일러각)
            t = i * 0.1  # 시간 변수
            orientation_data = [
                30 * np.sin(t * 0.5),  # pitch
                20 * np.cos(t * 0.3),  # roll  
                45 * np.sin(t * 0.2)   # yaw
            ]
            
            sensor_data = flex_data + orientation_data
            test_data.append(sensor_data)
        
        return test_data
    
    def save_demo_results(self, filepath: str = "unified_inference_demo_results.json"):
        """데모 결과 저장"""
        if not self.demo_results:
            print("❌ 저장할 데모 결과가 없습니다.")
            return
        
        results_data = {
            'timestamp': time.time(),
            'demo_type': 'unified_inference',
            'total_predictions': len(self.demo_results),
            'results': [
                {
                    'predicted_class': result.predicted_class,
                    'confidence': result.confidence,
                    'stability_score': result.stability_score,
                    'processing_time_ms': result.processing_time * 1000,
                    'timestamp': result.timestamp
                } for result in self.demo_results
            ]
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 데모 결과 저장: {filepath}")
    
    def run_comprehensive_demo(self):
        """종합 데모 실행"""
        print("🎪 UnifiedInferencePipeline 종합 데모 시작")
        print("=" * 80)
        
        try:
            # 1. 기본 추론 데모
            basic_results = self.demo_basic_inference()
            
            # 2. Madgwick 통합 데모
            madgwick_results = self.demo_madgwick_integration()
            
            # 3. 성능 비교 데모
            comparison_results = self.demo_performance_comparison()
            
            # 4. 실시간 시뮬레이션 데모
            realtime_stats = self.demo_realtime_simulation()
            
            # 5. 적응형 임계값 데모
            self.demo_adaptive_thresholding()
            
            # 결과 요약
            print("\n🎉 종합 데모 완료!")
            print("=" * 80)
            print(f"📊 총 수행한 테스트:")
            print(f"  ✅ 기본 추론: {len(basic_results)}개 예측")
            print(f"  ✅ Madgwick 통합: {len(madgwick_results)}개 예측")
            print(f"  ✅ 성능 비교: 완료")
            print(f"  ✅ 실시간 시뮬레이션: 완료")
            print(f"  ✅ 적응형 임계값: 완료")
            
            if comparison_results:
                improvements = comparison_results['improvements']
                print(f"\n🚀 성능 개선 요약:")
                print(f"  📈 FPS 개선: {improvements['fps']:+.1f}%")
                print(f"  ⚡ 지연시간 개선: {improvements['latency']:+.1f}%")
                print(f"  🎯 처리량 개선: {improvements['predictions']:+.1f}%")
            
            # 결과 저장
            self.save_demo_results()
            
        except Exception as e:
            print(f"❌ 데모 실행 중 오류 발생: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    demo = UnifiedInferenceDemo()
    demo.run_comprehensive_demo()
