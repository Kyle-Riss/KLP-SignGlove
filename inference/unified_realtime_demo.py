"""
SignGlove_HW unified 스타일 실시간 추론 데모
실제 데이터와 모델을 사용한 통합 추론 시스템 테스트
"""

import sys
import os
import time
import json
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
# from integrations.madgwick_adapter import MadgwickDataAdapter  # 제거됨

class UnifiedRealtimeDemo:
    """Unified 스타일 실시간 추론 데모"""
    
    def __init__(self):
        self.pipeline = None
        self.results_log = []
        
    def setup_unified_pipeline(self):
        """통합 추론 파이프라인 설정"""
        print("🚀 SignGlove_HW Unified 스타일 추론 파이프라인 초기화")
        print("=" * 70)
        
        # 고성능 설정으로 파이프라인 생성
        self.pipeline = create_unified_inference_pipeline(
            model_path="best_unified_model.pth",  # 새로 학습된 Unified 모델 사용
            config_path=None
        )
            
        print("✅ 통합 추론 파이프라인 초기화 완료")
            
        # 성능 통계 출력
        initial_stats = self.pipeline.get_performance_stats()
        print(f"📊 초기 시스템 상태:")
        print(f"  - 윈도우 크기: {initial_stats.get('window_size', 'N/A')}")
        print(f"  - 신뢰도 임계값: {initial_stats.get('confidence_threshold', 'N/A')}")
        print(f"  - 목표 지연시간: {initial_stats.get('target_latency_ms', 'N/A')}ms")
        
    def load_real_sensor_data(self) -> List[Dict]:
        """실제 센서 데이터 로드"""
        print("\n📁 실제 센서 데이터 로드 중...")
        
        sensor_data = []
        data_sources = [
            'integrations/SignGlove_HW/ㄱ_sample_data.csv',
            'integrations/SignGlove_HW/ㄴ_sample_data.csv',
            'integrations/SignGlove_HW/ㄷ_sample_data.csv',
            'integrations/SignGlove_HW/ㄹ_sample_data.csv',
            'integrations/SignGlove_HW/ㅁ_sample_data.csv',
            'integrations/SignGlove_HW/madgwick_demo_converted.csv'
        ]
        
        for data_file in data_sources:
            if os.path.exists(data_file):
                try:
                    df = pd.read_csv(data_file)
                    print(f"  📄 로드: {os.path.basename(data_file)} ({len(df)}개 샘플)")
                    
                    # Ground Truth 라벨 추출
                    filename = os.path.basename(data_file)
                    ground_truth = None
                    
                    # 파일명에서 라벨 추출 (디버깅 출력 추가)
                    print(f"  🔍 라벨 추출 중: {filename}")
                    
                    # 정확한 파일명 패턴 매핑 (Unified 데이터 포함)
                    if 'ㄱ_sample_data.csv' in filename or 'ㄱ_unified_data' in filename:
                        ground_truth = 'ㄱ'
                        print(f"    ✅ 라벨 추출: {ground_truth}")
                    elif 'ㄴ_sample_data.csv' in filename or 'ㄴ_unified_data' in filename:
                        ground_truth = 'ㄴ'
                        print(f"    ✅ 라벨 추출: {ground_truth}")
                    elif 'ㄷ_sample_data.csv' in filename or 'ㄷ_unified_data' in filename:
                        ground_truth = 'ㄷ'
                        print(f"    ✅ 라벨 추출: {ground_truth}")
                    elif 'ㄹ_sample_data.csv' in filename or 'ㄹ_unified_data' in filename:
                        ground_truth = 'ㄹ'
                        print(f"    ✅ 라벨 추출: {ground_truth}")
                    elif 'ㅁ_sample_data.csv' in filename or 'ㅁ_unified_data' in filename:
                        ground_truth = 'ㅁ'
                        print(f"    ✅ 라벨 추출: {ground_truth}")
                    elif 'ㅂ_unified_data' in filename:
                        ground_truth = 'ㅂ'
                        print(f"    ✅ 라벨 추출: {ground_truth}")
                    elif 'ㅅ_unified_data' in filename:
                        ground_truth = 'ㅅ'
                        print(f"    ✅ 라벨 추출: {ground_truth}")
                    elif 'ㅇ_unified_data' in filename:
                        ground_truth = 'ㅇ'
                        print(f"    ✅ 라벨 추출: {ground_truth}")
                    elif 'ㅈ_unified_data' in filename:
                        ground_truth = 'ㅈ'
                        print(f"    ✅ 라벨 추출: {ground_truth}")
                    elif 'ㅊ_unified_data' in filename:
                        ground_truth = 'ㅊ'
                        print(f"    ✅ 라벨 추출: {ground_truth}")
                    elif 'ㅋ_unified_data' in filename:
                        ground_truth = 'ㅋ'
                        print(f"    ✅ 라벨 추출: {ground_truth}")
                    elif 'ㅌ_unified_data' in filename:
                        ground_truth = 'ㅌ'
                        print(f"    ✅ 라벨 추출: {ground_truth}")
                    elif 'ㅍ_unified_data' in filename:
                        ground_truth = 'ㅍ'
                        print(f"    ✅ 라벨 추출: {ground_truth}")
                    elif 'ㅎ_unified_data' in filename:
                        ground_truth = 'ㅎ'
                        print(f"    ✅ 라벨 추출: {ground_truth}")
                    elif 'ㅏ_unified_data' in filename:
                        ground_truth = 'ㅏ'
                        print(f"    ✅ 라벨 추출: {ground_truth}")
                    elif 'ㅓ_unified_data' in filename:
                        ground_truth = 'ㅓ'
                        print(f"    ✅ 라벨 추출: {ground_truth}")
                    elif 'ㅗ_unified_data' in filename:
                        ground_truth = 'ㅗ'
                        print(f"    ✅ 라벨 추출: {ground_truth}")
                    elif 'ㅜ_unified_data' in filename:
                        ground_truth = 'ㅜ'
                        print(f"    ✅ 라벨 추출: {ground_truth}")
                    elif 'ㅡ_unified_data' in filename:
                        ground_truth = 'ㅡ'
                        print(f"    ✅ 라벨 추출: {ground_truth}")
                    elif 'ㅣ_unified_data' in filename:
                        ground_truth = 'ㅣ'
                        print(f"    ✅ 라벨 추출: {ground_truth}")
                    elif 'ㅑ_unified_data' in filename:
                        ground_truth = 'ㅑ'
                        print(f"    ✅ 라벨 추출: {ground_truth}")
                    elif 'ㅕ_unified_data' in filename:
                        ground_truth = 'ㅕ'
                        print(f"    ✅ 라벨 추출: {ground_truth}")
                    elif 'ㅛ_unified_data' in filename:
                        ground_truth = 'ㅛ'
                        print(f"    ✅ 라벨 추출: {ground_truth}")
                    elif 'ㅠ_unified_data' in filename:
                        ground_truth = 'ㅠ'
                        print(f"    ✅ 라벨 추출: {ground_truth}")
                    elif 'madgwick_demo_converted.csv' in filename:
                        ground_truth = 'ㄱ'  # Madgwick 데이터는 ㄱ으로 가정
                        print(f"    ✅ Madgwick 데이터 라벨: {ground_truth}")
                    elif 'test_madgwick_sample_data.csv' in filename:
                        ground_truth = 'ㄱ'  # 테스트 데이터는 ㄱ으로 가정
                        print(f"    ✅ 테스트 데이터 라벨: {ground_truth}")
                    elif 'imu_flex_20250807_150749.csv' in filename:
                        ground_truth = 'ㄱ'  # 기본 IMU 데이터는 ㄱ으로 가정
                        print(f"    ✅ IMU 데이터 라벨: {ground_truth}")
                    else:
                        print(f"    ❌ 알 수 없는 파일명: {filename}")
                        continue
                    
                    # 데이터 형태 확인 및 변환
                    samples_per_class = 200  # 각 클래스당 200개 샘플
                    class_samples = 0
                    
                    for idx, row in df.iterrows():
                        try:
                            # 클래스별 샘플 수 제한
                            if class_samples >= samples_per_class:
                                break
                            
                            # 기본 센서 데이터 (flex + orientation)
                            if all(col in df.columns for col in ['flex1', 'flex2', 'flex3', 'flex4', 'flex5']):
                                # KSL 형태 데이터
                                sensor_reading = {
                                    'timestamp': row.get('timestamp(ms)', time.time() * 1000) / 1000.0,
                                    'flex_data': [
                                        row['flex1'], row['flex2'], row['flex3'], 
                                        row['flex4'], row['flex5']
                                    ],
                                    'orientation_data': [
                                        row.get('pitch(°)', 0),
                                        row.get('roll(°)', 0),
                                        row.get('yaw(°)', 0)
                                    ],
                                    'source': f"ksl_{os.path.basename(data_file)}",
                                    'ground_truth': ground_truth,  # Ground Truth 추가
                                    'expected_class': ground_truth
                                }
                            else:
                                # Madgwick 형태 데이터 (이미 변환된 것)
                                sensor_reading = {
                                    'timestamp': row.get('timestamp(ms)', time.time() * 1000) / 1000.0,
                                    'flex_data': [
                                        row.get('flex1', 800), row.get('flex2', 820), 
                                        row.get('flex3', 810), row.get('flex4', 830), row.get('flex5', 850)
                                    ],
                                    'orientation_data': [
                                        row.get('pitch(°)', 0),
                                        row.get('roll(°)', 0),
                                        row.get('yaw(°)', 0)
                                    ],
                                    'source': f"madgwick_{os.path.basename(data_file)}",
                                    'ground_truth': ground_truth,  # Ground Truth 추가
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
    
    def run_unified_inference_demo(self):
        """통합 추론 데모 실행"""
        print("\n🎯 Unified 실시간 추론 데모 시작")
        print("=" * 70)
        
        # 실제 센서 데이터 로드
        sensor_data = self.load_real_sensor_data()
        
        if not sensor_data:
            print("❌ 센서 데이터가 없습니다.")
            return
        
        # 실시간 콜백 설정
        prediction_count = 0
        stable_prediction_count = 0
        class_predictions = {}
        accuracy_stats = {
            'correct': 0,
            'total': 0,
            'class_accuracy': {},
            'confusion_matrix': {}
        }
        
        def prediction_callback(result):
            nonlocal prediction_count, accuracy_stats
            prediction_count += 1
            
            # Ground Truth와 비교하여 정확도 계산
            ground_truth = result.source_data.ground_truth if hasattr(result.source_data, 'ground_truth') else None
            predicted_class = result.predicted_class
            
            if ground_truth and ground_truth != 'unknown':
                accuracy_stats['total'] += 1
                is_correct = (predicted_class == ground_truth)
                
                if is_correct:
                    accuracy_stats['correct'] += 1
                
                # 클래스별 정확도 계산
                if ground_truth not in accuracy_stats['class_accuracy']:
                    accuracy_stats['class_accuracy'][ground_truth] = {'correct': 0, 'total': 0}
                
                accuracy_stats['class_accuracy'][ground_truth]['total'] += 1
                if is_correct:
                    accuracy_stats['class_accuracy'][ground_truth]['correct'] += 1
                
                # 혼동 행렬 업데이트
                if ground_truth not in accuracy_stats['confusion_matrix']:
                    accuracy_stats['confusion_matrix'][ground_truth] = {}
                if predicted_class not in accuracy_stats['confusion_matrix'][ground_truth]:
                    accuracy_stats['confusion_matrix'][ground_truth][predicted_class] = 0
                accuracy_stats['confusion_matrix'][ground_truth][predicted_class] += 1
            
            # 클래스별 예측 통계
            if predicted_class not in class_predictions:
                class_predictions[predicted_class] = 0
            class_predictions[predicted_class] += 1
            
            # 로그 저장 (Ground Truth 포함)
            self.results_log.append({
                'timestamp': result.timestamp,
                'predicted_class': predicted_class,
                'ground_truth': ground_truth,
                'is_correct': is_correct if ground_truth and ground_truth != 'unknown' else None,
                'confidence': result.confidence,
                'stability_score': result.stability_score,
                'processing_time_ms': result.processing_time * 1000,
                'source': result.source_data.source if hasattr(result.source_data, 'source') else 'unknown'
            })
            
            # 주기적 출력 (정확도 포함)
            if prediction_count % 50 == 0:
                current_accuracy = (accuracy_stats['correct'] / accuracy_stats['total'] * 100) if accuracy_stats['total'] > 0 else 0
                print(f"📊 실시간 예측 #{prediction_count}: {predicted_class} "
                      f"(신뢰도: {result.confidence:.3f}, 안정성: {result.stability_score:.3f})")
                if ground_truth and ground_truth != 'unknown':
                    status = "✅" if is_correct else "❌"
                    print(f"  {status} Ground Truth: {ground_truth} | 현재 정확도: {current_accuracy:.1f}%")
        
        def stable_prediction_callback(result):
            nonlocal stable_prediction_count
            stable_prediction_count += 1
            
            # Ground Truth 정보 포함
            ground_truth = result.source_data.ground_truth if hasattr(result.source_data, 'ground_truth') else None
            predicted_class = result.predicted_class
            
            if ground_truth and ground_truth != 'unknown':
                is_correct = (predicted_class == ground_truth)
                status = "✅" if is_correct else "❌"
                print(f"🎯 안정적 예측 #{stable_prediction_count}: {predicted_class} "
                      f"(안정성: {result.stability_score:.3f}, 합의: {result.metadata.get('consensus_ratio', 0):.3f})")
                print(f"  {status} Ground Truth: {ground_truth}")
            else:
                print(f"🎯 안정적 예측 #{stable_prediction_count}: {predicted_class} "
                      f"(안정성: {result.stability_score:.3f}, 합의: {result.metadata.get('consensus_ratio', 0):.3f})")
        
        # 실시간 추론 시작
        self.pipeline.start_realtime_inference(
            prediction_callback=prediction_callback,
            stable_prediction_callback=stable_prediction_callback
        )
        
        print("🚀 실시간 추론 시작 - 센서 데이터 스트리밍 중...")
        
        # 센서 데이터 스트리밍 (실시간 시뮬레이션)
        start_time = time.time()
        processed_samples = 0
        
        try:
            for idx, data_dict in enumerate(sensor_data):
                # SensorReading 객체 생성
                sensor_reading = SensorReading(
                    timestamp=data_dict['timestamp'],
                    flex_data=data_dict['flex_data'],
                    orientation_data=data_dict['orientation_data'],
                    source=data_dict['source']
                )
                
                # Ground Truth 정보 추가
                sensor_reading.ground_truth = data_dict.get('ground_truth')
                
                # 추론 파이프라인에 데이터 추가
                success = self.pipeline.add_sensor_data(sensor_reading)
                
                if success:
                    processed_samples += 1
                
                # 실제 센서 주기 시뮬레이션 (20Hz = 50ms)
                time.sleep(0.05)
                
                # 진행률 표시
                if (idx + 1) % 100 == 0:
                    elapsed = time.time() - start_time
                    current_accuracy = (accuracy_stats['correct'] / accuracy_stats['total'] * 100) if accuracy_stats['total'] > 0 else 0
                    print(f"📈 진행률: {idx + 1}/{len(sensor_data)} "
                          f"({(idx + 1)/len(sensor_data)*100:.1f}%) - "
                          f"처리 시간: {elapsed:.1f}초 | 정확도: {current_accuracy:.1f}%")
                
                # 시간 제한 (60초)
                if time.time() - start_time > 60:
                    print("⏰ 60초 제한으로 데모 종료")
                    break
        
        except KeyboardInterrupt:
            print("\n⚠️  사용자에 의해 중단됨")
        
        finally:
            # 실시간 추론 중지
            self.pipeline.stop_realtime_inference()
        
        # 결과 분석 (정확도 포함)
        self._analyze_results_with_accuracy(start_time, processed_samples, class_predictions, 
                                          prediction_count, stable_prediction_count, accuracy_stats)
    
    def _analyze_results_with_accuracy(self, start_time, processed_samples, class_predictions, 
                                     prediction_count, stable_prediction_count, accuracy_stats):
        """정확도를 포함한 결과 분석 및 출력"""
        total_time = time.time() - start_time
        
        print("\n" + "="*70)
        print("📊 Unified 추론 데모 결과 분석 (라벨링 포함)")
        print("="*70)
        
        # 기본 통계
        print(f"🕒 실행 시간: {total_time:.2f}초")
        print(f"📄 처리된 샘플: {processed_samples}개")
        print(f"🎯 총 예측: {prediction_count}개")
        print(f"🏆 안정적 예측: {stable_prediction_count}개")
        
        if processed_samples > 0:
            print(f"📈 예측 비율: {prediction_count / processed_samples * 100:.1f}%")
            print(f"🎖️  안정성 비율: {stable_prediction_count / prediction_count * 100:.1f}%")
        
        # 정확도 분석
        if accuracy_stats['total'] > 0:
            overall_accuracy = accuracy_stats['correct'] / accuracy_stats['total'] * 100
            print(f"\n🎯 정확도 분석:")
            print(f"  📊 전체 정확도: {overall_accuracy:.2f}% ({accuracy_stats['correct']}/{accuracy_stats['total']})")
            
            # 클래스별 정확도
            print(f"  📈 클래스별 정확도:")
            for class_name, stats in accuracy_stats['class_accuracy'].items():
                class_acc = stats['correct'] / stats['total'] * 100
                print(f"    {class_name}: {class_acc:.1f}% ({stats['correct']}/{stats['total']})")
            
            # 혼동 행렬 출력
            print(f"  🔍 혼동 행렬:")
            for true_label in sorted(accuracy_stats['confusion_matrix'].keys()):
                print(f"    {true_label} -> ", end="")
                for pred_label in sorted(accuracy_stats['confusion_matrix'][true_label].keys()):
                    count = accuracy_stats['confusion_matrix'][true_label][pred_label]
                    print(f"{pred_label}:{count} ", end="")
                print()
        else:
            print(f"\n⚠️  라벨링된 데이터가 없어 정확도 계산 불가")
        
        # 클래스별 예측 분포
        if class_predictions:
            print(f"\n📊 클래스별 예측 분포:")
            for class_name, count in sorted(class_predictions.items()):
                percentage = count / prediction_count * 100 if prediction_count > 0 else 0
                print(f"  {class_name}: {count}개 ({percentage:.1f}%)")
        
        # 시스템 성능 통계
        final_stats = self.pipeline.get_performance_stats()
        print(f"\n⚡ 시스템 성능:")
        print(f"  🔥 평균 FPS: {final_stats.get('fps', 0):.1f}")
        print(f"  ⚡ 평균 지연시간: {final_stats.get('avg_latency_ms', 0):.2f}ms")
        print(f"  📊 총 프레임: {final_stats.get('total_frames', 0)}")
        print(f"  ❌ 오류 수: {final_stats.get('error_count', 0)}")
        
        # 버퍼 사용률
        buffer_util = final_stats.get('buffer_utilization', {})
        print(f"  💾 버퍼 사용률:")
        for buf_name, utilization in buffer_util.items():
            print(f"    - {buf_name}: {utilization*100:.1f}%")
        
        # 상세 타이밍 분석
        if 'inference_timing' in final_stats:
            timing = final_stats['inference_timing']
            print(f"  🕐 추론 타이밍:")
            print(f"    - 평균: {timing.get('avg_ms', 0):.2f}ms")
            print(f"    - 최소: {timing.get('min_ms', 0):.2f}ms")
            print(f"    - 최대: {timing.get('max_ms', 0):.2f}ms")
            print(f"    - 표준편차: {timing.get('std_ms', 0):.2f}ms")
    
    def save_results(self):
        """결과 저장"""
        if not self.results_log:
            print("💾 저장할 결과가 없습니다.")
            return
        
        # 결과 분석
        result_analysis = {
            'demo_info': {
                'timestamp': time.time(),
                'demo_type': 'unified_realtime_inference',
                'total_predictions': len(self.results_log)
            },
            'performance_summary': self.pipeline.get_performance_stats() if self.pipeline else {},
            'predictions': self.results_log
        }
        
        # JSON 저장
        output_file = 'unified_inference_results.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result_analysis, f, indent=2, ensure_ascii=False)
        
        print(f"💾 결과 저장 완료: {output_file}")
        
        # 성능 로그도 저장
        if self.pipeline:
            self.pipeline.save_performance_log('unified_performance_log.json')
            print(f"📈 성능 로그 저장: unified_performance_log.json")
    
    def run_complete_demo(self):
        """전체 데모 실행"""
        try:
            print("🎪 SignGlove_HW Unified 스타일 실시간 추론 데모")
            print("GitHub: https://github.com/KNDG01001/SignGlove_HW/tree/main/unified")
            print("="*80)
            
            # 1. 파이프라인 설정
            self.setup_unified_pipeline()
            
            # 2. 실시간 추론 데모
            self.run_unified_inference_demo()
            
            # 3. 결과 저장
            self.save_results()
            
            print("\n🎉 Unified 추론 데모 완료!")
            print("📁 결과 파일들이 생성되었습니다.")
            
        except Exception as e:
            print(f"❌ 데모 실행 중 오류 발생: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    demo = UnifiedRealtimeDemo()
    demo.run_complete_demo()
