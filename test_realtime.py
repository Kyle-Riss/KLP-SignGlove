#!/usr/bin/env python3
"""
실시간 수어 인식 시스템 간단 테스트
"""

import sys
import os
import time

# 프로젝트 모듈 임포트
sys.path.append(os.path.dirname(__file__))

def test_components():
    """각 컴포넌트 개별 테스트"""
    print("🧪 실시간 수어 인식 시스템 컴포넌트 테스트")
    print("=" * 60)
    
    # 1. 하드웨어 테스트
    print("1. 하드웨어 시뮬레이션 테스트...")
    try:
        from hardware.signglove_hw import SignGloveHardware
        
        hardware = SignGloveHardware('simulation')
        if hardware.connect():
            print("✅ 하드웨어 연결 성공")
            
            # 짧은 스트리밍 테스트
            data_count = 0
            def data_callback(sensor_data):
                nonlocal data_count
                data_count += 1
                if data_count <= 3:
                    print(f"   센서 데이터 {data_count}: {[f'{x:.1f}' for x in sensor_data[:3]]}...")
            
            hardware.start_streaming(data_callback)
            time.sleep(2)
            hardware.stop_streaming()
            hardware.disconnect()
            
            print(f"   총 {data_count}개 센서 데이터 수신")
        else:
            print("❌ 하드웨어 연결 실패")
            return False
            
    except Exception as e:
        print(f"❌ 하드웨어 테스트 오류: {e}")
        return False
    
    # 2. 추론 파이프라인 테스트
    print("\n2. 추론 파이프라인 테스트...")
    try:
        from inference.realtime import RealTimeInferencePipeline
        
        pipeline = RealTimeInferencePipeline(
            model_path='best_dl_model.pth',
            confidence_threshold=0.5
        )
        print("✅ 추론 파이프라인 초기화 성공")
        
        # 더미 데이터로 테스트
        for i in range(25):  # 윈도우 크기보다 많이
            dummy_data = [500 + i*10, 600 + i*5, 700 + i*3, 800 + i*2, 900 + i*1,
                         0.1*i, 0.2*i, 0.3*i]
            pipeline.add_sensor_data(dummy_data)
        
        # 예측 수행
        result = pipeline.predict_single(force_predict=True)
        if result:
            print(f"   예측 결과: {result['predicted_class']} (신뢰도: {result['confidence']:.3f})")
            print("✅ 추론 테스트 성공")
        else:
            print("❌ 예측 결과 없음")
            
    except Exception as e:
        print(f"❌ 추론 파이프라인 테스트 오류: {e}")
        return False
    
    # 3. TTS 엔진 테스트
    print("\n3. TTS 엔진 테스트...")
    try:
        from tts.engine import KoreanTTSEngine
        
        tts = KoreanTTSEngine()
        print("✅ TTS 엔진 초기화 성공")
        
        # 테스트 발음 (짧게)
        test_result = {'predicted_class': 'ㄱ', 'confidence': 0.9}
        print(f"   테스트 발음: {test_result['predicted_class']}")
        tts.speak_prediction_result(test_result)
        
        print("✅ TTS 테스트 성공")
        
    except Exception as e:
        print(f"❌ TTS 엔진 테스트 오류: {e}")
        return False
    
    print("\n🎉 모든 컴포넌트 테스트 성공!")
    return True

def test_integration():
    """통합 테스트"""
    print("\n🔄 통합 테스트 시작...")
    print("=" * 60)
    
    try:
        from hardware.signglove_hw import SignGloveHardware
        from inference.realtime import RealTimeInferencePipeline
        from tts.engine import KoreanTTSEngine
        
        # 컴포넌트 초기화
        hardware = SignGloveHardware('simulation')
        pipeline = RealTimeInferencePipeline('best_dl_model.pth', confidence_threshold=0.6)
        tts = KoreanTTSEngine()
        
        # 연결
        if not hardware.connect():
            print("❌ 하드웨어 연결 실패")
            return False
        
        print("✅ 모든 컴포넌트 초기화 완료")
        
        # 통합 테스트 실행
        prediction_count = 0
        stable_prediction_count = 0
        
        def sensor_callback(sensor_data):
            pipeline.add_sensor_data(sensor_data)
        
        def prediction_callback(result):
            nonlocal prediction_count
            prediction_count += 1
            if prediction_count <= 5:
                print(f"   예측 {prediction_count}: {result['predicted_class']} "
                      f"(신뢰도: {result['confidence']:.2f})")
        
        # 스트리밍 시작
        hardware.start_streaming(sensor_callback)
        pipeline.start_realtime_inference(prediction_callback)
        tts.start_async_speaking()
        
        print("📡 실시간 처리 시작... (5초간)")
        
        # 5초간 실행
        for i in range(50):  # 0.1초 * 50 = 5초
            time.sleep(0.1)
            
            # 안정적인 예측 체크
            stable = pipeline.get_stable_prediction(3)
            if stable and stable_prediction_count < 2:
                stable_prediction_count += 1
                print(f"🎯 안정적인 예측: {stable['predicted_class']} "
                      f"(신뢰도: {stable['confidence']:.2f})")
                
                # TTS 출력
                tts.speak_prediction_result({
                    'predicted_class': stable['predicted_class'],
                    'confidence': stable['confidence']
                })
        
        # 정리
        hardware.stop_streaming()
        pipeline.stop_realtime_inference()
        tts.stop_async_speaking()
        hardware.disconnect()
        
        # 성능 통계
        stats = pipeline.get_performance_stats()
        print(f"\n📊 성능 통계:")
        print(f"   총 예측 수: {prediction_count}")
        print(f"   안정적 예측: {stable_prediction_count}")
        print(f"   평균 추론 시간: {stats.get('avg_inference_time', 0)*1000:.1f}ms")
        print(f"   추론 FPS: {stats.get('fps', 0):.1f}")
        
        print("\n🎉 통합 테스트 성공!")
        return True
        
    except Exception as e:
        print(f"❌ 통합 테스트 오류: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 실시간 한국수어 인식 시스템 테스트 시작")
    print(f"시간: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 컴포넌트 테스트
    if test_components():
        # 통합 테스트
        test_integration()
    
    print(f"\n✅ 테스트 완료 - {time.strftime('%H:%M:%S')}")