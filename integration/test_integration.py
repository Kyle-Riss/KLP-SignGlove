#!/usr/bin/env python3
"""
SignGlove 통합 시스템 테스트
실제 하드웨어 없이도 테스트 가능한 시뮬레이션
"""

import time
import json
import numpy as np
from integration.signglove_client import (
    SignGloveAPIClient, 
    SignGloveIntegratedClient,
    SensorReading
)

def test_api_connection():
    """API 서버 연결 테스트"""
    print("🔗 API 서버 연결 테스트")
    print("="*40)
    
    api_client = SignGloveAPIClient("http://localhost:8000")
    
    # 서버 상태 확인
    if api_client._check_api_status():
        print("✅ API 서버 연결 성공")
        
        # 모델 정보 조회
        model_info = api_client.get_model_info()
        if model_info:
            print(f"📊 모델: {model_info.get('model_name', 'Unknown')}")
            print(f"🎯 정확도: {model_info.get('accuracy', 0):.2%}")
            print(f"📈 클래스 수: {model_info.get('num_classes', 0)}")
        else:
            print("❌ 모델 정보 조회 실패")
    else:
        print("❌ API 서버 연결 실패")
        return False
    
    return True

def test_prediction():
    """예측 기능 테스트"""
    print("\n🎯 예측 기능 테스트")
    print("="*40)
    
    api_client = SignGloveAPIClient("http://localhost:8000")
    
    # 테스트 데이터 생성 (24개 클래스별로)
    test_gestures = [
        # 자음 (14개)
        {'name': 'ㄱ', 'flex': [700, 700, 700, 700, 700]},
        {'name': 'ㄴ', 'flex': [800, 700, 700, 700, 700]},
        {'name': 'ㄷ', 'flex': [700, 800, 700, 700, 700]},
        {'name': 'ㄹ', 'flex': [700, 700, 800, 700, 700]},
        {'name': 'ㅁ', 'flex': [700, 700, 700, 800, 700]},
        {'name': 'ㅂ', 'flex': [700, 700, 700, 700, 800]},
        {'name': 'ㅅ', 'flex': [800, 800, 700, 700, 700]},
        {'name': 'ㅇ', 'flex': [700, 800, 800, 700, 700]},
        {'name': 'ㅈ', 'flex': [700, 700, 800, 800, 700]},
        {'name': 'ㅊ', 'flex': [700, 700, 700, 800, 800]},
        {'name': 'ㅋ', 'flex': [800, 700, 800, 700, 700]},
        {'name': 'ㅌ', 'flex': [700, 800, 700, 800, 700]},
        {'name': 'ㅍ', 'flex': [700, 700, 800, 700, 800]},
        {'name': 'ㅎ', 'flex': [800, 700, 700, 800, 700]},
        
        # 모음 (10개)
        {'name': 'ㅏ', 'flex': [900, 700, 700, 700, 700]},
        {'name': 'ㅑ', 'flex': [700, 900, 700, 700, 700]},
        {'name': 'ㅓ', 'flex': [700, 700, 900, 700, 700]},
        {'name': 'ㅕ', 'flex': [700, 700, 700, 900, 700]},
        {'name': 'ㅗ', 'flex': [700, 700, 700, 700, 900]},
        {'name': 'ㅛ', 'flex': [900, 900, 700, 700, 700]},
        {'name': 'ㅜ', 'flex': [700, 900, 900, 700, 700]},
        {'name': 'ㅠ', 'flex': [700, 700, 900, 900, 700]},
        {'name': 'ㅡ', 'flex': [700, 700, 700, 900, 900]},
        {'name': 'ㅣ', 'flex': [900, 700, 900, 700, 700]},
    ]
    
    results = []
    
    for gesture in test_gestures:
        # 테스트 데이터 생성
        test_data = SensorReading(
            timestamp=time.time(),
            pitch=np.random.uniform(-30, 30),
            roll=np.random.uniform(-30, 30),
            yaw=np.random.uniform(-30, 30),
            flex1=gesture['flex'][0],
            flex2=gesture['flex'][1],
            flex3=gesture['flex'][2],
            flex4=gesture['flex'][3],
            flex5=gesture['flex'][4]
        )
        
        # 예측 수행
        result = api_client.predict_gesture(test_data)
        
        if 'error' not in result:
            predicted = result.get('predicted_class', 'Unknown')
            confidence = result.get('confidence', 0.0)
            
            is_correct = predicted == gesture['name']
            results.append({
                'expected': gesture['name'],
                'predicted': predicted,
                'confidence': confidence,
                'correct': is_correct
            })
            
            status = "✅" if is_correct else "❌"
            print(f"{status} {gesture['name']} → {predicted} (신뢰도: {confidence:.3f})")
        else:
            print(f"❌ {gesture['name']} → 오류: {result['error']}")
            results.append({
                'expected': gesture['name'],
                'predicted': 'Error',
                'confidence': 0.0,
                'correct': False
            })
        
        time.sleep(0.1)  # API 요청 간격
    
    # 결과 요약
    correct_count = sum(1 for r in results if r['correct'])
    total_count = len(results)
    accuracy = correct_count / total_count if total_count > 0 else 0
    
    print(f"\n📊 테스트 결과 요약:")
    print(f"  총 테스트: {total_count}개")
    print(f"  정확한 예측: {correct_count}개")
    print(f"  정확도: {accuracy:.2%}")
    
    return accuracy > 0.5  # 50% 이상이면 성공

def test_word_prediction():
    """단어 예측 테스트"""
    print("\n📝 단어 예측 테스트")
    print("="*40)
    
    api_client = SignGloveAPIClient("http://localhost:8000")
    
    # 테스트 단어: "안녕하세요"
    test_word = "안녕하세요"
    word_gestures = ['ㅇ', 'ㅏ', 'ㄴ', 'ㄴ', 'ㅕ', 'ㅇ', 'ㅎ', 'ㅏ', 'ㅅ', 'ㅔ', 'ㅇ', 'ㅛ']
    
    print(f"테스트 단어: {test_word}")
    print(f"필요한 제스처: {' '.join(word_gestures)}")
    
    word_buffer = []
    
    for gesture in word_gestures:
        # 각 제스처에 대한 테스트 데이터 생성
        test_data = SensorReading(
            timestamp=time.time(),
            pitch=np.random.uniform(-30, 30),
            roll=np.random.uniform(-30, 30),
            yaw=np.random.uniform(-30, 30),
            flex1=np.random.uniform(700, 900),
            flex2=np.random.uniform(700, 900),
            flex3=np.random.uniform(700, 900),
            flex4=np.random.uniform(700, 900),
            flex5=np.random.uniform(700, 900)
        )
        
        # 제스처 예측
        result = api_client.predict_gesture(test_data)
        
        if 'error' not in result:
            predicted = result.get('predicted_class', 'Unknown')
            confidence = result.get('confidence', 0.0)
            
            word_buffer.append(predicted)
            print(f"  {gesture} → {predicted} (신뢰도: {confidence:.3f})")
        else:
            print(f"  {gesture} → 오류: {result['error']}")
            word_buffer.append('?')
        
        time.sleep(0.1)
    
    # 단어 완성 시뮬레이션
    current_word = ''.join(word_buffer)
    print(f"\n📝 완성된 단어: {current_word}")
    
    # 단어 예측 API 테스트
    test_data = SensorReading(
        timestamp=time.time(),
        pitch=0.0, roll=0.0, yaw=0.0,
        flex1=700, flex2=700, flex3=700, flex4=700, flex5=700
    )
    
    word_result = api_client.predict_word(test_data)
    if 'error' not in word_result:
        predicted_word = word_result.get('predicted_word', 'Unknown')
        print(f"API 단어 예측: {predicted_word}")
    else:
        print(f"단어 예측 오류: {word_result['error']}")

def test_integration_simulation():
    """통합 시스템 시뮬레이션 테스트"""
    print("\n🔄 통합 시스템 시뮬레이션 테스트")
    print("="*40)
    
    # 콜백 함수들
    detected_gestures = []
    completed_words = []
    
    def on_gesture_detected(gesture: str, confidence: float):
        detected_gestures.append({'gesture': gesture, 'confidence': confidence})
        print(f"🎯 감지: {gesture} (신뢰도: {confidence:.3f})")
    
    def on_word_completed(word: str):
        completed_words.append(word)
        print(f"📝 단어 완성: {word}")
    
    def on_error(error_msg: str):
        print(f"❌ 오류: {error_msg}")
    
    # 통합 클라이언트 생성 (하드웨어 없이)
    client = SignGloveIntegratedClient(
        hardware_port=None,  # 하드웨어 연결 없음
        api_url="http://localhost:8000",
        confidence_threshold=0.5  # 낮은 임계값으로 테스트
    )
    
    # 콜백 설정
    client.set_callbacks(
        on_gesture_detected=on_gesture_detected,
        on_word_completed=on_word_completed,
        on_error=on_error
    )
    
    # API 서버만 연결 테스트
    if client.api_client._check_api_status():
        print("✅ API 서버 연결 성공")
        
        # 시뮬레이션 데이터로 테스트
        print("🔄 시뮬레이션 데이터로 테스트 중...")
        
        # 24개 클래스 시뮬레이션
        test_gestures = ['ㄱ', 'ㄴ', 'ㄷ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅅ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ',
                        'ㅏ', 'ㅑ', 'ㅓ', 'ㅕ', 'ㅗ', 'ㅛ', 'ㅜ', 'ㅠ', 'ㅡ', 'ㅣ']
        
        for i, gesture in enumerate(test_gestures):
            # 시뮬레이션 데이터 생성
            test_data = SensorReading(
                timestamp=time.time(),
                pitch=np.random.uniform(-30, 30),
                roll=np.random.uniform(-30, 30),
                yaw=np.random.uniform(-30, 30),
                flex1=np.random.uniform(700, 900),
                flex2=np.random.uniform(700, 900),
                flex3=np.random.uniform(700, 900),
                flex4=np.random.uniform(700, 900),
                flex5=np.random.uniform(700, 900)
            )
            
            # API 호출 시뮬레이션
            result = client.api_client.predict_gesture(test_data)
            
            if 'error' not in result:
                predicted = result.get('predicted_class', 'Unknown')
                confidence = result.get('confidence', 0.0)
                
                # 콜백 호출 시뮬레이션
                if confidence > client.confidence_threshold:
                    on_gesture_detected(predicted, confidence)
                
                print(f"  {i+1:2d}/24: {gesture} → {predicted} ({confidence:.3f})")
            else:
                print(f"  {i+1:2d}/24: {gesture} → 오류")
            
            time.sleep(0.1)
        
        print(f"\n📊 시뮬레이션 결과:")
        print(f"  감지된 제스처: {len(detected_gestures)}개")
        print(f"  완성된 단어: {len(completed_words)}개")
        
        if detected_gestures:
            avg_confidence = np.mean([g['confidence'] for g in detected_gestures])
            print(f"  평균 신뢰도: {avg_confidence:.3f}")
    
    else:
        print("❌ API 서버 연결 실패")

def main():
    """메인 테스트 함수"""
    print("🧪 SignGlove 통합 시스템 테스트")
    print("="*60)
    
    # 1. API 서버 연결 테스트
    if not test_api_connection():
        print("\n❌ API 서버가 실행되지 않았습니다.")
        print("다음 명령어로 API 서버를 시작하세요:")
        print("  python server/main.py")
        return
    
    # 2. 예측 기능 테스트
    prediction_success = test_prediction()
    
    # 3. 단어 예측 테스트
    test_word_prediction()
    
    # 4. 통합 시스템 시뮬레이션
    test_integration_simulation()
    
    # 최종 결과
    print("\n" + "="*60)
    print("📋 테스트 결과 요약")
    print("="*60)
    
    if prediction_success:
        print("✅ 통합 시스템 테스트 성공!")
        print("🚀 실제 하드웨어와 연결하여 사용할 수 있습니다.")
        print("\n📖 사용법:")
        print("  1. API 서버 실행: python server/main.py")
        print("  2. 통합 클라이언트 실행: python integration/signglove_client.py")
        print("  3. SignGlove_HW 하드웨어 연결")
    else:
        print("⚠️  일부 테스트에서 문제가 발생했습니다.")
        print("🔧 모델 성능을 개선하거나 설정을 조정해보세요.")

if __name__ == "__main__":
    main()
