#!/usr/bin/env python3
"""
실시간 한국수어 인식 데모 시스템
- SignGlove 하드웨어에서 센서 데이터 수집
- 딥러닝 모델을 이용한 실시간 수어 인식
- TTS를 통한 음성 출력
- 실시간 시각화 및 모니터링
"""

import sys
import os
import time
import argparse
import threading
from collections import deque
from typing import Dict, List, Optional

# 프로젝트 모듈 임포트
sys.path.append(os.path.dirname(__file__))
from hardware.signglove_hw import SignGloveHardware
from inference.realtime import RealTimeInferencePipeline
from tts.engine import KoreanTTSEngine

class RealTimeSignLanguageDemo:
    """실시간 수어 인식 데모 시스템"""
    
    def __init__(self, model_path: str = 'best_dl_model.pth',
                 connection_type: str = 'simulation',
                 confidence_threshold: float = 0.75,
                 stability_window: int = 5):
        """
        Args:
            model_path: 학습된 모델 파일 경로
            connection_type: 하드웨어 연결 타입
            confidence_threshold: 예측 신뢰도 임계값
            stability_window: 안정성 체크 윈도우 크기
        """
        self.model_path = model_path
        self.connection_type = connection_type
        self.confidence_threshold = confidence_threshold
        self.stability_window = stability_window
        
        # 컴포넌트 초기화
        self.hardware = None
        self.inference_pipeline = None
        self.tts_engine = None
        
        # 실행 상태
        self.is_running = False
        self.demo_thread = None
        
        # 통계 및 모니터링
        self.total_predictions = 0
        self.successful_predictions = 0
        self.last_prediction = None
        self.last_stable_prediction = None
        self.prediction_history = deque(maxlen=100)
        
        # 성능 모니터링
        self.fps_counter = deque(maxlen=50)
        self.last_frame_time = time.time()
        
        print("실시간 수어 인식 데모 시스템 초기화")
    
    def initialize_components(self) -> bool:
        """모든 컴포넌트 초기화"""
        try:
            # 1. 하드웨어 초기화
            print("1. 하드웨어 초기화...")
            self.hardware = SignGloveHardware(self.connection_type)
            if not self.hardware.connect():
                print("❌ 하드웨어 연결 실패")
                return False
            print("✅ 하드웨어 연결 성공")
            
            # 2. 추론 파이프라인 초기화
            print("2. 추론 파이프라인 초기화...")
            self.inference_pipeline = RealTimeInferencePipeline(
                model_path=self.model_path,
                confidence_threshold=self.confidence_threshold
            )
            print("✅ 추론 파이프라인 초기화 완료")
            
            # 3. TTS 엔진 초기화
            print("3. TTS 엔진 초기화...")
            self.tts_engine = KoreanTTSEngine()
            self.tts_engine.start_async_speaking()
            print("✅ TTS 엔진 초기화 완료")
            
            return True
            
        except Exception as e:
            print(f"❌ 컴포넌트 초기화 오류: {e}")
            return False
    
    def start_demo(self):
        """데모 시작"""
        if not self.initialize_components():
            return False
        
        print("\n🚀 실시간 수어 인식 데모 시작!")
        print("=" * 60)
        print("사용법:")
        print("- 'q' + Enter: 종료")
        print("- 's' + Enter: 통계 보기")
        print("- 'c' + Enter: 신뢰도 임계값 변경")
        print("- 't' + Enter: TTS 테스트")
        print("=" * 60)
        
        self.is_running = True
        
        # 하드웨어 데이터 콜백 설정
        def sensor_data_callback(sensor_data):
            self.inference_pipeline.add_sensor_data(sensor_data)
        
        # 예측 결과 콜백 설정
        def prediction_callback(result):
            self._handle_prediction_result(result)
        
        # 하드웨어 스트리밍 시작
        self.hardware.start_streaming(sensor_data_callback)
        
        # 추론 파이프라인 시작
        self.inference_pipeline.start_realtime_inference(prediction_callback)
        
        # 메인 루프 및 UI 스레드 시작
        self.demo_thread = threading.Thread(target=self._demo_loop, daemon=True)
        self.demo_thread.start()
        
        # 사용자 입력 처리
        self._handle_user_input()
        
        return True
    
    def _demo_loop(self):
        """데모 메인 루프"""
        last_status_time = time.time()
        status_interval = 2.0  # 2초마다 상태 출력
        
        while self.is_running:
            try:
                current_time = time.time()
                
                # 주기적으로 상태 정보 출력
                if current_time - last_status_time >= status_interval:
                    self._print_status()
                    last_status_time = current_time
                
                # 안정적인 예측 체크
                stable_prediction = self.inference_pipeline.get_stable_prediction(
                    self.stability_window
                )
                
                if stable_prediction and stable_prediction != self.last_stable_prediction:
                    self._handle_stable_prediction(stable_prediction)
                    self.last_stable_prediction = stable_prediction
                
                time.sleep(0.1)  # CPU 사용량 제어
                
            except Exception as e:
                print(f"데모 루프 오류: {e}")
                continue
    
    def _handle_prediction_result(self, result: Dict):
        """개별 예측 결과 처리"""
        self.total_predictions += 1
        self.last_prediction = result
        self.prediction_history.append(result)
        
        # FPS 계산
        current_time = time.time()
        frame_time = current_time - self.last_frame_time
        self.fps_counter.append(1.0 / frame_time if frame_time > 0 else 0)
        self.last_frame_time = current_time
        
        # 높은 신뢰도 예측만 카운트
        if result['confidence'] >= self.confidence_threshold:
            self.successful_predictions += 1
    
    def _handle_stable_prediction(self, stable_prediction: Dict):
        """안정적인 예측 결과 처리"""
        predicted_class = stable_prediction['predicted_class']
        confidence = stable_prediction['confidence']
        stability = stable_prediction['stability']
        
        print(f"\n🎯 안정적인 예측: {predicted_class} "
              f"(신뢰도: {confidence:.2f}, 안정성: {stability:.2f})")
        
        # TTS 출력
        self.tts_engine.speak_prediction_result({
            'predicted_class': predicted_class,
            'confidence': confidence
        })
    
    def _print_status(self):
        """현재 상태 출력"""
        # 성능 통계
        inference_stats = self.inference_pipeline.get_performance_stats()
        hardware_status = self.hardware.get_connection_status()
        
        # FPS 계산
        avg_fps = sum(self.fps_counter) / len(self.fps_counter) if self.fps_counter else 0
        
        print(f"\n📊 실시간 상태 (시간: {time.strftime('%H:%M:%S')})")
        print(f"   하드웨어: {hardware_status['connection_type']} "
              f"(큐: {hardware_status['queue_size']})")
        print(f"   FPS: {avg_fps:.1f} | 추론 시간: {inference_stats.get('avg_inference_time', 0)*1000:.1f}ms")
        print(f"   총 예측: {self.total_predictions} | 성공: {self.successful_predictions}")
        
        if self.last_prediction:
            print(f"   최근 예측: {self.last_prediction['predicted_class']} "
                  f"({self.last_prediction['confidence']:.2f})")
        
        if self.last_stable_prediction:
            print(f"   안정 예측: {self.last_stable_prediction['predicted_class']} "
                  f"({self.last_stable_prediction['confidence']:.2f})")
    
    def _handle_user_input(self):
        """사용자 입력 처리"""
        while self.is_running:
            try:
                user_input = input().strip().lower()
                
                if user_input == 'q':
                    print("데모를 종료합니다...")
                    self.stop_demo()
                    break
                
                elif user_input == 's':
                    self._show_detailed_stats()
                
                elif user_input == 'c':
                    self._change_confidence_threshold()
                
                elif user_input == 't':
                    self._test_tts()
                
                elif user_input == 'h':
                    self._show_help()
                
            except KeyboardInterrupt:
                print("\n키보드 인터럽트 감지. 데모를 종료합니다...")
                self.stop_demo()
                break
            except EOFError:
                break
    
    def _show_detailed_stats(self):
        """상세 통계 표시"""
        print("\n📈 상세 통계")
        print("=" * 50)
        
        # 예측 성공률
        success_rate = (self.successful_predictions / self.total_predictions * 100 
                       if self.total_predictions > 0 else 0)
        print(f"예측 성공률: {success_rate:.1f}% ({self.successful_predictions}/{self.total_predictions})")
        
        # 클래스별 예측 분포
        if self.prediction_history:
            class_counts = {}
            for pred in self.prediction_history:
                if pred['confidence'] >= self.confidence_threshold:
                    class_name = pred['predicted_class']
                    class_counts[class_name] = class_counts.get(class_name, 0) + 1
            
            print("\n클래스별 예측 분포:")
            for class_name, count in sorted(class_counts.items()):
                print(f"  {class_name}: {count}회")
        
        # 성능 통계
        inference_stats = self.inference_pipeline.get_performance_stats()
        if inference_stats:
            print(f"\n성능 통계:")
            print(f"  평균 추론 시간: {inference_stats.get('avg_inference_time', 0)*1000:.2f}ms")
            print(f"  최대 추론 시간: {inference_stats.get('max_inference_time', 0)*1000:.2f}ms")
            print(f"  추론 FPS: {inference_stats.get('fps', 0):.1f}")
        
        print("=" * 50)
    
    def _change_confidence_threshold(self):
        """신뢰도 임계값 변경"""
        try:
            print(f"현재 신뢰도 임계값: {self.confidence_threshold}")
            new_threshold = float(input("새로운 임계값 (0.0-1.0): "))
            
            if 0.0 <= new_threshold <= 1.0:
                self.confidence_threshold = new_threshold
                self.inference_pipeline.set_confidence_threshold(new_threshold)
                print(f"신뢰도 임계값이 {new_threshold}로 변경되었습니다.")
            else:
                print("잘못된 값입니다. 0.0과 1.0 사이의 값을 입력하세요.")
                
        except ValueError:
            print("숫자를 입력하세요.")
    
    def _test_tts(self):
        """TTS 테스트"""
        test_chars = ['ㄱ', 'ㄴ', 'ㄷ', 'ㄹ', 'ㅁ']
        print("TTS 테스트 중...")
        
        for char in test_chars:
            print(f"발음: {char}")
            self.tts_engine.speak(char, blocking=True)
            time.sleep(0.5)
        
        print("TTS 테스트 완료")
    
    def _show_help(self):
        """도움말 표시"""
        print("\n📚 도움말")
        print("=" * 40)
        print("q: 종료")
        print("s: 상세 통계 보기")
        print("c: 신뢰도 임계값 변경")
        print("t: TTS 테스트")
        print("h: 도움말 (이 메시지)")
        print("=" * 40)
    
    def stop_demo(self):
        """데모 중지"""
        print("데모 종료 중...")
        self.is_running = False
        
        # 컴포넌트 정리
        if self.hardware:
            self.hardware.stop_streaming()
            self.hardware.disconnect()
        
        if self.inference_pipeline:
            self.inference_pipeline.stop_realtime_inference()
        
        if self.tts_engine:
            self.tts_engine.stop_async_speaking()
        
        # 스레드 종료 대기
        if self.demo_thread:
            self.demo_thread.join(timeout=2.0)
        
        print("✅ 데모 종료 완료")


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='실시간 한국수어 인식 데모')
    parser.add_argument('--model', type=str, default='best_dl_model.pth',
                       help='학습된 모델 파일 경로')
    parser.add_argument('--connection', type=str, default='simulation',
                       choices=['simulation', 'uart', 'wifi'],
                       help='하드웨어 연결 타입')
    parser.add_argument('--confidence', type=float, default=0.75,
                       help='예측 신뢰도 임계값 (0.0-1.0)')
    parser.add_argument('--stability', type=int, default=5,
                       help='안정성 체크 윈도우 크기')
    
    args = parser.parse_args()
    
    # 데모 시스템 생성 및 실행
    demo = RealTimeSignLanguageDemo(
        model_path=args.model,
        connection_type=args.connection,
        confidence_threshold=args.confidence,
        stability_window=args.stability
    )
    
    try:
        demo.start_demo()
    except KeyboardInterrupt:
        print("\n프로그램이 중단되었습니다.")
    finally:
        demo.stop_demo()


if __name__ == "__main__":
    main()