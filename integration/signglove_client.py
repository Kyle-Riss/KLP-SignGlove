#!/usr/bin/env python3
"""
SignGlove 통합 클라이언트
SignGlove_HW 하드웨어와 KLP-SignGlove API 서버 연결
"""

import serial
import time
import json
import requests
import threading
import queue
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import deque
import logging
import sys
import os

# 크로스 플랫폼 유틸리티 임포트
try:
    from .platform_utils import PlatformUtils
except ImportError:
    from platform_utils import PlatformUtils

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class SensorReading:
    """센서 데이터 구조체"""
    timestamp: float
    pitch: float
    roll: float
    yaw: float
    flex1: float
    flex2: float
    flex3: float
    flex4: float
    flex5: float
    accel_x: Optional[float] = None
    accel_y: Optional[float] = None
    accel_z: Optional[float] = None
    
    def to_dict(self) -> Dict:
        """API 요청용 딕셔너리 변환"""
        return {
            'timestamp': self.timestamp,
            'pitch': self.pitch,
            'roll': self.roll,
            'yaw': self.yaw,
            'flex1': self.flex1,
            'flex2': self.flex2,
            'flex3': self.flex3,
            'flex4': self.flex4,
            'flex5': self.flex5
        }
    
    def to_array(self) -> np.ndarray:
        """numpy 배열 변환"""
        return np.array([self.pitch, self.roll, self.yaw, 
                        self.flex1, self.flex2, self.flex3, 
                        self.flex4, self.flex5])

class SignGloveHardwareInterface:
    """SignGlove 하드웨어 인터페이스"""
    
    def __init__(self, port: str = None, baudrate: int = 115200):
        self.port = port
        self.baudrate = baudrate
        self.serial_conn = None
        self.is_connected = False
        self.data_queue = queue.Queue()
        self.is_running = False
        
        # 자동 포트 감지
        if port is None:
            self.port = self._auto_detect_port()
    
    def _auto_detect_port(self) -> str:
        """자동으로 아두이노 포트 감지 (크로스 플랫폼)"""
        # PlatformUtils 사용
        return PlatformUtils.find_arduino_port()
    
    def connect(self) -> bool:
        """하드웨어 연결 (크로스 플랫폼)"""
        try:
            if self.port is None:
                logger.error("연결할 포트가 없습니다.")
                return False
            
            # PlatformUtils를 사용한 플랫폼별 설정
            serial_config = PlatformUtils.get_serial_config()
            serial_config.update({
                'port': self.port,
                'baudrate': self.baudrate
            })
            
            self.serial_conn = serial.Serial(**serial_config)
            
            # 연결 확인
            time.sleep(2)
            if self.serial_conn.is_open:
                self.is_connected = True
                logger.info(f"하드웨어 연결 성공: {self.port}")
                
                # 연결 후 초기화 대기
                time.sleep(1)
                
                # 연결 테스트 (데이터 읽기 시도)
                try:
                    self.serial_conn.reset_input_buffer()
                    self.serial_conn.reset_output_buffer()
                    logger.info("시리얼 버퍼 초기화 완료")
                except Exception as e:
                    logger.warning(f"버퍼 초기화 실패: {e}")
                
                return True
            else:
                logger.error("하드웨어 연결 실패")
                return False
                
        except Exception as e:
            logger.error(f"하드웨어 연결 오류: {e}")
            return False
    
    def disconnect(self):
        """하드웨어 연결 해제"""
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()
        self.is_connected = False
        self.is_running = False
        logger.info("하드웨어 연결 해제")
    
    def _parse_sensor_data(self, line: str) -> Optional[SensorReading]:
        """센서 데이터 파싱"""
        try:
            # 헤더 라인 스킵
            if line.startswith('timestamp') or line.startswith('#'):
                return None
            
            # CSV 파싱
            parts = line.strip().split(',')
            
            # 9필드 형식 (기본)
            if len(parts) >= 9:
                return SensorReading(
                    timestamp=float(parts[0]),
                    pitch=float(parts[1]),
                    roll=float(parts[2]),
                    yaw=float(parts[3]),
                    flex1=float(parts[4]),
                    flex2=float(parts[5]),
                    flex3=float(parts[6]),
                    flex4=float(parts[7]),
                    flex5=float(parts[8])
                )
            
            # 12필드 형식 (가속도 포함)
            elif len(parts) >= 12:
                return SensorReading(
                    timestamp=float(parts[0]),
                    pitch=float(parts[1]),
                    roll=float(parts[2]),
                    yaw=float(parts[3]),
                    accel_x=float(parts[4]),
                    accel_y=float(parts[5]),
                    accel_z=float(parts[6]),
                    flex1=float(parts[7]),
                    flex2=float(parts[8]),
                    flex3=float(parts[9]),
                    flex4=float(parts[10]),
                    flex5=float(parts[11])
                )
            
            return None
            
        except Exception as e:
            logger.warning(f"센서 데이터 파싱 오류: {e} - {line}")
            return None
    
    def _read_sensor_data(self):
        """센서 데이터 읽기 스레드 (크로스 플랫폼)"""
        while self.is_running and self.is_connected:
            try:
                # 플랫폼별 데이터 읽기 최적화
                if PlatformUtils.is_windows():
                    # Windows: 더 안정적인 읽기 방식
                    if self.serial_conn.in_waiting > 0:
                        line = self.serial_conn.readline()
                        try:
                            line_str = line.decode('utf-8', errors='ignore').strip()
                            if line_str:
                                sensor_data = self._parse_sensor_data(line_str)
                                if sensor_data:
                                    self.data_queue.put(sensor_data)
                        except UnicodeDecodeError:
                            logger.debug("인코딩 오류 무시")
                    else:
                        time.sleep(0.01)  # 10ms 대기
                else:
                    # macOS/Linux: 기존 방식
                    if self.serial_conn.in_waiting > 0:
                        line = self.serial_conn.readline().decode('utf-8', errors='ignore').strip()
                        if line:
                            sensor_data = self._parse_sensor_data(line)
                            if sensor_data:
                                self.data_queue.put(sensor_data)
                    else:
                        time.sleep(0.01)  # 10ms 대기
                    
            except Exception as e:
                logger.error(f"센서 데이터 읽기 오류: {e}")
                time.sleep(0.1)
    
    def start_reading(self):
        """센서 데이터 읽기 시작"""
        if not self.is_connected:
            logger.error("하드웨어가 연결되지 않았습니다.")
            return False
        
        self.is_running = True
        self.read_thread = threading.Thread(target=self._read_sensor_data, daemon=True)
        self.read_thread.start()
        logger.info("센서 데이터 읽기 시작")
        return True
    
    def stop_reading(self):
        """센서 데이터 읽기 중지"""
        self.is_running = False
        logger.info("센서 데이터 읽기 중지")
    
    def get_latest_data(self) -> Optional[SensorReading]:
        """최신 센서 데이터 반환"""
        try:
            return self.data_queue.get_nowait()
        except queue.Empty:
            return None

class SignGloveAPIClient:
    """KLP-SignGlove API 클라이언트"""
    
    def __init__(self, api_url: str = "http://localhost:8000"):
        self.api_url = api_url
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'User-Agent': 'SignGlove-Client/1.0'
        })
        
        # API 상태 확인
        self._check_api_status()
    
    def _check_api_status(self):
        """API 서버 상태 확인"""
        try:
            response = self.session.get(f"{self.api_url}/health")
            if response.status_code == 200:
                logger.info("API 서버 연결 성공")
                return True
            else:
                logger.warning(f"API 서버 응답 오류: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"API 서버 연결 실패: {e}")
            return False
    
    def get_model_info(self) -> Dict:
        """모델 정보 조회"""
        try:
            response = self.session.get(f"{self.api_url}/model/info")
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"모델 정보 조회 실패: {response.status_code}")
                return {}
        except Exception as e:
            logger.error(f"모델 정보 조회 오류: {e}")
            return {}
    
    def predict_gesture(self, sensor_data: SensorReading) -> Dict:
        """제스처 예측"""
        try:
            payload = sensor_data.to_dict()
            response = self.session.post(f"{self.api_url}/predict", json=payload)
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"예측 요청 실패: {response.status_code}")
                return {'error': f'HTTP {response.status_code}'}
                
        except Exception as e:
            logger.error(f"예측 요청 오류: {e}")
            return {'error': str(e)}
    
    def predict_word(self, sensor_data: SensorReading) -> Dict:
        """단어 예측"""
        try:
            payload = sensor_data.to_dict()
            response = self.session.post(f"{self.api_url}/predict/word", json=payload)
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"단어 예측 요청 실패: {response.status_code}")
                return {'error': f'HTTP {response.status_code}'}
                
        except Exception as e:
            logger.error(f"단어 예측 요청 오류: {e}")
            return {'error': str(e)}

class SignGloveIntegratedClient:
    """SignGlove 통합 클라이언트"""
    
    def __init__(self, 
                 hardware_port: str = None,
                 api_url: str = "http://localhost:8000",
                 window_size: int = 20,
                 confidence_threshold: float = 0.7):
        
        # 하드웨어 인터페이스
        self.hardware = SignGloveHardwareInterface(port=hardware_port)
        
        # API 클라이언트
        self.api_client = SignGloveAPIClient(api_url)
        
        # 설정
        self.window_size = window_size
        self.confidence_threshold = confidence_threshold
        
        # 데이터 버퍼
        self.data_buffer = deque(maxlen=window_size * 2)
        self.prediction_history = deque(maxlen=5)
        
        # 상태
        self.is_running = False
        self.current_word = ""
        self.word_buffer = []
        
        # 콜백 함수들
        self.on_gesture_detected = None
        self.on_word_completed = None
        self.on_error = None
        
        logger.info("SignGlove 통합 클라이언트 초기화 완료")
    
    def set_callbacks(self, 
                     on_gesture_detected=None,
                     on_word_completed=None,
                     on_error=None):
        """콜백 함수 설정"""
        self.on_gesture_detected = on_gesture_detected
        self.on_word_completed = on_word_completed
        self.on_error = on_error
    
    def connect(self) -> bool:
        """하드웨어 및 API 연결"""
        # 하드웨어 연결
        if not self.hardware.connect():
            if self.on_error:
                self.on_error("하드웨어 연결 실패")
            return False
        
        # API 서버 상태 확인
        if not self.api_client._check_api_status():
            if self.on_error:
                self.on_error("API 서버 연결 실패")
            return False
        
        # 모델 정보 출력
        model_info = self.api_client.get_model_info()
        if model_info:
            logger.info(f"모델 정보: {model_info.get('model_name', 'Unknown')}")
            logger.info(f"정확도: {model_info.get('accuracy', 0):.2%}")
        
        return True
    
    def disconnect(self):
        """연결 해제"""
        self.stop()
        self.hardware.disconnect()
        logger.info("통합 클라이언트 연결 해제")
    
    def start(self):
        """실시간 추론 시작"""
        if not self.hardware.start_reading():
            if self.on_error:
                self.on_error("센서 데이터 읽기 시작 실패")
            return False
        
        self.is_running = True
        self.inference_thread = threading.Thread(target=self._inference_loop, daemon=True)
        self.inference_thread.start()
        
        logger.info("실시간 추론 시작")
        return True
    
    def stop(self):
        """실시간 추론 중지"""
        self.is_running = False
        self.hardware.stop_reading()
        logger.info("실시간 추론 중지")
    
    def _inference_loop(self):
        """추론 루프"""
        while self.is_running:
            try:
                # 센서 데이터 가져오기
                sensor_data = self.hardware.get_latest_data()
                if sensor_data is None:
                    time.sleep(0.01)
                    continue
                
                # 데이터 버퍼에 추가
                self.data_buffer.append(sensor_data.to_array())
                
                # 윈도우가 충분히 채워지면 추론
                if len(self.data_buffer) >= self.window_size:
                    # 제스처 예측
                    result = self.api_client.predict_gesture(sensor_data)
                    
                    if 'error' not in result:
                        self._process_gesture_result(result)
                    else:
                        if self.on_error:
                            self.on_error(f"추론 오류: {result['error']}")
                
                time.sleep(0.01)  # 10ms 간격
                
            except Exception as e:
                logger.error(f"추론 루프 오류: {e}")
                if self.on_error:
                    self.on_error(f"추론 루프 오류: {e}")
                time.sleep(0.1)
    
    def _process_gesture_result(self, result: Dict):
        """제스처 결과 처리"""
        predicted_class = result.get('predicted_class', '')
        confidence = result.get('confidence', 0.0)
        
        # 신뢰도 임계값 체크
        if confidence < self.confidence_threshold:
            return
        
        # 예측 히스토리에 추가
        self.prediction_history.append({
            'class': predicted_class,
            'confidence': confidence,
            'timestamp': time.time()
        })
        
        # 안정성 체크 (최근 3개가 같은지)
        if len(self.prediction_history) >= 3:
            recent_predictions = list(self.prediction_history)[-3:]
            recent_classes = [p['class'] for p in recent_predictions]
            
            if len(set(recent_classes)) == 1:  # 모두 같은 클래스
                # 콜백 호출
                if self.on_gesture_detected:
                    self.on_gesture_detected(predicted_class, confidence)
                
                # 단어 버퍼에 추가
                self.word_buffer.append(predicted_class)
                
                # 단어 완성 체크 (5개 글자마다)
                if len(self.word_buffer) >= 5:
                    word = ''.join(self.word_buffer)
                    if self.on_word_completed:
                        self.on_word_completed(word)
                    self.word_buffer = []
    
    def get_status(self) -> Dict:
        """현재 상태 반환"""
        return {
            'hardware_connected': self.hardware.is_connected,
            'api_connected': self.api_client._check_api_status(),
            'is_running': self.is_running,
            'buffer_size': len(self.data_buffer),
            'current_word': ''.join(self.word_buffer),
            'prediction_history': list(self.prediction_history)
        }

def main():
    """메인 실행 함수 - 데모"""
    print("🚀 SignGlove 통합 클라이언트 데모")
    print("="*50)
    
    # 콜백 함수들
    def on_gesture_detected(gesture: str, confidence: float):
        print(f"🎯 감지된 제스처: {gesture} (신뢰도: {confidence:.3f})")
    
    def on_word_completed(word: str):
        print(f"📝 완성된 단어: {word}")
    
    def on_error(error_msg: str):
        print(f"❌ 오류: {error_msg}")
    
    # 통합 클라이언트 생성
    client = SignGloveIntegratedClient(
        hardware_port=None,  # 자동 감지
        api_url="http://localhost:8000",
        confidence_threshold=0.7
    )
    
    # 콜백 설정
    client.set_callbacks(
        on_gesture_detected=on_gesture_detected,
        on_word_completed=on_word_completed,
        on_error=on_error
    )
    
    # 연결
    if not client.connect():
        print("❌ 연결 실패")
        return
    
    # 시작
    if not client.start():
        print("❌ 시작 실패")
        return
    
    print("✅ 실시간 추론 시작됨")
    print("🔄 센서 데이터를 기다리는 중... (Ctrl+C로 종료)")
    
    try:
        while True:
            # 상태 출력
            status = client.get_status()
            if status['buffer_size'] > 0:
                print(f"\r📊 버퍼: {status['buffer_size']} | 현재 단어: {status['current_word']}", end='')
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n🛑 사용자에 의해 중지됨")
    
    finally:
        client.disconnect()
        print("✅ 정상 종료")

if __name__ == "__main__":
    main()
