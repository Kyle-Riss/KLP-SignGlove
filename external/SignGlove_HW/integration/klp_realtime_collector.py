#!/usr/bin/env python3
"""
KLP-SignGlove 실시간 하드웨어 연동 수집기
Arduino 센서 데이터를 KLP-SignGlove API 서버로 실시간 전송
"""

import sys
import time
import serial
import threading
import numpy as np
import requests
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from collections import deque
from dataclasses import dataclass, asdict
import queue
import termios
import tty

# KLP-SignGlove API 설정
KLP_API_URL = "http://localhost:8000"
KLP_API_TOKEN = "demo_token_123"
KLP_HEADERS = {
    "Authorization": f"Bearer {KLP_API_TOKEN}",
    "Content-Type": "application/json"
}

# 데이터 버퍼링 설정
BUFFER_SIZE = 300  # KLP-SignGlove 모델이 요구하는 샘플 수
SAMPLING_RATE = 40  # Arduino 샘플링 레이트 (Hz)

@dataclass
class SignGloveSensorReading:
    """SignGlove 센서 읽기 데이터 구조"""
    timestamp_ms: int           # 아두이노 millis() 타임스탬프
    recv_timestamp_ms: int      # PC 수신 타임스탬프
    
    # IMU 데이터 (자이로스코프 - 오일러 각도)
    pitch: float               # Y축 회전 (도)
    roll: float                # X축 회전 (도) 
    yaw: float                 # Z축 회전 (도)
    
    # 플렉스 센서 데이터 (ADC 값)
    flex1: int                 # 엄지 (0-1023)
    flex2: int                 # 검지 (0-1023)
    flex3: int                 # 중지 (0-1023)
    flex4: int                 # 약지 (0-1023)
    flex5: int                 # 소지 (0-1023)
    
    # 계산된 Hz (실제 측정 주기)
    sampling_hz: float
    
    # 가속도 데이터 (IMU에서 실제 측정)
    accel_x: float         # X축 가속도 (g)
    accel_y: float         # Y축 가속도 (g)
    accel_z: float         # Z축 가속도 (g)

class KLP_SignGlove_RealtimeCollector:
    """KLP-SignGlove 실시간 하드웨어 연동 수집기"""
    
    def __init__(self):
        print("🚀 KLP-SignGlove 실시간 하드웨어 연동 수집기 초기화 중...")
        
        # 시리얼 통신 설정
        self.serial_port = None
        self.serial_connected = False
        
        # 데이터 버퍼링 (300 샘플)
        self.data_buffer = deque(maxlen=BUFFER_SIZE)
        self.buffer_lock = threading.Lock()
        
        # 실시간 추론 설정
        self.realtime_mode = False
        self.last_prediction_time = 0
        self.prediction_interval = 1.0  # 1초마다 추론
        
        # 통계
        self.total_samples = 0
        self.total_predictions = 0
        self.avg_processing_time = 0.0
        
        # KLP-SignGlove API 연결 테스트
        self.test_api_connection()
    
    def test_api_connection(self):
        """KLP-SignGlove API 서버 연결 테스트"""
        try:
            response = requests.get(f"{KLP_API_URL}/health", timeout=5)
            if response.status_code == 200:
                data = response.json()
                print(f"✅ KLP-SignGlove API 서버 연결 성공!")
                print(f"   서버 상태: {data.get('status', 'Unknown')}")
                print(f"   모델 로드: {data.get('model_loaded', False)}")
                print(f"   GPU 사용: {data.get('gpu_available', False)}")
                return True
            else:
                print(f"❌ API 서버 응답 오류: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ KLP-SignGlove API 서버 연결 실패: {e}")
            print(f"   서버가 실행 중인지 확인하세요: {KLP_API_URL}")
            return False
    
    def connect_arduino(self, port: str = None, baudrate: int = 115200):
        """Arduino 연결"""
        if port is None:
            # 자동 포트 감지
            import glob
            if sys.platform.startswith('win'):
                ports = glob.glob('COM*')
            else:
                ports = glob.glob('/dev/ttyUSB*') + glob.glob('/dev/ttyACM*')
            
            if not ports:
                print("❌ 사용 가능한 시리얼 포트가 없습니다.")
                return False
            
            port = ports[0]
            print(f"🔍 자동 감지된 포트: {port}")
        
        try:
            self.serial_port = serial.Serial(port, baudrate, timeout=1)
            self.serial_connected = True
            print(f"✅ Arduino 연결 성공: {port}")
            
            # 연결 직후 초기화
            time.sleep(2)
            self.serial_port.flushInput()
            
            return True
            
        except Exception as e:
            print(f"❌ Arduino 연결 실패: {e}")
            return False
    
    def disconnect_arduino(self):
        """Arduino 연결 해제"""
        if self.serial_port and self.serial_port.is_open:
            self.serial_port.close()
            self.serial_connected = False
            print("🔌 Arduino 연결 해제됨")
    
    def parse_arduino_data(self, line: str) -> Optional[SignGloveSensorReading]:
        """Arduino CSV 라인 파싱"""
        try:
            # CSV 형식: timestamp,pitch,roll,yaw,accel_x,accel_y,accel_z,flex1,flex2,flex3,flex4,flex5
            parts = line.strip().split(',')
            if len(parts) != 12:
                return None
            
            # 데이터 파싱
            timestamp_ms = int(parts[0])
            pitch = float(parts[1])
            roll = float(parts[2])
            yaw = float(parts[3])
            accel_x = float(parts[4])
            accel_y = float(parts[5])
            accel_z = float(parts[6])
            flex1 = int(parts[7])
            flex2 = int(parts[8])
            flex3 = int(parts[9])
            flex4 = int(parts[10])
            flex5 = int(parts[11])
            
            # 샘플링 레이트 계산
            if len(self.data_buffer) > 0:
                last_timestamp = self.data_buffer[-1].timestamp_ms
                time_diff = timestamp_ms - last_timestamp
                sampling_hz = 1000.0 / time_diff if time_diff > 0 else 0
            else:
                sampling_hz = SAMPLING_RATE
            
            return SignGloveSensorReading(
                timestamp_ms=timestamp_ms,
                recv_timestamp_ms=int(time.time() * 1000),
                pitch=pitch, roll=roll, yaw=yaw,
                flex1=flex1, flex2=flex2, flex3=flex3, flex4=flex4, flex5=flex5,
                sampling_hz=sampling_hz,
                accel_x=accel_x, accel_y=accel_y, accel_z=accel_z
            )
            
        except Exception as e:
            print(f"❌ 데이터 파싱 오류: {e}")
            return None
    
    def convert_to_klp_format(self, sensor_reading: SignGloveSensorReading) -> List[float]:
        """KLP-SignGlove 형식으로 데이터 변환"""
        # KLP-SignGlove 형식: [flex1, flex2, flex3, flex4, flex5, IMU_X, IMU_Y, IMU_Z]
        # Arduino 데이터를 정규화하여 변환
        
        # 플렉스 센서 정규화 (0-1023 → -1 to 1)
        flex1_norm = (sensor_reading.flex1 - 512) / 512.0
        flex2_norm = (sensor_reading.flex2 - 512) / 512.0
        flex3_norm = (sensor_reading.flex3 - 512) / 512.0
        flex4_norm = (sensor_reading.flex4 - 512) / 512.0
        flex5_norm = (sensor_reading.flex5 - 512) / 512.0
        
        # IMU 데이터 정규화 (각도 → -1 to 1)
        imu_x_norm = sensor_reading.pitch / 180.0
        imu_y_norm = sensor_reading.roll / 180.0
        imu_z_norm = sensor_reading.yaw / 180.0
        
        return [flex1_norm, flex2_norm, flex3_norm, flex4_norm, flex5_norm, 
                imu_x_norm, imu_y_norm, imu_z_norm]
    
    def add_to_buffer(self, sensor_reading: SignGloveSensorReading):
        """데이터 버퍼에 추가"""
        with self.buffer_lock:
            # KLP-SignGlove 형식으로 변환하여 버퍼에 추가
            klp_data = self.convert_to_klp_format(sensor_reading)
            self.data_buffer.append(klp_data)
            self.total_samples += 1
    
    def get_buffer_data(self) -> List[List[float]]:
        """버퍼 데이터 가져오기 (300x8 배열)"""
        with self.buffer_lock:
            if len(self.data_buffer) < BUFFER_SIZE:
                # 부족한 샘플은 마지막 샘플로 패딩
                if len(self.data_buffer) > 0:
                    last_sample = self.data_buffer[-1]
                    padding_samples = [last_sample] * (BUFFER_SIZE - len(self.data_buffer))
                    return list(self.data_buffer) + padding_samples
                else:
                    return []
            else:
                return list(self.data_buffer)
    
    def send_to_klp_server(self, sensor_data: List[List[float]]) -> Optional[Dict]:
        """KLP-SignGlove 서버로 데이터 전송"""
        try:
            request_data = {
                "sensor_data": sensor_data,
                "class_names": None
            }
            
            start_time = time.time()
            response = requests.post(
                f"{KLP_API_URL}/predict",
                json=request_data,
                headers=KLP_HEADERS,
                timeout=10
            )
            
            processing_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                self.total_predictions += 1
                self.avg_processing_time = (
                    (self.avg_processing_time * (self.total_predictions - 1) + processing_time) 
                    / self.total_predictions
                )
                
                return {
                    "success": True,
                    "predicted_class": result.get("predicted_class", "Unknown"),
                    "confidence": result.get("confidence", 0.0),
                    "processing_time": processing_time,
                    "server_time": result.get("processing_time", 0.0)
                }
            else:
                print(f"❌ API 서버 오류: {response.status_code}")
                return {"success": False, "error": f"HTTP {response.status_code}"}
                
        except Exception as e:
            print(f"❌ API 호출 오류: {e}")
            return {"success": False, "error": str(e)}
    
    def start_checkpoint_mode(self):
        """체크포인트 모드 시작"""
        if not self.serial_connected:
            print("❌ Arduino가 연결되지 않았습니다.")
            return
        
        print("🎯 체크포인트 모드 시작...")
        print("   - 300 샘플 수집 후 수동 추론")
        print("   - Enter 키로 추론 실행")
        print("   - Ctrl+C로 종료")
        
        self.realtime_mode = True
        
        try:
            while self.realtime_mode:
                if self.serial_port.in_waiting > 0:
                    line = self.serial_port.readline().decode('utf-8').strip()
                    if line:
                        # 데이터 파싱
                        sensor_reading = self.parse_arduino_data(line)
                        if sensor_reading:
                            # 버퍼에 추가
                            self.add_to_buffer(sensor_reading)
                            
                            # 버퍼 상태 표시
                            if self.total_samples % 50 == 0:
                                buffer_fill = len(self.data_buffer) / BUFFER_SIZE * 100
                                print(f"📊 버퍼: {buffer_fill:.1f}% ({len(self.data_buffer)}/{BUFFER_SIZE})")
                            
                            # 300 샘플이 모이면 추론 준비 완료
                            if len(self.data_buffer) >= BUFFER_SIZE:
                                print(f"✅ 추론 준비 완료! Enter 키를 눌러 수화 인식을 시작하세요.")
                                break
                
                time.sleep(0.01)  # 10ms 대기
            
            # 체크포인트 추론
            while self.realtime_mode:
                try:
                    input("Enter 키를 눌러 수화 인식 실행...")
                    
                    if len(self.data_buffer) >= BUFFER_SIZE:
                        buffer_data = self.get_buffer_data()
                        result = self.send_to_klp_server(buffer_data)
                        
                        if result and result.get("success"):
                            print(f"🤟 수화 인식 결과: {result['predicted_class']} "
                                  f"(신뢰도: {result['confidence']:.3f})")
                        else:
                            print(f"❌ 추론 실패: {result.get('error', 'Unknown error')}")
                        
                        # 버퍼 초기화 (새로운 체크포인트 준비)
                        with self.buffer_lock:
                            self.data_buffer.clear()
                        print("🔄 버퍼 초기화됨. 새로운 체크포인트 준비 중...")
                    else:
                        print(f"❌ 버퍼 부족: {len(self.data_buffer)}/{BUFFER_SIZE}")
                        
                except KeyboardInterrupt:
                    break
                    
        except KeyboardInterrupt:
            print("\n⏹️ 체크포인트 모드 종료")
            self.realtime_mode = False
    
    def print_status(self):
        """현재 상태 출력"""
        print("\n📊 KLP-SignGlove 실시간 수집기 상태")
        print("=" * 50)
        print(f"Arduino 연결: {'✅' if self.serial_connected else '❌'}")
        print(f"API 서버 연결: {'✅' if self.test_api_connection() else '❌'}")
        print(f"실시간 모드: {'✅' if self.realtime_mode else '❌'}")
        print(f"버퍼 상태: {len(self.data_buffer)}/{BUFFER_SIZE} ({len(self.data_buffer)/BUFFER_SIZE*100:.1f}%)")
        print(f"총 샘플: {self.total_samples}")
        print(f"총 추론: {self.total_predictions}")
        print(f"평균 처리 시간: {self.avg_processing_time:.3f}s")
        print("=" * 50)
    
    def run_interactive(self):
        """대화형 모드 실행"""
        print("🎮 KLP-SignGlove 실시간 수집기 대화형 모드")
        print("=" * 50)
        print("명령어:")
        print("  C: Arduino 연결/재연결")
        print("  R: 실시간 모드 시작")
        print("  S: 상태 확인")
        print("  Q: 종료")
        print("=" * 50)
        
        while True:
            try:
                if sys.platform == 'win32':
                    if msvcrt.kbhit():
                        key = msvcrt.getch().decode('utf-8').upper()
                    else:
                        time.sleep(0.1)
                        continue
                else:
                    # Linux/Mac 키보드 입력
                    import select
                    if select.select([sys.stdin], [], [], 0.1)[0]:
                        key = sys.stdin.read(1).upper()
                    else:
                        continue
                
                if key == 'C':
                    port = input("포트 입력 (Enter로 자동 감지): ").strip() or None
                    if self.connect_arduino(port):
                        print("✅ Arduino 연결 성공!")
                    else:
                        print("❌ Arduino 연결 실패!")
                
                elif key == 'R':
                    if self.serial_connected:
                        self.start_realtime_mode()
                    else:
                        print("❌ 먼저 Arduino를 연결하세요 (C 키)")
                
                elif key == 'S':
                    self.print_status()
                
                elif key == 'Q':
                    print("👋 프로그램 종료")
                    break
                    
            except KeyboardInterrupt:
                print("\n👋 프로그램 종료")
                break
        
        self.disconnect_arduino()

def main():
    """메인 함수"""
    collector = KLP_SignGlove_RealtimeCollector()
    
    if len(sys.argv) > 1:
        # 명령행 인수로 포트 지정
        port = sys.argv[1]
        if collector.connect_arduino(port):
            collector.start_realtime_mode()
        else:
            print("❌ Arduino 연결 실패")
    else:
        # 대화형 모드
        collector.run_interactive()

if __name__ == "__main__":
    main()
