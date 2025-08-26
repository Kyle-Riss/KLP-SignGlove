# SignGlove 통합 시스템 가이드

**SignGlove_HW 하드웨어와 KLP-SignGlove API 서버를 연결하는 통합 시스템**

## 🎯 개요

이 통합 시스템은 [SignGlove_HW](https://github.com/KNDG01001/SignGlove_HW) 하드웨어에서 실시간으로 센서 데이터를 수집하고, KLP-SignGlove API 서버를 통해 AI 추론을 수행하여 완전한 수화 인식 시스템을 구축합니다.

## 🏗️ 시스템 아키텍처

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   SignGlove_HW  │───▶│  통합 클라이언트  │───▶│  KLP-SignGlove  │
│   하드웨어       │    │  (signglove_     │    │  API 서버       │
│   (아두이노)     │    │   client.py)    │    │  (FastAPI)      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │   실시간 추론    │
                       │   결과 출력      │
                       └─────────────────┘
```

## 📋 요구사항

### 하드웨어 요구사항
- **SignGlove_HW**: 아두이노 + IMU 센서 + Flex 센서
- **펌웨어**: `imu_flex_serial.ino` 업로드 완료
- **연결**: USB 시리얼 연결

### 소프트웨어 요구사항
```bash
# Python 패키지
pip install pyserial requests numpy pandas

# KLP-SignGlove API 서버 실행
python server/main.py
```

## 🚀 빠른 시작

### 1. API 서버 시작
```bash
# KLP-SignGlove 디렉토리에서
python server/main.py
```

### 2. 통합 클라이언트 실행
```bash
# 새로운 터미널에서
python integration/signglove_client.py
```

### 3. 하드웨어 연결
- SignGlove_HW를 USB로 연결
- 자동 포트 감지 또는 수동 포트 지정

## 📊 데이터 흐름

### 1. 하드웨어 데이터 수집
```
SignGlove_HW → 시리얼 통신 → CSV 형식
timestamp,pitch,roll,yaw,flex1,flex2,flex3,flex4,flex5
```

### 2. 데이터 파싱 및 전처리
```python
# 9필드 형식 (기본)
SensorReading(
    timestamp=1234567890,
    pitch=10.5,
    roll=-5.2,
    yaw=15.8,
    flex1=512,
    flex2=678,
    flex3=723,
    flex4=834,
    flex5=567
)

# 12필드 형식 (가속도 포함)
SensorReading(
    timestamp=1234567890,
    pitch=10.5,
    roll=-5.2,
    yaw=15.8,
    accel_x=0.123,
    accel_y=-0.045,
    accel_z=0.987,
    flex1=512,
    flex2=678,
    flex3=723,
    flex4=834,
    flex5=567
)
```

### 3. API 서버 추론
```python
# POST /predict
{
    "timestamp": 1234567890,
    "pitch": 10.5,
    "roll": -5.2,
    "yaw": 15.8,
    "flex1": 512,
    "flex2": 678,
    "flex3": 723,
    "flex4": 834,
    "flex5": 567
}

# 응답
{
    "predicted_class": "ㄱ",
    "confidence": 0.95,
    "all_probabilities": [...],
    "timestamp": 1234567890.123
}
```

## 🔧 고급 설정

### 포트 수동 지정
```python
client = SignGloveIntegratedClient(
    hardware_port="/dev/tty.usbmodem14101",  # macOS
    # hardware_port="COM3",                  # Windows
    api_url="http://localhost:8000"
)
```

### 신뢰도 임계값 조정
```python
client = SignGloveIntegratedClient(
    confidence_threshold=0.8,  # 더 엄격한 필터링
    window_size=30            # 더 큰 윈도우
)
```

### 콜백 함수 커스터마이징
```python
def custom_gesture_handler(gesture: str, confidence: float):
    print(f"🎯 {gesture} (신뢰도: {confidence:.3f})")
    # TTS, LED 제어, 로깅 등

def custom_word_handler(word: str):
    print(f"📝 단어 완성: {word}")
    # 단어 저장, 번역 등

client.set_callbacks(
    on_gesture_detected=custom_gesture_handler,
    on_word_completed=custom_word_handler
)
```

## 📈 성능 최적화

### 1. 실시간 처리 최적화
- **버퍼 크기**: 윈도우 크기에 맞게 조정
- **스레드 관리**: 비동기 데이터 처리
- **메모리 효율성**: 큐 기반 스트리밍

### 2. 안정성 향상
- **연결 재시도**: 자동 재연결 메커니즘
- **오류 처리**: 포괄적인 예외 처리
- **데이터 검증**: 센서 데이터 유효성 검사

### 3. 정확도 개선
- **신뢰도 필터링**: 임계값 기반 필터링
- **안정성 체크**: 연속 예측 일치성 확인
- **노이즈 제거**: 이동 평균 필터링

## 🧪 테스트 및 디버깅

### 1. 하드웨어 연결 테스트
```python
from integration.signglove_client import SignGloveHardwareInterface

hw = SignGloveHardwareInterface()
if hw.connect():
    print("✅ 하드웨어 연결 성공")
    hw.start_reading()
    
    # 10초간 데이터 수집
    import time
    start_time = time.time()
    while time.time() - start_time < 10:
        data = hw.get_latest_data()
        if data:
            print(f"센서: {data.to_dict()}")
        time.sleep(0.1)
    
    hw.disconnect()
else:
    print("❌ 하드웨어 연결 실패")
```

### 2. API 서버 테스트
```python
from integration.signglove_client import SignGloveAPIClient

api = SignGloveAPIClient()
model_info = api.get_model_info()
print(f"모델: {model_info}")

# 테스트 예측
test_data = SensorReading(
    timestamp=time.time(),
    pitch=0.0, roll=0.0, yaw=0.0,
    flex1=700, flex2=700, flex3=700, flex4=700, flex5=700
)

result = api.predict_gesture(test_data)
print(f"예측 결과: {result}")
```

### 3. 통합 시스템 테스트
```python
from integration.signglove_client import SignGloveIntegratedClient

def test_callback(gesture, confidence):
    print(f"테스트 콜백: {gesture} ({confidence:.3f})")

client = SignGloveIntegratedClient()
client.set_callbacks(on_gesture_detected=test_callback)

if client.connect():
    client.start()
    time.sleep(30)  # 30초 테스트
    client.disconnect()
```

## 🔍 문제 해결

### 일반적인 문제들

#### 1. 하드웨어 연결 실패
```bash
# 포트 확인
ls /dev/tty.*  # macOS/Linux
# 또는
mode           # Windows

# 권한 문제 (Linux)
sudo chmod 666 /dev/ttyUSB0
```

#### 2. API 서버 연결 실패
```bash
# 서버 상태 확인
curl http://localhost:8000/health

# 포트 확인
netstat -an | grep 8000
```

#### 3. 데이터 파싱 오류
- CSV 형식 확인
- 헤더 라인 처리
- 필드 수 검증

#### 4. 추론 성능 저하
- 신뢰도 임계값 조정
- 윈도우 크기 최적화
- 모델 재학습 고려

## 📊 모니터링 및 로깅

### 실시간 상태 모니터링
```python
while True:
    status = client.get_status()
    print(f"하드웨어: {status['hardware_connected']}")
    print(f"API: {status['api_connected']}")
    print(f"버퍼: {status['buffer_size']}")
    print(f"현재 단어: {status['current_word']}")
    time.sleep(1)
```

### 로그 레벨 조정
```python
import logging
logging.basicConfig(level=logging.DEBUG)  # 상세 로그
```

## 🔮 향후 개선 계획

### 1. 웹 인터페이스
- 실시간 시각화 대시보드
- 설정 관리 UI
- 성능 모니터링

### 2. 클라우드 연동
- 원격 모니터링
- 데이터 백업
- 다중 사용자 지원

### 3. 고급 기능
- 음성 합성 (TTS)
- 번역 서비스 연동
- 학습 데이터 자동 수집

## 📚 참고 자료

- [SignGlove_HW GitHub](https://github.com/KNDG01001/SignGlove_HW)
- [KLP-SignGlove API 문서](http://localhost:8000/docs)
- [시리얼 통신 가이드](https://pyserial.readthedocs.io/)
- [FastAPI 문서](https://fastapi.tiangolo.com/)

---

**🤟 SignGlove 통합 시스템 - 하드웨어와 AI의 완벽한 조화**
