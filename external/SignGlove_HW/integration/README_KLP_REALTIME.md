# KLP-SignGlove 실시간 하드웨어 연동 수집기

## 🎯 개요

**SignGlove_HW + KLP-SignGlove 통합 시스템**
Arduino 센서 데이터를 KLP-SignGlove API 서버로 실시간 전송하여 수화 인식을 수행합니다.

## 🔧 하드웨어 요구사항

- **Arduino Nano 33 IoT** (LSM6DS3 IMU 내장)
- **플렉스 센서 5개** (A0, A1, A2, A3, A6 핀 연결)
- **USB 시리얼 연결** (115200 baud)

## 📋 소프트웨어 요구사항

### 1. KLP-SignGlove API 서버 실행
```bash
# KLP-SignGlove-Clean 폴더에서
cd api
python3 api_server.py
```

### 2. 의존성 설치
```bash
# SignGlove_HW/integration 폴더에서
pip install -r requirements_klp_realtime.txt
```

## 🚀 사용법

### 1. 기본 실행 (대화형 모드)
```bash
python3 klp_realtime_collector.py
```

### 2. 특정 포트로 직접 실행
```bash
python3 klp_realtime_collector.py /dev/ttyACM0
```

### 3. 대화형 모드 명령어
```
C: Arduino 연결/재연결
R: 실시간 모드 시작
S: 상태 확인
Q: 종료
```

## 🔄 데이터 흐름

```
🤖 Arduino Nano 33 IoT
    ↓ USB Serial (115200 baud)
    ↓ CSV 데이터 스트림
💻 KLP_SignGlove_RealtimeCollector
    ↓ 데이터 버퍼링 (300 샘플)
    ↓ KLP-SignGlove 형식 변환
    ↓ HTTP POST 요청
🌐 KLP-SignGlove API Server
    ↓ RGRU 모델 추론
    ↓ 실시간 결과 반환
📱 실시간 수화 인식 결과
```

## 📊 데이터 형식 변환

### Arduino CSV 형식 (입력)
```csv
timestamp,pitch,roll,yaw,accel_x,accel_y,accel_z,flex1,flex2,flex3,flex4,flex5
12345,15.2,-8.5,45.0,0.1,-0.2,9.8,512,678,445,567,489
```

### KLP-SignGlove 형식 (출력)
```python
# 8채널 정규화된 데이터
[flex1_norm, flex2_norm, flex3_norm, flex4_norm, flex5_norm, 
 imu_x_norm, imu_y_norm, imu_z_norm]

# 정규화 방법
flex_norm = (flex_raw - 512) / 512.0      # 0-1023 → -1 to 1
imu_norm = angle_degrees / 180.0           # 각도 → -1 to 1
```

## ⚙️ 설정 옵션

### 버퍼링 설정
```python
BUFFER_SIZE = 300        # KLP-SignGlove 모델 요구 샘플 수
SAMPLING_RATE = 40       # Arduino 샘플링 레이트 (Hz)
PREDICTION_INTERVAL = 1.0 # 추론 간격 (초)
```

### API 설정
```python
KLP_API_URL = "http://localhost:8000"     # API 서버 주소
KLP_API_TOKEN = "demo_token_123"          # 인증 토큰
```

## 📈 실시간 모니터링

### 버퍼 상태
```
📊 버퍼: 45.3% (136/300)
📊 버퍼: 67.7% (203/300)
📊 버퍼: 100.0% (300/300)
```

### 추론 결과
```
🤟 수화 인식: ㄱ (신뢰도: 0.956) [0.023s]
🤟 수화 인식: ㅏ (신뢰도: 0.892) [0.018s]
🤟 수화 인식: ㄴ (신뢰도: 0.978) [0.021s]
```

## 🔍 문제 해결

### 1. Arduino 연결 실패
```bash
# 포트 확인
ls /dev/ttyUSB* /dev/ttyACM*

# 권한 문제
sudo chmod 666 /dev/ttyACM0
```

### 2. API 서버 연결 실패
```bash
# KLP-SignGlove API 서버 실행 확인
curl http://localhost:8000/health

# 포트 충돌 확인
netstat -tulpn | grep :8000
```

### 3. 데이터 파싱 오류
- Arduino 펌웨어 확인 (imu_flex_serial.ino)
- CSV 형식 검증
- 시리얼 통신 속도 확인 (115200 baud)

## 📁 파일 구조

```
SignGlove_HW/integration/
├── klp_realtime_collector.py      # 메인 수집기
├── requirements_klp_realtime.txt  # 의존성
├── README_KLP_REALTIME.md        # 이 파일
└── imu_flex_serial.ino           # Arduino 펌웨어
```

## 🎮 고급 기능

### 1. 자동 포트 감지
- Windows: COM1, COM2, ...
- Linux/Mac: /dev/ttyUSB0, /dev/ttyACM0, ...

### 2. 실시간 통계
- 총 샘플 수
- 총 추론 수
- 평균 처리 시간
- 버퍼 상태

### 3. 에러 처리
- 연결 끊김 자동 감지
- 데이터 손실 방지
- API 서버 오류 처리

## 🔮 향후 계획

- [ ] 웹 기반 실시간 모니터링
- [ ] 데이터 저장 옵션 (H5 + API)
- [ ] 다중 Arduino 지원
- [ ] 실시간 시각화
- [ ] 클라우드 API 연동

## 📞 지원

**KLP-SignGlove + SignGlove_HW 통합 시스템**
- **데이터 수집**: SignGlove_HW
- **수화 인식**: KLP-SignGlove RGRU 모델
- **실시간 처리**: HTTP API 연동

---

**🤟 SignGlove Project - Making Sign Language Accessible Through Technology**
