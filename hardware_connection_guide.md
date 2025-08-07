# 🤖 SignGlove 하드웨어 연결 가이드

## 🔌 실제 하드웨어 연결 방법

### 1. UART/USB 시리얼 연결
```bash
# macOS/Linux에서 장치 확인
ls /dev/tty*

# 일반적인 장치명:
# macOS: /dev/tty.usbserial-xxxxx 또는 /dev/cu.usbmodem-xxxxx
# Linux: /dev/ttyUSB0, /dev/ttyACM0
# Windows: COM3, COM4 등

# UART 모드로 데모 실행
python realtime_demo.py --connection uart --model best_dl_model.pth
```

### 2. WiFi/TCP 연결
```bash
# SignGlove WiFi 설정 후
python realtime_demo.py --connection wifi --model best_dl_model.pth
```

### 3. 하드웨어 데이터 포맷

SignGlove에서 전송해야 하는 데이터 형식:

#### CSV 형식 (권장)
```
flex1,flex2,flex3,flex4,flex5,pitch,roll,yaw
854.18,868.01,870.32,889.92,915.28,-0.85,-0.06,0.12
840.72,856.67,853.25,876.03,891.98,-0.85,0.0,0.12
```

#### JSON 형식 (대안)
```json
{"flex1": 854.18, "flex2": 868.01, "flex3": 870.32, "flex4": 889.92, "flex5": 915.28, "pitch": -0.85, "roll": -0.06, "yaw": 0.12}
```

## 📊 현재 시뮬레이션 데이터 구성

실제 수집된 샘플 데이터:
- **ㄱ_sample_data.csv**: 1411줄 (ㄱ 자음 데이터)
- **ㄴ_sample_data.csv**: 1411줄 (ㄴ 자음 데이터)  
- **ㄷ_sample_data.csv**: 1411줄 (ㄷ 자음 데이터)
- **ㄹ_sample_data.csv**: 1411줄 (ㄹ 자음 데이터)
- **ㅁ_sample_data.csv**: 1411줄 (ㅁ 자음 데이터)

**총 7055개 샘플 → 834개 윈도우로 변환됨**

## 🛠️ 하드웨어 요구사항

### SignGlove 사양
- **Flex 센서**: 5개 (각 손가락)
- **IMU 센서**: 3축 (pitch, roll, yaw)
- **통신**: UART (115200 baud) 또는 WiFi
- **샘플링 주파수**: 50Hz 권장

### 센서 범위
- **Flex 센서**: 0-1024 (10비트 ADC)
- **IMU 각도**: -180° ~ +180°
- **데이터 형식**: float (소수점 2자리 권장)

## 🔍 실제 vs 시뮬레이션 비교

| 구분 | 시뮬레이션 모드 | 실제 하드웨어 |
|------|----------------|---------------|
| **데이터 소스** | CSV 파일 반복 | 실시간 센서 |
| **지연시간** | 없음 | ~20ms |
| **노이즈** | 없음 | 실제 센서 노이즈 |
| **연결 안정성** | 100% | 환경에 따라 변동 |
| **활용도** | 개발/테스트 | 실제 사용 |

## 🚀 실제 하드웨어 연동 체크리스트

### 1. 하드웨어 준비
- [ ] SignGlove 전원 연결
- [ ] UART 케이블 또는 WiFi 설정
- [ ] 센서 보정 완료
- [ ] 데이터 전송 테스트

### 2. 소프트웨어 설정
- [ ] 포트/IP 확인
- [ ] 보드레이트 설정 (115200)
- [ ] 데이터 형식 검증
- [ ] 실시간 모드 테스트

### 3. 성능 검증
- [ ] 지연시간 < 50ms
- [ ] 데이터 손실 < 1%
- [ ] 예측 정확도 > 85%
- [ ] 안정성 테스트 통과

## 🔧 문제 해결

### 연결 문제
```python
# 포트 스캔 스크립트
import serial.tools.list_ports

ports = serial.tools.list_ports.comports()
for port in ports:
    print(f"포트: {port.device}, 설명: {port.description}")
```

### 데이터 검증
```python
# 데이터 형식 검증
def validate_sensor_data(data):
    if len(data) != 8:
        return False
    
    # Flex 센서 범위 체크 (0-1024)
    for i in range(5):
        if not (0 <= data[i] <= 1024):
            return False
    
    # IMU 범위 체크 (-180 ~ 180)
    for i in range(5, 8):
        if not (-180 <= data[i] <= 180):
            return False
    
    return True
```

## 📈 실제 사용 시나리오

1. **개발자 테스트**: 시뮬레이션 모드로 알고리즘 검증
2. **하드웨어 통합**: UART/WiFi로 실제 연결
3. **사용자 체험**: 실시간 수어 인식 데모
4. **제품 배포**: 최적화된 실시간 시스템