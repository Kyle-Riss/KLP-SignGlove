# SignGlove 기술 명세서

## 🔧 하드웨어 명세

### 마이크로컨트롤러
- **모델**: Arduino Nano 33 IoT
- **프로세서**: SAMD21 Cortex-M0+ (48MHz)
- **메모리**: 256KB Flash, 32KB SRAM
- **입력/출력**: 14개 디지털 핀, 8개 아날로그 핀
- **통신**: USB, UART, SPI, I2C, WiFi, Bluetooth
- **내장 센서**: IMU (LSM6DS3)

### 센서 구성
```
┌─────────────────────────────────────┐
│           센서 배치도               │
├─────────────────────────────────────┤
│  SF15~150 Flex 센서 5개:            │
│  • 엄지 (Thumb)    - A0             │
│  • 검지 (Index)    - A1             │
│  • 중지 (Middle)   - A2             │
│  • 약지 (Ring)     - A3             │
│  • 소지 (Pinky)    - A4             │
│                                     │
│  내장 IMU 센서:                     │
│  • LSM6DS3        - 내장            │
│  • 3축 가속도계 + 3축 자이로스코프   │
│                                     │
│  통신 모듈:                         │
│  • WiFi/Bluetooth - 내장            │
└─────────────────────────────────────┘
```

### SF15~150 Flex 센서 스펙
- **내구성**: 100만 회 이상
- **초기 저항**: 10MΩ 이상 (무부하)
- **테스트 전압**: 3.3VDC
- **응답 시간**: 1ms 미만
- **복구 시간**: 15ms 미만
- **전자기 간섭**: 발생하지 않음
- **정전기 방전**: 무감응

### 전력 관리
- **배터리**: 3.7V Li-Po 1000mAh
- **충전 회로**: TP4056 충전 모듈
- **전압 조정**: 3.3V 레귤레이터 (Arduino Nano 33 IoT 내장)
- **예상 사용시간**: 10-12시간 (내장 모듈로 인한 효율성 향상)

## 💻 소프트웨어 아키텍처

### 시스템 구조
```
┌─────────────────────────────────────┐
│           SignGlove 시스템          │
├─────────────────────────────────────┤
│  하드웨어 계층                      │
│  ├── Arduino 펌웨어                 │
│  ├── 센서 데이터 수집               │
│  └── 블루투스 통신                  │
│                                     │
│  통신 계층                          │
│  ├── 시리얼 통신 프로토콜           │
│  ├── 데이터 패킷 구조               │
│  └── 에러 처리                      │
│                                     │
│  처리 계층                          │
│  ├── 데이터 전처리                  │
│  ├── 특징 추출                      │
│  ├── 모델 추론                      │
│  └── 후처리                         │
│                                     │
│  출력 계층                          │
│  ├── 텍스트 변환                    │
│  ├── 음성 합성                      │
│  └── 시각적 피드백                  │
└─────────────────────────────────────┘
```

### 데이터 플로우
```
센서 데이터 → 전처리 → 특징 추출 → 모델 추론 → 후처리 → 출력
     ↓           ↓         ↓         ↓         ↓       ↓
  36Hz        노이즈    24개      LSTM/GRU   연속성   텍스트
  샘플링      제거     특징      +Attention  검증    변환
```

## 🧠 모델 구조 상세

### 입력 데이터
- **원본 센서**: 8개 (Flex 5개 + IMU 3개)
- **1차 미분**: 8개 (속도)
- **2차 미분**: 8개 (가속도)
- **총 특징**: 24개
- **시퀀스 길이**: 300 프레임 (8.3초 @ 36Hz)

### 모델 아키텍처
```python
class SignGloveModel(nn.Module):
    def __init__(self):
        # 입력: (batch_size, 300, 24)
        self.lstm = nn.LSTM(24, 64, 2, bidirectional=True, batch_first=True)
        self.attention = nn.MultiheadAttention(128, 8, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.layer_norm = nn.LayerNorm(128)
        self.classifier = nn.Linear(128, 24)
    
    def forward(self, x):
        # LSTM 처리
        lstm_out, _ = self.lstm(x)  # (batch, 300, 128)
        
        # Attention 적용
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # 정규화 및 드롭아웃
        out = self.layer_norm(attn_out + lstm_out)
        out = self.dropout(out)
        
        # 마지막 시점 선택
        final_out = out[:, -1, :]  # (batch, 128)
        
        # 분류
        logits = self.classifier(final_out)  # (batch, 24)
        return logits
```

### 하이퍼파라미터
- **학습률**: 0.001
- **배치 크기**: 32
- **에포크**: 100
- **옵티마이저**: Adam
- **스케줄러**: ReduceLROnPlateau
- **정규화**: Dropout 0.2, L2 1e-4

## 📊 데이터 처리 파이프라인

### 전처리 단계
1. **노이즈 제거**: 이동평균 필터
2. **정규화**: Min-Max 정규화
3. **이상치 제거**: IQR 방법
4. **특징 추출**: 미분 계산

### 특징 엔지니어링
```python
def extract_features(sensor_data):
    # 원본 특징 (8개)
    original = sensor_data
    
    # 1차 미분 (속도)
    velocity = np.diff(sensor_data, axis=0, prepend=sensor_data[0:1])
    
    # 2차 미분 (가속도)
    acceleration = np.diff(velocity, axis=0, prepend=velocity[0:1])
    
    # 결합
    features = np.concatenate([original, velocity, acceleration], axis=1)
    return features  # (300, 24)
```

## 🔄 실시간 처리 시스템

### 데이터 수집
- **샘플링 주기**: 27.8ms (36Hz)
- **버퍼 크기**: 300 프레임
- **처리 지연**: < 100ms
- **통신 프로토콜**: UART 115200 bps

### 추론 파이프라인
```python
class RealTimeInference:
    def __init__(self, model_path):
        self.model = torch.load(model_path)
        self.buffer = np.zeros((300, 24))
        self.buffer_idx = 0
        
    def process_frame(self, sensor_data):
        # 버퍼 업데이트
        self.buffer[self.buffer_idx] = sensor_data
        self.buffer_idx = (self.buffer_idx + 1) % 300
        
        # 특징 추출
        features = self.extract_features(self.buffer)
        
        # 추론
        with torch.no_grad():
            logits = self.model(features.unsqueeze(0))
            prediction = torch.argmax(logits, dim=1)
            
        return prediction.item()
```

## 📱 사용자 인터페이스

### GUI 구성
```
┌─────────────────────────────────────┐
│           SignGlove GUI             │
├─────────────────────────────────────┤
│  연결 상태: [연결됨] [연결 끊김]     │
│                                     │
│  실시간 인식:                       │
│  ┌─────────────────────────────────┐ │
│  │        인식된 텍스트            │ │
│  │                                 │ │
│  │        ㄱ ㄴ ㄷ ㄹ ㅁ           │ │
│  └─────────────────────────────────┘ │
│                                     │
│  설정:                              │
│  • 모델 선택: [LSTM] [GRU] [Transformer] │
│  • 민감도: [●●●●○]                  │
│  • 출력 방식: [텍스트] [음성]        │
│                                     │
│  [시작] [중지] [설정] [도움말]       │
└─────────────────────────────────────┘
```

### API 인터페이스
```python
class SignGloveAPI:
    def __init__(self):
        self.inference = RealTimeInference()
        
    def start_recognition(self):
        """인식 시작"""
        pass
        
    def stop_recognition(self):
        """인식 중지"""
        pass
        
    def get_prediction(self):
        """현재 예측 결과 반환"""
        pass
        
    def set_model(self, model_type):
        """모델 변경"""
        pass
```

## 🔒 보안 및 안전성

### 데이터 보안
- **로컬 처리**: 모든 데이터는 로컬에서 처리
- **암호화**: 블루투스 통신 암호화
- **개인정보**: 수집하지 않음

### 안전성 고려사항
- **전기 안전**: 저전압 사용 (3.3V)
- **화재 안전**: Li-Po 배터리 안전 회로
- **사용자 안전**: 부드러운 소재 사용

## 📈 성능 최적화

### 하드웨어 최적화
- **전력 관리**: 슬립 모드 구현
- **메모리 최적화**: 버퍼 크기 조정
- **처리 속도**: 인터럽트 기반 처리

### 소프트웨어 최적화
- **모델 경량화**: 양자화, 프루닝
- **추론 최적화**: ONNX 변환
- **메모리 관리**: 가비지 컬렉션 최적화

## 🧪 테스트 계획

### 단위 테스트
- **센서 데이터 검증**
- **전처리 함수 테스트**
- **모델 추론 테스트**

### 통합 테스트
- **하드웨어-소프트웨어 통합**
- **실시간 처리 테스트**
- **사용자 시나리오 테스트**

### 성능 테스트
- **정확도 측정**
- **지연시간 측정**
- **전력 소비 측정**

---

**문서 버전**: v2.0  
**작성일**: 2024년 9월 15일  
**최종 수정일**: 2024년 9월 15일  
**작성자**: SignGlove 기술팀
