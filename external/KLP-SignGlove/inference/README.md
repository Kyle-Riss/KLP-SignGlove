# SignGlove 추론 시스템

실시간 한국수어 인식을 위한 SignGlove 추론 시스템입니다.

## 📁 파일 구조

```
inference/
├── signglove_inference.py    # 핵심 추론 시스템
├── api_server.py             # FastAPI 웹 서버
├── test_client.py            # API 테스트 클라이언트
├── config.json               # 설정 파일
└── README.md                 # 이 파일
```

## 🚀 빠른 시작

### 1. 추론 시스템 직접 실행

```bash
# 추론 시스템 데모 실행
poetry run python signglove_inference.py
```

### 2. API 서버 실행

```bash
# API 서버 시작 (백그라운드)
poetry run python api_server.py &

# 또는 포그라운드에서 실행
poetry run python api_server.py
```

### 3. 클라이언트 테스트

```bash
# API 서버 테스트
poetry run python test_client.py
```

## 🔧 API 엔드포인트

### 기본 정보
- **서버 주소**: `http://localhost:8000`
- **API 문서**: `http://localhost:8000/docs`

### 주요 엔드포인트

#### 1. 헬스 체크
```bash
GET /health
```

#### 2. 시스템 정보
```bash
GET /info
```

#### 3. 라벨 목록
```bash
GET /labels
```

#### 4. 단일 예측
```bash
POST /predict
Content-Type: application/json

{
  "sensor_data": [
    {
      "flex1": 0.5,
      "flex2": 0.3,
      "flex3": 0.7,
      "flex4": 0.2,
      "flex5": 0.8,
      "pitch": 0.1,
      "roll": 0.4,
      "yaw": 0.6
    }
    // ... 20개 센서 데이터
  ]
}
```

#### 5. 배치 예측
```bash
POST /predict/batch
Content-Type: application/json

{
  "sensor_data": [
    // 여러 시퀀스의 센서 데이터
  ]
}
```

#### 6. 실시간 예측
```bash
# 실시간 예측 시작
POST /realtime/start

# 센서 데이터 추가
POST /realtime/add
Content-Type: application/json

{
  "flex1": 0.5,
  "flex2": 0.3,
  "flex3": 0.7,
  "flex4": 0.2,
  "flex5": 0.8,
  "pitch": 0.1,
  "roll": 0.4,
  "yaw": 0.6
}

# 버퍼 초기화
POST /realtime/reset
```

## 📊 응답 형식

### 예측 응답
```json
{
  "predicted_label": "ㅌ",
  "confidence": 0.129,
  "probabilities": {
    "ㄱ": 0.045,
    "ㄴ": 0.052,
    "ㄷ": 0.048,
    // ... 모든 라벨의 확률
  },
  "timestamp": 1756449657.3618548
}
```

### 시스템 정보
```json
{
  "model_path": "/path/to/model.pth",
  "device": "cuda",
  "total_parameters": 189473,
  "labels": ["ㄱ", "ㄴ", "ㄷ", ...],
  "config": {
    "input_features": 8,
    "sequence_length": 20,
    "num_classes": 24,
    "hidden_dim": 48,
    "num_layers": 1,
    "dropout": 0.3
  }
}
```

## ⚙️ 설정

`config.json` 파일에서 다음 설정을 변경할 수 있습니다:

```json
{
  "model_config": {
    "input_features": 8,
    "sequence_length": 20,
    "num_classes": 24,
    "hidden_dim": 48,
    "num_layers": 1,
    "dropout": 0.3
  },
  "inference_config": {
    "confidence_threshold": 0.7,
    "smoothing_window": 5,
    "prediction_delay": 0.1,
    "buffer_size": 20,
    "sample_rate": 50.0
  },
  "api_config": {
    "host": "0.0.0.0",
    "port": 8000,
    "debug": false,
    "workers": 1
  }
}
```

## 🔍 사용 예시

### Python 클라이언트

```python
import requests
import json

# API 서버 주소
base_url = "http://localhost:8000"

# 헬스 체크
response = requests.get(f"{base_url}/health")
print(response.json())

# 시스템 정보 조회
response = requests.get(f"{base_url}/info")
info = response.json()
print(f"모델: {info['model_path']}")
print(f"디바이스: {info['device']}")
print(f"라벨: {info['labels']}")

# 단일 예측
sensor_data = [
    {
        "flex1": 0.5, "flex2": 0.3, "flex3": 0.7, "flex4": 0.2, "flex5": 0.8,
        "pitch": 0.1, "roll": 0.4, "yaw": 0.6
    }
    # ... 20개 데이터
]

response = requests.post(
    f"{base_url}/predict",
    json={"sensor_data": sensor_data}
)
result = response.json()
print(f"예측: {result['predicted_label']}")
print(f"신뢰도: {result['confidence']}")
```

### 실시간 예측

```python
import requests
import time

# 실시간 예측 시작
requests.post(f"{base_url}/realtime/start")

# 센서 데이터를 실시간으로 추가
for i in range(30):
    sensor_data = {
        "flex1": 0.5, "flex2": 0.3, "flex3": 0.7, "flex4": 0.2, "flex5": 0.8,
        "pitch": 0.1, "roll": 0.4, "yaw": 0.6
    }
    
    response = requests.post(f"{base_url}/realtime/add", json=sensor_data)
    result = response.json()
    
    if result['prediction']:
        print(f"예측: {result['prediction']['predicted_label']}")
    
    time.sleep(0.1)  # 100ms 간격
```

## 🎯 성능 정보

- **모델 파라미터**: 189,473개
- **추론 시간**: ~160ms (단일 예측)
- **지원 라벨**: 24개 한글 자음/모음
- **입력 형태**: 8축 센서 데이터 (flex1-5, pitch, roll, yaw)
- **시퀀스 길이**: 20개 샘플

## 🔧 문제 해결

### 1. 모델 로드 오류
- 모델 파일 경로 확인
- PyTorch 버전 호환성 확인

### 2. API 서버 연결 오류
- 서버가 실행 중인지 확인
- 포트 8000이 사용 가능한지 확인

### 3. 예측 정확도 낮음
- 신뢰도 임계값 조정
- 센서 데이터 전처리 확인

## 📝 라벨 매핑

| 인덱스 | 라벨 | 설명 |
|--------|------|------|
| 0-13 | ㄱ,ㄴ,ㄷ,ㄹ,ㅁ,ㅂ,ㅅ,ㅇ,ㅈ,ㅊ,ㅋ,ㅌ,ㅍ,ㅎ | 자음 |
| 14-23 | ㅏ,ㅑ,ㅓ,ㅕ,ㅗ,ㅛ,ㅜ,ㅠ,ㅡ,ㅣ | 모음 |

## 🚀 배포

### Docker 배포 (선택사항)

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["python", "api_server.py"]
```

### 시스템 서비스 (Linux)

```bash
# 서비스 파일 생성
sudo nano /etc/systemd/system/signglove.service

[Unit]
Description=SignGlove API Server
After=network.target

[Service]
Type=simple
User=signglove
WorkingDirectory=/path/to/signglove
ExecStart=/usr/bin/python api_server.py
Restart=always

[Install]
WantedBy=multi-user.target

# 서비스 시작
sudo systemctl enable signglove
sudo systemctl start signglove
```

## 📞 지원

문제가 발생하면 다음을 확인하세요:
1. 로그 파일 확인
2. API 문서 참조 (`http://localhost:8000/docs`)
3. 설정 파일 검증
4. 모델 파일 무결성 확인




