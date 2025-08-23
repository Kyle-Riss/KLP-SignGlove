# SignGlove 추론 API 명세서

## 📋 개요

SignGlove 추론 API는 실시간 한국수어 인식을 위한 RESTful API 서버입니다. 센서 데이터를 입력받아 24개 한국어 자음/모음 클래스를 예측합니다.

- **API 버전**: 1.0.0
- **기본 URL**: `http://localhost:8000`
- **프로토콜**: HTTP/HTTPS
- **데이터 형식**: JSON

## 🎯 모델 정보

### 기본 모델 정보
- **모델 이름**: SignGlove Unified Model
- **모델 버전**: 1.0.0
- **테스트 정확도**: 96.60%
- **지원 클래스**: 24개 (자음 14개 + 모음 10개)
- **입력 특성**: 8개 (flex1-5 + pitch, roll, yaw)
- **윈도우 크기**: 20개 샘플

### 지원 클래스 목록
```
자음 (14개): ㄱ, ㄴ, ㄷ, ㄹ, ㅁ, ㅂ, ㅅ, ㅇ, ㅈ, ㅊ, ㅋ, ㅌ, ㅍ, ㅎ
모음 (10개): ㅏ, ㅑ, ㅓ, ㅕ, ㅗ, ㅛ, ㅜ, ㅠ, ㅡ, ㅣ
```

## 🔗 엔드포인트

### 1. 루트 엔드포인트

#### `GET /`
API 서버의 기본 정보를 반환합니다.

**응답 예시:**
```json
{
  "message": "SignGlove 추론 API 서버",
  "version": "1.0.0",
  "status": "running",
  "docs": "/docs"
}
```

### 2. 헬스 체크

#### `GET /health`
서버 상태와 모델 로드 상태를 확인합니다.

**응답 예시:**
```json
{
  "status": "healthy",
  "timestamp": 1692800000.0,
  "model_loaded": true,
  "performance_stats": {
    "fps": 125.5,
    "avg_latency_ms": 0.8,
    "total_predictions": 1500,
    "buffer_utilization": 0.75,
    "confidence_threshold": 0.7
  }
}
```

### 3. 모델 정보

#### `GET /model/info`
학습된 모델의 상세 정보를 반환합니다.

**응답 예시:**
```json
{
  "model_name": "SignGlove Unified Model",
  "model_version": "1.0.0",
  "accuracy": 0.966,
  "num_classes": 24,
  "supported_classes": ["ㄱ", "ㄴ", "ㄷ", "ㄹ", "ㅁ", "ㅂ", "ㅅ", "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ", "ㅏ", "ㅑ", "ㅓ", "ㅕ", "ㅗ", "ㅛ", "ㅜ", "ㅠ", "ㅡ", "ㅣ"],
  "input_features": 8,
  "window_size": 20
}
```

### 4. 성능 통계

#### `GET /model/performance`
실시간 성능 통계를 반환합니다.

**응답 예시:**
```json
{
  "fps": 125.5,
  "avg_latency_ms": 0.8,
  "total_predictions": 1500,
  "buffer_utilization": 0.75,
  "confidence_threshold": 0.7
}
```

### 5. 단일 예측

#### `POST /predict`
단일 센서 데이터로 제스처를 예측합니다.

**요청 형식:**
```json
{
  "timestamp": 1692800000.0,
  "pitch": 15.5,
  "roll": -44.6,
  "yaw": 8.8,
  "flex1": 765,
  "flex2": 772,
  "flex3": 834,
  "flex4": 852,
  "flex5": 861,
  "source": "api"
}
```

**응답 형식:**
```json
{
  "predicted_class": "ㄱ",
  "confidence": 0.987,
  "stability_score": 0.95,
  "processing_time_ms": 0.8,
  "timestamp": 1692800000.1
}
```

### 6. 배치 예측

#### `POST /predict/batch`
여러 센서 데이터를 배치로 처리하여 예측합니다.

**요청 형식:**
```json
{
  "sensor_data": [
    {
      "timestamp": 1692800000.0,
      "pitch": 15.5,
      "roll": -44.6,
      "yaw": 8.8,
      "flex1": 765,
      "flex2": 772,
      "flex3": 834,
      "flex4": 852,
      "flex5": 861
    },
    {
      "timestamp": 1692800000.1,
      "pitch": 15.6,
      "roll": -44.7,
      "yaw": 8.9,
      "flex1": 766,
      "flex2": 773,
      "flex3": 835,
      "flex4": 853,
      "flex5": 862
    }
  ],
  "window_size": 20,
  "stride": 10
}
```

**응답 형식:**
```json
[
  {
    "predicted_class": "ㄱ",
    "confidence": 0.987,
    "stability_score": 0.95,
    "processing_time_ms": 0.8,
    "timestamp": 1692800000.1
  },
  {
    "predicted_class": "ㄱ",
    "confidence": 0.992,
    "stability_score": 0.98,
    "processing_time_ms": 0.7,
    "timestamp": 1692800000.2
  }
]
```

### 7. 안정적 예측

#### `POST /predict/stable`
안정성 체크를 포함한 예측을 수행합니다.

**요청 형식:** (단일 예측과 동일)

**응답 형식:**
```json
{
  "predicted_class": "ㄱ",
  "confidence": 0.987,
  "stability_score": 0.95,
  "processing_time_ms": 1.2,
  "timestamp": 1692800000.1
}
```

### 8. 설정 관리

#### `POST /config/confidence`
신뢰도 임계값을 설정합니다.

**요청 형식:**
```json
{
  "threshold": 0.8
}
```

**응답 형식:**
```json
{
  "message": "신뢰도 임계값이 0.8로 설정되었습니다"
}
```

### 9. 버퍼 관리

#### `POST /buffer/clear`
데이터 버퍼를 초기화합니다.

**응답 형식:**
```json
{
  "message": "버퍼가 초기화되었습니다"
}
```

### 10. 지원 클래스 조회

#### `GET /classes`
지원하는 클래스 목록을 반환합니다.

**응답 형식:**
```json
{
  "consonants": ["ㄱ", "ㄴ", "ㄷ", "ㄹ", "ㅁ", "ㅂ", "ㅅ", "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ"],
  "vowels": ["ㅏ", "ㅑ", "ㅓ", "ㅕ", "ㅗ", "ㅛ", "ㅜ", "ㅠ", "ㅡ", "ㅣ"],
  "all_classes": ["ㄱ", "ㄴ", "ㄷ", "ㄹ", "ㅁ", "ㅂ", "ㅅ", "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ", "ㅏ", "ㅑ", "ㅓ", "ㅕ", "ㅗ", "ㅛ", "ㅜ", "ㅠ", "ㅡ", "ㅣ"]
}
```

## 📊 데이터 형식

### 센서 데이터 필드

| 필드명 | 타입 | 설명 | 범위 | 단위 |
|--------|------|------|------|------|
| `timestamp` | float | 타임스탬프 | - | 초 |
| `pitch` | float | 피치 각도 | -180 ~ 180 | 도 |
| `roll` | float | 롤 각도 | -180 ~ 180 | 도 |
| `yaw` | float | 요 각도 | -180 ~ 180 | 도 |
| `flex1` | int | 플렉스 센서 1 | 0 ~ 1023 | - |
| `flex2` | int | 플렉스 센서 2 | 0 ~ 1023 | - |
| `flex3` | int | 플렉스 센서 3 | 0 ~ 1023 | - |
| `flex4` | int | 플렉스 센서 4 | 0 ~ 1023 | - |
| `flex5` | int | 플렉스 센서 5 | 0 ~ 1023 | - |
| `source` | string | 데이터 소스 | - | - |

### 예측 결과 필드

| 필드명 | 타입 | 설명 | 범위 |
|--------|------|------|------|
| `predicted_class` | string | 예측된 클래스 | 24개 클래스 중 하나 |
| `confidence` | float | 예측 신뢰도 | 0.0 ~ 1.0 |
| `stability_score` | float | 안정성 점수 | 0.0 ~ 1.0 |
| `processing_time_ms` | float | 처리 시간 | 밀리초 |
| `timestamp` | float | 응답 타임스탬프 | 초 |

## 🚀 사용 예시

### Python 클라이언트 예시

```python
import requests
import json

# API 기본 URL
BASE_URL = "http://localhost:8000"

# 1. 서버 상태 확인
response = requests.get(f"{BASE_URL}/health")
print("서버 상태:", response.json())

# 2. 모델 정보 조회
response = requests.get(f"{BASE_URL}/model/info")
model_info = response.json()
print("모델 정확도:", model_info["accuracy"])

# 3. 단일 예측
sensor_data = {
    "timestamp": 1692800000.0,
    "pitch": 15.5,
    "roll": -44.6,
    "yaw": 8.8,
    "flex1": 765,
    "flex2": 772,
    "flex3": 834,
    "flex4": 852,
    "flex5": 861,
    "source": "python_client"
}

response = requests.post(f"{BASE_URL}/predict", json=sensor_data)
result = response.json()
print(f"예측 결과: {result['predicted_class']} (신뢰도: {result['confidence']:.3f})")

# 4. 배치 예측
batch_data = {
    "sensor_data": [sensor_data, sensor_data],  # 여러 데이터
    "window_size": 20,
    "stride": 10
}

response = requests.post(f"{BASE_URL}/predict/batch", json=batch_data)
results = response.json()
for i, result in enumerate(results):
    print(f"배치 {i+1}: {result['predicted_class']} (신뢰도: {result['confidence']:.3f})")
```

### cURL 예시

```bash
# 1. 서버 상태 확인
curl -X GET "http://localhost:8000/health"

# 2. 모델 정보 조회
curl -X GET "http://localhost:8000/model/info"

# 3. 단일 예측
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "timestamp": 1692800000.0,
    "pitch": 15.5,
    "roll": -44.6,
    "yaw": 8.8,
    "flex1": 765,
    "flex2": 772,
    "flex3": 834,
    "flex4": 852,
    "flex5": 861,
    "source": "curl"
  }'

# 4. 신뢰도 임계값 설정
curl -X POST "http://localhost:8000/config/confidence" \
  -H "Content-Type: application/json" \
  -d '{"threshold": 0.8}'
```

## ⚠️ 오류 코드

| HTTP 상태 코드 | 설명 | 해결 방법 |
|---------------|------|----------|
| 200 | 성공 | - |
| 400 | 잘못된 요청 | 요청 데이터 형식 확인 |
| 503 | 서비스 불가 | 모델 로드 상태 확인 |
| 500 | 서버 내부 오류 | 서버 로그 확인 |

### 오류 응답 형식

```json
{
  "detail": "오류 메시지"
}
```

## 🔧 성능 최적화

### 권장 설정
- **신뢰도 임계값**: 0.7 ~ 0.9
- **배치 크기**: 10 ~ 50개
- **요청 간격**: 10ms 이상
- **연결 유지**: Keep-Alive 사용

### 성능 지표
- **평균 지연시간**: < 1ms
- **처리량**: > 100 FPS
- **메모리 사용량**: ~100MB
- **CPU 사용률**: < 20%

## 📚 추가 문서

- **API 문서**: `http://localhost:8000/docs` (Swagger UI)
- **ReDoc 문서**: `http://localhost:8000/redoc`
- **소스 코드**: `server/main.py`

## 🔄 버전 관리

| 버전 | 날짜 | 주요 변경사항 |
|------|------|-------------|
| 1.0.0 | 2025-08-23 | 초기 릴리스 |

---

**SignGlove Project** - Making Sign Language Accessible Through Technology
