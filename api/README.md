# API (Application Programming Interface)

이 폴더는 SignGlove 프로젝트의 REST API 서버와 클라이언트를 포함합니다.

## 파일 목록

### `api_server.py`
- **기능**: SignGlove REST API 서버
- **사용법**: `python3 api_server.py`
- **포트**: 8000
- **인증**: Bearer Token (demo_token_123)

### `api_client_test.py`
- **기능**: API 서버 테스트 클라이언트
- **사용법**: `python3 api_client_test.py`
- **목적**: API 기능 검증 및 성능 테스트

### `requirements_api.txt`
- **기능**: API 서버 의존성 목록
- **설치**: `pip install -r requirements_api.txt`

## API 엔드포인트

### 기본 엔드포인트
- `GET /` - 서버 정보
- `GET /health` - 건강 상태 확인
- `GET /docs` - API 문서 (Swagger UI)
- `GET /redoc` - API 문서 (ReDoc)

### 데이터 엔드포인트
- `GET /class-info` - 클래스 정보 조회
- `GET /models` - 모델 목록 조회
- `GET /performance` - 성능 통계 조회

### 추론 엔드포인트
- `POST /predict` - 단일 추론
- `POST /batch-predict` - 배치 추론

## 사용 방법

### 1. API 서버 시작
```bash
cd api
python3 api_server.py
```

### 2. API 클라이언트 테스트
```bash
cd api
python3 api_client_test.py
```

### 3. API 문서 확인
브라우저에서 `http://localhost:8000/docs` 접속

## API 요청 예시

### 단일 추론
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Authorization: Bearer demo_token_123" \
  -H "Content-Type: application/json" \
  -d '{
    "sensor_data": [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], ...]
  }'
```

### 건강 상태 확인
```bash
curl -X GET "http://localhost:8000/health"
```

## 인증

API는 Bearer Token 인증을 사용합니다:
- **토큰**: `demo_token_123`
- **헤더**: `Authorization: Bearer demo_token_123`

## 성능 지표

- **평균 처리 시간**: ~0.013초
- **성공률**: 100%
- **지원 클래스**: 24개 (한글 자음/모음)
- **모델 정확도**: 95.48%

## 주요 특징

- **FastAPI 기반**: 현대적이고 빠른 API 프레임워크
- **자동 문서화**: Swagger UI 및 ReDoc 지원
- **CORS 지원**: 크로스 오리진 요청 허용
- **실시간 통계**: 요청 수, 성공률, 처리 시간 모니터링
- **배치 처리**: 다중 요청 동시 처리 지원
- **GPU 가속**: CUDA 지원으로 빠른 추론

## 의존성

- **fastapi**: API 프레임워크
- **uvicorn**: ASGI 서버
- **torch**: PyTorch 딥러닝 프레임워크
- **numpy**: 수치 계산
- **pandas**: 데이터 처리
- **scikit-learn**: 전처리기
- **pydantic**: 데이터 검증

## 배포

### 개발 환경
```bash
python3 api_server.py
```

### 프로덕션 환경
```bash
uvicorn api_server:app --host 0.0.0.0 --port 8000 --workers 4
```

## 모니터링

API 서버는 다음 정보를 실시간으로 제공합니다:
- 서버 가동 시간
- 총 요청 수
- 성공한 예측 수
- 평균 처리 시간
- 정확도
- GPU 사용 가능 여부



