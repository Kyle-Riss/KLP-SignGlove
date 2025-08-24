# KLP-SignGlove 🧤

**한국어 수화 인식 시스템 - 센서 기반 실시간 수화 번역**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 🎯 프로젝트 개요

KLP-SignGlove는 **센서 기반 실시간 한국어 수화 인식 시스템**입니다. 손목에 착용하는 센서 장치를 통해 24개의 한국어 자음과 모음을 실시간으로 인식하고 번역합니다.

### 🔑 핵심 성능 (2024년 8월 최신)

- **🎯 전체 정확도**: **99.49%** (60,000개 샘플 테스트)
- **⚡ 실시간 정확도**: **100.00%** (연속 데이터 시뮬레이션)
- **📊 클래스별 성능**: 모든 24개 클래스 99% 이상
- **🚀 처리 속도**: 실시간 센서 데이터 처리
- **📈 데이터셋**: 600개 Episode 파일, 60,000개 샘플

## 🏗️ 시스템 아키텍처

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   센서 데이터   │───▶│  전처리 파이프라인 │───▶│  딥러닝 모델    │
│   (IMU 센서)    │    │  (정규화, 윈도우) │    │ (CNN+LSTM+Attention) │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                       │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   결과 출력     │◀───│  후처리         │◀───│  예측 결과      │
│   (한글 문자)   │    │  (신뢰도 필터링) │    │ (24개 클래스)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🎮 데모 실행

### 1. 실시간 시뮬레이션 (24개 클래스)
```bash
python inference/episode_realtime_demo_v2.py
```

### 2. 전체 데이터 테스트 (60,000개 샘플)
```bash
python inference/episode_full_data_demo.py
```

### 3. 통합 실시간 데모
```bash
python inference/unified_realtime_demo.py
```

## 🌐 API 서버

### FastAPI 기반 REST API 서버

**서버 실행:**
```bash
python server/main.py
```

**서버 주소:** `http://localhost:8000`

### 📡 API 엔드포인트

#### **기본 정보**
- `GET /` - 서버 상태 확인
- `GET /health` - 헬스 체크
- `GET /docs` - Swagger UI 문서
- `GET /redoc` - ReDoc 문서

#### **모델 정보**
- `GET /model/info` - 모델 정보 조회
- `GET /model/performance` - 성능 통계 조회

#### **추론 API**
- `POST /predict` - 단일 센서 데이터로 제스처 예측
- `POST /predict/batch` - 배치 센서 데이터로 제스처 예측
- `POST /predict/word` - 단어 인식 (글자 → 단어)

#### **단어 인식**
- `GET /word/status` - 단어 인식 상태 조회
- `POST /word/clear` - 현재 단어 초기화

### 🔧 API 사용 예시

#### **1. 서버 상태 확인**
```bash
curl http://localhost:8000/
# {"message":"SignGlove 추론 API 서버","version":"1.0.0","status":"running","docs":"/docs"}
```

#### **2. 모델 정보 조회**
```bash
curl http://localhost:8000/model/info
# {"model_name":"SignGlove Balanced Episode Model","model_version":"2.0.0","accuracy":0.9989,...}
```

#### **3. 단일 예측**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "timestamp": 1234567890.123,
    "pitch": 45.0,
    "roll": 30.0,
    "yaw": 60.0,
    "flex1": 100,
    "flex2": 150,
    "flex3": 200,
    "flex4": 180,
    "flex5": 120
  }'
```

#### **4. 단어 인식**
```bash
curl -X POST "http://localhost:8000/predict/word" \
  -H "Content-Type: application/json" \
  -d '{
    "timestamp": 1234567890.123,
    "pitch": 45.0,
    "roll": 30.0,
    "yaw": 60.0,
    "flex1": 100,
    "flex2": 150,
    "flex3": 200,
    "flex4": 180,
    "flex5": 120
  }'
```

### 📊 API 서버 성능

- **모델**: SignGlove Balanced Episode Model v2.0.0
- **정확도**: 99.89%
- **지원 클래스**: 24개 (14개 자음 + 10개 모음)
- **실시간 처리**: 센서 데이터 실시간 추론
- **단어 인식**: 글자 → 단어 변환 시스템

## 🧠 모델 학습

### 균형잡힌 Episode 모델 학습
```bash
python training/train_balanced_episode.py
```

**학습 결과:**
- **테스트 정확도**: 99.89%
- **Early Stopping**: 41 에포크
- **데이터셋**: 120,000개 윈도우 (24개 클래스 × 5,000개)
- **모델 파일**: `best_balanced_episode_model.pth`

## 📊 최신 성능 결과

### 🎯 전체 데이터 테스트 (60,000개 샘플)
```
📈 전체 성능:
  총 테스트: 60,000개
  정확한 예측: 59,696개
  전체 정확도: 99.49%

📋 클래스별 성능:
ㄱ: 99.6% | ㄴ: 99.6% | ㄷ: 99.8% | ㄹ: 99.4% | ㅁ: 99.3%
ㅂ: 99.6% | ㅅ: 99.4% | ㅇ: 99.6% | ㅈ: 99.4% | ㅊ: 99.4%
ㅋ: 99.4% | ㅌ: 99.5% | ㅍ: 99.6% | ㅎ: 100.0% | ㅏ: 99.7%
ㅑ: 99.6% | ㅓ: 99.4% | ㅕ: 99.2% | ㅗ: 99.2% | ㅛ: 99.7%
ㅜ: 99.5% | ㅠ: 99.4% | ㅡ: 99.6% | ㅣ: 99.7%
```

### ⚡ 실시간 시뮬레이션 (744개 샘플)
```
🎮 실시간 추론 시뮬레이션 결과:
  총 테스트: 744개
  정확한 예측: 744개
  전체 정확도: 100.00%
  모든 24개 클래스: 100% 정확도
```

## 🔧 주요 구성 요소

### 1. 데이터 처리
- **EpisodeSensorReading**: 센서 데이터 구조체
- **EpisodeInferencePipeline**: 실시간 추론 파이프라인
- **BalancedEpisodeDataset**: 균형잡힌 데이터셋 로더

### 2. 모델 아키텍처
- **CNN + LSTM + Attention**: 시계열 패턴 학습
- **입력**: 8개 센서 채널 (가속도계, 자이로스코프, 오일러 각도)
- **출력**: 24개 클래스 (14개 자음 + 10개 모음)

### 3. 학습 전략
- **과적합 방지**: Dropout 0.5, Weight Decay 1e-3
- **조기 종료**: Patience 10, 최고 모델 저장
- **데이터 증강**: 클래스별 5,000개 샘플 균형

## 📈 학습 결과

### 균형잡힌 Episode 모델 학습 성과
```
🎯 최종 모델 평가
============================================================
📊 테스트 정확도: 99.89%
✅ 혼동 행렬 저장: balanced_episode_confusion_matrix.png

📊 클래스별 성능:
ㄱ: 정확도=1.000, 재현율=1.000, F1=1.000
ㄴ: 정확도=1.000, 재현율=1.000, F1=1.000
ㄷ: 정확도=1.000, 재현율=1.000, F1=1.000
ㄹ: 정확도=0.986, 재현율=1.000, F1=0.993
ㅁ: 정확도=1.000, 재현율=1.000, F1=1.000
ㅂ: 정확도=1.000, 재현율=1.000, F1=1.000
ㅅ: 정확도=1.000, 재현율=0.992, F1=0.996
ㅇ: 정확도=1.000, 재현율=1.000, F1=1.000
ㅈ: 정확도=0.992, 재현율=1.000, F1=0.996
ㅊ: 정확도=1.000, 재현율=1.000, F1=1.000
ㅋ: 정확도=1.000, 재현율=1.000, F1=1.000
ㅌ: 정확도=1.000, 재현율=0.994, F1=0.997
ㅍ: 정확도=1.000, 재현율=1.000, F1=1.000
ㅎ: 정확도=1.000, 재현율=1.000, F1=1.000
ㅏ: 정확도=1.000, 재현율=1.000, F1=1.000
ㅑ: 정확도=1.000, 재현율=1.000, F1=1.000
ㅓ: 정확도=1.000, 재현율=1.000, F1=1.000
ㅕ: 정확도=1.000, 재현율=0.992, F1=0.996
ㅗ: 정확도=1.000, 재현율=0.997, F1=0.999
ㅛ: 정확도=1.000, 재현율=1.000, F1=1.000
ㅜ: 정확도=0.997, 재현율=1.000, F1=0.999
ㅠ: 정확도=1.000, 재현율=1.000, F1=1.000
ㅡ: 정확도=1.000, 재현율=1.000, F1=1.000
ㅣ: 정확도=1.000, 재현율=1.000, F1=1.000
```

## 📁 프로젝트 구조

```
KLP-SignGlove/
├── 📁 inference/                 # 추론 관련
│   ├── episode_realtime_demo_v2.py      # 실시간 시뮬레이션 V2
│   ├── episode_full_data_demo.py        # 전체 데이터 테스트
│   ├── unified_realtime_demo.py         # 통합 실시간 데모
│   ├── episode_inference.py             # Episode 추론 파이프라인
│   └── unified_inference.py             # 통합 추론 파이프라인
├── 📁 training/                  # 학습 관련
│   ├── train_balanced_episode.py        # 균형잡힌 Episode 학습
│   ├── train_with_full_episode_data.py  # 전체 Episode 데이터 학습
│   └── label_mapping.py                 # 라벨 매핑
├── 📁 server/                    # API 서버
│   └── main.py                          # FastAPI 서버
├── 📁 models/                    # 모델 정의
│   └── deep_learning.py                 # CNN+LSTM+Attention 모델
├── 📁 integrations/              # 외부 데이터 통합
│   └── SignGlove_HW/                    # GitHub 데이터셋
├── 📁 data/                      # 데이터 관련
│   └── processed/                       # 전처리된 데이터
├── 📄 best_balanced_episode_model.pth   # 최고 성능 모델
├── 📄 balanced_episode_training_curves.png  # 학습 곡선
├── 📄 balanced_episode_confusion_matrix.png # 혼동 행렬
└── 📄 README.md                  # 프로젝트 문서
```

## 🆕 새로운 수화 추가

새로운 수화 동작을 추가하려면:

1. **데이터 수집**: 센서 데이터를 CSV 형식으로 저장
2. **라벨 매핑**: `training/label_mapping.py`에 새 클래스 추가
3. **모델 재학습**: `python training/train_balanced_episode.py`
4. **성능 검증**: `python inference/episode_realtime_demo_v2.py`

## 🚀 실제 실행 예시

### 1. 모델 학습
```bash
$ python training/train_balanced_episode.py
🚀 균형잡힌 Episode 데이터로 모델 재학습 시작
📁 발견된 Episode 파일: 600개
✅ 총 120000개 윈도우 로드 완료
🎯 학습 시작 (총 80 에포크)
...
🎉 균형잡힌 Episode 모델 학습 완료!
✅ 최고 모델: best_balanced_episode_model.pth
```

### 2. 실시간 시뮬레이션
```bash
$ python inference/episode_realtime_demo_v2.py
📁 발견된 Episode 파일: 600개
🎮 실시간 추론 시뮬레이션 (연속 Episode 데이터)
🔍 ㄱ 연속 테스트 (총 50개 샘플):
  ✅ 샘플 20: ㄱ → ㄱ (신뢰도: 1.000)
  ✅ 샘플 21: ㄱ → ㄱ (신뢰도: 1.000)
  ...
📊 ㄱ 정확도: 100.0% (31/31)
...
🎯 전체 실시간 시뮬레이션 결과:
  총 테스트: 744개
  정확한 예측: 744개
  전체 정확도: 100.00%
```

### 3. 전체 데이터 테스트
```bash
$ python inference/episode_full_data_demo.py
📁 발견된 Episode 파일: 600개
✅ 총 60000개 센서 데이터 로드 완료
📈 전체 성능:
  총 테스트: 60,000개
  정확한 예측: 59,696개
  전체 정확도: 99.49%
```

## 🔑 주요 기능

- **🎯 고정밀 인식**: 24개 한국어 자음/모음 99% 이상 정확도
- **⚡ 실시간 처리**: 센서 데이터 실시간 추론
- **🔄 연속 데이터**: 시계열 패턴 학습으로 안정적 성능
- **📊 대규모 테스트**: 60,000개 샘플 검증
- **🎮 시뮬레이션**: 실제 사용 환경과 유사한 테스트
- **📈 시각화**: 학습 곡선, 혼동 행렬 제공

## 📊 프로젝트 현황 & 로드맵

### ✅ 완료된 기능
- [x] 24개 한국어 자음/모음 인식
- [x] 실시간 센서 데이터 처리
- [x] CNN+LSTM+Attention 모델 구현
- [x] 대규모 데이터셋 학습 (60,000개 샘플)
- [x] 실시간 시뮬레이션 (100% 정확도)
- [x] 과적합 방지 및 모델 최적화
- [x] 클래스별 성능 분석
- [x] FastAPI 기반 REST API 서버
- [x] 단어 인식 시스템 (글자 → 단어)
- [x] 실시간 추론 API 엔드포인트

### 🚧 진행 중인 기능
- [ ] 웹 인터페이스 개발
- [ ] 모바일 앱 연동
- [ ] 추가 수화 동작 확장

### 📋 향후 계획
- [ ] 문장 단위 수화 인식
- [ ] 다국어 지원
- [ ] 클라우드 기반 서비스
- [ ] 하드웨어 최적화

## 🤝 기여하기

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 `LICENSE` 파일을 참조하세요.

## 📞 연락처

프로젝트 링크: [https://github.com/your-username/KLP-SignGlove](https://github.com/your-username/KLP-SignGlove)

---

**KLP-SignGlove** - 한국어 수화 인식의 새로운 표준 🧤✨
