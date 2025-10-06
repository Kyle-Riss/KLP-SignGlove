# KLP-SignGlove: 한국어 수화 인식 프로젝트

한국어 수화 자모(자음, 모음) 인식을 위한 센서 장갑 기반 실시간 분류 시스템입니다.

## 🎯 프로젝트 개요

**목표**: 한국어 수화 자모 24개 클래스(자음 14개 + 모음 10개)를 실시간으로 인식하는 시스템 개발

**성과**: 
- 현재 성능: 97.5% (598 샘플 실험 기준)
- 목표 성능: 98.5% (데이터 확장 후)
- 실시간 추론 준비: 추론 전용 엔진(inference/) 분리 완료

## 📊 데이터셋 정보

- 클래스: 24개 (자음 14 + 모음 10)
- 타임스텝: 87
- 채널: 8 (flex1~5, pitch, roll, yaw)
- 전처리: 0.0 패딩, 트렁케이션/균등 리샘플링(옵션), StandardScaler 정규화

### 클래스 목록
- 자음 (14개): ㄱ, ㄴ, ㄷ, ㄹ, ㅁ, ㅂ, ㅅ, ㅇ, ㅈ, ㅊ, ㅋ, ㅌ, ㅍ, ㅎ
- 모음 (10개): ㅏ, ㅑ, ㅓ, ㅕ, ㅗ, ㅛ, ㅜ, ㅠ, ㅡ, ㅣ

## 🔧 최적 모델 설정

### 현재 모델 (598개 데이터셋 - 97.5% 성능)
```python
{
    'hidden_size': 48,
    'num_layers': 1,
    'dropout': 0.15,
    'dense_size': 96,
    'learning_rate': 0.0003,
    'batch_size': 16,
    'weight_decay': 0.001,
    'max_epochs': 100,
    'early_stopping_patience': 30
}
```

### 확장 모델 (데이터 확장 시 예상 성능)
```python
{
    'hidden_size': 64,
    'num_layers': 2,
    'dropout': 0.2,
    'dense_size': 128,
    'learning_rate': 0.001,
    'batch_size': 32,
    'weight_decay': 0.0001,
    'max_epochs': 150,
    'early_stopping_patience': 40
}
```

## 🏗️ 프로젝트 구조

```
KLP-SignGlove-Clean/
├── src/
│   ├── models/                 # 모델 구현
│   │   ├── MSCSGRUModels.py    # CNN-GRU 변형들
│   │   ├── GRUModels.py        # GRU 관련
│   │   ├── LSTMModels.py       # LSTM 관련
│   │   ├── EncoderModels.py    # Transformer/CNN 인코더
│   │   ├── generalModels.py    # 공통 유틸
│   │   └── LightningModel.py   # Lightning 모듈
│   ├── misc/
│   │   └── DynamicDataModule.py
├── inference/                 # 추론 전용 시스템(분리)
│   ├── engine.py
│   ├── models/
│   └── utils/
├── best_model/
│   ├── best_model.ckpt        # 체크포인트
│   └── scaler.pkl             # 훈련 시 저장한 StandardScaler
├── final_results/
│   ├── results.json
│   └── project_summary.txt
├── requirements.txt
└── README.md
```

## 🚀 시작하기

### 1. 환경 설정
```bash
# 저장소 클론
git clone <repository-url>
cd KLP-SignGlove-Clean

# 의존성 설치
pip install -r requirements.txt
```

### 2. 데이터 준비
SignGlove_HW 프로젝트의 unified 데이터셋을 사용하세요.

### 3. 모델 훈련
(훈련 스크립트는 프로젝트 진행 상황에 맞게 구성)

### 4. 모델 로드 및 추론 (분리된 추론 엔진 사용)
```python
from inference import SignGloveInference
import numpy as np

engine = SignGloveInference(
    model_path='best_model/best_model.ckpt',
    device='cpu'  # 단건 예측은 CPU 저지연 권장
)

raw = np.random.randn(87, 8)
result = engine.predict_single(raw)
engine.print_prediction(result)
```

## 📈 성능 비교 (예시)
| 모델 | 초기 손실 | 특징 |
|------|-----------|------|
| CNN-GRU | ~3.58 | 단순/빠름 |
| CNN-StackedGRU | ~3.52 | 중간 복잡도 |
| MS-GRU | ~3.58 | 멀티스케일 특징 |
| MS-StackedGRU | ~3.47 | 표현력 높음 |

## 🔬 기술적 특징
- 멀티스케일 CNN: kernel 3/5/7 병렬 추출
- GRU 아키텍처: 단일/스택 구성 지원
- 전처리: 0.0 패딩, 훈련 스케일러 강제 사용(scaler.pkl), 가변 길이 리샘플링 옵션

## 🎯 주요 성과
- CNN-GRU 계열 4종 구현 및 분리된 추론 시스템 완성
- 실시간 추론 준비(경량 경로 및 API)

## 🚀 향후 계획
- 데이터셋 확장 및 성능 검증
- 하이퍼파라미터 최적화 및 과적합 방지
- 실시간 UI/서비스 연동

## 📚 참고 자료
- ASL-Sign-Research (아이디어 참고)
- SignGlove_HW (하드웨어 프로젝트)

## 📄 라이선스
MIT
