# SignGlove: 한글 자음/모음 인식 시스템

## 📋 프로젝트 개요

SignGlove는 **데이터 장갑(Data Glove)**을 사용하여 **한글 자음과 모음**을 실시간으로 인식하는 딥러닝 시스템입니다. Flex 센서와 IMU 센서 데이터를 활용하여 24개의 한글 문자(14개 자음, 10개 모음)를 높은 정확도로 분류합니다.

## 🎯 주요 성과

- **평균 신뢰도**: 86.3%
- **처리 속도**: 0.7ms (평균 추론 시간)
- **인식 문자**: 24개 한글 자음/모음 (ㄱ,ㄴ,ㄷ,ㄹ,ㅁ,ㅂ,ㅅ,ㅇ,ㅈ,ㅊ,ㅋ,ㅌ,ㅍ,ㅎ,ㅏ,ㅑ,ㅓ,ㅕ,ㅗ,ㅛ,ㅜ,ㅠ,ㅡ,ㅣ)
- **모델 정확도**: 94.78% (검증 데이터 기준)
- **과적합 해결**: 우수한 일반화 성능 (과적합 지수: 0.1766)

## 🏗️ 시스템 아키텍처

### 모델 구조
- **기본 모델**: RGRU (정규화된 GRU 기반)
- **입력**: 8차원 센서 데이터 (5개 Flex + 3개 IMU)
- **시퀀스 길이**: 300 (패딩/자르기 적용)
- **출력**: 24개 클래스 확률 분포

### 핵심 구성 요소
1. **RGRU**: 메인 분류 모델
2. **ClassDiscriminator**: ㄹ/ㅕ 후처리 필터 (Random Forest) -> 데이터 보완 시 Post filter는 수정 가능합니다
3. **데이터 전처리**: 정규화 및 시퀀스 패딩
4. **실시간 추론**: 배치 및 단일 파일 처리

## 📊 데이터 구조

### 센서 데이터
- **Flex 센서**: 5개 (0-1023 범위 → 0-1 정규화)
- **IMU 센서**: 3개 (가속도계, 절댓값 후 정규화)
- **총 특성**: 8차원

### 데이터 폴더 구조
```
real_data_filtered/
├── ㄱ/
│   ├── 1/
│   ├── 2/
│   ├── 3/
│   ├── 4/
│   └── 5/
├── ㄴ/
│   └── ...
├── ...
└── ㅣ/
    └── ...
```

### 데이터 분할 방식
- **훈련**: 각 폴더에서 3개씩 (60%)
- **검증**: 각 폴더에서 1개씩 (20%)
- **테스트**: 각 폴더에서 1개씩 (20%)
- **총 데이터**: 575개 파일

## 🚀 사용 방법

### 1. 환경 설정

```bash
# 필요한 패키지 설치
pip install torch torchvision torchaudio
pip install pandas numpy matplotlib seaborn
pip install scikit-learn
```

### 2. 모델 훈련

```bash
# 개선된 모델 훈련
python3 improved_training.py
```

### 3. 추론 실행

```bash
# 정규화된 데이터로 추론 (권장)
python3 corrected_filtered_inference.py

# 원시 데이터로 추론 (전처리 필요)
python3 enhanced_inference_with_improved_model.py
```

### 4. 결과 분석

```bash
# 추론 정확도 분석
python3 analyze_inference_accuracy.py

# 학습 곡선 생성
python3 real_training_curves.py
```

## 📁 파일 구조

### 핵심 파일
```
KLP-SignGlove-Clean/
├── README.md                           # 프로젝트 문서
├── improved_model_architecture.py      # 모델 아키텍처 정의
├── improved_training.py                # 모델 훈련 시스템
├── corrected_filtered_inference.py     # 정규화된 데이터 추론
├── class_discriminator.py              # ㄹ/ㅕ 차별화기
├── analyze_inference_accuracy.py       # 추론 정확도 분석
├── real_training_curves.py             # 학습 곡선 생성
├── models/
│   └── improved_regularized_model.pth  # 훈련된 모델
└── data/
    ├── real_data/                      # 원시 센서 데이터
    └── real_data_filtered/             # 정규화된 데이터
```

### 결과 파일
- `corrected_filtered_results.json`: 추론 결과
- `real_training_curves.png`: 학습 곡선
- `inference_accuracy_analysis.png`: 정확도 분석
- `training_report.txt`: 훈련 보고서

## 🔧 모델 아키텍처

### RGRU
```python
class RGRU(nn.Module):
    def __init__(self, input_size=8, hidden_size=96, num_classes=24, dropout=0.5):
        # 입력 정규화
        self.input_norm = nn.LayerNorm(input_size)
        
        # 특성 추출기
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_size, hidden_size), nn.ReLU(), nn.Dropout(dropout), nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Dropout(dropout), nn.LayerNorm(hidden_size)
        )
        
        # GRU 레이어
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True, dropout=dropout)
        
        # 어텐션 메커니즘
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2), nn.Tanh(),
            nn.Linear(hidden_size // 2, 1)
        )
        
        # 분류기
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2), nn.ReLU(), nn.Dropout(dropout), nn.LayerNorm(hidden_size // 2),
            nn.Linear(hidden_size // 2, hidden_size // 4), nn.ReLU(), nn.Dropout(dropout), nn.LayerNorm(hidden_size // 4),
            nn.Linear(hidden_size // 4, num_classes)
        )
```

### 주요 특징
- **LayerNorm**: 입력 및 중간 레이어 정규화
- **Dropout**: 과적합 방지 (0.5)
- **GRU**: 시퀀스 데이터 처리
- **Attention**: 중요 시점 집중
- **Multi-layer Classifier**: 복잡한 패턴 학습

## 📈 성능 지표

### 클래스별 정확도 (신뢰도 기준)
| 클래스 | 정확도 | 샘플 수 | 상태 |
|--------|--------|---------|------|
| ㄴ | 99.4% | 25 | ✅ |
| ㅌ | 99.0% | 25 | ✅ |
| ㅎ | 98.9% | 24 | ✅ |
| ㅊ | 98.9% | 25 | ✅ |
| ㅋ | 98.7% | 24 | ✅ |
| ㄱ | 97.7% | 25 | ✅ |
| ㄷ | 97.8% | 19 | ✅ |
| ㅇ | 97.9% | 25 | ✅ |
| ㅗ | 98.4% | 25 | ✅ |
| ㅣ | 95.8% | 25 | ✅ |
| ㅂ | 96.4% | 25 | ✅ |
| ㅁ | 94.3% | 25 | ✅ |
| ㅍ | 87.1% | 25 | ⚠️ |
| ㅛ | 89.0% | 25 | ⚠️ |
| ㅓ | 91.6% | 25 | ⚠️ |
| ㅑ | 85.7% | 25 | ⚠️ |
| ㄹ | 61.7% | 42 | ❌ |
| ㅅ | 67.8% | 25 | ❌ |
| ㅈ | 70.2% | 25 | ❌ |
| ㅏ | 56.6% | 29 | ❌ |
| ㅜ | 56.9% | 19 | ❌ |
| ㅕ | 50.8% | 1 | ❌ |

### 전체 성능
- **높은 신뢰도 (>80%)**: 424개 (73.7%)
- **중간 신뢰도 (60-80%)**: 92개 (16.0%)
- **낮은 신뢰도 (<60%)**: 59개 (10.3%)

## 🔍 후처리 필터

### ㄹ/ㅕ 차별화기
- **모델**: Random Forest
- **특성**: Flex5_mean, Flex3_mean
- **목적**: ㄹ과 ㅕ의 혼동 해결
- **정확도**: 100% (훈련 데이터 기준)

## ⚠️ 주의사항

### 데이터 전처리
1. **원시 데이터 사용 시**: 반드시 정규화 필요
   - Flex 센서: 0-1023 → 0-1
   - IMU 센서: 절댓값 후 최대값 정규화
2. **정규화된 데이터 사용**: 바로 추론 가능

### 모델 로딩
- 체크포인트 키: `model_state_dict`
- 모델 타입: `RGRU`
- 입력 형태: `(batch_size, sequence_length, features)`

## 🛠️ 문제 해결

### 일반적인 오류
1. **모델 로딩 실패**: `fix_model_loading.py` 실행
2. **데이터 불일치**: `real_data_filtered` 사용
3. **메모리 부족**: 배치 크기 조정

### 성능 개선
1. **낮은 정확도 클래스**: 추가 데이터 수집
2. **과적합**: Dropout 비율 조정
3. **느린 추론**: 모델 경량화

## 📚 참고 자료

### 논문 및 기술
- GRU (Gated Recurrent Unit)
- Attention Mechanism
- Layer Normalization
- Dropout Regularization

### 관련 프로젝트
- Sign Language Recognition
- Gesture Recognition
- Sensor Data Processing

## 🤝 기여 방법

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

## 👥 팀원

프로젝트 관리자: minuum
KLP-SignGlove 팀: Kyle-Riss
SignGlove_HW 팀: KNDG01001

## 📞 문의

프로젝트에 대한 문의사항이 있으시면 이슈를 생성해주세요.

---

**SignGlove: 한글 자음/모음 인식의 새로운 패러다임** 🚀
