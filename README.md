# KLP-SignGlove: 한국어 수화 인식 프로젝트

한국어 수화 자모(자음, 모음) 인식을 위한 센서 장갑 기반 실시간 분류 시스템입니다.

## 🎯 프로젝트 개요

**목표**: 한국어 수화 자모 24개 클래스(자음 14개 + 모음 10개)를 실시간으로 인식하는 시스템 개발

**특징**: 
- **멀티스케일 CNN + GRU 하이브리드 모델**: 다양한 시간 스케일의 패턴을 동시에 학습
- **4가지 모델 변형**: CNN-GRU, CNN-StackedGRU, MS-GRU, MS-StackedGRU
- **모듈화된 구조**: 각 모델 타입별로 분리된 파일 구조
- **실시간 추론**: 경량 모델로 실시간 처리 가능

## 📊 데이터셋 정보

### SignGlove (우리) 데이터셋
- **총 샘플 수**: 7,200개
- **클래스 수**: 24개 (자음 14개 + 모음 10개)
- **타임스텝**: 87개
- **센서 채널**: 8개 (flex1-5 + pitch, roll, yaw)
- **샘플링 주파수**: 32.1 Hz
- **데이터 분할**: 훈련 70%, 검증 15%, 테스트 15%
- **전처리 방식**: 패딩/트렁케이션(0.0 패딩) + StandardScaler 정규화

### 클래스 목록
- **자음 (14개)**: ㄱ, ㄴ, ㄷ, ㄹ, ㅁ, ㅂ, ㅅ, ㅇ, ㅈ, ㅊ, ㅋ, ㅌ, ㅍ, ㅎ
- **모음 (10개)**: ㅏ, ㅑ, ㅓ, ㅕ, ㅗ, ㅛ, ㅜ, ㅠ, ㅡ, ㅣ

## 🏗️ 모델 아키텍처

### 1. CNN-GRU (Single-scale CNN + Single GRU)
```python
Conv1D(kernel=3) → BatchNorm → ReLU → MaxPool → Dropout
         ↓
Single GRU(64) → Dropout → Dense(128) → Dense(24)
```

### 2. CNN-StackedGRU (Single-scale CNN + Stacked GRU)
```python
Conv1D(kernel=3) → BatchNorm → ReLU → MaxPool → Dropout
         ↓
GRU1(64) → Dropout → GRU2(64) → Dropout → Dense(128) → Dense(24)
```

### 3. MS-GRU (Multi-scale CNN + Single GRU)
```python
Tower1(kernel=3) ┐
Tower2(kernel=5) ├→ Concat → BatchNorm → ReLU → MaxPool → Dropout
Tower3(kernel=7) ┘
         ↓
Single GRU(64) → Dropout → Dense(128) → Dense(24)
```

### 4. MS-StackedGRU (Multi-scale CNN + Stacked GRU)
```python
Tower1(kernel=3) ┐
Tower2(kernel=5) ├→ Concat → BatchNorm → ReLU → MaxPool → Dropout
Tower3(kernel=7) ┘
         ↓
GRU1(64) → Dropout → GRU2(64) → Dropout → Dense(128) → Dense(24)
```

## 📁 프로젝트 구조

```
KLP-SignGlove-Clean/
├── src/
│   ├── models/                    # 모델 구현
│   │   ├── MSCSGRUModels.py      # 4가지 CNN-GRU 변형 (메인)
│   │   ├── GRUModels.py          # GRU 관련 모델들
│   │   ├── LSTMModels.py         # LSTM 관련 모델들
│   │   ├── EncoderModels.py      # Transformer/CNN 인코더들
│   │   ├── generalModels.py      # 공통 모델 클래스
│   │   └── LightningModel.py     # PyTorch Lightning 기본 클래스
│   └── misc/
│       └── DynamicDataModule.py  # 데이터 로더 및 전처리
├── best_model/                   # 최고 성능 모델
│   ├── best_model.ckpt
│   └── results.json
├── final_results/               # 최종 실험 결과
│   ├── results.json
│   └── project_summary.txt
├── requirements.txt             # 의존성 패키지
├── README.md                   # 프로젝트 문서
├── README_our_version.md       # 상세 기술 문서
└── LICENSE                     # MIT 라이선스
```

## 🚀 시작하기

### 1. 환경 설정
```bash
# 저장소 클론
git clone https://github.com/Kyle-Riss/KLP-SignGlove.git
cd KLP-SignGlove-Clean

# 의존성 설치
pip install -r requirements.txt
```

### 2. 데이터 준비
SignGlove_HW 프로젝트의 unified 데이터셋을 사용합니다:
```bash
git clone https://github.com/KNDG01001/SignGlove_HW.git
# 데이터는 /home/billy/25-1kp/SignGlove_HW/datasets/unified/ 에 위치
```

### 3. 모델 테스트
```python
import torch
from src.models.MSCSGRUModels import CNNGRU, CNNStackedGRU, MSCGRU, MSCSGRU

# 테스트 데이터
batch_size, time_steps, input_channels = 4, 87, 8
num_classes = 34
x = torch.randn(batch_size, time_steps, input_channels)
x_mask = torch.ones(batch_size, time_steps)
y_targets = torch.randint(0, num_classes, (batch_size,))

# 모델 선택 및 테스트
model = MSCGRU(learning_rate=1e-3, input_size=8, hidden_size=64, classes=34)
logits, loss = model(x, x_mask, y_targets)
print(f"Output shape: {logits.shape}, Loss: {loss.item():.4f}")
```

### 4. 데이터 로더 테스트
```python
from src.misc.DynamicDataModule import DynamicDataModule

# 데이터 모듈 초기화
data_module = DynamicDataModule(
    data_dir="/home/billy/25-1kp/SignGlove_HW/datasets/unified",
    time_steps=87,
    batch_size=32
)

# 데이터 로드
data_module.setup(stage='fit')
train_loader = data_module.train_dataloader()

# 배치 확인
for batch in train_loader:
    print(f"Batch shape: {batch['measurement'].shape}")
    print(f"Labels: {batch['label']}")
    break
```

## 📈 모델 성능 비교

### 초기 손실값 (무작위 초기화)
| 모델 | 초기 손실 | 특징 |
|------|-----------|------|
| **CNN-GRU** | ~3.58 | 단순하고 빠름 |
| **CNN-StackedGRU** | ~3.52 | 중간 복잡도 |
| **MS-GRU** | ~3.58 | 멀티스케일 특징 추출 |
| **MS-StackedGRU** | ~3.47 | 가장 높은 표현력 |

### 모델 선택 가이드
- **빠른 실험**: `CNNGRU` - 가장 단순하고 빠름
- **균형잡힌 성능**: `MSCGRU` - 멀티스케일 + 단일 GRU
- **최고 성능**: `MSCSGRU` - 멀티스케일 + 스택 GRU
- **중간 복잡도**: `CNNStackedGRU` - 단일 CNN + 스택 GRU

## 🔬 기술적 특징

### 멀티스케일 CNN
- **3개 타워 병렬 처리**: kernel_size 3, 5, 7로 다양한 시간 스케일 패턴 추출
- **미세 패턴 (kernel=3)**: 짧은 시간 동안의 센서값 변화
- **중간 패턴 (kernel=5)**: 중간 시간 동안의 패턴
- **거시 패턴 (kernel=7)**: 긴 시간에 걸친 전체적인 패턴

### GRU 아키텍처
- **단일 GRU**: 파라미터 수 적음, 빠른 학습
- **스택 GRU**: 2층 구조로 복잡한 시간 의존성 학습

### 데이터 전처리
- **타임스텝 정규화**: 가변 길이 → 87 타임스텝
- **스케일링**: StandardScaler 적용
- **층화 샘플링**: 클래스 비율 유지하며 분할

## 🎯 주요 성과

✅ **4가지 모델 변형 구현**: 다양한 복잡도와 성능 트레이드오프 제공  
✅ **모듈화된 구조**: 각 모델 타입별로 분리된 파일 구조  
✅ **새로운 데이터셋 지원**: SignGlove_HW unified 데이터셋 (7,200개 샘플, 34개 클래스)  
✅ **멀티스케일 특징 추출**: 다양한 시간 스케일 패턴 동시 학습  
✅ **실시간 추론 준비**: 경량 모델로 실시간 처리 가능  

## 🚀 향후 계획

### 단기 계획
- [ ] 각 모델별 성능 평가 및 비교
- [ ] 하이퍼파라미터 최적화
- [ ] 과적합 방지 기법 적용
- [ ] 실시간 추론 시스템 구축

### 장기 계획
- [ ] 웹 기반 수화 통역 서비스
- [ ] 모바일 앱 개발
- [ ] 다국어 수화 지원 확장
- [ ] 클라우드 기반 서비스 구축

## 📚 참고 자료

- [ASL-Sign-Research](https://github.com/adityamakkar000/ASL-Sign-Research): 원본 ASL 프로젝트
- [SignGlove_HW](https://github.com/KNDG01001/SignGlove_HW): 하드웨어 구현 프로젝트
- [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/): 모델 프레임워크

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

## 🤝 기여하기

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📞 문의

프로젝트에 대한 문의사항이 있으시면 이슈를 생성해 주세요.