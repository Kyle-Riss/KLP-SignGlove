# KLP-SignGlove: 한국어 수화 인식 프로젝트

한국어 수화 자모(자음, 모음) 인식을 위한 센서 장갑 기반 실시간 분류 시스템입니다.

## 🎯 프로젝트 개요

**목표**: 한국어 수화 자모 24개 클래스(자음 14개 + 모음 10개)를 실시간으로 인식하는 시스템 개발

**성과**: 
- **현재 성능**: 97.5% 정확도 (598개 샘플)
- **목표 성능**: 98.5% 정확도 (3000개 샘플 예정)
- **데이터셋**: 598개 → 3000개 확장 예정
- **실시간 추론**: 경량 모델로 실시간 처리 가능

## 📊 데이터셋 정보

### 현재 데이터셋 (598개)
- **총 샘플 수**: 598개
- **클래스 수**: 24개 (자음 14개 + 모음 10개)
- **타임스텝**: 100개 (3.12초 지속시간)
- **센서 채널**: 8개 (flex1-5 + pitch, roll, yaw)
- **샘플링 주파수**: 32.1 Hz
- **데이터 분할**: 훈련 59.9%, 검증 20.1%, 테스트 20.1%

### 확장 예정 데이터셋 (3000개)
- **총 샘플 수**: 3000개 (5배 증가)
- **클래스 수**: 24개 (동일)
- **타임스텝**: 100개 (동일)
- **센서 채널**: 8개 (동일)
- **예상 성능**: 98.5% (현재 97.5% 대비 +1.0% 향상)

### 클래스 목록
- **자음 (14개)**: ㄱ, ㄴ, ㄷ, ㄹ, ㅁ, ㅂ, ㅅ, ㅇ, ㅈ, ㅊ, ㅋ, ㅌ, ㅍ, ㅎ
- **모음 (10개)**: ㅏ, ㅑ, ㅓ, ㅕ, ㅗ, ㅛ, ㅜ, ㅠ, ㅡ, ㅣ

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

### 확장 모델 (3000개 데이터셋 - 예상 98.5% 성능)
```python
{
    'hidden_size': 64,        # 48 → 64 (더 큰 모델)
    'num_layers': 2,          # 1 → 2 (더 깊은 네트워크)
    'dropout': 0.2,           # 0.15 → 0.2 (과적합 방지)
    'dense_size': 128,        # 96 → 128 (더 큰 Dense)
    'learning_rate': 0.001,   # 0.0003 → 0.001 (더 큰 학습률)
    'batch_size': 32,         # 16 → 32 (더 큰 배치)
    'weight_decay': 0.0001,   # 0.001 → 0.0001 (약한 정규화)
    'max_epochs': 150,        # 100 → 150 (더 많은 에포크)
    'early_stopping_patience': 40  # 30 → 40 (더 긴 patience)
}
```

## 🏗️ 프로젝트 구조

```
KLP-SignGlove-Clean/
├── src/
│   ├── models/                 # 모델 구현
│   │   ├── GRU.py             # 1층 GRU
│   │   ├── LSTM.py            # 1층 LSTM
│   │   ├── encoder.py         # Transformer Encoder
│   │   ├── generalModels.py   # 공통 모델 클래스
│   │   └── LightningModel.py  # PyTorch Lightning 기본 클래스
│   ├── misc/
│   │   └── DynamicDataModule.py  # 데이터 로더
│   └── StackedGRUModel.py     # 최고 성능 모델
├── best_model/
│   ├── best_model.ckpt        # 최고 성능 모델 체크포인트
│   └── results.json           # 성능 결과
├── final_results/
│   ├── results.json           # 최종 실험 결과
│   └── project_summary.txt    # 프로젝트 요약
├── optimal_config.py          # 현재 최적 하이퍼파라미터 설정
├── optimal_config_3000.py     # 3000개 데이터셋용 설정
├── train_optimal_model.py     # 현재 모델 훈련 스크립트
├── train_3000_dataset.py      # 3000개 데이터셋용 훈련 스크립트
├── requirements.txt           # 의존성 패키지
└── README.md                  # 프로젝트 문서
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
SignGlove-DataAnalysis 폴더를 프로젝트 루트에 배치하세요.

### 3. 모델 훈련

#### 현재 데이터셋 (598개) 훈련
```bash
# 현재 최적 모델 훈련
python train_optimal_model.py
```

#### 확장 데이터셋 (3000개) 훈련
```bash
# 3000개 데이터셋용 최적화 모델 훈련
python train_3000_dataset.py
```

### 4. 모델 로드 및 추론
```python
import torch
from src.StackedGRUModel import StackedGRULightning
from optimal_config import OPTIMAL_CONFIG

# 모델 로드
model = StackedGRULightning.load_from_checkpoint('best_model/best_model.ckpt')
model.eval()

# 추론
with torch.no_grad():
    predictions = model(input_data)
```

## 📈 성능 비교

### 현재 성능 (598개 데이터셋)
| 모델 | 테스트 정확도 | 특징 |
|------|---------------|------|
| **StackedGRU** | **97.5%** | 🥇 최고 성능 + 빠른 속도 |
| GRU | 89.2% | 안정적 성능 + 매우 빠른 속도 |
| LSTM | ~90% | 균형잡힌 성능 |
| Encoder | ~94% | Transformer 기반 |

### 예상 성능 (3000개 데이터셋)
| 모델 | 예상 정확도 | 개선 요인 |
|------|-------------|-----------|
| **StackedGRU (확장)** | **98.5%** | 더 큰 모델 + 더 많은 데이터 |
| GRU (확장) | ~92% | 더 큰 모델 + 더 많은 데이터 |
| LSTM (확장) | ~94% | 더 큰 모델 + 더 많은 데이터 |
| Encoder (확장) | ~96% | 더 큰 모델 + 더 많은 데이터 |

## 🔬 기술적 특징

### 데이터 전처리
- **타임스텝 정규화**: 가변 길이 → 100 타임스텝
- **스케일링**: StandardScaler 적용
- **데이터 증강**: 시드 기반 재현 가능한 분할

### 모델 아키텍처
- **StackedGRU**: 2층 GRU + Dense 레이어
- **드롭아웃**: 0.15 (과적합 방지)
- **정규화**: Weight Decay 0.001

### 훈련 전략
- **옵티마이저**: AdamW
- **스케줄러**: ReduceLROnPlateau
- **조기 종료**: 30 에포크 patience

## 🎯 주요 성과

✅ **성능 목표 초과 달성**: 70% → 97.5% (39% 초과)  
✅ **데이터 로더 버그 수정**: 클래스 중복 문제 해결  
✅ **하이퍼파라미터 최적화**: 체계적인 실험을 통한 최적 설정 도출  
✅ **실시간 추론 준비**: 경량 모델로 실시간 처리 가능  
✅ **SignGlove_HW 호환성**: 하드웨어 프로젝트와 완벽 연동  

## 🚀 향후 계획

### 단기 계획
- [x] 현재 데이터셋 최적화 완료 (598개, 97.5%)
- [ ] 3000개 데이터셋 훈련 및 검증
- [ ] 확장 모델 성능 평가
- [ ] 실시간 추론 시스템 구축
- [ ] 사용자 인터페이스 개발

### 장기 계획
- [ ] 웹 기반 수화 통역 서비스
- [ ] 모바일 앱 개발
- [ ] 다국어 수화 지원 확장
- [ ] 클라우드 기반 서비스 구축

## 📚 참고 자료

- [ASL-Sign-Research](https://github.com/adityamakkar000/ASL-Sign-Research): 원본 ASL 프로젝트
- [SignGlove_HW](https://github.com/your-username/SignGlove_HW): 하드웨어 구현 프로젝트

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