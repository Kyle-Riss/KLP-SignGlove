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
- **총 샘플 수**: 1,444개
- **클래스 수**: 24개 (자음 14개 + 모음 10개)
- **타임스텝**: 87개
- **센서 채널**: 8개 (flex1-5 + pitch, roll, yaw)
- **샘플링 주파수**: 32.1 Hz
- **데이터 분할**: 훈련 60%, 검증 20%, 테스트 20%
- **전처리 방식**: 제로 패딩(0.0) + StandardScaler 정규화

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
│   ├── experiments/              # 학습 스크립트
│   │   ├── LightningTrain.py     # PyTorch Lightning 학습
│   │   └── ModelTrain.sh         # 모델별 학습 실행
│   └── misc/
│       └── DynamicDataModule.py  # 데이터 로더 및 전처리
├── visualizations/              # 학습 곡선 시각화
│   ├── GRU/                     # GRU 모델 결과
│   ├── CNNGRU/                  # CNNGRU 모델 결과
│   └── MSCSGRU/                 # MSCSGRU 모델 결과
├── analyze_training.py          # 학습 결과 분석 스크립트
├── requirements.txt             # 의존성 패키지
├── README.md                   # 프로젝트 문서
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

### 3. 모델 학습 및 평가
```bash
# 개별 모델 학습
cd src/experiments
bash ModelTrain.sh RNN GRU
bash ModelTrain.sh MSCSGRU CNNGRU
bash ModelTrain.sh MSCSGRU MSCSGRU

# 학습 결과 시각화
cd ../../
python3 analyze_training.py -model GRU
python3 analyze_training.py -model CNNGRU
python3 analyze_training.py -model MSCSGRU
```

### 4. 모델 성능 확인
```bash
# 학습 로그에서 최종 성능 확인
tail -20 training_output_RNN_GRU.log
tail -20 training_output_MSCSGRU_CNNGRU.log
tail -20 training_output_MSCSGRU_MSCSGRU.log

# 시각화 파일 확인
ls visualizations/GRU/
ls visualizations/CNNGRU/
ls visualizations/MSCSGRU/
```

## 📈 모델 성능 비교

### 최종 학습 결과 (Early Stopping 적용)
| 모델 | Train Loss | Val Loss | Train Acc | Val Acc | Epochs | 특징 |
|------|------------|----------|-----------|---------|--------|------|
| **GRU** | 0.027 | 0.032 | 99.7% | 99.0% | 58 | 정상적 학습 패턴 |
| **CNNGRU** | 0.138 | 0.052 | 95.3% | 99.3% | 100 | 큰 Loss Gap (과도한 정규화) |
| **MSCSGRU** | 0.092 | 0.043 | 98.0% | 99.3% | 58 | 균형잡힌 성능 |

### 모델 선택 가이드
- **최고 안정성**: `GRU` - 정상적인 학습 패턴, 높은 안정성
- **균형잡힌 성능**: `MSCSGRU` - 멀티스케일 + 적절한 정규화
- **빠른 학습**: `CNNGRU` - 단일 CNN + GRU, 빠른 수렴

### 핵심 개선사항
✅ **데이터 누수 해결**: 독립적인 Train/Val/Test 분할  
✅ **Early Stopping**: 과적합 방지 및 효율적 학습  
✅ **정상적 학습 패턴**: Train > Val 성능 관계 복원  
✅ **X축 최적화**: 실제 학습 epoch까지만 시각화

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

✅ **3가지 핵심 모델 구현**: GRU, CNNGRU, MSCSGRU 모델 비교  
✅ **데이터 누수 해결**: 독립적인 Train/Val/Test 분할로 신뢰성 있는 평가  
✅ **Early Stopping**: 과적합 방지 및 효율적 학습  
✅ **실시간 시각화**: 학습 곡선 실시간 모니터링  
✅ **정상적 학습 패턴**: Train > Val 성능 관계 복원  
✅ **멀티스케일 특징 추출**: 다양한 시간 스케일 패턴 동시 학습  

## 🚀 향후 계획

### 단기 계획
- [x] 각 모델별 성능 평가 및 비교 (완료)
- [x] Early Stopping 구현 (완료)
- [x] 데이터 누수 문제 해결 (완료)
- [ ] 하이퍼파라미터 최적화
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