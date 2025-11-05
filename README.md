# SignGlove: 한국 수어 인식 딥러닝 프로젝트

> Multi-Scale 3D CNN + GRU 기반 한국 수어 손동작 인식 시스템  
> **최고 성능: 99.13% 정확도** 달성

---

## 📋 목차

1. [프로젝트 개요](#-프로젝트-개요)
2. [데이터셋](#-데이터셋)
3. [실험 과정 및 결과](#-실험-과정-및-결과)
4. [모델 성능](#-모델-성능)
5. [모델 아키텍처](#-모델-아키텍처)
6. [추론 시스템](#-추론-시스템)
7. [하드웨어 통합](#-하드웨어-통합)
8. [프로젝트 구조](#-프로젝트-구조)
9. [Quick Start](#-quick-start)

---

## 🎯 프로젝트 개요

### 목표
8채널 센서 데이터를 활용한 한국 수어 24개 자모(자음 14개 + 모음 10개) 인식 시스템 개발

### 핵심 성과
- 🥇 **최고 정확도**: **99.13%** (MS3DGRU)
- 📊 **테스트셋 성능**: 577개 샘플 기준
- 💾 **모델 크기**: 723KB (MS3DGRU 체크포인트)
- ⚡ **추론 속도**: CPU 기준 ~10ms/샘플
- 🎯 **F1-Score**: 0.9913

---

## 📊 데이터셋

### 데이터 구성
- **총 샘플**: 2,884개 (unified 데이터셋)
- **클래스**: 24개
  - **자음** (14개): ㄱ, ㄴ, ㄷ, ㄹ, ㅁ, ㅂ, ㅅ, ㅇ, ㅈ, ㅊ, ㅋ, ㅌ, ㅍ, ㅎ
  - **모음** (10개): ㅏ, ㅑ, ㅓ, ㅕ, ㅗ, ㅛ, ㅜ, ㅠ, ㅡ, ㅣ
- **센서**: 8채널
  - **Flex 센서** (5개): flex1, flex2, flex3, flex4, flex5
  - **IMU 센서** (3개): pitch, roll, yaw
- **시퀀스 길이**: 최대 87 타임스텝 (padding 처리)
- **데이터 경로**: `/home/billy/25-1kp/SignGlove-DataAnalysis/unified/unified`

### 데이터 전처리
- **정규화**: StandardScaler 적용 (훈련 데이터로 학습된 scaler)
- **Padding**: 시퀀스 길이 87로 통일 (짧은 시퀀스는 zero-padding)
- **Scaler 파일**: `archive/checkpoints_backup/checkpoints_backup/scaler.pkl`

---

## 🔬 실험 과정 및 결과

### Phase 1: 기본 모델 비교 (GRU, StackedGRU, LSTM)

**목적**: 다양한 기본 RNN 모델들의 성능 비교

**실험 모델:**
- `GRU`: 기본 GRU (1층, hidden_size=64)
- `StackedGRU`: 다층 GRU (2층, hidden_size=64)
- `LSTM` / `StackedLSTM`: LSTM 변형

**결과:**
- **GRU**: 98.79% 정확도 (가장 안정적이고 높은 성능)
- **StackedGRU**: 93.93% 정확도 (파라미터 대비 성능 낮음)

**결론**: 단일 GRU가 더 효율적이고 성능이 우수

---

### Phase 2: CNN-GRU 하이브리드 모델

**목적**: CNN으로 공간 특성을 추출한 후 GRU로 시계열 처리

**실험 모델:**
- `CNNGRU`: 단일 스케일 1D CNN + GRU
- `CNNStackedGRU`: 1D CNN + Stacked GRU
- `MSCGRU`: Multi-Scale 1D CNN + GRU
- `MSCSGRU`: Multi-Scale 1D CNN + Stacked GRU

**결과:**
- 1D CNN이 시간 패턴은 잘 포착하지만 공간 특성 활용 미흡
- MSCGRU가 98.44%로 좋은 성능 (GRU와 유사)
- **결론**: 1D CNN만으로는 한계 확인

---

### Phase 3: 3D CNN 모델 개발

**목적**: 센서 간 공간적 상호작용을 학습하기 위한 3D CNN 도입

#### 3.1 2D CNN 시도 (MS2DGRU)
**아이디어**: 센서를 2D 공간으로 배치하여 2D CNN 적용
- **결과**: ❌ 성능 저하 (공간 구조가 임의적)
- **교훈**: 센서 배치가 의미 있는 공간 구조를 만들지 못함

#### 3.2 3D CNN 개발 (MS3DGRU)
**아이디어**: 
- 8개 센서를 4x2 공간으로 재배치
- 시간-공간 특성을 동시에 학습하는 3D CNN 적용
- Multi-Scale CNN (3x3x3, 5x5x5, 7x7x7 커널) 병렬 처리

**구조:**
```
입력: (batch, time, 8) → (batch, time, 4, 2)
     ↓
3D CNN (Multi-Scale): 3개 타워 병렬
  - Tower 1: Conv3d(3x3x3) - 세밀한 특성
  - Tower 2: Conv3d(5x5x5) - 중간 특성  
  - Tower 3: Conv3d(7x7x7) - 거시적 특성
     ↓
결합 → MaxPool3d((2, 4, 2)) → GRU → 분류기
```

**최적화 과정:**
1. 초기: 차원 불일치 오류 → `permute`와 `contiguous().view()` 조정
2. MaxPool3d 커널 크기: `(2, 4, 2)`로 시간/공간 차원 최적화
3. Dropout 조정: 0.1이 최적
4. 추가 Conv 레이어: 2단계 CNN으로 성능 향상

**최종 결과:**
- ✅ **99.13% 정확도 달성** (테스트셋 577개 샘플 기준)
- **F1-Score**: 0.9913
- **Precision**: 0.9919
- **Recall**: 0.9913
- **안정성**: 매우 안정적 (클래스 불균형 없음)

#### 3.3 MS3DStackedGRU 개발
**목적**: 3D CNN + Stacked GRU로 추가 성능 향상 시도

**결과:**
- **97.92% 정확도** (MS3DGRU보다 낮음)
- 파라미터 증가 (167K vs 58K)
- **결론**: 단일 GRU가 더 효율적

---

### Phase 4: 고급 기법 적용

#### 4.1 Residual Connections
**모델**: `ResidualGRU`
- ResNet 스타일 residual connection 적용
- **결과**: 98.78% (MS3DGRU보다 낮음)

#### 4.2 Attention Mechanisms
**모델**: `AttentionGRU`, `TransformerGRU`
- Multi-head Attention 적용
- **결과**: ~95% (기대 이하)

#### 4.3 Sensor-Aware 모델
**모델**: `SensorAwareGRU`, `SensorAwareCNNGRU`, `SensorAwareMultiScaleGRU`, `SensorAware3DGRU`
- Yaw/Pitch/Roll과 Flex 1-5를 분리 처리
- **결과**: ❌ 성능 저하 (~90-96%)
- **이유**: 센서 간 강한 상관관계로 분리가 오히려 정보 손실

**교훈**: 
- 센서 간 상관관계가 강해서 통합 처리가 더 효과적
- 공간적 구조가 명확하지 않은 경우 센서 분리 효과 없음

---

### Phase 5: Unified 데이터셋 재훈련

**목적**: 통합된 데이터셋으로 최종 모델 재훈련

**데이터셋:**
- **unified**: 전체 2,884개 샘플 통합
- **훈련/검증/테스트 분할**: 동적 분할 (DynamicDataModule)

**재훈련 모델:**
1. **GRU** (1층): 98.79% 정확도
2. **StackedGRU** (2층): 93.93% 정확도
3. **MS3DGRU**: 99.13% 정확도
4. **MS3DStackedGRU**: 97.92% 정확도

**최종 선택 모델: MS3DGRU**
- ✅ 최고 성능 (99.13%)
- ✅ 안정적인 성능
- ✅ 적절한 모델 크기 (723KB)

---

### Phase 6: 하이퍼파라미터 최적화

**최적화 항목:**
- **Learning Rate**: 0.0001, 0.001, 0.005, 0.01
- **Batch Size**: 32, 64, 128
- **Dropout**: 0.01, 0.05, 0.1, 0.2
- **Epochs**: 100 (Early Stopping 적용)

**MS3DGRU 최적 설정:**
- Learning Rate: 0.001
- Batch Size: 64
- Dropout: 0.1
- Hidden Size: 64
- CNN Filters: 32

---

## 📊 모델 성능

### 최종 모델 성능 비교 (테스트셋 577개 샘플)

| 순위 | 모델 | Accuracy | F1-Score | Precision | Recall | 파라미터 | 체크포인트 경로 |
|------|------|----------|----------|-----------|--------|----------|----------------|
| 🥇 1 | **MS3DGRU** | **99.13%** | **0.9913** | 0.9919 | 0.9913 | 58,840 | `best_model/ms3dgru_best.ckpt` |
| 🥈 2 | **GRU** | **98.79%** | **0.9879** | 0.9881 | 0.9879 | ~25K | `checkpoints/best_model_epoch=epoch=92_val/loss=val/loss=0.04.ckpt` |
| 🥉 3 | **MS3DStackedGRU** | **97.92%** | **0.9792** | 0.9803 | 0.9792 | 167,032 | `checkpoints/best_model_epoch=epoch=82_val/loss=val/loss=0.05.ckpt` |
| 4 | **StackedGRU** | 93.93% | 0.9367 | 0.9470 | 0.9393 | 50,584 | `checkpoints/best_model_epoch=epoch=68_val/loss=val/loss=0.19.ckpt` |

### 성능 분석

**클래스별 성능:**
- 모든 클래스에서 균형적인 성능 (클래스 불균형 없음)
- Accuracy와 F1-Score가 거의 동일 (모든 클래스의 Precision ≈ Recall)
- 오분류율: 0.87% (MS3DGRU 기준)

**모델 효율성:**
- MS3DGRU: 최고 성능 + 적절한 모델 크기 (723KB)
- GRU: 높은 성능 + 작은 모델 크기 (598KB)
- MS3DStackedGRU: 높은 성능이지만 모델 크기 큼 (2.0MB)

---

## 🏗️ 모델 아키텍처

### MS3DGRU (Multi-Scale 3D CNN + GRU)

**아키텍처:**
```
입력: (batch, timesteps=87, channels=8)
     ↓
Reshape: (batch, timesteps=87, height=4, width=2)
     ↓
Multi-Scale 3D CNN (3개 타워 병렬):
  ├─ Tower 1: Conv3d(3x3x3) → Conv3d(3x3x3) → BatchNorm → ReLU
  ├─ Tower 2: Conv3d(5x5x5) → Conv3d(3x3x3) → BatchNorm → ReLU
  └─ Tower 3: Conv3d(7x7x7) → Conv3d(3x3x3) → BatchNorm → ReLU
     ↓
Concatenate → MaxPool3d(2, 4, 2)
     ↓
Flatten → GRU(hidden_size=64) → Dropout(0.1)
     ↓
Output Layers → 24 classes
```

**하이퍼파라미터:**
- Input Size: 8 (센서 채널)
- Hidden Size: 64
- CNN Filters: 32
- Dropout: 0.1
- Target Timesteps: 87

---

## 🚀 추론 시스템

### 추론 가중치 파일 위치

| 모델 | 체크포인트 경로 | 크기 | 상태 |
|------|----------------|------|------|
| **MS3DGRU** | `best_model/ms3dgru_best.ckpt` | 723KB | ✅ 사용 중 |
| **GRU** | `checkpoints/best_model_epoch=epoch=92_val/loss=val/loss=0.04.ckpt` | 598KB | ✅ 사용 중 |
| **MS3DStackedGRU** | `checkpoints/best_model_epoch=epoch=82_val/loss=val/loss=0.05.ckpt` | 2.0MB | ✅ 사용 중 |
| **StackedGRU** | `checkpoints/best_model_epoch=epoch=68_val/loss=val/loss=0.19.ckpt` | 609KB | ✅ 사용 중 |
| **Scaler** | `archive/checkpoints_backup/checkpoints_backup/scaler.pkl` | 641B | ✅ 필수 |

### 추론 시스템 구조

```
inference/
├── engine.py              # 통합 추론 엔진 (고수준 API)
├── models/                # 추론용 모델 정의
│   ├── ms3dgru_inference.py
│   ├── gru_inference.py
│   ├── stackedgru_inference.py
│   ├── ms3dstackedgru_inference.py
│   └── mscsgru_inference.py
├── utils/                 # 전처리/후처리 유틸리티
│   ├── preprocessor.py
│   └── postprocessor.py
└── examples/              # 사용 예제
    ├── generate_confusion_matrices.py  # 혼동 행렬 생성
    ├── single_predict.py      # 단일 샘플 예측
    ├── batch_predict.py       # 배치 예측
    └── test_all_models.py     # 전체 모델 테스트
```

### Python API 사용법

#### 1. 기본 사용법

```python
import numpy as np
from inference import SignGloveInference

# 추론 엔진 초기화
engine = SignGloveInference(
    model_path='best_model/ms3dgru_best.ckpt',
    model_type='MS3DGRU',
    scaler_path='archive/checkpoints_backup/checkpoints_backup/scaler.pkl',
    device='cpu'  # 또는 'cuda'
)

# 센서 데이터 준비 (Shape: (timesteps, 8))
# 채널 순서: [flex1, flex2, flex3, flex4, flex5, pitch, roll, yaw]
raw_data = np.random.randn(87, 8).astype(np.float32)

# 예측 수행
result = engine.predict_single(raw_data, top_k=5, normalize=False)

# 결과 출력
print(f"예측 클래스: {result['predicted_class']}")
print(f"확률: {result['confidence']:.4f}")
print("\n상위 5개 예측:")
for i, pred in enumerate(result['top_k_predictions'], 1):
    print(f"  {i}. {pred['class']}: {pred['confidence']:.4f}")
```

#### 2. 여러 모델 비교

```python
from inference import SignGloveInference

models_config = {
    'MS3DGRU': {
        'path': 'best_model/ms3dgru_best.ckpt',
        'type': 'MS3DGRU',
        'cnn_filters': 32,
        'dropout': 0.1
    },
    'GRU': {
        'path': 'checkpoints/best_model_epoch=epoch=92_val/loss=val/loss=0.04.ckpt',
        'type': 'GRU',
        'hidden_size': 64,
        'layers': 1,
        'dropout': 0.2
    }
}

results = {}
for model_name, config in models_config.items():
    engine = SignGloveInference(
        model_path=config['path'],
        model_type=config['type'],
        scaler_path='archive/checkpoints_backup/checkpoints_backup/scaler.pkl',
        **{k: v for k, v in config.items() if k not in ['path', 'type']}
    )
    
    result = engine.predict_single(raw_data, normalize=False)
    results[model_name] = result
    print(f"{model_name}: {result['predicted_class']} ({result['confidence']:.4f})")
```

#### 3. 혼동 행렬 생성

```bash
# 모든 모델의 혼동 행렬 생성
python inference/examples/generate_confusion_matrices.py

# 생성되는 파일:
# - visualizations/confusion_matrices/confusion_matrix_gru.png
# - visualizations/confusion_matrices/confusion_matrix_ms3dgru.png
# - visualizations/confusion_matrices/confusion_matrix_stackedgru.png
# - visualizations/confusion_matrices/confusion_matrix_ms3dstackedgru.png
# - visualizations/confusion_matrices/confusion_matrix_comparison.png
# - visualizations/confusion_matrices/confusion_matrix_grid_all_models.png
# - visualizations/confusion_matrices/confusion_matrix_summary.txt
```

### 명령행 사용법

#### 단일 샘플 예측
```bash
python inference/examples/single_predict.py \
    --model best_model/ms3dgru_best.ckpt \
    --csv sensor_data.csv
```

#### 배치 예측
```bash
python inference/examples/batch_predict.py \
    --model best_model/ms3dgru_best.ckpt \
    --dir ./sensor_data/
```

---

## 🔧 하드웨어 통합

### 하드웨어 시스템과의 연동

이 프로젝트는 **SignGlove_HW** 하드웨어 시스템과 함께 사용됩니다.

#### 1. 데이터 수집

하드웨어에서 센서 데이터를 수집하면 H5 파일 형식으로 저장됩니다:
```
unified/
├── ㄱ/
│   ├── sample_001.h5
│   ├── sample_002.h5
│   └── ...
├── ㄴ/
└── ...
```

#### 2. 추론 가중치 준비

하드웨어 시스템에 다음 파일들을 배포해야 합니다:

**필수 파일:**
1. **체크포인트 파일**: `best_model/ms3dgru_best.ckpt` (723KB)
2. **Scaler 파일**: `archive/checkpoints_backup/checkpoints_backup/scaler.pkl` (641B)

**선택적 파일:**
- 다른 모델 체크포인트 (GRU, StackedGRU, MS3DStackedGRU)

#### 3. 하드웨어에서 추론 실행

##### Python 환경 설정

```bash
# 하드웨어 시스템에 Python 환경 설정
pip install torch torchvision torchaudio
pip install numpy scikit-learn
```

##### 추론 스크립트 예시

```python
# hardware_inference.py
import numpy as np
from inference import SignGloveInference

# 추론 엔진 초기화 (하드웨어에서는 CPU 사용)
engine = SignGloveInference(
    model_path='best_model/ms3dgru_best.ckpt',
    model_type='MS3DGRU',
    scaler_path='archive/checkpoints_backup/checkpoints_backup/scaler.pkl',
    device='cpu'
)

def predict_from_sensors(sensor_data):
    """
    하드웨어 센서 데이터를 받아서 예측
    
    Args:
        sensor_data: numpy array, shape (timesteps, 8)
                     채널 순서: [flex1, flex2, flex3, flex4, flex5, pitch, roll, yaw]
    
    Returns:
        predicted_class: str, 예측된 자모 (예: 'ㄱ', 'ㅏ')
        confidence: float, 확률
    """
    result = engine.predict_single(sensor_data, normalize=False)
    return result['predicted_class'], result['confidence']

# 실시간 센서 데이터 처리 예시
def process_realtime_sensor_data(sensor_buffer):
    """
    실시간 센서 버퍼에서 데이터를 받아 처리
    
    Args:
        sensor_buffer: list of (timestamp, sensor_values) tuples
    """
    # 버퍼를 numpy array로 변환
    timesteps = len(sensor_buffer)
    sensor_array = np.array([values for _, values in sensor_buffer], dtype=np.float32)
    
    # 예측
    predicted_class, confidence = predict_from_sensors(sensor_array)
    
    return predicted_class, confidence
```

##### 하드웨어 통합 예시 (Arduino/ESP32)

```cpp
// Arduino/ESP32 예시 코드 구조
// 센서 데이터를 수집하고 Python 스크립트로 전송

void loop() {
    // 센서 데이터 수집
    float flex1 = readFlexSensor(1);
    float flex2 = readFlexSensor(2);
    float flex3 = readFlexSensor(3);
    float flex4 = readFlexSensor(4);
    float flex5 = readFlexSensor(5);
    float pitch, roll, yaw;
    readIMU(&pitch, &roll, &yaw);
    
    // 센서 데이터를 버퍼에 저장
    sensor_buffer.add({
        flex1, flex2, flex3, flex4, flex5,
        pitch, roll, yaw
    });
    
    // 버퍼가 충분히 쌓이면 (예: 87 timesteps)
    if (sensor_buffer.size() >= 87) {
        // Python 스크립트로 데이터 전송
        send_to_python(sensor_buffer);
        sensor_buffer.clear();
    }
}
```

#### 4. 실시간 추론 파이프라인

```
하드웨어 센서 → 데이터 수집 → 전처리 → 추론 엔진 → 예측 결과 → 출력
     ↓              ↓            ↓           ↓           ↓
  Arduino       H5 파일     Scaler 적용   MS3DGRU    자모 출력
  / ESP32       또는 CSV    (normalize)   모델
```

#### 5. 성능 최적화

**CPU 추론 최적화:**
- 배치 처리: 여러 샘플을 한 번에 처리
- 모델 경량화: GRU 모델 사용 (598KB, 98.79% 정확도)
- 추론 속도: CPU 기준 ~10ms/샘플

**메모리 최적화:**
- 단일 추론 시 CPU 사용 (GPU 메모리 불필요)
- 모델은 한 번만 로드하고 재사용

#### 6. 배포 체크리스트

하드웨어 시스템 배포 시 다음을 확인하세요:

- [ ] 체크포인트 파일 배포 (`best_model/ms3dgru_best.ckpt`)
- [ ] Scaler 파일 배포 (`archive/checkpoints_backup/checkpoints_backup/scaler.pkl`)
- [ ] Python 환경 설정 (torch, numpy, scikit-learn)
- [ ] 추론 엔진 코드 배포 (`inference/` 폴더)
- [ ] 센서 데이터 형식 확인 (8채널, 순서: flex1-5, pitch, roll, yaw)
- [ ] 시퀀스 길이 처리 (87 timesteps, padding 필요 시)

---

## 📁 프로젝트 구조

```
KLP-SignGlove-Clean/
├── src/                           # 소스 코드
│   ├── models/                    # 모델 정의
│   │   ├── GRUModels.py           # GRU, StackedGRU
│   │   ├── MultiScale3DGRUModels.py  # MS3DGRU, MS3DStackedGRU
│   │   ├── MSCSGRUModels.py       # MSCGRU, MSCSGRU, CNNGRU
│   │   ├── AdvancedGRUModels.py   # AttentionGRU, ResidualGRU, TransformerGRU
│   │   ├── AGRUModels.py         # AGRU (실험 예정)
│   │   ├── EncoderModels.py       # TransformerEncoder, CNNEncoder
│   │   ├── LSTMModels.py          # LSTM, StackedLSTM
│   │   └── LightningModel.py      # PyTorch Lightning 베이스 클래스
│   ├── experiments/
│   │   └── LightningTrain.py      # 학습 스크립트
│   └── misc/
│       ├── DynamicDataModule.py   # 데이터 로더
│       ├── data_loader.py         # 파일 찾기/로딩
│       ├── data_preprocessor.py   # 전처리 (Scaler 생성)
│       └── dataset.py             # PyTorch Dataset
│
├── inference/                     # 추론 시스템
│   ├── engine.py                  # 통합 추론 엔진
│   ├── models/                    # 추론용 모델
│   │   ├── ms3dgru_inference.py
│   │   ├── gru_inference.py
│   │   ├── stackedgru_inference.py
│   │   ├── ms3dstackedgru_inference.py
│   │   └── mscsgru_inference.py
│   ├── utils/                     # 전처리/후처리
│   │   ├── preprocessor.py
│   │   └── postprocessor.py
│   └── examples/                  # 사용 예제
│       ├── generate_confusion_matrices.py  # 혼동 행렬 생성
│       ├── single_predict.py      # 단일 샘플 예측
│       ├── batch_predict.py       # 배치 예측
│       └── test_all_models.py     # 전체 모델 테스트
│
├── scripts/                       # 분석/테스트 스크립트
│   ├── generate_scaler.py         # Scaler 생성
│   ├── eval_quick.py              # 빠른 평가
│   ├── visualize_*.py             # 시각화 스크립트
│   └── analyze_*.py               # 분석 스크립트
│
├── checkpoints/                   # 체크포인트 (훈련된 모델)
│   ├── best_model_epoch=epoch=92_val/loss=val/loss=0.04.ckpt  # GRU
│   ├── best_model_epoch=epoch=68_val/loss=val/loss=0.19.ckpt  # StackedGRU
│   └── best_model_epoch=epoch=82_val/loss=val/loss=0.05.ckpt  # MS3DStackedGRU
│
├── best_model/                    # 최고 성능 모델 (간편 접근)
│   └── ms3dgru_best.ckpt          # MS3DGRU (99.13%)
│
├── archive/                       # 보관 파일
│   └── checkpoints_backup/
│       └── checkpoints_backup/
│           └── scaler.pkl         # 훈련용 Scaler (필수)
│
├── visualizations/                # 시각화 결과
│   └── confusion_matrices/        # 혼동 행렬
│
├── lightning_logs/                # PyTorch Lightning 로그
│   ├── GRU/                       # GRU 학습 로그
│   ├── StackedGRU/                # StackedGRU 학습 로그
│   ├── MS3DGRU/                   # MS3DGRU 학습 로그
│   └── MS3DStackedGRU/           # MS3DStackedGRU 학습 로그
│
├── requirements.txt               # Python 패키지 의존성
├── LICENSE                        # 라이선스
└── README.md                      # 이 파일
```

---

## 🚀 Quick Start

### 1. 설치

```bash
# 저장소 클론
git clone <repository-url>
cd KLP-SignGlove-Clean

# 의존성 설치
pip install -r requirements.txt
```

### 2. Scaler 생성 (필수)

```bash
# 훈련 데이터로부터 Scaler 생성
python scripts/generate_scaler.py \
    --data_dir /home/billy/25-1kp/SignGlove-DataAnalysis/unified/unified \
    --output_path archive/checkpoints_backup/checkpoints_backup/scaler.pkl \
    --target_timesteps 87
```

### 3. 모델 훈련

```bash
# MS3DGRU 학습 (최고 성능 모델)
python src/experiments/LightningTrain.py \
    -model MS3DGRU \
    -epochs 100 \
    -batch_size 64 \
    -lr 0.001 \
    -hidden_size 64 \
    -data_dir /home/billy/25-1kp/SignGlove-DataAnalysis/unified/unified

# GRU 학습 (경량 모델)
python src/experiments/LightningTrain.py \
    -model GRU \
    -epochs 100 \
    -batch_size 64 \
    -lr 0.001 \
    -layers 1 \
    -hidden_size 64 \
    -data_dir /home/billy/25-1kp/SignGlove-DataAnalysis/unified/unified
```

### 4. 추론 테스트

```bash
# 혼동 행렬 생성 (모든 모델 성능 평가)
python inference/examples/generate_confusion_matrices.py

# 단일 샘플 예측
python inference/examples/single_predict.py \
    --model best_model/ms3dgru_best.ckpt \
    --csv sensor_data.csv
```

### 5. 하드웨어 배포

```bash
# 하드웨어 시스템에 배포할 파일 복사
cp best_model/ms3dgru_best.ckpt /path/to/hardware/
cp archive/checkpoints_backup/checkpoints_backup/scaler.pkl /path/to/hardware/
cp -r inference/ /path/to/hardware/
```

---

## 📊 핵심 교훈

### 1. 3D CNN이 효과적인 이유
- ✅ **센서 간 상호작용 학습**: 8개 센서를 공간적으로 배치하여 센서 간 관계 포착
- ✅ **시간-공간 특성 동시 처리**: 3D CNN이 시간과 공간을 동시에 모델링
- ✅ **Multi-Scale 특성**: 다양한 커널 크기로 세밀한 패턴부터 거시적 패턴까지 포착

### 2. 1D CNN의 한계
- ❌ 시간 패턴만 포착 (공간 특성 미활용)
- ❌ 센서 간 상호작용 학습 어려움
- 결과: 98.44% (GRU와 유사)

### 3. Sensor-Aware 접근의 문제
- ❌ 센서 분리가 오히려 정보 손실
- ❌ 센서 간 강한 상관관계로 통합 처리 필요
- 결과: ~90-96% (GRU보다 낮음)

### 4. 모델 복잡도와 성능 트레이드오프
- MS3DGRU (58K params): 99.13% ✅
- MS3DStackedGRU (167K params): 97.92% (더 복잡하지만 성능 낮음)
- 결론: 단순한 구조가 더 효과적

---

## 📦 Requirements

```
torch>=1.10.0
pytorch-lightning>=1.5.0
numpy>=1.20.0
matplotlib>=3.3.0
seaborn>=0.11.0
scikit-learn>=0.24.0
pandas>=1.2.0
tqdm>=4.60.0
```

---

## 🔗 관련 프로젝트

이 프로젝트는 SignGlove 시스템의 딥러닝 모델 훈련 및 추론 부분을 담당합니다. 전체 SignGlove 프로젝트는 다음 레포지토리로 구성되어 있습니다:

### 하드웨어 (HW)
**[SignGlove_HW](https://github.com/KNDG01001/SignGlove_HW)** - 센서 기반 수화 통역 장갑 하드웨어
- 아두이노 펌웨어 및 데이터 수집 시스템
- 센서 데이터 수집 및 전처리
- H5 에피소드 파일 형식 지원
- 실시간 데이터 수집 인터페이스

### 데이터 분석 (Data Analysis)
**[SignGlove-DataAnalysis](https://github.com/wodu2s/SignGlove-DataAnalysis)** - 데이터 분석 및 시각화
- 데이터셋 분석 및 통계
- 특성 분석 및 시각화
- 데이터 품질 평가

### 메인 프로젝트 (PM - Project Merge)
**[SignGlove](https://github.com/minuum/SignGlove)** - 최종 병합된 메인 프로젝트
- 전체 시스템 통합
- 실시간 추론 + TTS 통합
- FastAPI 서버 및 웹 인터페이스
- 최종 배포 버전

### 현재 프로젝트
**KLP-SignGlove-Clean** (이 레포지토리) - 딥러닝 모델 훈련 및 추론
- 모델 아키텍처 구현 및 실험
- 학습 파이프라인
- 추론 시스템 및 엔진
- 성능 평가 및 분석

---

## 📞 문의 및 기여

프로젝트 완료일: 2025-11-05  
최종 모델: MS3DGRU (99.13% accuracy, 58,840 parameters)

---

*이 프로젝트는 체계적인 실험과 분석을 통해 완성되었습니다.*
