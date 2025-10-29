# SignGlove: 한국 수어 인식 딥러닝 프로젝트

> Multi-Scale 3D CNN + GRU 기반 한국 수어 손동작 인식 시스템  
> **최고 성능: 99.13% 정확도** 달성

---

## 🎯 프로젝트 개요

### 목표
8채널 센서 데이터를 활용한 한국 수어 24개 자모(자음 14개 + 모음 10개) 인식 시스템 개발

### 데이터셋
- **총 샘플**: 2,884개 (yubeen: 1,440개, jaeyeon: 1,440개, combined: 2,884개)
- **클래스**: 24개 (자음 14개: ㄱ-ㅎ, 모음 10개: ㅏ-ㅣ)
- **센서**: 8채널
  - Flex 센서: flex1, flex2, flex3, flex4, flex5 (5개)
  - IMU 센서: pitch, roll, yaw (3개)
- **시퀀스 길이**: 최대 87 타임스텝
- **데이터 경로**: `/home/billy/25-1kp/SignGlove_HW/datasets/unified`

### 최종 성과
- 🥇 **최고 정확도**: **99.13%** (MS3DGRU, combined 데이터셋)
- 🥈 **2위 모델**: GRU 98.44% / MS3DStackedGRU 98.78%
- 💾 **모델 크기**: 723KB (체크포인트)
- 📊 **F1-Score**: 0.9913

---

## 📊 실험 결과 및 모델 성능

### 전체 모델 비교표

| 순위 | 모델 | Test Accuracy | Test F1 | Test Loss | Parameters | 특징 |
|------|------|---------------|---------|-----------|------------|------|
| 🥇 1 | **MS3DGRU** | **99.13%** | 0.9913 | 0.052 | 58,840 | Multi-Scale 3D CNN + GRU |
| 🥈 2 | **GRU** | **98.44%** | 0.9844 | 0.061 | 74,776 | 기본 GRU 모델 (안정적) |
| 🥈 2 | **MS3DStackedGRU** | **98.78%** | 0.9878 | 0.045 | 167,032 | 3D CNN + Stacked GRU |
| 🥉 3 | **MSCGRU** | **98.44%** | 0.9844 | 0.046 | ~100K | Multi-Scale 1D CNN + GRU |
| 4 | **ResidualGRU** | 98.78% | - | - | ~75K | Residual Connection |
| 5 | **MSCSGRU** | 98.09% | - | - | ~140K | Multi-Scale 1D CNN + Stacked GRU |
| 6 | **StackedGRU** | 97.06% | 0.9698 | 0.092 | 50,584 | 2층 Stacked GRU |

### 최종 선택 모델: **MS3DGRU**

**선정 이유:**
1. ✅ **최고 성능** (99.13% 정확도)
2. ✅ **안정적인 성능** (여러 데이터셋에서 일관된 결과)
3. ✅ **공간-시간 특성 학습** (3D CNN으로 센서 간 상호작용 포착)
4. ✅ **적절한 모델 크기** (58,840 파라미터)

---

## 🔬 실험 과정

### Phase 1: 기본 모델 비교

**목적**: 다양한 기본 모델들의 성능 비교

**실험 모델:**
- `GRU`: 기본 GRU (2층)
- `StackedGRU`: 다층 GRU (안정적이지만 성능 낮음)
- `LSTM` / `StackedLSTM`: LSTM 변형

**결과:**
- GRU가 가장 안정적이고 높은 성능 (98.44%)
- StackedGRU는 파라미터 대비 성능이 낮음

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
- MSCGRU가 98.44%로 좋은 성능 (GRU와 동일)
- 1D CNN만으로는 한계 확인

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
3. Dropout 조정: 0.1이 최적 (0.0: 96.53%, 0.1: 98.96%, 0.2: 98.44%)
4. 추가 Conv 레이어: 2단계 CNN으로 성능 향상 (97.57% → 98.44%)

**결과:**
- ✅ **99.13% 정확도 달성**
- 안정성: 5회 실행 모두 98.78% ±0.0% (매우 안정적)
- 데이터셋별:
  - yubeen: 98.78%
  - jaeyeon: 98.78%
  - combined: 99.13%

#### 3.3 MS3DStackedGRU 개발
**목적**: 3D CNN + Stacked GRU로 추가 성능 향상 시도

**최적화 과정:**
- 과적합 방지: Dropout 강화 → ❌ 성능 하락
- CNN 구조 개선: 추가 Conv 레이어 → ✅ 98.44% 달성
- Bidirectional GRU: 98.09%
- Attention: 98.09%
- 하이퍼파라미터 튜닝: 98.26%

**결과:**
- 최고 성능: 98.44% (MS3DGRU보다 낮음)
- 파라미터 증가 (167K vs 58K)
- 결론: 단일 GRU가 더 효율적

---

### Phase 4: 고급 기법 적용

#### 4.1 Residual Connections
**모델**: `ResidualGRU`
- ResNet 스타일 residual connection 적용
- Gradient flow 개선
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

### Phase 5: 데이터셋 확장 및 검증

**목적**: 다양한 데이터셋에서 모델 안정성 검증

**데이터셋:**
1. **yubeen**: 첫 1,440개 샘플
2. **jaeyeon**: 다음 1,440개 샘플
3. **combined**: 전체 2,884개 샘플

**검증 결과:**

| 모델 | yubeen | jaeyeon | combined | 평균 | 안정성 |
|------|--------|---------|----------|------|--------|
| GRU | 98.44% | 98.44% | 98.44% | 98.44% | 매우 안정 |
| StackedGRU | 97.06% | 97.06% | 97.06% | 97.06% | 매우 안정 |
| MS3DGRU | **98.78%** | **98.78%** | **99.13%** | **98.90%** | 매우 안정 |
| MS3DStackedGRU | 98.78% | 95.14% | 98.44% | 97.45% | 불안정 |

**결론**: MS3DGRU가 모든 데이터셋에서 일관되고 최고 성능

---

### Phase 6: 하이퍼파라미터 최적화

**최적화 항목:**
- Learning Rate: 0.0001, 0.001, 0.005, 0.01
- Batch Size: 32, 64, 128
- Dropout: 0.01, 0.05, 0.1, 0.2
- Epochs: 100 (Early Stopping 적용)

**MS3DGRU 최적 설정:**
- Learning Rate: 0.001
- Batch Size: 64
- Dropout: 0.1
- Hidden Size: 64
- CNN Filters: 32

---

### Phase 7: 통계적 검증

**목적**: 모델의 안정성과 통계적 유의성 검증

**방법**: 5회 실행 (서로 다른 random seed)

**결과:**
- MS3DGRU: 98.78% ± 0.0% (매우 안정)
- StackedGRU: 91.85% ± 변동 (불안정)

**결론**: MS3DGRU는 매우 안정적인 성능

---

## 🚀 추론 시스템 사용법

### 시스템 구조

```
inference/
├── engine.py              # 통합 추론 엔진 (고수준 API)
├── models/                # 추론용 모델 정의
│   ├── ms3dgru_inference.py
│   ├── gru_inference.py
│   ├── ms3dstackedgru_inference.py
│   └── mscsgru_inference.py
├── utils/                 # 전처리/후처리 유틸리티
│   ├── preprocessor.py
│   └── postprocessor.py
├── examples/              # 사용 예제
│   ├── single_predict.py      # 단일 샘플 예측
│   ├── batch_predict.py       # 배치 예측
│   ├── ms3dgru_predict.py     # MS3DGRU 전용 예제
│   └── best_models_demo.py    # 최고 성능 모델 데모
├── best_models/           # 훈련된 모델 체크포인트
│   └── ms3dgru_best.ckpt      # MS3DGRU (99.13%)
└── performance_visualizations/  # 성능 시각화 결과
```

---

### 1. 기본 사용법 (Python API)

#### 1.1 단일 샘플 예측

```python
import numpy as np
from inference import SignGloveInference

# 1. 추론 엔진 초기화
engine = SignGloveInference(
    model_path='inference/best_models/ms3dgru_best.ckpt',
    model_type='MS3DGRU',  # 또는 'GRU', 'MS3DStackedGRU', 'MSCSGRU'
    device='cpu',  # 또는 'cuda'
    input_size=8,
    hidden_size=64,
    classes=24,
    cnn_filters=32,
    dropout=0.1
)

# 2. 센서 데이터 준비
# Shape: (timesteps, 8)
# 채널 순서: [flex1, flex2, flex3, flex4, flex5, pitch, roll, yaw]
raw_data = np.random.randn(87, 8).astype(np.float32)  # 예시 데이터

# 3. 예측 수행
result = engine.predict_single(raw_data, top_k=5)

# 4. 결과 출력
engine.print_prediction(result)
# 출력 예시:
# 🎯 예측 클래스: ㄱ
# 📈 확률: 0.9845
# 📋 상위 5개 예측:
#     1. ㄱ: 0.9845
#     2. ㅂ: 0.0102
#     3. ㅁ: 0.0031
#     4. ㄴ: 0.0012
#     5. ㄷ: 0.0005
```

#### 1.2 CSV 파일에서 예측

```python
import pandas as pd
from inference import SignGloveInference

# 추론 엔진 초기화
engine = SignGloveInference(
    model_path='inference/best_models/ms3dgru_best.ckpt',
    model_type='MS3DGRU'
)

# CSV 파일 로드
df = pd.read_csv('sensor_data.csv')

# 센서 컬럼 추출 (필수: flex1-5, pitch, roll, yaw)
sensor_columns = ['flex1', 'flex2', 'flex3', 'flex4', 'flex5', 'pitch', 'roll', 'yaw']
raw_data = df[sensor_columns].values

# 예측
result = engine.predict_single(raw_data)
print(f"예측: {result['predicted_class']}, 확률: {result['confidence']:.2%}")
```

#### 1.3 배치 예측

```python
import numpy as np
from inference import SignGloveInference

# 추론 엔진 초기화
engine = SignGloveInference(
    model_path='inference/best_models/ms3dgru_best.ckpt',
    model_type='MS3DGRU'
)

# 여러 샘플 준비 (길이가 다를 수 있음)
raw_data_list = [
    np.random.randn(87, 8).astype(np.float32),  # 샘플 1
    np.random.randn(75, 8).astype(np.float32),  # 샘플 2 (다른 길이 가능)
    np.random.randn(90, 8).astype(np.float32),  # 샘플 3
]

# 배치 예측
results = engine.predict_batch(raw_data_list, top_k=3)

# 결과 확인
for i, result in enumerate(results, 1):
    print(f"샘플 {i}: {result['predicted_class']} ({result['confidence']:.2%})")
```

#### 1.4 상세 정보 포함 예측

```python
# 예측과 함께 상세 정보 반환
detailed_result = engine.predict_with_details(raw_data)

print(f"예측 클래스: {detailed_result['predicted_class']}")
print(f"확률: {detailed_result['confidence']:.4f}")
print(f"입력 shape: {detailed_result['input_shape']}")
print(f"처리 시간: {detailed_result['processing_time']:.4f}초")
print("\n상위 5개 예측:")
for i, pred in enumerate(detailed_result['top_k_predictions'], 1):
    print(f"  {i}. {pred['class']}: {pred['confidence']:.4f}")
```

---

### 2. 명령행 사용법

#### 2.1 단일 샘플 예측 스크립트

```bash
# CSV 파일에서 예측
python inference/examples/single_predict.py \
    --model inference/best_models/ms3dgru_best.ckpt \
    --csv sensor_data.csv

# 랜덤 데이터로 테스트
python inference/examples/single_predict.py \
    --model inference/best_models/ms3dgru_best.ckpt \
    --test
```

#### 2.2 배치 예측 스크립트

```bash
# 여러 CSV 파일 배치 예측
python inference/examples/batch_predict.py \
    --model inference/best_models/ms3dgru_best.ckpt \
    --csvs file1.csv file2.csv file3.csv

# 디렉토리의 모든 CSV 파일 예측
python inference/examples/batch_predict.py \
    --model inference/best_models/ms3dgru_best.ckpt \
    --dir ./sensor_data/

# 랜덤 데이터로 테스트 (배치 크기 10)
python inference/examples/batch_predict.py \
    --model inference/best_models/ms3dgru_best.ckpt \
    --test 10
```

#### 2.3 MS3DGRU 전용 예제

```bash
# MS3DGRU 모델의 모든 기능 데모
python inference/examples/ms3dgru_predict.py

# 최고 성능 모델 데모
python inference/examples/best_models_demo.py
```

---

### 3. Confusion Matrix 생성

```bash
# MS3DGRU 모델의 전체 테스트셋 성능 평가
python scripts/generate_confusion_matrix.py

# 생성되는 파일:
# - inference/performance_visualizations/real_test_confusion_matrix_ms3dgru_final.png
# - inference/performance_visualizations/real_test_class_accuracy_ms3dgru_final.png
# - inference/performance_visualizations/real_test_report_ms3dgru_final.txt
```

**결과 예시:**
```
정확도: 99.13%
F1-Score (Macro): 0.9913
F1-Score (Weighted): 0.9913

✅ 정확 예측: 572개 (99.13%)
❌ 오분류: 5개 (0.87%)
```

---

### 4. 성능 벤치마크

```python
import time
import numpy as np
from inference import SignGloveInference

engine = SignGloveInference(
    model_path='inference/best_models/ms3dgru_best.ckpt',
    model_type='MS3DGRU'
)

# 단일 샘플 추론 시간 측정
raw_data = np.random.randn(87, 8).astype(np.float32)
n_iterations = 100

start_time = time.time()
for _ in range(n_iterations):
    _ = engine.predict_single(raw_data, return_all_info=False)
single_time = (time.time() - start_time) / n_iterations

print(f"단일 샘플 추론 시간: {single_time*1000:.2f}ms")
print(f"초당 추론 가능 횟수: {1/single_time:.1f} samples/sec")

# 배치 추론 시간 측정
batch_sizes = [1, 4, 8, 16, 32]
for batch_size in batch_sizes:
    batch_data = [np.random.randn(87, 8).astype(np.float32) for _ in range(batch_size)]
    start_time = time.time()
    _ = engine.predict_batch(batch_data)
    batch_time = (time.time() - start_time) / batch_size
    print(f"배치 크기 {batch_size:2d}: 샘플당 {batch_time*1000:.2f}ms")
```

---

### 5. 지원 모델 타입

| 모델 타입 | 설명 | 정확도 | 체크포인트 경로 |
|-----------|------|--------|----------------|
| `MS3DGRU` | Multi-Scale 3D CNN + GRU | **99.13%** | `inference/best_models/ms3dgru_best.ckpt` |
| `GRU` | 기본 GRU (2층) | 98.44% | `checkpoints/best_model_epoch=66_val/loss=0.06.ckpt` |
| `MS3DStackedGRU` | 3D CNN + Stacked GRU | 98.78% | (체크포인트 필요) |
| `MSCSGRU` | Multi-Scale 1D CNN + Stacked GRU | 98.09% | (체크포인트 필요) |

**사용 예시:**
```python
# MS3DGRU (권장)
engine = SignGloveInference(
    model_path='inference/best_models/ms3dgru_best.ckpt',
    model_type='MS3DGRU'
)

# GRU
engine = SignGloveInference(
    model_path='checkpoints/best_model_epoch=66_val/loss=0.06.ckpt',
    model_type='GRU'
)
```

---

### 6. 입력 데이터 형식

#### 6.1 NumPy 배열
```python
# Shape: (timesteps, 8)
# timesteps: 50-120 (87 권장)
# 8 channels: [flex1, flex2, flex3, flex4, flex5, pitch, roll, yaw]
raw_data = np.array([
    [1.0, 0.5, 0.3, 0.2, 0.1, 0.0, 0.0, 0.0],  # timestep 0
    [1.0, 0.6, 0.4, 0.3, 0.2, 0.1, 0.0, 0.0],  # timestep 1
    # ... 최대 87 timesteps
], dtype=np.float32)
```

#### 6.2 CSV 파일
```csv
flex1,flex2,flex3,flex4,flex5,pitch,roll,yaw
1.0,0.5,0.3,0.2,0.1,0.0,0.0,0.0
1.0,0.6,0.4,0.3,0.2,0.1,0.0,0.0
...
```

**필수 컬럼:**
- `flex1`, `flex2`, `flex3`, `flex4`, `flex5`: 굽힘 센서 (0.0-1.0)
- `pitch`, `roll`, `yaw`: IMU 센서 (각도 값)

---

### 7. 출력 결과 형식

#### 7.1 단일 예측 결과
```python
result = {
    'predicted_class': 'ㄱ',  # 예측된 클래스 (한글 자모)
    'confidence': 0.9845,     # 확률 (0.0-1.0)
    'top_k_predictions': [    # 상위 k개 예측
        {'class': 'ㄱ', 'confidence': 0.9845},
        {'class': 'ㅂ', 'confidence': 0.0102},
        {'class': 'ㅁ', 'confidence': 0.0031},
        # ...
    ]
}
```

#### 7.2 배치 예측 결과
```python
results = [
    {'predicted_class': 'ㄱ', 'confidence': 0.9845, ...},  # 샘플 1
    {'predicted_class': 'ㅏ', 'confidence': 0.9721, ...},  # 샘플 2
    # ...
]
```

---

### 8. 고급 기능

#### 8.1 디바이스 선택
```python
# CPU 사용
engine = SignGloveInference(
    model_path='inference/best_models/ms3dgru_best.ckpt',
    model_type='MS3DGRU',
    device='cpu'  # 기본값
)

# GPU 사용 (CUDA 사용 가능 시)
engine = SignGloveInference(
    model_path='inference/best_models/ms3dgru_best.ckpt',
    model_type='MS3DGRU',
    device='cuda'
)

# 자동 감지
engine = SignGloveInference(
    model_path='inference/best_models/ms3dgru_best.ckpt',
    model_type='MS3DGRU',
    device=None  # 자동으로 cuda 또는 cpu 선택
)
```

#### 8.2 StandardScaler 사용
```python
# 훈련 시 사용한 스케일러가 있는 경우
engine = SignGloveInference(
    model_path='inference/best_models/ms3dgru_best.ckpt',
    model_type='MS3DGRU',
    scaler_path='checkpoints/training_scaler.pkl'  # 스케일러 파일 경로
)
```

#### 8.3 모델 정보 확인
```python
info = engine.get_model_info()
print(f"모델 타입: {info['model_type']}")
print(f"파라미터 수: {info['total_parameters']:,}")
print(f"클래스 수: {info['classes']}")
print(f"디바이스: {info['device']}")
```

---

### 9. 문제 해결

#### 9.1 모델 로드 오류
```python
# 체크포인트 파일 경로 확인
import os
checkpoint_path = 'inference/best_models/ms3dgru_best.ckpt'
if not os.path.exists(checkpoint_path):
    print(f"❌ 체크포인트를 찾을 수 없습니다: {checkpoint_path}")
```

#### 9.2 입력 데이터 형식 오류
```python
# 올바른 shape 확인
assert raw_data.shape[1] == 8, f"입력 채널 수가 8개가 아닙니다: {raw_data.shape}"
assert len(raw_data.shape) == 2, f"입력은 2D 배열이어야 합니다: {raw_data.shape}"
```

#### 9.3 클래스 이름 확인
```python
# 지원되는 클래스 목록
CLASS_NAMES = [
    'ㄱ', 'ㄴ', 'ㄷ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅅ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ',
    'ㅏ', 'ㅑ', 'ㅓ', 'ㅕ', 'ㅗ', 'ㅛ', 'ㅜ', 'ㅠ', 'ㅡ', 'ㅣ'
]
```

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
│   │   └── LightningModel.py      # PyTorch Lightning 베이스 클래스
│   ├── experiments/
│   │   └── LightningTrain.py      # 학습 스크립트
│   └── misc/
│       ├── DynamicDataModule.py   # 데이터 로더
│       ├── data_loader.py         # 파일 찾기/로딩
│       ├── data_preprocessor.py   # 전처리
│       └── dataset.py             # PyTorch Dataset
│
├── inference/                     # 추론 시스템
│   ├── engine.py                  # 통합 추론 엔진
│   ├── models/                    # 추론용 모델
│   ├── utils/                     # 전처리/후처리
│   ├── examples/                  # 사용 예제
│   ├── best_models/               # 최고 성능 모델
│   │   └── ms3dgru_best.ckpt      # MS3DGRU (99.13%)
│   └── performance_visualizations/  # 성능 시각화
│
├── scripts/                       # 분석/테스트 스크립트
│   ├── generate_confusion_matrix.py  # Confusion Matrix 생성
│   ├── test_inference_with_engine.py # 추론 엔진 테스트
│   └── analyze_*.py               # 분석 스크립트
│
├── checkpoints/                   # 체크포인트
│   ├── best_model_epoch=66_val/   # 최고 성능 (99.13%)
│   └── training_scaler.pkl        # 데이터 스케일러
│
├── best_model/                    # 최고 성능 모델 (간편 접근)
│   └── ms3dgru_best.ckpt
│
├── lightning_logs/                # PyTorch Lightning 로그
│   ├── MS3DGRU/                   # MS3DGRU 학습 로그
│   ├── GRU/                       # GRU 학습 로그
│   └── StackedGRU/                # StackedGRU 학습 로그
│
├── visualizations/                # 시각화
│   ├── efficiency_analysis/       # 모델 효율성 비교
│   └── noise_robustness/          # 노이즈 견고성 분석
│
├── archive/                       # 보관 파일
│   ├── agru_research/             # AGRU 연구 관련
│   ├── lightning_logs_backup/     # 이전 로그
│   └── checkpoints_backup/        # 이전 체크포인트
│
└── README.md                      # 이 파일
```

---

## 🚀 Quick Start

### 설치
```bash
pip install -r requirements.txt
```

### 학습
```bash
# MS3DGRU 학습 (최고 성능 모델)
python src/experiments/LightningTrain.py \
    -model MS3DGRU \
    -epochs 100 \
    -batch_size 64 \
    -lr 0.001 \
    -hidden_size 64

# GRU 학습 (기본 모델)
python src/experiments/LightningTrain.py \
    -model GRU \
    -epochs 100 \
    -batch_size 64 \
    -lr 0.001
```

### 추론 테스트
```bash
# Confusion Matrix 생성 (전체 테스트셋 평가)
python scripts/generate_confusion_matrix.py

# 단일 샘플 예측
python inference/examples/single_predict.py \
    --model inference/best_models/ms3dgru_best.ckpt \
    --csv sensor_data.csv
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
- 결과: 98.44% (GRU와 동일)

### 3. Sensor-Aware 접근의 문제
- ❌ 센서 분리가 오히려 정보 손실
- ❌ 센서 간 강한 상관관계로 통합 처리 필요
- 결과: ~90-96% (GRU보다 낮음)

### 4. 모델 복잡도와 성능 트레이드오프
- MS3DGRU (58K params): 99.13% ✅
- MS3DStackedGRU (167K params): 98.78% (더 복잡하지만 성능 낮음)
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

## 📄 License

See `LICENSE` file for details.

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

프로젝트 완료일: 2025-10-29  
최종 모델: MS3DGRU (99.13% accuracy, 58,840 parameters)

---

*이 프로젝트는 체계적인 실험과 분석을 통해 완성되었습니다.*
