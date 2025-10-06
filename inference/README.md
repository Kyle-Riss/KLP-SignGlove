# SignGlove 추론 시스템

훈련된 SignGlove 모델을 사용한 추론 전용 시스템입니다.
훈련 코드와 완전히 분리되어 있으며, 경량화되고 배포에 최적화되어 있습니다.

## 🎯 주요 특징

- **완전 분리**: 훈련 코드와 독립적으로 작동
- **경량화**: 추론에 필요한 코드만 포함
- **사용 편의성**: 간단한 API로 쉽게 사용
- **성능 최적화**: 빠른 추론 속도
- **유연성**: 단일/배치 예측 모두 지원

## 📂 폴더 구조

```
inference/
├── __init__.py                    # 패키지 초기화
├── engine.py                      # 통합 추론 엔진
├── models/
│   ├── __init__.py
│   └── mscsgru_inference.py      # 추론 전용 MSCSGRU 모델
├── utils/
│   ├── __init__.py
│   ├── preprocessor.py           # 전처리 유틸리티
│   └── postprocessor.py          # 후처리 유틸리티
├── examples/
│   ├── single_predict.py         # 단일 샘플 예측 예제
│   └── batch_predict.py          # 배치 예측 예제
└── README.md                      # 이 파일
```

## 🚀 빠른 시작

### 1. 단일 샘플 예측

```python
import numpy as np
from inference import SignGloveInference

# 추론 엔진 초기화
engine = SignGloveInference(
    model_path='best_model/best_model.ckpt',
    model_type='MSCSGRU',
    device='cpu'
)

# 센서 데이터 준비 (timesteps, 8 channels)
raw_data = np.random.randn(87, 8)

# 예측
result = engine.predict_single(raw_data)
engine.print_prediction(result)
```

### 2. 배치 예측

```python
# 여러 샘플 준비
raw_data_list = [
    np.random.randn(87, 8),
    np.random.randn(87, 8),
    np.random.randn(87, 8)
]

# 배치 예측
results = engine.predict_batch(raw_data_list)

for i, result in enumerate(results, 1):
    print(f"샘플 {i}: {result['predicted_class']} ({result['confidence']:.4f})")
```

### 3. 상세 예측

```python
# 상세 정보를 포함한 예측
detailed_result = engine.predict_with_details(raw_data)

print(f"예측 클래스: {detailed_result['predicted_class']}")
print(f"확률: {detailed_result['confidence']:.4f}")
print(f"상위 5개 예측: {detailed_result['top_k_predictions']}")
print(f"모든 클래스 확률: {detailed_result['all_class_probabilities']}")
```

## 📊 입력 데이터 형식

### 센서 데이터 구조

```python
# Shape: (timesteps, 8)
# - timesteps: 가변 길이 (자동으로 87로 조정)
# - 8 channels: [flex1, flex2, flex3, flex4, flex5, yaw, pitch, roll]

raw_data = np.array([
    [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],  # timestep 1
    [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],  # timestep 2
    # ... (가변 길이)
])
```

### 가변 길이 입력

추론 엔진은 자동으로 입력 길이를 조정합니다:

- **짧은 입력** (< 87): 패딩 추가
- **긴 입력** (> 87): 트렁케이션
- **정확한 입력** (= 87): 그대로 사용

## 📈 출력 형식

### 기본 예측 결과

```python
result = {
    'predicted_class': 'ㄱ',           # 예측된 클래스
    'predicted_class_idx': 0,          # 클래스 인덱스
    'confidence': 0.95,                # 예측 확률
    'top_k_predictions': [             # 상위 K개 예측
        {'class': 'ㄱ', 'class_idx': 0, 'confidence': 0.95},
        {'class': 'ㄴ', 'class_idx': 1, 'confidence': 0.03},
        # ...
    ]
}
```

## 🔧 API 문서

### SignGloveInference

```python
engine = SignGloveInference(
    model_path: str,                # 체크포인트 파일 경로
    model_type: str = 'MSCSGRU',    # 모델 타입
    input_size: int = 8,            # 입력 채널 수
    hidden_size: int = 64,          # 히든 사이즈
    classes: int = 24,              # 클래스 수
    cnn_filters: int = 32,          # CNN 필터 수
    dropout: float = 0.3,           # 드롭아웃 비율
    target_timesteps: int = 87,     # 타임스텝 길이
    device: str = None,             # 디바이스 ('cuda', 'cpu', None=자동)
    class_names: List[str] = None   # 클래스 이름 리스트
)
```

#### 주요 메서드

- **`predict_single(raw_data, top_k=5)`**: 단일 샘플 예측
- **`predict_batch(raw_data_list, top_k=5)`**: 배치 예측
- **`predict_with_details(raw_data)`**: 상세 정보를 포함한 예측
- **`get_model_info()`**: 모델 정보 반환
- **`print_prediction(prediction)`**: 예측 결과 출력

## 🎓 클래스 목록

24개의 한국어 수화 자모:

### 자음 (14개)
ㄱ, ㄴ, ㄷ, ㄹ, ㅁ, ㅂ, ㅅ, ㅇ, ㅈ, ㅊ, ㅋ, ㅌ, ㅍ, ㅎ

### 모음 (10개)
ㅏ, ㅑ, ㅓ, ㅕ, ㅗ, ㅛ, ㅜ, ㅠ, ㅡ, ㅣ

## 💡 사용 예제

### 실제 센서 데이터 사용

```python
import pandas as pd

# CSV 파일에서 센서 데이터 로딩
sensor_data = pd.read_csv('sensor_data.csv')

# 필요한 컬럼만 추출
columns = ['flex1', 'flex2', 'flex3', 'flex4', 'flex5', 'pitch', 'roll', 'yaw']
raw_data = sensor_data[columns].values

# 추론
result = engine.predict_single(raw_data)
```

### 대용량 배치 처리

```python
def predict_large_batch(engine, raw_data_list, chunk_size=32):
    """대용량 배치를 청크로 나누어 처리"""
    all_results = []
    
    for i in range(0, len(raw_data_list), chunk_size):
        chunk = raw_data_list[i:i + chunk_size]
        results = engine.predict_batch(chunk)
        all_results.extend(results)
        
        print(f'처리 완료: {len(all_results)}/{len(raw_data_list)}')
    
    return all_results
```

## 🔍 모델 정보 확인

```python
# 모델 정보 출력
info = engine.get_model_info()

print(f"모델 타입: {info['model_type']}")
print(f"파라미터 수: {info['total_parameters']:,}")
print(f"클래스 수: {info['classes']}")
print(f"디바이스: {info['device']}")
print(f"클래스 목록: {info['class_names']}")
```

## 📝 추가 예제

더 많은 예제는 `examples/` 폴더를 참고하세요:

- `single_predict.py`: 단일 샘플 예측 예제
- `batch_predict.py`: 배치 예측 예제

## ⚠️ 주의사항

1. **모델 체크포인트**: 추론 전에 훈련된 모델 체크포인트가 필요합니다
2. **디바이스**: GPU 사용 시 CUDA가 설치되어 있어야 합니다
3. **입력 형식**: 센서 데이터는 반드시 8개 채널이어야 합니다
4. **메모리**: 대용량 배치는 청크로 나누어 처리하세요

## 🚀 성능 최적화

### GPU 사용

```python
# GPU 사용 (CUDA 설치 필요)
engine = SignGloveInference(
    model_path='best_model/best_model.ckpt',
    device='cuda'
)
```

### 배치 크기 조정

```python
# 메모리에 맞게 배치 크기 조정
optimal_batch_size = 32  # GPU 메모리에 따라 조정
```

## 📞 문의

문제가 발생하거나 질문이 있으시면 이슈를 생성해 주세요.

