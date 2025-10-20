# Scale-Aware GRU Models

## 개요

**Scale-Aware GRU**는 Multi-Scale CNN의 각 타워(kernel=3,5,7)에 **독립적인 GRU 가중치**를 할당하여, 스케일별 중요도를 학습하는 혁신적인 아키텍처입니다.

---

## 🎯 핵심 아이디어

### 기존 MS-CSGRU의 한계
```python
# 기존: 3개 타워를 단순 Concatenate
conv_out = torch.cat([t3, t5, t7], dim=1)  # (batch, 96, time)
gru_out, _ = GRU(conv_out)  # 단일 가중치로 처리
```

**문제점:**
- 모든 스케일이 동일한 가중치로 처리됨
- 어떤 스케일이 중요한지 알 수 없음
- 해석 가능성 부족

### Scale-Aware GRU의 혁신
```python
# 개선: 각 스케일에 독립적인 가중치
z_t = sigmoid(W_3 @ t3 + W_5 @ t5 + W_7 @ t7 + U_z @ h_prev)
```

**장점:**
- ✅ 각 스케일의 중요도를 독립적으로 학습
- ✅ 해석 가능성 향상 (어떤 스케일이 중요한지 분석 가능)
- ✅ 더 풍부한 특징 표현

---

## 📊 모델 비교

| 모델 | 파라미터 | 추론 시간 | 특징 |
|------|----------|-----------|------|
| **기존 MSCSGRU** | 71,800 | 3.99ms | 단일 가중치 |
| **MSCSGRU_ScaleAware** | 95,992 | 8.45ms | 스케일별 가중치 (Sigmoid/Tanh) |
| **MSCSGRU_ScaleHard** | 95,992 | 8.46ms | 스케일별 가중치 (Hard 함수) |
| **MSCGRU_ScaleAware** | 46,648 | 4.57ms | 단일 GRU + 스케일별 가중치 |

### 파라미터 증가 분석
- 기존 → Scale-Aware: **+33.7%** (24,192 파라미터 증가)
- 이유: 각 게이트(update, reset, hidden)마다 3개 가중치 행렬 필요

### 속도 분석
- Scale-Aware는 **2.1배 느림** (3.99ms → 8.45ms)
- 이유: 3배 많은 Linear 연산
- **Hard 함수 사용 시 속도 차이 거의 없음** (8.45ms vs 8.46ms)

---

## 🔬 수학적 정의

### 기존 GRU
```
z_t = σ(W_z @ x_t + U_z @ h_{t-1} + b_z)
r_t = σ(W_r @ x_t + U_r @ h_{t-1} + b_r)
h̃_t = tanh(W_h @ x_t + U_h @ (r_t ⊙ h_{t-1}) + b_h)
h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t
```

### Scale-Aware GRU
```
z_t = σ(W_z3 @ t3 + W_z5 @ t5 + W_z7 @ t7 + U_z @ h_{t-1} + b_z)
r_t = σ(W_r3 @ t3 + W_r5 @ t5 + W_r7 @ t7 + U_r @ h_{t-1} + b_r)
h̃_t = tanh(W_h3 @ t3 + W_h5 @ t5 + W_h7 @ t7 + U_h @ (r_t ⊙ h_{t-1}) + b_h)
h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t
```

**차이점:**
- 단일 가중치 `W_z` → 3개 가중치 `W_z3, W_z5, W_z7`
- 각 스케일이 게이트에 독립적으로 기여

---

## 💡 Hard 함수 최적화

### Hard 함수란?
```python
# 기존 Sigmoid (지수 연산 필요)
def sigmoid(x):
    return 1 / (1 + exp(-x))

# HardSigmoid (선형 연산만)
def hard_sigmoid(x):
    return clamp(x * 0.2 + 0.5, 0, 1)
```

### 장점
- ✅ **지수 연산 제거** → 임베디드 시스템에 최적
- ✅ **미분 간단** → 역전파 빠름
- ✅ **메모리 효율적**

### 단점
- ⚠️ **표현력 감소** → 정확도 소폭 하락 가능
- ⚠️ **Dead Neuron** 문제 가능

---

## 📈 스케일 중요도 분석

### 초기화 직후 (학습 전)
```
Scale 3 (kernel=3): 6.5241
Scale 5 (kernel=5): 6.5470
Scale 7 (kernel=7): 6.6049
```

**관찰:**
- 초기화 시 모든 스케일이 비슷한 가중치
- 학습 후 차이가 벌어질 것으로 예상

### 학습 후 예상 패턴
```
예시 1 (짧은 동작 중요):
  Scale 3: 8.2  ← 가장 높음
  Scale 5: 6.1
  Scale 7: 4.5

예시 2 (긴 동작 중요):
  Scale 3: 4.3
  Scale 5: 6.8
  Scale 7: 9.1  ← 가장 높음
```

---

## 🚀 사용 방법

### 1. 기본 사용
```python
from src.models.MSCSGRUModels_ScaleAware import MSCSGRU_ScaleAware

# 모델 생성
model = MSCSGRU_ScaleAware(
    learning_rate=1e-3,
    input_size=8,
    hidden_size=64,
    classes=24,
    use_hard_functions=False  # Sigmoid/Tanh 사용
)

# 학습
logits, loss = model(x, x_padding, y_targets)
loss.backward()
optimizer.step()
```

### 2. Hard 함수 버전 (임베디드 최적화)
```python
from src.models.MSCSGRUModels_ScaleAware import MSCSGRU_ScaleHard

# Hard 함수 강제 활성화
model = MSCSGRU_ScaleHard(
    learning_rate=1e-3,
    input_size=8,
    classes=24
)
```

### 3. 스케일 중요도 분석
```python
# 학습 후 분석
importance = model.get_scale_importance()

print("Update Gate 가중치:")
print(f"  Scale 3: {importance['gru_layer1']['update_gate']['scale_3']:.4f}")
print(f"  Scale 5: {importance['gru_layer1']['update_gate']['scale_5']:.4f}")
print(f"  Scale 7: {importance['gru_layer1']['update_gate']['scale_7']:.4f}")
```

---

## 🧪 실험 계획

### Phase 1: 정확도 검증
```bash
# Scale-Aware 모델 학습
python3 src/experiments/LightningTrain.py \
    -model MSCSGRU_ScaleAware \
    -model_type MSCSGRU_ScaleAware

# 목표: 기존 MSCSGRU와 동등 이상의 정확도 (99%+)
```

### Phase 2: 스케일 중요도 분석
```python
# 학습 완료 후
importance = model.get_scale_importance()

# 시각화
import matplotlib.pyplot as plt
scales = ['Scale 3', 'Scale 5', 'Scale 7']
weights = [importance['gru_layer1']['update_gate'][f'scale_{i}'] 
           for i in [3, 5, 7]]

plt.bar(scales, weights)
plt.title('스케일별 중요도')
plt.savefig('scale_importance.png')
```

### Phase 3: Hard 함수 성능 평가
```bash
# Hard 함수 버전 학습
python3 src/experiments/LightningTrain.py \
    -model MSCSGRU_ScaleHard \
    -model_type MSCSGRU_ScaleAware

# 목표: 정확도 98%+ 유지, 속도 향상
```

---

## 📊 예상 결과

### 정확도 예측
| 모델 | 예상 정확도 | 근거 |
|------|-------------|------|
| MSCSGRU | 99.3% | 기존 결과 |
| MSCSGRU_ScaleAware | **99.4-99.5%** | 더 풍부한 특징 표현 |
| MSCSGRU_ScaleHard | **98.5-99.0%** | Hard 함수로 인한 소폭 하락 |

### 속도 예측
| 환경 | 기존 | Scale-Aware | ScaleHard |
|------|------|-------------|-----------|
| **CPU** | 10ms | 20ms | 15ms |
| **GPU** | 5ms | 8ms | 7ms |
| **임베디드** | 100ms | 150ms | **80ms** ← 최적 |

---

## 🎯 장단점 요약

### 장점 ✅
1. **해석 가능성**: 어떤 스케일이 중요한지 분석 가능
2. **표현력 향상**: 스케일별 독립적 학습
3. **임베디드 최적화**: Hard 함수로 계산량 감소
4. **유연성**: Sigmoid/Tanh ↔ Hard 함수 전환 가능

### 단점 ⚠️
1. **파라미터 증가**: +33.7% (71K → 96K)
2. **속도 저하**: 2.1배 느림 (CPU 기준)
3. **과적합 위험**: 데이터가 적으면 문제
4. **Hard 함수 정확도**: 1-2% 하락 가능

---

## 🔮 향후 계획

### 단기
- [ ] 실제 데이터로 학습 및 평가
- [ ] 스케일 중요도 시각화
- [ ] 기존 모델과 성능 비교

### 중기
- [ ] ONNX 변환 및 최적화
- [ ] INT8 양자화
- [ ] TensorRT 최적화

### 장기
- [ ] 임베디드 시스템 배포
- [ ] MCU 포팅 (STM32, ESP32)
- [ ] 실시간 추론 검증

---

## 📚 참고 자료

### 코드 파일
- `src/models/ScaleAwareGRU.py` - Scale-Aware GRU Cell 구현
- `src/models/MSCSGRUModels_ScaleAware.py` - 전체 모델 통합

### 관련 논문
- GRU: Learning Phrase Representations using RNN Encoder-Decoder (Cho et al., 2014)
- Hard Functions: Searching for MobileNetV3 (Howard et al., 2019)

---

## 🤝 기여

이 모델은 실험적 구현입니다. 피드백과 개선 제안을 환영합니다!

1. 정확도 개선 아이디어
2. 속도 최적화 방법
3. 새로운 스케일 조합
4. 임베디드 배포 경험

---

## 📞 문의

프로젝트 관련 문의는 이슈를 생성해주세요.

