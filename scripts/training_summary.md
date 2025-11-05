# 학습 필요 여부 및 체크포인트 상태 정리

## 📊 현재 체크포인트 상태

### ✅ 사용 가능한 체크포인트

| 모델 | 체크포인트 경로 | 실제 모델 타입 | 상태 | 파라미터 수 |
|------|---------------|--------------|------|------------|
| **StackedGRU** | `archive/checkpoints_backup/checkpoints_backup/GRU_best.ckpt` | StackedGRU (2 layers) | ✅ 사용 가능 | 74,776 |
| **MS3DGRU** | `best_model/ms3dgru_best.ckpt` | MS3DGRU | ✅ 사용 가능 (99.13% 정확도) | 59,228 |

### ❌ 누락된 체크포인트

| 모델 | 상태 | 이유 |
|------|------|------|
| **GRU** | ❌ 없음 | `GRU_best.ckpt`는 실제로 StackedGRU임 |
| **MS3DStackedGRU** | ❌ 없음 | MS3DGRU 체크포인트를 사용하지만 구조 불일치 |

---

## 🎯 학습 필요 여부

### 1. **GRU** (단일 레이어) - **새로 학습 필요** ✅

**이유:**
- `GRU_best.ckpt`는 실제로 StackedGRU입니다 (2개 레이어: l0, l1)
- 실제 단일 레이어 GRU 체크포인트가 없습니다

**학습 방법:**
```bash
python src/experiments/LightningTrain.py \
    -model GRU \
    -epochs 100 \
    -batch_size 64 \
    -data_dir /home/billy/25-1kp/SignGlove-DataAnalysis/unified/unified \
    -seed 1337 \
    -layers 1 \
    -hidden_size 64 \
    -lr 1e-3
```

**주의사항:**
- `-layers 1`로 설정하여 단일 레이어 GRU 생성
- 학습 후 체크포인트를 `archive/checkpoints_backup/checkpoints_backup/GRU_single_best.ckpt`로 저장

---

### 2. **StackedGRU** - **이미 있음** ✅

**상태:**
- `archive/checkpoints_backup/checkpoints_backup/GRU_best.ckpt`가 실제로 StackedGRU입니다
- 이미 사용 가능하므로 추가 학습 불필요

**현재 체크포인트:**
- 경로: `archive/checkpoints_backup/checkpoints_backup/GRU_best.ckpt`
- 레이어: 2개 (l0, l1)
- 파라미터 수: 74,776
- Epoch: 65

**권장사항:**
- 혼동 행렬 생성 시 이 체크포인트를 StackedGRU로 사용
- 파일명을 `StackedGRU_best.ckpt`로 변경하는 것을 권장

---

### 3. **MS3DStackedGRU** - **새로 학습 필요** ✅

**이유:**
- MS3DGRU 체크포인트를 사용하지만 구조 불일치:
  ```
  size mismatch for tower2.0.weight: 
  copying a param with shape torch.Size([32, 1, 5, 5, 5]) 
  from checkpoint, the shape in current model is torch.Size([32, 1, 3, 5, 3]).
  ```
- MS3DStackedGRU 전용 체크포인트가 없습니다

**학습 방법:**
```bash
python src/experiments/LightningTrain.py \
    -model MS3DStackedGRU \
    -epochs 100 \
    -batch_size 64 \
    -data_dir /home/billy/25-1kp/SignGlove-DataAnalysis/unified/unified \
    -seed 1337 \
    -cnn_filters 32 \
    -hidden_size 64 \
    -lr 1e-3
```

---

## 📋 권장 학습 순서

1. **GRU 학습** (단일 레이어)
   - 가장 기본 모델
   - 다른 모델과 성능 비교 가능

2. **MS3DStackedGRU 학습**
   - MS3DGRU의 성능 향상 버전
   - 구조 불일치 문제 해결

3. **StackedGRU** (선택사항)
   - 이미 있지만, 더 나은 성능을 위해 재학습 가능

---

## 🔧 Scaler 파일 문제

**문제:**
- Scaler 파일(`scaler.pkl`)이 없어서 추론 시 정규화가 적용되지 않음
- 성능 저하 가능성

**해결 방안:**
1. 학습 시 scaler 자동 저장
2. 또는 별도 스크립트로 scaler 생성

---

## ✅ 최종 권장사항

### 즉시 학습 필요:
1. **GRU** (단일 레이어) - 혼동 행렬 생성에 필요
2. **MS3DStackedGRU** - 모델 구조 불일치 해결

### 이미 있음:
- **StackedGRU** - `GRU_best.ckpt` 사용
- **MS3DGRU** - `ms3dgru_best.ckpt` 사용 (99.13% 정확도)

### 추가 작업:
- Scaler 파일 생성 또는 자동 저장 기능 추가






