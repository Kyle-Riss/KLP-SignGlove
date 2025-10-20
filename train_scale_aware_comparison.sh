#!/bin/bash
# Scale-Aware GRU 모델 비교 학습 스크립트

echo "🚀 Scale-Aware GRU 모델 비교 학습 시작"
echo "========================================================================"

# 공통 설정
EPOCHS=100
BATCH_SIZE=32
LR=0.001
HIDDEN_SIZE=64

# 1. 기존 MSCSGRU (Baseline)
echo ""
echo "📊 1/4: 기존 MSCSGRU 학습 (Baseline)"
echo "------------------------------------------------------------------------"
python3 src/experiments/LightningTrain.py \
    -model MSCSGRU \
    -model_type MSCSGRU \
    -epochs $EPOCHS \
    -batch_size $BATCH_SIZE \
    -lr $LR \
    -hidden_size $HIDDEN_SIZE \
    -description "Baseline MSCSGRU for comparison" \
    2>&1 | tee training_output_MSCSGRU_baseline.log

echo "✅ 기존 MSCSGRU 학습 완료"

# 2. MSCSGRU_ScaleAware (Sigmoid/Tanh)
echo ""
echo "📊 2/4: MSCSGRU_ScaleAware 학습 (Sigmoid/Tanh)"
echo "------------------------------------------------------------------------"
python3 src/experiments/LightningTrain.py \
    -model MSCSGRU_ScaleAware \
    -model_type MSCSGRU_ScaleAware \
    -epochs $EPOCHS \
    -batch_size $BATCH_SIZE \
    -lr $LR \
    -hidden_size $HIDDEN_SIZE \
    -description "Scale-Aware GRU with Sigmoid/Tanh" \
    2>&1 | tee training_output_MSCSGRU_ScaleAware.log

echo "✅ MSCSGRU_ScaleAware 학습 완료"

# 3. MSCSGRU_ScaleHard (HardSigmoid/HardTanh)
echo ""
echo "📊 3/4: MSCSGRU_ScaleHard 학습 (Hard Functions)"
echo "------------------------------------------------------------------------"
python3 src/experiments/LightningTrain.py \
    -model MSCSGRU_ScaleHard \
    -model_type MSCSGRU_ScaleAware \
    -epochs $EPOCHS \
    -batch_size $BATCH_SIZE \
    -lr $LR \
    -hidden_size $HIDDEN_SIZE \
    -description "Scale-Aware GRU with Hard Functions" \
    2>&1 | tee training_output_MSCSGRU_ScaleHard.log

echo "✅ MSCSGRU_ScaleHard 학습 완료"

# 4. MSCGRU_ScaleAware (Single GRU)
echo ""
echo "📊 4/4: MSCGRU_ScaleAware 학습 (Single GRU)"
echo "------------------------------------------------------------------------"
python3 src/experiments/LightningTrain.py \
    -model MSCGRU_ScaleAware \
    -model_type MSCSGRU_ScaleAware \
    -epochs $EPOCHS \
    -batch_size $BATCH_SIZE \
    -lr $LR \
    -hidden_size $HIDDEN_SIZE \
    -description "Scale-Aware Single GRU" \
    2>&1 | tee training_output_MSCGRU_ScaleAware_single.log

echo "✅ MSCGRU_ScaleAware (Single) 학습 완료"

echo ""
echo "========================================================================"
echo "🎉 모든 모델 학습 완료!"
echo "========================================================================"
echo ""
echo "📊 학습 로그 파일:"
echo "  - training_output_MSCSGRU_baseline.log"
echo "  - training_output_MSCSGRU_ScaleAware.log"
echo "  - training_output_MSCSGRU_ScaleHard.log"
echo "  - training_output_MSCGRU_ScaleAware_single.log"
echo ""
echo "📈 다음 단계:"
echo "  python3 analyze_scale_aware_results.py"

