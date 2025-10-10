#!/bin/bash

# 모델별 학습 스크립트
# 사용법: bash ModelTrain.sh [MODEL_TYPE] [MODEL_NAME]

# 기본 설정
MODEL_TYPE=${1:-"RNN"}  # RNN, ENCODER, MSCSGRU
MODEL_NAME=${2:-"GRU"}  # GRU, LSTM, TransformerEncoder, CNNEncoder, MSCSGRU 등

# 공통 하이퍼파라미터
lr=0.001
hidden_size=64
time_steps=87
batch_size=64
epochs=100

# 모델별 특화 설정
case $MODEL_TYPE in
    "RNN")
        case $MODEL_NAME in
            "GRU")
                layers=2
                echo "🚀 GRU 모델 학습 시작..."
                ;;
            "LSTM")
                layers=2
                echo "🚀 LSTM 모델 학습 시작..."
                ;;
            "StackedGRU")
                layers=3
                echo "🚀 StackedGRU 모델 학습 시작..."
                ;;
            "StackedLSTM")
                layers=3
                echo "🚀 StackedLSTM 모델 학습 시작..."
                ;;
            *)
                echo "❌ 지원하지 않는 RNN 모델: $MODEL_NAME"
                echo "지원 모델: GRU, LSTM, StackedGRU, StackedLSTM"
                exit 1
                ;;
        esac
        ;;
    "ENCODER")
        case $MODEL_NAME in
            "TransformerEncoder")
                num_heads=8
                num_layers=2
                echo "🚀 TransformerEncoder 모델 학습 시작..."
                ;;
            "CNNEncoder")
                echo "🚀 CNNEncoder 모델 학습 시작..."
                ;;
            "HybridEncoder")
                num_heads=8
                num_layers=2
                echo "🚀 HybridEncoder 모델 학습 시작..."
                ;;
            *)
                echo "❌ 지원하지 않는 ENCODER 모델: $MODEL_NAME"
                echo "지원 모델: TransformerEncoder, CNNEncoder, HybridEncoder"
                exit 1
                ;;
        esac
        ;;
    "MSCSGRU")
        case $MODEL_NAME in
            "MSCGRU")
                echo "🚀 MSCGRU 모델 학습 시작..."
                ;;
            "CNNGRU")
                echo "🚀 CNNGRU 모델 학습 시작..."
                ;;
            "CNNStackedGRU")
                echo "🚀 CNNStackedGRU 모델 학습 시작..."
                ;;
            "MSCSGRU")
                echo "🚀 MSCSGRU 모델 학습 시작..."
                ;;
            *)
                echo "❌ 지원하지 않는 MSCSGRU 모델: $MODEL_NAME"
                echo "지원 모델: MSCGRU, CNNGRU, CNNStackedGRU, MSCSGRU"
                exit 1
                ;;
        esac
        ;;
    *)
        echo "❌ 지원하지 않는 모델 타입: $MODEL_TYPE"
        echo "지원 타입: RNN, ENCODER, MSCSGRU"
        exit 1
        ;;
esac

# 로그 파일명 설정
LOG_FILE="../../training_output_${MODEL_TYPE}_${MODEL_NAME}.log"

echo "📊 학습 설정:"
echo "  모델 타입: $MODEL_TYPE"
echo "  모델 이름: $MODEL_NAME"
echo "  학습률: $lr"
echo "  히든 사이즈: $hidden_size"
echo "  배치 사이즈: $batch_size"
echo "  에포크: $epochs"
echo "  로그 파일: $LOG_FILE"
echo ""

# 학습 실행
python3 LightningTrain.py \
    -model $MODEL_NAME \
    -model_type $MODEL_TYPE \
    -hidden_size $hidden_size \
    -lr $lr \
    -time_steps $time_steps \
    -batch_size $batch_size \
    -epochs $epochs \
    ${layers:+ -layers $layers} \
    ${num_heads:+ -num_heads $num_heads} \
    ${num_layers:+ -num_layers $num_layers} \
    2>&1 | tee $LOG_FILE

echo "✅ $MODEL_NAME 모델 학습 완료!"
echo "📁 로그 파일: $LOG_FILE"


