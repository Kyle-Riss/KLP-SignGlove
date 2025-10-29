#!/bin/bash

echo "⏳ 학습 완료 대기 중..."
echo ""

# GRU 학습 완료 확인
while true; do
    if grep -q "test/accuracy" gru_training.log 2>/dev/null; then
        GRU_ACC=$(grep "test/accuracy" gru_training.log | tail -1 | awk '{print $5}')
        echo "✅ GRU 학습 완료! Test Accuracy: $GRU_ACC"
        break
    fi
    sleep 5
done

# MS-CSGRU 학습 완료 확인
while true; do
    if grep -q "test/accuracy" mscsgru_training.log 2>/dev/null; then
        MSCSGRU_ACC=$(grep "test/accuracy" mscsgru_training.log | tail -1 | awk '{print $5}')
        echo "✅ MS-CSGRU 학습 완료! Test Accuracy: $MSCSGRU_ACC"
        break
    fi
    sleep 5
done

echo ""
echo "🎉 모든 모델 학습 완료!"
echo "📊 이제 노이즈 견고성 분석을 시작합니다..."
echo ""

# 노이즈 견고성 분석 실행
python3 analyze_noise_robustness.py




