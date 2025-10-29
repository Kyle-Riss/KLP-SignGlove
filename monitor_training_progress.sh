#!/bin/bash
# 모든 모델의 학습 진행 상황을 모니터링하는 스크립트

echo "🚀 모델 재학습 진행 상황 모니터링"
echo "=================================="
echo ""

# GRU 진행 상황
echo "📊 GRU 모델:"
if [ -f gru_retrain.log ]; then
    tail -n 3 gru_retrain.log | grep -E "(Epoch|val/accuracy|test/accuracy)" || echo "  아직 학습 시작 안 됨"
else
    echo "  로그 파일 없음"
fi
echo ""

# MS-CSGRU 진행 상황
echo "📊 MS-CSGRU 모델:"
if [ -f mscsgru_retrain.log ]; then
    tail -n 3 mscsgru_retrain.log | grep -E "(Epoch|val/accuracy|test/accuracy)" || echo "  아직 학습 시작 안 됨"
else
    echo "  로그 파일 없음"
fi
echo ""

# A-GRU 진행 상황
echo "📊 A-GRU 모델:"
if [ -f agru_retrain.log ]; then
    tail -n 3 agru_retrain.log | grep -E "(Epoch|val/accuracy|test/accuracy)" || echo "  아직 학습 시작 안 됨"
else
    echo "  로그 파일 없음"
fi
echo ""

# 프로세스 확인
echo "🔍 실행 중인 학습 프로세스:"
ps aux | grep "LightningTrain.py" | grep -v grep | wc -l | xargs -I {} echo "  {} 개의 프로세스 실행 중"
echo ""

echo "💡 Tip: watch -n 10 ./monitor_training_progress.sh"




