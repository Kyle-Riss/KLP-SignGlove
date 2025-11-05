#!/bin/bash
set -e

run() {
    mdl=$1
    data=$2
    log=$3
    extra=$4
    echo "[START] $mdl -> $data (epochs=100)" | tee -a "$log"
    CUDA_VISIBLE_DEVICES= python3 -u src/experiments/LightningTrain.py -model "$mdl" -epochs 100 -batch_size 64 -data_dir "$data" $extra >> "$log" 2>&1
    echo "[DONE] $mdl -> $data" | tee -a "$log"
}

# MS3DGRU
# run MS3DGRU /home/billy/25-1kp/SignGlove-DataAnalysis/unified/unified lightning_logs/seq_ms3dgru_unified.log ''  # ✅ 완료
run MS3DGRU /home/billy/25-1kp/SignGlove-DataAnalysis/unified/unified_HB lightning_logs/seq_ms3dgru_yubeen.log ''
run MS3DGRU /home/billy/25-1kp/SignGlove-DataAnalysis/unified/unified_JY/JY lightning_logs/seq_ms3dgru_jaeyeon.log ''

# GRU
run GRU /home/billy/25-1kp/SignGlove-DataAnalysis/unified/unified lightning_logs/seq_gru_unified.log '-layers 2 -hidden_size 64 -lr 1e-3'
run GRU /home/billy/25-1kp/SignGlove-DataAnalysis/unified/unified_HB lightning_logs/seq_gru_yubeen.log '-layers 2 -hidden_size 64 -lr 1e-3'
run GRU /home/billy/25-1kp/SignGlove-DataAnalysis/unified/unified_JY/JY lightning_logs/seq_gru_jaeyeon.log '-layers 2 -hidden_size 64 -lr 1e-3'

# StackedGRU
run StackedGRU /home/billy/25-1kp/SignGlove-DataAnalysis/unified/unified lightning_logs/seq_stackedgru_unified.log '-layers 2 -hidden_size 64 -lr 1e-3'
run StackedGRU /home/billy/25-1kp/SignGlove-DataAnalysis/unified/unified_HB lightning_logs/seq_stackedgru_yubeen.log '-layers 2 -hidden_size 64 -lr 1e-3'
run StackedGRU /home/billy/25-1kp/SignGlove-DataAnalysis/unified/unified_JY/JY lightning_logs/seq_stackedgru_jaeyeon.log '-layers 2 -hidden_size 64 -lr 1e-3'

# MS3DStackedGRU
run MS3DStackedGRU /home/billy/25-1kp/SignGlove-DataAnalysis/unified/unified lightning_logs/seq_ms3dstk_unified.log '-layers 2 -hidden_size 64 -lr 1e-3'
run MS3DStackedGRU /home/billy/25-1kp/SignGlove-DataAnalysis/unified/unified_HB lightning_logs/seq_ms3dstk_yubeen.log '-layers 2 -hidden_size 64 -lr 1e-3'
run MS3DStackedGRU /home/billy/25-1kp/SignGlove-DataAnalysis/unified/unified_JY/JY lightning_logs/seq_ms3dstk_jaeyeon.log '-layers 2 -hidden_size 64 -lr 1e-3'

echo "All training completed!"

