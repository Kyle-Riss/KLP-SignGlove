#!/bin/bash
set -e

# 5개의 서로 다른 시드
SEEDS=(1337 42 1234 5678 9999)
NUM_RUNS=${#SEEDS[@]}

echo "============================================================"
echo "Multi-Seed Training (${NUM_RUNS} runs per model-dataset)"
echo "Seeds: ${SEEDS[@]}"
echo "============================================================"
echo ""

run() {
    mdl=$1
    data=$2
    seed=$3
    run_idx=$4
    log=$5
    extra=$6
    
    echo "[RUN $run_idx/$NUM_RUNS] $mdl -> $data (seed=$seed)" | tee -a "$log"
    CUDA_VISIBLE_DEVICES= python3 -u src/experiments/LightningTrain.py \
        -model "$mdl" \
        -epochs 100 \
        -batch_size 64 \
        -data_dir "$data" \
        -seed "$seed" \
        $extra \
        >> "$log" 2>&1
    
    echo "[DONE RUN $run_idx/$NUM_RUNS] $mdl -> $data (seed=$seed)" | tee -a "$log"
    echo "" | tee -a "$log"
}

# 모델-데이터셋 조합
MODELS=("MS3DGRU" "GRU" "StackedGRU" "MS3DStackedGRU")
DATASETS=(
    "/home/billy/25-1kp/SignGlove-DataAnalysis/unified/unified:unified"
    "/home/billy/25-1kp/SignGlove-DataAnalysis/unified/unified_HB:yubeen"
    "/home/billy/25-1kp/SignGlove-DataAnalysis/unified/unified_JY/JY:jaeyeon"
)

# 각 모델-데이터셋 조합에 대해 5번 실행
for model in "${MODELS[@]}"; do
    for dataset_info in "${DATASETS[@]}"; do
        IFS=':' read -r data_path dataset_name <<< "$dataset_info"
        
        # 각 시드에 대해 실행
        for run_idx in $(seq 1 $NUM_RUNS); do
            seed=${SEEDS[$((run_idx-1))]}
            
            # 로그 파일명 생성 (모델_데이터셋_seed_런번호.log)
            log_file="lightning_logs/multi_seed_${model,,}_${dataset_name}_seed${seed}_run${run_idx}.log"
            
            # 모델별 추가 파라미터
            if [ "$model" == "GRU" ] || [ "$model" == "StackedGRU" ] || [ "$model" == "MS3DStackedGRU" ]; then
                extra_params="-layers 2 -hidden_size 64 -lr 1e-3"
            else
                extra_params=""
            fi
            
            # 실행
            run "$model" "$data_path" "$seed" "$run_idx" "$log_file" "$extra_params"
        done
    done
done

echo "============================================================"
echo "✅ All multi-seed training completed!"
echo "============================================================"

