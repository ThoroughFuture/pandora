#!/bin/bash

function find_free_port() {
    python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()'
}

export CUDA_VISIBLE_DEVICES=1,2,3,4

NUM_GPUS=4

CONFIGS=(
    "linear_768_marker_all_1_Scan"
    "linear_768_marker_all_14_Scan"
    "linear_768_marker_all_18_Scan"
    "linear_768_marker_all_5_Scan"
)

MAIN_SCRIPT="train_ddp.py"
LOG_DIR="./nohop_txt/bash_logs"
# ===========================================

mkdir -p $LOG_DIR

for cfg in "${CONFIGS[@]}"; do
    echo "=========================================="
    echo "Starting task: $cfg"
    echo "Using GPUs: $NUM_GPUS"
    echo "=========================================="

    LOG_FILE="$LOG_DIR/${cfg}.log"

    FREE_PORT=$(find_free_port)
    echo "Using port: $FREE_PORT"

    torchrun \
        --nproc_per_node=$NUM_GPUS \
        --master_port=$FREE_PORT \
        $MAIN_SCRIPT \
        --yml_opt_path "$cfg" \
        2>&1 | tee $LOG_FILE

    echo "Task $cfg finished. Log saved to $LOG_FILE"
done

echo "All tasks done."