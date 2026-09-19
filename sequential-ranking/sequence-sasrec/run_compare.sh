#!/bin/bash
set -e

DATA_PATH="./data/sample_movielens.csv"
SAVE_ROOT="./outputs_compare"

if [ ! -f "$DATA_PATH" ]; then
    python generate_sample_movielens.py \
        --save_path "$DATA_PATH" \
        --num_users 1000 \
        --num_items 2000
fi

python train_sequence_models.py \
    --data_path "$DATA_PATH" \
    --save_root "$SAVE_ROOT" \
    --model all \
    --epochs 10 \
    --batch_size 256 \
    --topk 20

python plot_sequence_results.py \
    --save_root "$SAVE_ROOT" \
    --topk 20

echo "[Done] Sequence recommendation comparison finished."
