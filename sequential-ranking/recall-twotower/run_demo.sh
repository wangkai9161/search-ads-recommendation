#!/bin/bash
set -e

DATA_PATH="./data/sample_movielens.csv"

if [ ! -f "$DATA_PATH" ]; then
    python generate_sample_movielens.py --save_path "$DATA_PATH" --num_users 1000 --num_items 2000
fi

python train_twotower.py --data_path "$DATA_PATH" --epochs 10 --batch_size 512 --topk 20
