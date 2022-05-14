#!/bin/bash

LOG_DIR="/private/home/millicentli/BERT-kNN/preprocess/log"

if [ ! -d "$LOG_DIR" ]; then
    mkdir -p "$LOG_DIR"
fi

for n in {0..99}
do
    sbatch --job-name="enc_context" \
        --output="$LOG_DIR/%j.out" \
        --error="$LOG_DIR/%j.err" \
        --partition=learnlab,learnfair \
        --nodes=1 \
        --gres=gpu:1 \
        --mem=450gb \
        --time=1000 \
        --cpus-per-gpu=20 \
        -C volta32gb \
        --wrap=". /public/apps/anaconda3/2021.05/etc/profile.d/conda.sh; 
                conda activate bert_knn; 
                cd /private/home/millicentli/BERT-kNN;
                python preprocess/encode_context.py --dump $n"
done