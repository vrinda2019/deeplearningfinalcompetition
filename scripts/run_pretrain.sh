#!/bin/bash
# -----------------------------------------
# run_pretrain.sh
# -----------------------------------------
# Usage: bash run_pretrain.sh
# This script launches self-supervised training on burst/HPC.

# -----------------------------------------
# Request GPU resources (adjust as needed)
# -----------------------------------------
# Example for 1 A100 GPU for 4 hours:
# srun --account=csci_ga_2572-2025fa \
#      --partition=c12m85-a100-1 \
#      --gres=gpu:1 \
#      --time=04:00:00 \
#      --pty /bin/bash

# -----------------------------------------
# Load Conda
# -----------------------------------------
# Adjust path if Miniconda is installed elsewhere
source /home/vt2370/miniconda3/etc/profile.d/conda.sh

# Activate your environment
conda activate ssl_env

# -----------------------------------------
# Set paths
# -----------------------------------------
CONFIG_FILE="./configs/pretrain.yaml"
OUTPUT_DIR="./artifacts"

# Make sure output directories exist
mkdir -p $OUTPUT_DIR/logs
mkdir -p $OUTPUT_DIR/checkpoints

# -----------------------------------------
# Run training
# -----------------------------------------
python code/train_ssl.py \
    --config $CONFIG_FILE \
    --output_dir $OUTPUT_DIR \
    --device cuda

# Optional: redirect logs
# python code/train_ssl.py \
#    --config $CONFIG_FILE \
#    --output_dir $OUTPUT_DIR \
#    --device cuda \
#    > $OUTPUT_DIR/logs/train.log 2>&1




