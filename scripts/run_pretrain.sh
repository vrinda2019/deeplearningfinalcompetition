#!/bin/bash
# =========================================
# Run self-supervised training on Burst HPC
# =========================================

# Request: interactive shell or Slurm (if needed, adjust GPU/time in srun)
# Example for Slurm:
# srun --account=csci_ga_2572-2025fa --partition=c12m85-a100-1 --gres=gpu:1 --time=08:00:00 --pty /bin/bash

# -----------------------------
# 1. Initialize Conda
# -----------------------------
CONDA_DIR="/home/vt2370/miniconda3"
source $CONDA_DIR/etc/profile.d/conda.sh
conda activate ssl_env

# -----------------------------
# 2. Set paths
# -----------------------------
DATA_DIR="/scratch/vt2370/fall2025_deeplearning/cc3m_all/train"
CONFIG_FILE="./configs/pretrain.yaml"
OUTPUT_DIR="./artifacts"

mkdir -p $OUTPUT_DIR

# -----------------------------
# 3. Launch training
# -----------------------------
echo "Starting SSL training..."
python code/train_ssl.py \
    --config $CONFIG_FILE \
    --data_dir $DATA_DIR \
    --output_dir $OUTPUT_DIR

echo "Training finished!"
