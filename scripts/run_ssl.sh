#!/bin/bash
#SBATCH --account=csci_ga_2572-2025fa
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --job-name=ssl_pretrain
#SBATCH --output=ssl_pretrain_%j.out
#SBATCH --error=ssl_pretrain_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --chdir=/scratch/vt2370/fall2025_deeplearning/deeplearning_repo

echo "=== SSL Pretraining Job Started ==="
echo "Running on: $(hostname)"

# -------------------------------
# Activate environment
# -------------------------------
source /home/vt2370/miniconda3/etc/profile.d/conda.sh
conda activate ssl_env

# -------------------------------
# Config & checkpoint paths
# -------------------------------
CONFIG="configs/pretrain96.yaml"
SAVE_DIR="./artif"

# -------------------------------
# Find latest checkpoint
# -------------------------------
echo "Looking for checkpoints in: $SAVE_DIR"
LATEST_CKPT=$(ls -1t $SAVE_DIR/checkpoint_epoch_*.pth 2>/dev/null | head -n 1)

# -------------------------------
# Resume or start new training
# -------------------------------
if [ -f "$LATEST_CKPT" ]; then
    echo "Resuming from checkpoint:"
    echo "  $LATEST_CKPT"
    python code/train_sslpy --config $CONFIG --resume $LATEST_CKPT
else
    echo "No checkpoint found. Starting fresh training."
    python code/train_ssl.py --config $CONFIG
fi

echo "=== SSL Pretraining Job Finished ==="
