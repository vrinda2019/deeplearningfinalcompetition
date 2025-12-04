#!/bin/bash
#SBATCH --account=csci_ga_2572-2025fa
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --job-name=ssl_pretrain
#SBATCH --output=ssl_pretrain.out
#SBATCH --error=ssl_pretrain.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

echo "Starting SSL pretraining job..."

# -------------------------------
# Paths
# -------------------------------
CONFIG="configs/pretrain.yaml"
SAVE_DIR=$(grep save_dir $CONFIG | awk '{print $2}')
LATEST_CKPT=""

# -------------------------------
# Auto-detect latest checkpoint
# -------------------------------
if [ -d "$SAVE_DIR" ]; then
    LATEST_CKPT=$(ls -1t $SAVE_DIR/checkpoint_epoch_*.pth 2>/dev/null | head -n 1)
fi

# -------------------------------
# Decide command (resume vs fresh)
# -------------------------------
if [ -f "$LATEST_CKPT" ]; then
    echo "Resuming from checkpoint: $LATEST_CKPT"
    python train_ssl.py --config $CONFIG --resume $LATEST_CKPT
else
    echo "Starting fresh training (no checkpoint found)."
    python train_ssl.py --config $CONFIG
fi

echo "Training complete."
