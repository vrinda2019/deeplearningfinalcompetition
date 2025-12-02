#!/bin/bash
#SBATCH --account=csci_ga_2572-2025fa   # Your account
#SBATCH --partition=c12m85-a100-1       # GPU partition
#SBATCH --gres=gpu:1                     # Number of GPUs
#SBATCH --time=08:00:00                  # Max runtime
#SBATCH --job-name=ssl_pretrain
#SBATCH --output=logs/ssl_pretrain_%j.log
#SBATCH --chdir=/scratch/vt2370/fall2025_deeplearning/deeplearning_repo

# --------------------------
# Load conda environment
# --------------------------
echo "Activating conda environment..."
source /home/vt2370/miniconda3/etc/profile.d/conda.sh
conda activate ssl_env

# Check Python and CUDA
echo "Python version: $(python --version)"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"

# --------------------------
# NumPy fix (if needed)
# --------------------------
pip install --upgrade "numpy<2"

# --------------------------
# Config & output directories
# --------------------------
CONFIG_FILE="configs/pretrain.yaml"
OUTPUT_DIR="./artifacts"

mkdir -p $OUTPUT_DIR
mkdir -p logs

echo "Using config: $CONFIG_FILE"
echo "Output directory: $OUTPUT_DIR"

# --------------------------
# Start SSL pretraining
# --------------------------
echo "Starting SSL pretraining..."
python code/train_ssl.py --config $CONFIG_FILE

echo "Training finished!"
