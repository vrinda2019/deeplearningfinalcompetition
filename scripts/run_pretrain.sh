#!/bin/bash
#SBATCH --account=csci_ga_2572-2025fa
#SBATCH --partition=g2-standard-12
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --job-name=ssl_pretrain
#SBATCH --output=logs/pretrain_%j.out
#SBATCH --requeue

# Load modules / conda environment
module load anaconda
source activate my_ssl_env

# Run training
python code/train_ssl.py --config configs/pretrain.yaml



