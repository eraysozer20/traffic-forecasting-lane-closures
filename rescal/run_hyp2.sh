#!/bin/bash
#SBATCH --job-name=rescal_hyp2
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=08:00:00
#SBATCH --partition=tue.gpu.q
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

set -euo pipefail

# =========================
# Environment
# =========================
module load anaconda/2022.10-pth39
eval "$(conda shell.bash hook)"
conda activate gwnet_env

cd "$SLURM_SUBMIT_DIR"

# create dirs if missing
mkdir -p logs save data/processed result
mkdir -p data/processed/Tyson save/Tyson
mkdir -p data/processed/PEMSD8 save/PEMSD8
mkdir -p data/processed/SIM_GRID save/SIM_GRID

echo "============================================"
echo "Running HYP2: Baseline + ResCAL (shared splits)"
echo "Script: main_hyp2.py"
echo "Node: $(hostname)"
echo "GPU:"
nvidia-smi || true
echo "============================================"

# =========================
# Config
# =========================
DATASET="Tyson"     # Tyson | PEMSD8 | SIM_GRID
MODEL="wz"             # wz | no
EPOCHS=50
BATCH=16
LR=0.001

echo ""
echo "Dataset  : ${DATASET}"
echo "Model    : ${MODEL}"
echo "Epochs   : ${EPOCHS}"
echo "Batch    : ${BATCH}"
echo "LR       : ${LR}"
echo ""

# =========================
# Run
# =========================
python -u main_hyp2.py \
  --dataset_choice "${DATASET}" \
  --model_variant "${MODEL}" \
  --epochs "${EPOCHS}"

echo ""
echo "Run finished."
echo "============================================"
