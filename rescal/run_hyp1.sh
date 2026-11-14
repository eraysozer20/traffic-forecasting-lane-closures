#!/bin/bash
#SBATCH --job-name=main_hyp1_rescal
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=08:00:00
#SBATCH --partition=tue.gpu.q
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

set -euo pipefail

module load anaconda/2022.10-pth39
eval "$(conda shell.bash hook)"
conda activate gwnet_env

cd "$SLURM_SUBMIT_DIR"

mkdir -p logs save data/processed result

nvidia-smi || true
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', 'cuda' if torch.cuda.is_available() else 'cpu')"

# ----------------------------
# Config (edit these)
# ----------------------------
DATASET="Tyson"     # Tyson | PEMSD8 | SIM_GRID
MODEL="wz"          # wz | no

T=12
P=6
BS=16
LR=0.001
EPOCHS=50

A_RATIO=0.8
B_RATIO=0.5

# Optional: if you want to force a clean run each job, set to 1
CLEAN_START=0

# ----------------------------
# Cleanup (optional)
# ----------------------------
if [[ "${CLEAN_START}" -eq 1 ]]; then
  echo "[CLEAN_START] Removing caches + checkpoints for ${DATASET}/${MODEL}"
  rm -f "save/${DATASET}/${MODEL}/baseline_params.pt" || true
  rm -f "save/${DATASET}/${MODEL}/rescal_params.pt" || true
  rm -rf "data/processed/${DATASET}/${MODEL}/hyp1_residuals" || true
fi

echo "============================================"
echo "Running main_hyp1.py"
echo "dataset=${DATASET}, model=${MODEL}"
echo "T=${T}, P=${P}, batch_size=${BS}, lr=${LR}, epochs=${EPOCHS}"
echo "hyp1_a_ratio=${A_RATIO}, hyp1_b_ratio=${B_RATIO}"
echo "============================================"

python -u main_hyp1.py \
  --dataset_choice "${DATASET}" \
  --model_variant "${MODEL}" \
  --epochs "${EPOCHS}" \
  --hyp1_a_ratio "${A_RATIO}" \
  --hyp1_b_ratio "${B_RATIO}" \
  --enable-cuda

echo "Done."
