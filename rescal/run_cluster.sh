#!/bin/bash
#SBATCH --job-name=stgcn_lam_sweep_serial
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
mkdir -p data/processed/SIM_GRID save/SIM_GRID

nvidia-smi || true

DATASET="SIM_GRID"
MODEL="wz"
LAMBDAS=(0.3 0.5 0.7 0.9 1.1 1.3 1.5 2.0)

echo "============================================"
echo "Serial sweep: dataset=${DATASET}, model=${MODEL}"
echo "Lambdas: ${LAMBDAS[*]}"
echo "============================================"

for LAMBDA in "${LAMBDAS[@]}"; do
  echo ""
  echo "--------------------------------------------"
  echo "Running lambda_=${LAMBDA}"
  echo "--------------------------------------------"

  python -u main.py \
    --dataset_choice "${DATASET}" \
    --model_variant "${MODEL}" \
    --lambda_ "${LAMBDA}"

  echo "[CLEANUP] Removing cached processed + checkpoints for ${DATASET}..."
  rm -rf "data/processed/${DATASET}/"*
  rm -rf "save/${DATASET}/"*
  echo "[CLEANUP] Done."
done

echo "All runs finished."
