#!/bin/bash
#SBATCH --job-name=main3_lstm_rescal
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --partition=tue.gpu.q
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-8

set -euo pipefail

module load anaconda/2022.10-pth39
eval "$(conda shell.bash hook)"
conda activate gwnet_env

cd "$SLURM_SUBMIT_DIR"

mkdir -p logs save data/processed result

DATASETS=(
  Tyson
  PEMSD3SUB
  PEMSD4SUB
  PEMSD5
  PEMSD6
  PEMSD8SUB
  PEMSD10
  PEMSD11SUB
  PEMSD12SUB
)

DATASET="${DATASETS[$SLURM_ARRAY_TASK_ID]}"
BASE_MODEL="lstm"
MODEL="wz"

EPOCHS=20
NUM_RUNS=1
BASE_SEED=42

BASELINE_CLEAN_START=0
RESCAL_CLEAN_START=1
REUSE_RESIDUAL_CACHE=1

BASELINE_BATCH=32
RESCAL_BATCH=32
RESCAL_EVAL_BATCH=32

CSV_NAME="rescal_results_${DATASET}_${BASE_MODEL}.csv"

MODES=(
  rescal_base_only
  rescal_base_plus_fold
  rescal_base_plus_fold_no_closure
)

echo "============================================"
echo "Running main3.py"
echo "dataset=${DATASET}"
echo "base_model=${BASE_MODEL}"
echo "epochs=${EPOCHS}, runs=${NUM_RUNS}"
echo "batch=${BASELINE_BATCH}, rescal_batch=${RESCAL_BATCH}, rescal_eval_batch=${RESCAL_EVAL_BATCH}"
echo "modes=${MODES[*]}"
echo "============================================"

echo "SLURM job id: $SLURM_JOB_ID"
echo "SLURM array task id: $SLURM_ARRAY_TASK_ID"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not_set}"

echo "===== GPU BEFORE ====="
nvidia-smi || true

python - <<EOF
import torch
print("CUDA available:", torch.cuda.is_available())
print("Device:", "cuda" if torch.cuda.is_available() else "cpu")
EOF

python -u main3.py \
  --dataset_choice "${DATASET}" \
  --base_model "${BASE_MODEL}" \
  --model_variant "${MODEL}" \
  --epochs "${EPOCHS}" \
  --num_runs "${NUM_RUNS}" \
  --base_seed "${BASE_SEED}" \
  --ablation_csv "${CSV_NAME}" \
  --ablation_modes "${MODES[@]}" \
  --baseline_clean_start "${BASELINE_CLEAN_START}" \
  --rescal_clean_start "${RESCAL_CLEAN_START}" \
  --reuse_residual_cache "${REUSE_RESIDUAL_CACHE}" \
  --batch_size "${BASELINE_BATCH}" \
  --rescal_batch_size "${RESCAL_BATCH}" \
  --rescal_eval_batch_size "${RESCAL_EVAL_BATCH}" \
  --enable-cuda

echo "===== GPU AFTER ====="
nvidia-smi || true

echo "Done: ${DATASET}"