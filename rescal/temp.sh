#!/bin/bash
#SBATCH --job-name=abl_time_off
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --time=07:00:00
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

DATASETS=(Tyson PEMSD3SUB PEMSD4SUB PEMSD5 PEMSD6 PEMSD8SUB PEMSD10 PEMSD11SUB PEMSD12SUB)

DATASET="${DATASETS[$SLURM_ARRAY_TASK_ID]}"

python -u main3.py \
  --dataset_choice "${DATASET}" \
  --base_model stgcn \
  --model_variant wz \
  --epochs 20 \
  --num_runs 3 \
  --base_seed 42 \
  --ablation_csv "ablation_time_off_${DATASET}.csv" \
  --ablation_modes rescal_base_plus_fold \
  --batch_size 32 \
  --rescal_batch_size 32 \
  --rescal_eval_batch_size 32 \
  --baseline_clean_start 0 \
  --rescal_clean_start 1 \
  --reuse_residual_cache 1 \
  --adj_type binary \
  --closure_representation weighted \
  --num_folds 5 \
  --rescal_use_time_features 0 \
  --rescal_residual_history 6 \
  --enable-cuda