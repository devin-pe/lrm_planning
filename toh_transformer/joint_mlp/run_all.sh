#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --job-name=joint_mlp_patching
#SBATCH --ntasks=1
#SBATCH --time=6:00:00
#SBATCH --output=/home/dpereira/lrm_planning/work/joint_mlp_patching_%A.out

set -euo pipefail

module purge
module load 2025
module load Python/3.13.1-GCCcore-14.2.0
module load CUDA/12.8.0
source /home/dpereira/lrm_planning/env/bin/activate

REPO_ROOT=/home/dpereira/lrm_planning
cd "$REPO_ROOT"

export PYTHONUNBUFFERED=1

LAYER=${LAYER:-5}
EPOCHS=${EPOCHS:-500}
DEVICE=${DEVICE:-cuda}
# Bottleneck sizes to sweep; accepts space- or comma-separated (e.g. "32,64").
H_DIMS=${H_DIMS:-8 16}

mkdir -p outputs/joint_mlp_probes work

echo "========== Joint MLP Probes + Patching =========="
echo "LAYER=$LAYER EPOCHS=$EPOCHS DEVICE=$DEVICE H_DIMS=$H_DIMS"
echo "Python: $(which python)"; python --version
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "================================================="

for H_DIM in $(echo "$H_DIMS" | tr ',' ' '); do
  for REGIME in sequential_cls_first joint sequential_dist_first; do
    CKPT="outputs/joint_mlp_probes/regime_${REGIME}_layer${LAYER}_h${H_DIM}.pt"
    PATCH_JSON="outputs/joint_mlp_probes/patching_${REGIME}_layer${LAYER}_h${H_DIM}.json"

    echo "--- TRAIN regime=$REGIME h_dim=$H_DIM ---"
    python toh_transformer/joint_mlp/train.py \
        --regime "$REGIME" \
        --layer "$LAYER" \
        --h_dim "$H_DIM" \
        --epochs "$EPOCHS" \
        --device "$DEVICE" \
        --output "$CKPT"

    echo "--- PATCH regime=$REGIME h_dim=$H_DIM ---"
    python toh_transformer/joint_mlp/patching.py \
        --checkpoint "$CKPT" \
        --layer "$LAYER" \
        --device "$DEVICE" \
        --output "$PATCH_JSON"
  done
done

echo "--- REPORT ---"
python toh_transformer/joint_mlp/report.py --layer "$LAYER" --h_dims "$H_DIMS"

echo "Joint MLP probes + patching completed."
