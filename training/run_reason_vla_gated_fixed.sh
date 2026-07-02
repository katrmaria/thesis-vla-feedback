#!/bin/bash
#SBATCH -A NAISS2025-22-1583 -p alvis
#SBATCH -J reason-vla-gfix             # Job name
#SBATCH -t 3-00:00:00                  # Walltime: 3 days
#SBATCH -N 1                           # 1 node
#SBATCH --gpus-per-node=A40:1          # 1x A40 (48GB)
#SBATCH -o logs/reason_vla_gfix_%j.out # stdout
#SBATCH -e logs/reason_vla_gfix_%j.err # stderr

# ============================================================================
# ReasonVLA Training — gated-fixed variant
#
# Identical to run_reason_vla.sh except it launches reason_vla_gated_fixed_train.py,
# which monkey-patches reason_vla.ReasonVLA -> ReasonVLAGatedFixed before main().
# Only the `gated` feedback path differs (identity gate init + bounded hint).
# All other feedback modes (additive, film, adaln, scaled) behave exactly as
# the original — but you should still launch additive runs via the original
# script, this one is just for the gated A/B test.
#
# Usage (same flags as run_reason_vla.sh):
#   sbatch --gpus-per-node=A100:1 --time=15:00:00 \
#     run_reason_vla_gated_fixed.sh \
#     --stage 1 --vla-path openvla/openvla-7b-finetuned-libero-goal \
#     --dataset-name libero_goal_no_noops \
#     --feedback-mode gated --hidden-layer 24 \
#     --lr 2e-5 --max-steps 4950
# ============================================================================

# ============================================================================
# Defaults — overridden by named flags below; final values are echoed before run
# ============================================================================
PROJECT_DIR="/mimer/NOBACKUP/groups/robot_unforseen/mariakat"
VENV_DIR="$PROJECT_DIR/venvs/openvla"
WORK_DIR="/cephyr/users/mariakat/Alvis/openvla"

NUM_GPUS=0
STAGE=""
VLA_PATH="openvla/openvla-7b-finetuned-libero-spatial"
DATA_ROOT_DIR="$PROJECT_DIR/data/modified_libero_rlds"
DATASET_NAME="libero_spatial_no_noops"
OUTPUT_DIR="$PROJECT_DIR/runs/reason_vla"
BATCH_SIZE=1
GRAD_ACCUM=32
LR="2e-5"
MAX_STEPS=16560
SAVE_STEPS=1650
HIDDEN_LAYER=-1
FEEDBACK_MODE="gated"   # this script only makes sense with gated; default reflects intent
LORA_RANK=32
IMAGE_AUG=true

REMAINING_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --num-gpus)         NUM_GPUS="$2"; shift 2 ;;
        --stage)            STAGE="$2"; shift 2 ;;
        --vla-path)         VLA_PATH="$2"; shift 2 ;;
        --data-root-dir)    DATA_ROOT_DIR="$2"; shift 2 ;;
        --dataset-name)     DATASET_NAME="$2"; shift 2 ;;
        --output-dir)       OUTPUT_DIR="$2"; shift 2 ;;
        --batch-size)       BATCH_SIZE="$2"; shift 2 ;;
        --grad-accum-steps) GRAD_ACCUM="$2"; shift 2 ;;
        --lr)               LR="$2"; shift 2 ;;
        --max-steps)        MAX_STEPS="$2"; shift 2 ;;
        --save-steps)       SAVE_STEPS="$2"; shift 2 ;;
        --hidden-layer)     HIDDEN_LAYER="$2"; shift 2 ;;
        --feedback-mode)    FEEDBACK_MODE="$2"; shift 2 ;;
        --lora-rank)        LORA_RANK="$2"; shift 2 ;;
        --no-image-aug)     IMAGE_AUG=false; shift ;;
        --image-aug)        IMAGE_AUG=true; shift ;;
        *)                  REMAINING_ARGS+=("$1"); shift ;;
    esac
done
set -- "${REMAINING_ARGS[@]}"

if [ "$FEEDBACK_MODE" != "gated" ]; then
    echo "WARNING: FEEDBACK_MODE='$FEEDBACK_MODE' but this script only patches the gated path."
    echo "         Non-gated modes will behave identically to the original script — use that instead."
fi

export HF_HOME="/mimer/NOBACKUP/groups/robot_unforseen/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export TOKENIZERS_PARALLELISM=false
export RUN_NAME="reason-vla-gfix-${SLURM_JOB_ID:-local}"

export PYTHONPATH="$WORK_DIR/openvla_repo:$WORK_DIR:$PYTHONPATH"

module purge
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
source "$VENV_DIR/bin/activate"

echo "============================================"
echo "Job ID:        ${SLURM_JOB_ID:-local}"
echo "Node:          ${SLURM_NODELIST:-$(hostname)}"
echo "GPU:           $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Variant:       gated-fixed (identity gate init + bounded hint)"
echo "Stage:         ${STAGE:-both}"
echo "VLA:           $VLA_PATH"
echo "Dataset:       $DATASET_NAME"
echo "Data root:     $DATA_ROOT_DIR"
echo "Output:        $OUTPUT_DIR"
echo "Batch size:    $BATCH_SIZE"
echo "Grad accum:    $GRAD_ACCUM"
echo "Effective bs:  $((BATCH_SIZE * GRAD_ACCUM))"
echo "LR:            $LR"
echo "Max steps:     $MAX_STEPS"
echo "Save steps:    $SAVE_STEPS"
echo "Hidden layer:  $HIDDEN_LAYER"
echo "Feedback mode: $FEEDBACK_MODE"
echo "LoRA rank:     $LORA_RANK"
echo "Image aug:     $IMAGE_AUG"
if [ ${#REMAINING_ARGS[@]} -gt 0 ]; then
    echo "Extra args:    ${REMAINING_ARGS[*]}"
fi
echo "============================================"

mkdir -p logs

TRAIN_ARGS=(
    --vla-path "$VLA_PATH"
    --data-root-dir "$DATA_ROOT_DIR"
    --dataset-name "$DATASET_NAME"
    --output-dir "$OUTPUT_DIR"
    --batch-size "$BATCH_SIZE"
    --lr "$LR"
    --max-steps "$MAX_STEPS"
    --save-steps "$SAVE_STEPS"
    --lora-rank "$LORA_RANK"
    --grad-accum-steps "$GRAD_ACCUM"
    --hidden-layer "$HIDDEN_LAYER"
    --feedback-mode "$FEEDBACK_MODE"
)
if [ "$IMAGE_AUG" = "true" ]; then
    TRAIN_ARGS+=(--image-aug)
fi
if [ -n "$STAGE" ]; then
    TRAIN_ARGS+=(--training-stage "$STAGE")
fi
TRAIN_ARGS+=("${REMAINING_ARGS[@]}")

if [ "$NUM_GPUS" -gt 0 ]; then
    torchrun --standalone --nproc_per_node="$NUM_GPUS" \
        "$WORK_DIR/reason_vla_gated_fixed_train.py" "${TRAIN_ARGS[@]}"
else
    python "$WORK_DIR/reason_vla_gated_fixed_train.py" "${TRAIN_ARGS[@]}"
fi

echo "Job finished at $(date)"
