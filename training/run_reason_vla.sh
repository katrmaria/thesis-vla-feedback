#!/bin/bash
#SBATCH -A NAISS2025-22-1583 -p alvis
#SBATCH -J reason-vla                  # Job name
#SBATCH -t 3-00:00:00                  # Walltime: 3 days
#SBATCH -N 1                           # 1 node
#SBATCH --gpus-per-node=A40:1          # 1x A40 (48GB)
#SBATCH -o logs/reason_vla_%j.out      # stdout
#SBATCH -e logs/reason_vla_%j.err      # stderr

# ============================================================================
# ReasonVLA Training
#
# Dataset: libero_spatial_no_noops
#   - 432 episodes, 52,970 samples, ~123 steps/episode
#
# Hyperparameters:
#   - virtual_bs = batch_size × grad_accum × n_gpus = 1 × 32 × 1 = 32
#   - 1 epoch = 52,970 / 32 = 1,656 steps
#   - 33,000 steps = ~20 epochs
#   - batch_size=1 required for correct first-pass truncation
#
# Usage:
#   sbatch run_reason_vla.sh --stage 1               # stage 1 only
#   sbatch run_reason_vla.sh --stage 2               # stage 2
#   sbatch run_reason_vla.sh --stage 1 --max-steps 5000  # quick test
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
FEEDBACK_MODE="additive"
LORA_RANK=32
IMAGE_AUG=true

# Parse named flags. Unknown flags fall through to REMAINING_ARGS so reason_vla.py
# still receives them — but anything captured here is set ONCE and visible in the
# echo block below, so you never have to guess what argparse actually saw.
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

export HF_HOME="/mimer/NOBACKUP/groups/robot_unforseen/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export TOKENIZERS_PARALLELISM=false
export RUN_NAME="reason-vla-${SLURM_JOB_ID:-local}"

# Add openvla to Python path (for prismatic imports)
export PYTHONPATH="$WORK_DIR/openvla_repo:$WORK_DIR:$PYTHONPATH"

# Load modules
module purge
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

# Activate venv
source "$VENV_DIR/bin/activate"

echo "============================================"
echo "Job ID:        ${SLURM_JOB_ID:-local}"
echo "Node:          ${SLURM_NODELIST:-$(hostname)}"
echo "GPU:           $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
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

# Build args (each flag appears exactly once)
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

# Run
if [ "$NUM_GPUS" -gt 0 ]; then
    torchrun --standalone --nproc_per_node="$NUM_GPUS" \
        "$WORK_DIR/reason_vla.py" "${TRAIN_ARGS[@]}"
else
    python "$WORK_DIR/reason_vla.py" "${TRAIN_ARGS[@]}"
fi

echo "Job finished at $(date)"
