#!/bin/bash
#SBATCH -A NAISS2025-22-1583 -p alvis
#SBATCH -J rvla-projcrossattn          # Job name
#SBATCH -t 1-00:00:00                  # Walltime: 1 day
#SBATCH -N 1                           # 1 node
#SBATCH --gpus-per-node=A100:1         # 1x A100 (40GB)
#SBATCH -o logs/reason_vla_projcrossattn_%j.out
#SBATCH -e logs/reason_vla_projcrossattn_%j.err

# ============================================================================
# ReasonVLA Projector Cross-Attention Training
#
# Cross-attention at the projector output.
# Projected patches (Q) attend to instruction token hidden states (K, V)
# extracted from layer `hidden_layer` of a full multimodal forward.
#
# Dataset: libero_spatial_no_noops
#   - 432 episodes, 52,970 samples
#   - virtual_bs = 1 × 32 × 1 = 32
#   - 1 epoch = 52,970 / 32 = 1,656 steps
#
# Usage:
#   sbatch run_reason_vla_projector_crossattn.sh --stage 1
#   sbatch run_reason_vla_projector_crossattn.sh --stage 1 --hidden-layer 15   # ablation
#   sbatch run_reason_vla_projector_crossattn.sh --stage 1 --max-steps 10      # debug
# ============================================================================

# Parse --stage from args
NUM_GPUS=0
STAGE=""
REMAINING_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --num-gpus) NUM_GPUS="$2"; shift 2 ;;
        --stage) STAGE="$2"; shift 2 ;;
        *) REMAINING_ARGS+=("$1"); shift ;;
    esac
done
set -- "${REMAINING_ARGS[@]}"

# Paths
PROJECT_DIR="/mimer/NOBACKUP/groups/robot_unforseen/mariakat"
VENV_DIR="$PROJECT_DIR/venvs/openvla"
WORK_DIR="/cephyr/users/mariakat/Alvis/openvla"
DATA_ROOT_DIR="$PROJECT_DIR/data/modified_libero_rlds"
DATASET_NAME="libero_spatial_no_noops"
OUTPUT_DIR="$PROJECT_DIR/runs/reason_vla_projector_crossattn"
VLA_PATH="openvla/openvla-7b-finetuned-libero-spatial"

export HF_HOME="/mimer/NOBACKUP/groups/robot_unforseen/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export TOKENIZERS_PARALLELISM=false
export RUN_NAME="rvla-projcrossattn-${SLURM_JOB_ID:-local}"

export PYTHONPATH="$WORK_DIR/openvla_repo:$WORK_DIR:$PYTHONPATH"

module purge
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
source "$VENV_DIR/bin/activate"

BATCH_SIZE=1
GRAD_ACCUM=32
LR="1e-4"
MAX_STEPS=8280      # 5 epochs (1 epoch = 1,656 steps at effective bs=32)
SAVE_STEPS=1650     # Checkpoint every epoch: 1650, 3300, 4950, 6600, 8250
HIDDEN_LAYER=7      # End of instruction encoding phase (pure linguistic K/V)

echo "============================================"
echo "Job ID:        ${SLURM_JOB_ID:-local}"
echo "Node:          ${SLURM_NODELIST:-$(hostname)}"
echo "GPU:           $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Stage:         ${STAGE:-both}"
echo "Script:        reason_vla_projector_crossattn.py"
echo "VLA:           $VLA_PATH"
echo "Dataset:       $DATASET_NAME"
echo "Batch size:    $BATCH_SIZE"
echo "Grad accum:    $GRAD_ACCUM"
echo "Effective bs:  $((BATCH_SIZE * GRAD_ACCUM))"
echo "LR:            $LR"
echo "Max steps:     $MAX_STEPS"
echo "Hidden layer:  $HIDDEN_LAYER"
echo "Output:        $OUTPUT_DIR"
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
    --image-aug
    --lora-rank 16
    --grad-accum-steps "$GRAD_ACCUM"
    --hidden-layer "$HIDDEN_LAYER"
    "$@"
)

if [ -n "$STAGE" ]; then
    TRAIN_ARGS+=(--training-stage "$STAGE")
fi

if [ "$NUM_GPUS" -gt 0 ]; then
    torchrun --standalone --nproc_per_node="$NUM_GPUS" \
        "$WORK_DIR/reason_vla_projector_crossattn.py" "${TRAIN_ARGS[@]}"
else
    python "$WORK_DIR/reason_vla_projector_crossattn.py" "${TRAIN_ARGS[@]}"
fi

echo "Job finished at $(date)"
