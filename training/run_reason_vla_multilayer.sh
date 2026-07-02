#!/bin/bash
#SBATCH -A NAISS2025-22-1583 -p alvis
#SBATCH -J reason-vla-multilayer        # Job name
#SBATCH -t 2-00:00:00                   # Walltime: 2 days
#SBATCH -N 1                            # 1 node
#SBATCH --gpus-per-node=A40:1           # 1x A40 (48GB)
#SBATCH -o logs/reason_vla_multilayer_%j.out  # stdout
#SBATCH -e logs/reason_vla_multilayer_%j.err  # stderr

# ============================================================================
# Variant: multi-layer hint injection (hints at ViT blocks 0, 8, 16)
#
# Submit jobs:
#   sbatch run_reason_vla_multilayer.sh                         # both stages
#   sbatch run_reason_vla_multilayer.sh --stage 1               # stage 1 only
#   sbatch run_reason_vla_multilayer.sh --stage 2               # stage 2 only
#   sbatch run_reason_vla_multilayer.sh --max-steps 300         # quick test
#   sbatch run_reason_vla_multilayer.sh --inject-layers 0 4 8 12 16  # custom layers
#
# Multi-GPU:
#   #SBATCH --gpus-per-node=A100:4
#   sbatch run_reason_vla_multilayer.sh --num-gpus 4
# ============================================================================

# Parse --num-gpus and --stage from args
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
OUTPUT_DIR="$PROJECT_DIR/runs/reason_vla_multilayer"
VLA_PATH="openvla/openvla-7b-finetuned-libero-spatial"

export HF_HOME="/mimer/NOBACKUP/groups/robot_unforseen/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export TOKENIZERS_PARALLELISM=false
export RUN_NAME="reason-vla-multilayer-${SLURM_JOB_ID:-local}"

# Add openvla to Python path (for prismatic imports)
export PYTHONPATH="$WORK_DIR/openvla_repo:$WORK_DIR:$PYTHONPATH"

# Load modules
module purge
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

# Activate venv
source "$VENV_DIR/bin/activate"

echo "============================================"
echo "Job ID:   ${SLURM_JOB_ID:-local}"
echo "Node:     ${SLURM_NODELIST:-$(hostname)}"
echo "GPU:      $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Stage:    ${STAGE:-both}"
echo "Variant:  Multi-layer hints (inject at ViT blocks 0, 8, 16)"
echo "VLA:      $VLA_PATH"
echo "Dataset:  $DATASET_NAME"
echo "Data dir: $DATA_ROOT_DIR"
echo "Output:   $OUTPUT_DIR"
if [ "$NUM_GPUS" -gt 0 ]; then
    echo "Multi-GPU: $NUM_GPUS GPUs (torchrun)"
fi
echo "============================================"

mkdir -p logs

# Build args
TRAIN_ARGS=(
    --vla-path "$VLA_PATH"
    --data-root-dir "$DATA_ROOT_DIR"
    --dataset-name "$DATASET_NAME"
    --output-dir "$OUTPUT_DIR"
    --batch-size 1
    --lr 2e-5
    --max-steps 200000
    --save-steps 5000
    --image-aug
    --lora-rank 32
    --grad-accum-steps 8
    "$@"
)

# Add stage if specified
if [ -n "$STAGE" ]; then
    TRAIN_ARGS+=(--training-stage "$STAGE")
fi

# Run
if [ "$NUM_GPUS" -gt 0 ]; then
    torchrun --standalone --nproc_per_node="$NUM_GPUS" \
        "$WORK_DIR/reason_vla_multilayer.py" "${TRAIN_ARGS[@]}"
else
    python "$WORK_DIR/reason_vla_multilayer.py" "${TRAIN_ARGS[@]}"
fi

echo "Job finished at $(date)"
