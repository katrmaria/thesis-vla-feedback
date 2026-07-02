#!/bin/bash
# ===========================================================================
# LIBERO Simulation Evaluation for ReasonVLA (reason_vla.py) on Alvis
# Supports both original LIBERO and LIBERO-PRO perturbed evaluations.
#
# Usage:
#   Args: <task_suite> <checkpoint.pth> <stage> [num_trials] [seed] [base_model] [pro] [lora_dir] [hidden_layer]
#
#   # Stage 1 eval (5 trials, quick test)
#   sbatch run_libero_eval_reason_vla.sh libero_spatial /path/to/final_checkpoint_stage1.pth 1 5 7
#
#   # Stage 1 eval (50 trials)
#   sbatch run_libero_eval_reason_vla.sh libero_spatial /path/to/final_checkpoint_stage1.pth 1 50 7
#
#   # LIBERO-PRO eval (edit evaluation_config.yaml first, e.g. set use_swap: true)
#   sbatch run_libero_eval_reason_vla.sh libero_spatial /path/to/final_checkpoint_stage1.pth 1 50 7 openvla/openvla-7b-finetuned-libero-spatial pro
#
#   # Stage 2 eval (needs lora_dir as 8th arg)
#   sbatch run_libero_eval_reason_vla.sh libero_spatial /path/to/final_checkpoint_stage2.pth 2 50 7 openvla/openvla-7b-finetuned-libero-spatial "" /path/to/lora_dir
#
#   # Stage 1 with hidden layer 24
#   sbatch run_libero_eval_reason_vla.sh libero_spatial /path/to/final_checkpoint_stage1.pth 1 50 7 openvla/openvla-7b-finetuned-libero-spatial "" "" 24
#
#   # 3 seeds
#   for SEED in 7 42 123; do
#     sbatch run_libero_eval_reason_vla.sh libero_spatial /path/to/final_checkpoint_stage1.pth 1 50 $SEED
#   done
# ===========================================================================

#SBATCH --job-name=reasonvla-eval
#SBATCH --account=NAISS2025-22-1583
#SBATCH --time=12:00:00
#SBATCH --partition=alvis
#SBATCH --gpus-per-node=A100:1
#SBATCH --output=/cephyr/users/mariakat/Alvis/openvla/logs/%x_%j.out
#SBATCH --error=/cephyr/users/mariakat/Alvis/openvla/logs/%x_%j.err

# ---- Arguments ----
TASK_SUITE="${1:-libero_spatial}"
CHECKPOINT="${2}"              # Path to .pth file
STAGE="${3}"                   # 1 or 2
NUM_TRIALS="${4:-50}"
SEED="${5:-7}"
BASE_MODEL="${6:-openvla/openvla-7b-finetuned-libero-spatial}"
USE_PRO="${7:-}"               # Pass "pro" to enable LIBERO-PRO perturbation eval
LORA_DIR="${8:-}"              # Directory with LoRA adapter files (required for stage 2)
HIDDEN_LAYER="${9:--1}"        # LLM hidden layer for reasoning (-1=last, 24=action decision layer)
MERGE_PATH="${10:-}"           # Path to second .pth checkpoint to merge with (optional)
MERGE_LORA_DIR="${11:-}"       # Path to second LoRA adapter dir to merge with (optional, stage 2)
FEEDBACK_MODE="${12:-additive}" # Feedback mode: additive, film, gated, adaln

# ---- Paths ----
ALVIS_HOME=/cephyr/users/mariakat/Alvis
WORK_DIR=$ALVIS_HOME/openvla
OPENVLA_REPO=$WORK_DIR/openvla_repo
LIBERO_PRO_DIR=$OPENVLA_REPO/experiments/robot/libero/LIBERO-PRO

# ---- Rendering backend for headless GPU simulation ----
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export NUMBA_CACHE_DIR=/tmp/numba_cache
export HF_HOME="/mimer/NOBACKUP/groups/robot_unforseen/.cache/huggingface"
export ALVIS_HOME

# ---- LIBERO config (prevents interactive prompt in batch jobs) ----
# Use LIBERO-PRO BDDL files only for perturbation eval;
# vanilla eval uses original LIBERO (single-object scenes).
if [ "$USE_PRO" = "pro" ]; then
    export LIBERO_CONFIG_PATH="$LIBERO_PRO_DIR/.libero_config"
    LIBERO_CONFIG_FILE="$LIBERO_CONFIG_PATH/config.yaml"
    if [ ! -f "$LIBERO_CONFIG_FILE" ]; then
        mkdir -p "$LIBERO_CONFIG_PATH"
        LIBERO_DATA="$LIBERO_PRO_DIR/libero/libero"
        cat > "$LIBERO_CONFIG_FILE" <<YAMLEOF
benchmark_root: $LIBERO_DATA
bddl_files: $LIBERO_DATA/bddl_files
init_states: $LIBERO_DATA/init_files
datasets: $LIBERO_DATA/../datasets
assets: $LIBERO_DATA/assets
YAMLEOF
        echo "Created LIBERO-PRO config at $LIBERO_CONFIG_FILE"
    fi
else
    VANILLA_LIBERO="/mimer/NOBACKUP/groups/robot_unforseen/mariakat/LIBERO_repro/libero/libero"
    export LIBERO_CONFIG_PATH="/mimer/NOBACKUP/groups/robot_unforseen/mariakat/LIBERO_repro/.libero_config"
    LIBERO_CONFIG_FILE="$LIBERO_CONFIG_PATH/config.yaml"
    if [ ! -f "$LIBERO_CONFIG_FILE" ]; then
        mkdir -p "$LIBERO_CONFIG_PATH"
        cat > "$LIBERO_CONFIG_FILE" <<YAMLEOF
benchmark_root: $VANILLA_LIBERO
bddl_files: $VANILLA_LIBERO/bddl_files
init_states: $VANILLA_LIBERO/init_files
datasets: $VANILLA_LIBERO/../datasets
assets: $VANILLA_LIBERO/assets
YAMLEOF
        echo "Created vanilla LIBERO config at $LIBERO_CONFIG_FILE"
    fi
fi

# ---- Load modules + venv ----
ml purge
ml load Python/3.10.8-GCCcore-12.2.0
ml load CUDA/12.1.1
export CUDA_HOME=$EBROOTCUDA
source /mimer/NOBACKUP/groups/robot_unforseen/mariakat/venvs/venv_libero_repro/bin/activate

# ---- PYTHONPATH and script selection ----
EVAL_CONFIG_ARG=""
if [ "$USE_PRO" = "pro" ]; then
    export PYTHONPATH="$OPENVLA_REPO:$WORK_DIR:$LIBERO_PRO_DIR:$PYTHONPATH"
    EVAL_SCRIPT="$WORK_DIR/run_libero_eval_reason_vla_patched.py"
    EVAL_CONFIG_FILE="${EVAL_CONFIG:-$LIBERO_PRO_DIR/evaluation_config.yaml}"
    EVAL_CONFIG_ARG="--evaluation_config_path $EVAL_CONFIG_FILE"
else
    export PYTHONPATH="$OPENVLA_REPO:$WORK_DIR:/mimer/NOBACKUP/groups/robot_unforseen/mariakat/LIBERO_repro:$PYTHONPATH"
    EVAL_SCRIPT="$WORK_DIR/run_libero_eval_reason_vla.py"
fi

# ---- Validate ----
if [ -z "$CHECKPOINT" ]; then
    echo "ERROR: checkpoint path is required"
    echo "Usage: sbatch run_libero_eval_reason_vla.sh <task_suite> <checkpoint.pth> <stage> [num_trials] [seed] [base_model] [pro] [lora_dir] [hidden_layer]"
    exit 1
fi
if [ -z "$STAGE" ] || { [ "$STAGE" != "1" ] && [ "$STAGE" != "2" ]; }; then
    echo "ERROR: Stage must be 1 or 2"
    exit 1
fi
if [ "$STAGE" = "2" ] && [ -z "$LORA_DIR" ]; then
    echo "ERROR: --lora_dir (arg 8) is required for stage 2"
    exit 1
fi

# ---- Extract run/job ID from checkpoint path (e.g. reason-vla-6112746) ----
RUN_ID=$(echo "$CHECKPOINT" | grep -oP 'reason-vla-\K[0-9]+' || basename "$(dirname "$CHECKPOINT")")

# ---- Extract checkpoint name (e.g. checkpoint-2500 or final_checkpoint_stage1) ----
CKPT_BASE=$(basename "$CHECKPOINT" .pth)

# ---- Output directory ----
MERGE_TAG=""
if [ -n "$MERGE_PATH" ]; then
    MERGE_CKPT_BASE=$(basename "$MERGE_PATH" .pth)
    MERGE_TAG="_merged_${MERGE_CKPT_BASE}"
fi

if [ "$USE_PRO" = "pro" ]; then
    PERTURB_SUFFIX=$(grep -E '^use_[a-z]+: *true' "$EVAL_CONFIG_FILE" | head -1 | sed 's/^use_//;s/:.*$//')
    case "$PERTURB_SUFFIX" in
        swap) SUFFIX="swap" ;;
        object) SUFFIX="object" ;;
        language) SUFFIX="lan" ;;
        task) SUFFIX="task" ;;
        environment) SUFFIX="env" ;;
        *) SUFFIX="pro" ;;
    esac
    LOCAL_LOG_DIR="$WORK_DIR/rollouts/reasonvla_${RUN_ID}_${CKPT_BASE}${MERGE_TAG}_${TASK_SUITE}_stage${STAGE}_pro_${SUFFIX}/seed${SEED}"
else
    LOCAL_LOG_DIR="$WORK_DIR/rollouts/reasonvla_${RUN_ID}_${CKPT_BASE}${MERGE_TAG}_${TASK_SUITE}_stage${STAGE}/seed${SEED}"
fi

# ---- Create log/output directories ----
mkdir -p "$WORK_DIR/logs"
mkdir -p "$LOCAL_LOG_DIR"

# ---- Run ----
cd "$OPENVLA_REPO"

echo "============================================"
echo "LIBERO ReasonVLA Eval — Job $SLURM_JOB_ID"
echo "Task suite:  $TASK_SUITE"
echo "Checkpoint:  $CHECKPOINT"
echo "Stage:       $STAGE"
echo "LoRA dir:    ${LORA_DIR:-none}"
echo "Base model:  $BASE_MODEL"
echo "Hidden layer: $HIDDEN_LAYER"
echo "Merge path:  ${MERGE_PATH:-none}"
echo "Feedback:    $FEEDBACK_MODE"
echo "Trials/task: $NUM_TRIALS"
echo "Seed:        $SEED"
echo "LIBERO-PRO:  ${USE_PRO:-no}"
echo "Eval script: $EVAL_SCRIPT"
echo "Log dir:     $LOCAL_LOG_DIR"
echo "GPU:         $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Python:      $(which python)"
echo "Start time:  $(date)"
echo "============================================"

LORA_ARGS=""
if [ -n "$LORA_DIR" ]; then
    LORA_ARGS="--lora_dir $LORA_DIR"
fi

MERGE_ARGS=""
if [ -n "$MERGE_PATH" ]; then
    MERGE_ARGS="--merge_path $MERGE_PATH"
fi
if [ -n "$MERGE_LORA_DIR" ]; then
    MERGE_ARGS="$MERGE_ARGS --merge_lora_dir $MERGE_LORA_DIR"
fi

# Always pass --feedback_mode; only pass --merge_args for vanilla eval
EXTRA_ARGS="--feedback_mode $FEEDBACK_MODE"
if [ "$USE_PRO" != "pro" ]; then
    EXTRA_ARGS="$EXTRA_ARGS $MERGE_ARGS"
fi

python "$EVAL_SCRIPT" \
    --base_model "$BASE_MODEL" \
    --checkpoint_path "$CHECKPOINT" \
    --stage "$STAGE" \
    --task_suite_name "$TASK_SUITE" \
    --center_crop True \
    --num_trials_per_task "$NUM_TRIALS" \
    --seed "$SEED" \
    --local_log_dir "$LOCAL_LOG_DIR" \
    --hidden_layer "$HIDDEN_LAYER" \
    $EVAL_CONFIG_ARG \
    $LORA_ARGS \
    $EXTRA_ARGS
EXIT_CODE=$?

echo ""
echo "============================================"
echo "Finished: $(date)"
echo "Exit code: $EXIT_CODE"
echo "============================================"
exit $EXIT_CODE
