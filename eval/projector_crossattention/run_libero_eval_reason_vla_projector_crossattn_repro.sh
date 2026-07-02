#!/bin/bash
# ===========================================================================
# LIBERO Simulation Evaluation for ReasonVLAProjectorCrossAttn on Alvis
# REPRO version: uses venv_libero_repro (correct MuJoCo) and LIBERO_repro
#
# Usage:
#   Args: <task_suite> <checkpoint.pth> <stage> [num_trials] [seed] [base_model] [pro] [lora_dir] [hidden_layer]
#
#   # Stage 1 vanilla eval (50 trials, hidden layer 7 — training default)
#   sbatch run_libero_eval_reason_vla_projector_crossattn_repro.sh libero_spatial /path/to/checkpoint-4950.pth 1 50 7 openvla/openvla-7b-finetuned-libero-spatial "" "" 7
#
#   # LIBERO-PRO eval (pick one of use_* in eval_configs/eval_config_<swap|object|language|task>.yaml)
#   EVAL_CONFIG=/cephyr/users/mariakat/Alvis/openvla/eval_configs/eval_config_swap.yaml \
#     sbatch run_libero_eval_reason_vla_projector_crossattn_repro.sh libero_spatial /path/to/checkpoint-4950.pth 1 50 7 \
#       openvla/openvla-7b-finetuned-libero-spatial pro "" 7
#
#   # Stage 2 eval (needs lora_dir as 8th arg)
#   sbatch run_libero_eval_reason_vla_projector_crossattn_repro.sh libero_spatial /path/to/checkpoint-4950.pth 2 50 7 \
#     openvla/openvla-7b-finetuned-libero-spatial "" /path/to/lora_dir 7
# ===========================================================================

#SBATCH --job-name=projcrossattn-eval
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
HIDDEN_LAYER="${9:-7}"         # LLM hidden layer for text-token extraction (training default: 7)

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

# ---- LIBERO config ----
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

# ---- Load modules + venv (REPRO environment) ----
ml purge
ml load Python/3.10.8-GCCcore-12.2.0
ml load CUDA/12.1.1
export CUDA_HOME=$EBROOTCUDA
source /mimer/NOBACKUP/groups/robot_unforseen/mariakat/venvs/venv_libero_repro/bin/activate

# ---- PYTHONPATH and script selection ----
EVAL_CONFIG_ARG=""
if [ "$USE_PRO" = "pro" ]; then
    export PYTHONPATH="$OPENVLA_REPO:$WORK_DIR:$LIBERO_PRO_DIR:$PYTHONPATH"
    EVAL_SCRIPT="$WORK_DIR/run_libero_eval_reason_vla_projector_crossattn_patched.py"
    EVAL_CONFIG_FILE="${EVAL_CONFIG:-$LIBERO_PRO_DIR/evaluation_config.yaml}"
    EVAL_CONFIG_ARG="--evaluation_config_path $EVAL_CONFIG_FILE"
else
    export PYTHONPATH="$OPENVLA_REPO:$WORK_DIR:/mimer/NOBACKUP/groups/robot_unforseen/mariakat/LIBERO_repro:$PYTHONPATH"
    EVAL_SCRIPT="$WORK_DIR/run_libero_eval_reason_vla_projector_crossattn.py"
fi

# ---- Validate ----
if [ -z "$CHECKPOINT" ]; then
    echo "ERROR: checkpoint path is required"
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

# ---- Extract run/job ID from checkpoint path (folder prefix: rvla-projcrossattn-<jobid>) ----
RUN_ID=$(echo "$CHECKPOINT" | grep -oP 'rvla-projcrossattn-\K[0-9]+' || basename "$(dirname "$CHECKPOINT")")

# ---- Extract checkpoint name ----
CKPT_BASE=$(basename "$CHECKPOINT" .pth)

# ---- Output directory ----
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
    LOCAL_LOG_DIR="$WORK_DIR/rollouts/projcrossattn_${RUN_ID}_${CKPT_BASE}_${TASK_SUITE}_stage${STAGE}_pro_${SUFFIX}/seed${SEED}"
else
    LOCAL_LOG_DIR="$WORK_DIR/rollouts/projcrossattn_${RUN_ID}_${CKPT_BASE}_${TASK_SUITE}_stage${STAGE}/seed${SEED}"
fi

# ---- Create log/output directories ----
mkdir -p "$WORK_DIR/logs"
mkdir -p "$LOCAL_LOG_DIR"

# ---- Run ----
cd "$OPENVLA_REPO"

echo "============================================"
echo "LIBERO ReasonVLAProjectorCrossAttn Eval — Job $SLURM_JOB_ID"
echo "Task suite:    $TASK_SUITE"
echo "Checkpoint:    $CHECKPOINT"
echo "Stage:         $STAGE"
echo "LoRA dir:      ${LORA_DIR:-none}"
echo "Base model:    $BASE_MODEL"
echo "Hidden layer:  $HIDDEN_LAYER"
echo "Trials/task:   $NUM_TRIALS"
echo "Seed:          $SEED"
echo "LIBERO-PRO:    ${USE_PRO:-no}"
echo "Eval script:   $EVAL_SCRIPT"
echo "Log dir:       $LOCAL_LOG_DIR"
echo "GPU:           $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Python:        $(which python)"
echo "Start time:    $(date)"
echo "============================================"

LORA_ARGS=""
if [ -n "$LORA_DIR" ]; then
    LORA_ARGS="--lora_dir $LORA_DIR"
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
    $LORA_ARGS
EXIT_CODE=$?

echo ""
echo "============================================"
echo "Finished: $(date)"
echo "Exit code: $EXIT_CODE"
echo "============================================"
exit $EXIT_CODE
