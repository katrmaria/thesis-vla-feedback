#!/bin/bash
# ===========================================================================
# LIBERO Simulation Evaluation on Alvis (with LIBERO-PRO support)
#
# Supports both original LIBERO and LIBERO-PRO perturbed evaluations.
# The perturbation type is controlled via evaluation_config.yaml.
#
# Usage:
#   # Original LIBERO eval (no perturbation)
#   sbatch run_libero_eval.sh libero_spatial openvla/openvla-7b-finetuned-libero-spatial 50 7
#
#   # LIBERO-PRO perturbation eval (pass config path as arg 6)
#   sbatch run_libero_eval.sh libero_spatial openvla/openvla-7b-finetuned-libero-spatial 50 7 pro /path/to/eval_config_swap.yaml
#
#   # All 3 remaining perturbations
#   sbatch run_libero_eval.sh libero_spatial openvla/openvla-7b-finetuned-libero-spatial 50 7 pro /cephyr/users/mariakat/Alvis/openvla/eval_configs/eval_config_swap.yaml
#   sbatch run_libero_eval.sh libero_spatial openvla/openvla-7b-finetuned-libero-spatial 50 7 pro /cephyr/users/mariakat/Alvis/openvla/eval_configs/eval_config_object.yaml
#   sbatch run_libero_eval.sh libero_spatial openvla/openvla-7b-finetuned-libero-spatial 50 7 pro /cephyr/users/mariakat/Alvis/openvla/eval_configs/eval_config_language.yaml
# ===========================================================================

#SBATCH --job-name=openvla-eval
#SBATCH --account=NAISS2025-22-1583
#SBATCH --time=12:00:00
#SBATCH --partition=alvis
#SBATCH --gpus-per-node=A40:1
#SBATCH --output=/cephyr/users/mariakat/Alvis/openvla/logs/%x_%j.out
#SBATCH --error=/cephyr/users/mariakat/Alvis/openvla/logs/%x_%j.err

# ---- Arguments ----
TASK_SUITE="${1:-libero_spatial}"
CHECKPOINT="${2:-openvla/openvla-7b-finetuned-libero-spatial}"
NUM_TRIALS="${3:-50}"
SEED="${4:-7}"
USE_PRO="${5:-}"  # Pass "pro" to enable LIBERO-PRO perturbation eval
EVAL_CONFIG="${6:-}"  # Path to evaluation config YAML (for PRO perturbations)

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
    VANILLA_LIBERO="$ALVIS_HOME/LIBERO/libero/libero"
    export LIBERO_CONFIG_PATH="$ALVIS_HOME/LIBERO/.libero_config"
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
source /mimer/NOBACKUP/groups/robot_unforseen/mariakat/venvs/venv_libero_eval/bin/activate

# ---- PYTHONPATH and script selection ----
if [ "$USE_PRO" = "pro" ]; then
    export PYTHONPATH="$OPENVLA_REPO:$WORK_DIR:$LIBERO_PRO_DIR:$PYTHONPATH"
    EVAL_SCRIPT="experiments/robot/libero/run_libero_eval_patched.py"
    EVAL_CONFIG_FILE="${EVAL_CONFIG:-$LIBERO_PRO_DIR/evaluation_config.yaml}"
    EVAL_CONFIG_ARG="--evaluation_config_path $EVAL_CONFIG_FILE"
else
    export PYTHONPATH="$OPENVLA_REPO:$WORK_DIR:$ALVIS_HOME/LIBERO:$PYTHONPATH"
    EVAL_SCRIPT="experiments/robot/libero/run_libero_eval.py"
    EVAL_CONFIG_ARG=""
fi

# ---- Determine output directory based on suite + perturbation ----
if [ "$USE_PRO" = "pro" ]; then
    # Detect which perturbation is active from the config
    PERTURB_SUFFIX=$(grep -E '^use_[a-z]+: *true' "$EVAL_CONFIG_FILE" | head -1 | sed 's/^use_//;s/:.*$//')
    case "$PERTURB_SUFFIX" in
        swap) SUFFIX="swap" ;;
        object) SUFFIX="object" ;;
        language) SUFFIX="lan" ;;
        task) SUFFIX="task" ;;
        environment) SUFFIX="env" ;;
        *) SUFFIX="pro" ;;
    esac
    MODEL_TAG=$(basename "$CHECKPOINT")
    LOCAL_LOG_DIR="$WORK_DIR/rollouts/${MODEL_TAG}_${TASK_SUITE}_pro_${SUFFIX}/seed${SEED}"
else
    MODEL_TAG=$(basename "$CHECKPOINT")
    LOCAL_LOG_DIR="$WORK_DIR/rollouts/${MODEL_TAG}_${TASK_SUITE}/seed${SEED}"
fi

# ---- Create log/output directories (outside repo) ----
mkdir -p "$WORK_DIR/logs"
mkdir -p "$LOCAL_LOG_DIR"

# ---- Run ----
cd "$OPENVLA_REPO"

echo "============================================"
echo "LIBERO Evaluation — Job $SLURM_JOB_ID"
echo "Task suite:  $TASK_SUITE"
echo "Checkpoint:  $CHECKPOINT"
echo "Trials/task: $NUM_TRIALS"
echo "Seed:        $SEED"
echo "LIBERO-PRO:  ${USE_PRO:-no}"
echo "Log dir:     $LOCAL_LOG_DIR"
echo "GPU:         $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Python:      $(which python)"
echo "Start time:  $(date)"
echo "============================================"

export LIBERO_ROLLOUT_DIR="$LOCAL_LOG_DIR"
python "$EVAL_SCRIPT" \
    --model_family openvla \
    --pretrained_checkpoint "$CHECKPOINT" \
    --task_suite_name "$TASK_SUITE" \
    --center_crop True \
    --num_trials_per_task "$NUM_TRIALS" \
    --seed "$SEED" \
    --local_log_dir "$LOCAL_LOG_DIR" \
    $EVAL_CONFIG_ARG
EXIT_CODE=$?

echo ""
echo "============================================"
echo "Finished: $(date)"
echo "Exit code: $EXIT_CODE"
echo "============================================"
exit $EXIT_CODE
