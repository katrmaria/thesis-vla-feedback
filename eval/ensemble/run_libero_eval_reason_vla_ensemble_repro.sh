#!/bin/bash
# ===========================================================================
# LIBERO Ensemble Evaluation for ReasonVLA (multiple models averaged)
# REPRO version: uses venv_libero_repro + LIBERO_repro / LIBERO-PRO
#
# Usage:
#   # Vanilla ensemble eval
#   sbatch run_libero_eval_reason_vla_ensemble_repro.sh \
#     libero_spatial 50 7 "" "" \
#     "ckpt1.pth,1,none,-1,additive" \
#     "ckpt2.pth,2,lora_dir,24,additive"
#
#   # Task-perturbation ensemble eval
#   EVAL_CONFIG=/cephyr/users/mariakat/Alvis/openvla/eval_configs/eval_config_task.yaml \
#     sbatch run_libero_eval_reason_vla_ensemble_repro.sh \
#       libero_spatial 50 7 pro "" \
#       "ckpt1.pth,1,none,-1,additive" \
#       "ckpt2.pth,2,lora_dir,24,additive"
# ===========================================================================

#SBATCH --job-name=rvla-ensemble-eval
#SBATCH --account=NAISS2025-22-1583
#SBATCH --time=14:00:00
#SBATCH --partition=alvis
#SBATCH --gpus-per-node=A100:1
#SBATCH --output=/cephyr/users/mariakat/Alvis/openvla/logs/%x_%j.out
#SBATCH --error=/cephyr/users/mariakat/Alvis/openvla/logs/%x_%j.err

TASK_SUITE="${1:-libero_spatial}"
NUM_TRIALS="${2:-50}"
SEED="${3:-7}"
USE_PRO="${4:-}"        # Pass "pro" to enable LIBERO-PRO perturbation eval
RESERVED="${5:-}"       # Reserved for future use; pass "" for now
shift 5

ALVIS_HOME=/cephyr/users/mariakat/Alvis
WORK_DIR=$ALVIS_HOME/openvla
OPENVLA_REPO=$WORK_DIR/openvla_repo
LIBERO_PRO_DIR=$OPENVLA_REPO/experiments/robot/libero/LIBERO-PRO

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export NUMBA_CACHE_DIR=/tmp/numba_cache
export HF_HOME="/mimer/NOBACKUP/groups/robot_unforseen/.cache/huggingface"
export ALVIS_HOME

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

ml purge
ml load Python/3.10.8-GCCcore-12.2.0
ml load CUDA/12.1.1
export CUDA_HOME=$EBROOTCUDA
source /mimer/NOBACKUP/groups/robot_unforseen/mariakat/venvs/venv_libero_repro/bin/activate

EVAL_CONFIG_ARG=""
if [ "$USE_PRO" = "pro" ]; then
    export PYTHONPATH="$OPENVLA_REPO:$WORK_DIR:$LIBERO_PRO_DIR:$PYTHONPATH"
    EVAL_SCRIPT="$WORK_DIR/run_libero_eval_reason_vla_ensemble_patched.py"
    EVAL_CONFIG_FILE="${EVAL_CONFIG:-$LIBERO_PRO_DIR/evaluation_config.yaml}"
    EVAL_CONFIG_ARG="--evaluation_config_path $EVAL_CONFIG_FILE"
else
    export PYTHONPATH="$OPENVLA_REPO:$WORK_DIR:/mimer/NOBACKUP/groups/robot_unforseen/mariakat/LIBERO_repro:$PYTHONPATH"
    EVAL_SCRIPT="$WORK_DIR/run_libero_eval_reason_vla_ensemble.py"
fi

# Output directory
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
    LOCAL_LOG_DIR="$WORK_DIR/rollouts/ensemble_${SLURM_JOB_ID}_${TASK_SUITE}_pro_${SUFFIX}/seed${SEED}"
else
    LOCAL_LOG_DIR="$WORK_DIR/rollouts/ensemble_${SLURM_JOB_ID}_${TASK_SUITE}/seed${SEED}"
fi

mkdir -p "$WORK_DIR/logs"
mkdir -p "$LOCAL_LOG_DIR"

cd "$OPENVLA_REPO"

echo "============================================"
echo "LIBERO ReasonVLA Ensemble Eval — Job $SLURM_JOB_ID"
echo "Task suite:  $TASK_SUITE"
echo "Trials/task: $NUM_TRIALS"
echo "Seed:        $SEED"
echo "LIBERO-PRO:  ${USE_PRO:-no}"
echo "Eval script: $EVAL_SCRIPT"
echo "Configs:     $@"
echo "Log dir:     $LOCAL_LOG_DIR"
echo "GPU:         $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Start time:  $(date)"
echo "============================================"

python "$EVAL_SCRIPT" \
    --base_model "${BASE_MODEL:-openvla/openvla-7b-finetuned-libero-spatial}" \
    --task_suite_name "$TASK_SUITE" \
    --center_crop True \
    --num_trials_per_task "$NUM_TRIALS" \
    --seed "$SEED" \
    --local_log_dir "$LOCAL_LOG_DIR" \
    $EVAL_CONFIG_ARG \
    --ensemble_configs "$@"

echo ""
echo "Finished: $(date)"
