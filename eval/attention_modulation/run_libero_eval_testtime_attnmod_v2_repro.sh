#!/bin/bash
# ===========================================================================
# LIBERO eval for test-time attention modulation V2 (NO TRAINING).
# REPRO version: uses venv_libero_repro (correct MuJoCo) and LIBERO_repro.
#
# Methods: song, logit_bias, attn_gate
#
# Usage:
#   Args: <task_suite> <method> [method_args...] [num_trials] [seed] [base_model]
#
#   # Song et al. faithful (suppress bottom 20% at L28)
#   sbatch run_libero_eval_testtime_attnmod_v2_repro.sh libero_spatial song 0.2 0.1 50 7
#
#   # Logit bias at L10-15, beta=1.0
#   sbatch run_libero_eval_testtime_attnmod_v2_repro.sh libero_spatial logit_bias 1.0 50 7
#
#   # Attn gate at L10-15
#   sbatch run_libero_eval_testtime_attnmod_v2_repro.sh libero_spatial attn_gate 50 7
# ===========================================================================

#SBATCH --job-name=ttamv2-repro
#SBATCH --account=NAISS2025-22-1583
#SBATCH --time=12:00:00
#SBATCH --partition=alvis
#SBATCH --gpus-per-node=A100:1
#SBATCH --output=/cephyr/users/mariakat/Alvis/openvla/logs/%x_%j.out
#SBATCH --error=/cephyr/users/mariakat/Alvis/openvla/logs/%x_%j.err

# ---- Arguments ----
TASK_SUITE="${1:-libero_spatial}"
METHOD="${2:-song}"

# Method-specific args parsed based on METHOD
if [ "$METHOD" = "song" ]; then
    RHO="${3:-0.2}"
    LAM="${4:-0.1}"
    PRE_LAYER="${5:-15}"
    POST_LAYER="${6:-27}"
    REVIEW_LAYER="${7:-28}"
    NUM_TRIALS="${8:-50}"
    SEED="${9:-7}"
    BASE_MODEL="${10:-openvla/openvla-7b-finetuned-libero-spatial}"
    METHOD_ARGS="--rho $RHO --lam $LAM --pre_layer $PRE_LAYER --post_layer $POST_LAYER --review_layer $REVIEW_LAYER"
    EXPERIMENT="song_rho${RHO}_lam${LAM}_pre${PRE_LAYER}_post${POST_LAYER}_rev${REVIEW_LAYER}"
elif [ "$METHOD" = "logit_bias" ]; then
    BETA="${3:-1.0}"
    NUM_TRIALS="${4:-50}"
    SEED="${5:-7}"
    BASE_MODEL="${6:-openvla/openvla-7b-finetuned-libero-spatial}"
    METHOD_ARGS="--beta $BETA"
    EXPERIMENT="logitbias_beta${BETA}"
elif [ "$METHOD" = "attn_gate" ]; then
    NUM_TRIALS="${3:-50}"
    SEED="${4:-7}"
    BASE_MODEL="${5:-openvla/openvla-7b-finetuned-libero-spatial}"
    METHOD_ARGS=""
    EXPERIMENT="attngate"
elif [ "$METHOD" = "vit_bias" ]; then
    BETA="${3:-1.0}"
    NUM_TRIALS="${4:-50}"
    SEED="${5:-7}"
    BASE_MODEL="${6:-openvla/openvla-7b-finetuned-libero-spatial}"
    METHOD_ARGS="--beta $BETA"
    EXPERIMENT="vitbias_beta${BETA}"
else
    echo "ERROR: Unknown method '$METHOD'. Use: song, logit_bias, attn_gate, vit_bias"
    exit 1
fi

# ---- Paths ----
ALVIS_HOME=/cephyr/users/mariakat/Alvis
WORK_DIR=$ALVIS_HOME/openvla
OPENVLA_REPO=$WORK_DIR/openvla_repo

# ---- Rendering ----
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export NUMBA_CACHE_DIR=/tmp/numba_cache
export HF_HOME="/mimer/NOBACKUP/groups/robot_unforseen/.cache/huggingface"
export ALVIS_HOME

# ---- LIBERO config (repro) ----
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
fi

# ---- Modules + venv (REPRO environment) ----
ml purge
ml load Python/3.10.8-GCCcore-12.2.0
ml load CUDA/12.1.1
export CUDA_HOME=$EBROOTCUDA
source /mimer/NOBACKUP/groups/robot_unforseen/mariakat/venvs/venv_libero_repro/bin/activate

# ---- PYTHONPATH ----
export PYTHONPATH="$OPENVLA_REPO:$WORK_DIR:/mimer/NOBACKUP/groups/robot_unforseen/mariakat/LIBERO_repro:$PYTHONPATH"

# ---- Output dir ----
LOCAL_LOG_DIR="$WORK_DIR/rollouts/repro_v2_${EXPERIMENT}_${TASK_SUITE}/seed${SEED}"
mkdir -p "$WORK_DIR/logs"
mkdir -p "$LOCAL_LOG_DIR"

cd "$OPENVLA_REPO"

echo "============================================"
echo "LIBERO Test-Time Attention Modulation V2 (REPRO) — Job $SLURM_JOB_ID"
echo "Task suite:    $TASK_SUITE"
echo "Method:        $METHOD"
echo "Base model:    $BASE_MODEL"
echo "Method args:   $METHOD_ARGS"
echo "Trials/task:   $NUM_TRIALS"
echo "Seed:          $SEED"
echo "Log dir:       $LOCAL_LOG_DIR"
echo "Venv:          venv_libero_repro"
echo "GPU:           $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Start time:    $(date)"
echo "============================================"

python "$WORK_DIR/run_libero_eval_testtime_attnmod_v2.py" \
    --base_model "$BASE_MODEL" \
    --task_suite_name "$TASK_SUITE" \
    --center_crop True \
    --num_trials_per_task "$NUM_TRIALS" \
    --seed "$SEED" \
    --local_log_dir "$LOCAL_LOG_DIR" \
    --method "$METHOD" \
    $METHOD_ARGS
EXIT_CODE=$?

echo ""
echo "============================================"
echo "Finished: $(date)"
echo "Exit code: $EXIT_CODE"
echo "============================================"
exit $EXIT_CODE
