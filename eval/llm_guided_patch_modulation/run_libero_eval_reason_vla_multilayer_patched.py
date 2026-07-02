"""
run_libero_eval_reason_vla_multilayer_patched.py

Runs the ReasonVLA Multi-Layer model (from reason_vla_multilayer.py) with LIBERO-PRO perturbations.

How it works:
  - Pass --task_suite_name with the base suite (e.g., libero_spatial)
  - Pass --evaluation_config_path pointing to LIBERO-PRO/evaluation_config.yaml
  - The config's use_swap/use_object/use_language/use_task/use_environment flags
    determine which perturbation to apply
  - The script appends the perturbation suffix automatically

Usage:
    python run_libero_eval_reason_vla_multilayer_patched.py \
        --base_model openvla/openvla-7b-finetuned-libero-spatial \
        --checkpoint_path /path/to/final_checkpoint_stage1.pth \
        --stage 1 \
        --task_suite_name libero_spatial \
        --center_crop True \
        --num_trials_per_task 50 \
        --seed 7 \
        --evaluation_config_path /path/to/evaluation_config.yaml
"""

import atexit
import os
import sys
import argparse
import hashlib
import traceback
import imageio
import numpy as np
import tensorflow as tf
import tqdm
import yaml
from PIL import Image
from libero.libero import benchmark

# ---- Path setup ----
ALVIS_HOME = os.environ.get("ALVIS_HOME", "/cephyr/users/mariakat/Alvis")
WORK_DIR = os.path.join(ALVIS_HOME, "openvla")
OPENVLA_REPO = os.path.join(WORK_DIR, "openvla_repo")

sys.path.insert(0, OPENVLA_REPO)
sys.path.insert(0, WORK_DIR)

# Import LIBERO-PRO perturbation module (required for this script)
LIBERO_PRO_DIR = os.path.join(OPENVLA_REPO, "experiments", "robot", "libero", "LIBERO-PRO")
sys.path.append(LIBERO_PRO_DIR)
import perturbation

from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    quat2axisangle,
)
from experiments.robot.openvla_utils import crop_and_resize
from experiments.robot.robot_utils import (
    DATE_TIME,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)

# ---- ReasonVLA Multi-Layer model ----
from reason_vla_multilayer import ReasonVLA

# ---- Verify import paths ----
import experiments.robot.libero.libero_utils as _lu
import libero.libero.benchmark as _bm
import reason_vla_multilayer as _rv
print("=" * 60)
print("IMPORT PATH CHECK")
print(f"  libero.libero.benchmark: {_bm.__file__}")
print(f"  libero_utils:            {_lu.__file__}")
print(f"  reason_vla_multilayer:   {_rv.__file__}")
print(f"  perturbation:            {perturbation.__file__}")
print("=" * 60)
del _lu, _bm, _rv


# ---- Constants ----
ACTION_DIM = 7

# =============================================================================
# LIBERO-PRO: Task max steps for all suite variants
# =============================================================================
TASK_MAX_STEPS = {
    # Original suites
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
    # Temporary (combined perturbations)
    "libero_goal_temp": 300,
    "libero_spatial_temp": 220,
    "libero_10_temp": 520,
    "libero_object_temp": 280,
    # Language perturbation
    "libero_goal_lan": 300,
    "libero_spatial_lan": 220,
    "libero_10_lan": 520,
    "libero_object_lan": 280,
    # Object perturbation
    "libero_goal_object": 300,
    "libero_spatial_object": 220,
    "libero_10_object": 520,
    "libero_object_object": 280,
    # Swap (position) perturbation
    "libero_goal_swap": 300,
    "libero_spatial_swap": 220,
    "libero_10_swap": 520,
    "libero_object_swap": 280,
    # Task perturbation
    "libero_goal_task": 300,
    "libero_spatial_task": 220,
    "libero_10_task": 520,
    "libero_object_task": 280,
    # Environment perturbation
    "libero_goal_env": 300,
    "libero_spatial_env": 220,
    "libero_10_env": 520,
    "libero_object_env": 280,
}

BASE_SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90")


def save_rollout_video(rollout_images, idx, success, task_description, log_file=None, output_dir="."):
    """Saves an MP4 replay of an episode to the specified output directory."""
    os.makedirs(output_dir, exist_ok=True)
    processed_task = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]
    mp4_path = os.path.join(output_dir, f"episode={idx}--success={success}--task={processed_task}.mp4")
    video_writer = imageio.get_writer(mp4_path, fps=30)
    for img in rollout_images:
        video_writer.append_data(img)
    video_writer.close()
    print(f"Saved rollout MP4 at path {mp4_path}")
    if log_file is not None:
        log_file.write(f"Saved rollout MP4 at path {mp4_path}\n")
    return mp4_path


def get_base_suite(task_suite_name: str) -> str:
    """Extract base suite name for unnorm_key.
    E.g., 'libero_spatial_swap' -> 'libero_spatial'
    """
    for base in BASE_SUITES:
        if task_suite_name == base or task_suite_name.startswith(base + "_"):
            return base
    return task_suite_name


def check_unnorm_key(task_suite_name, model_vla):
    """Set and validate the action un-normalization key.
    Perturbed suites use the base suite's norm stats.
    Returns the validated unnorm_key.
    """
    unnorm_key = get_base_suite(task_suite_name)

    if unnorm_key not in model_vla.norm_stats and f"{unnorm_key}_no_noops" in model_vla.norm_stats:
        unnorm_key = f"{unnorm_key}_no_noops"
    assert unnorm_key in model_vla.norm_stats, (
        f"Action un-norm key '{unnorm_key}' not found in model norm_stats! "
        f"Available keys: {list(model_vla.norm_stats.keys())}"
    )
    return unnorm_key


def get_action_from_model(model, obs, task_label, unnorm_key, center_crop=False):
    """Get action from the ReasonVLA model."""
    image = Image.fromarray(obs["full_image"]).convert("RGB")

    if center_crop:
        crop_scale = 0.9
        image_tf = tf.convert_to_tensor(np.array(image))
        orig_dtype = image_tf.dtype
        image_tf = tf.image.convert_image_dtype(image_tf, tf.float32)
        image_tf = crop_and_resize(image_tf, crop_scale, batch_size=1)
        image_tf = tf.clip_by_value(image_tf, 0, 1)
        image_tf = tf.image.convert_image_dtype(image_tf, orig_dtype, saturate=True)
        image = Image.fromarray(image_tf.numpy()).convert("RGB")

    action = model.generate(image, task_label, unnorm_key=unnorm_key)
    return action


def main():
    parser = argparse.ArgumentParser(description="LIBERO-PRO eval for ReasonVLA Multi-Layer (reason_vla_multilayer.py)")
    parser.add_argument("--base_model", type=str, default="openvla/openvla-7b-finetuned-libero-spatial",
                        help="Base OpenVLA checkpoint (HuggingFace name or path)")
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Path to .pth checkpoint file (e.g. final_checkpoint_stage1.pth)")
    parser.add_argument("--task_suite_name", type=str, default="libero_spatial",
                        help="Task suite name (e.g., libero_spatial, libero_object, libero_goal)")
    parser.add_argument("--center_crop", type=str, default="True",
                        help="Center crop images (True if model trained with augmentations)")
    parser.add_argument("--num_trials_per_task", type=int, default=50)
    parser.add_argument("--num_steps_wait", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--local_log_dir", type=str, default="./experiments/logs",
                        help="Local directory for eval logs and videos")
    parser.add_argument("--run_id_note", type=str, default=None)
    parser.add_argument("--stage", type=int, required=True, choices=[1, 2],
                        help="Which stage checkpoint to load (1 or 2)")
    parser.add_argument("--lora_dir", type=str, default=None,
                        help="Directory containing LoRA adapter files (required for stage 2)")
    parser.add_argument("--evaluation_config_path", type=str, required=True,
                        help="Path to LIBERO-PRO evaluation_config.yaml")
    parser.add_argument("--hidden_layer", type=int, default=-1,
                        help="LLM hidden layer for reasoning extraction (-1=last, 23=action decision layer)")
    parser.add_argument("--inject_layers", type=int, nargs="+", default=None,
                        help="ViT layer indices to inject hints at (default: 0 8 16)")
    args = parser.parse_args()

    center_crop = args.center_crop.lower() in ("true", "1", "yes")
    if "image_aug" in args.checkpoint_path:
        assert center_crop, "Expecting center_crop==True because model was trained with image augmentations!"

    # Set random seed
    set_seed_everywhere(args.seed)

    # =========================================================================
    # LIBERO-PRO: Handle perturbation config (from LIBERO-PRO README)
    # =========================================================================
    task_suite_name = args.task_suite_name

    with open(args.evaluation_config_path, "r", encoding="utf-8") as f:
        evaluation_cfg = yaml.safe_load(f)
    assert evaluation_cfg is not None, (
        f"evaluation_config.yaml at {args.evaluation_config_path} is empty or malformed!"
    )

    bddl_base_path = evaluation_cfg.get("bddl_files_path", "").rstrip("/")
    evaluation_cfg["bddl_files_path"] = bddl_base_path + "/" + task_suite_name
    evaluation_cfg["task_suite_name"] = task_suite_name

    use_swap = evaluation_cfg.get("use_swap", False)
    use_object = evaluation_cfg.get("use_object", False)
    use_language = evaluation_cfg.get("use_language", False)
    use_task = evaluation_cfg.get("use_task", False)
    use_environment = evaluation_cfg.get("use_environment", False)

    print("=" * 60)
    print("LIBERO-PRO CONFIG")
    print(f"  evaluation_config_path: {args.evaluation_config_path}")
    print(f"  bddl_base_path:        {bddl_base_path}")
    print(f"  bddl_files_path:       {evaluation_cfg['bddl_files_path']}")
    print(f"  init_file_dir:         {evaluation_cfg.get('init_file_dir', 'NOT SET')}")
    print(f"  task_suite_name:       {task_suite_name}")
    print(f"  use_swap={use_swap}, use_object={use_object}, use_language={use_language}, use_task={use_task}, use_environment={use_environment}")
    print("=" * 60)

    # Step 1: Multiple perturbation flags are True -> combined perturbation (_temp)
    if sum([use_swap, use_object, use_language, use_task, use_environment]) > 1:
        bddl_file_path = bddl_base_path + "/" + task_suite_name + "_temp/"
        init_file_path = evaluation_cfg.get("init_file_dir", "").rstrip("/") + "/" + task_suite_name + "_temp/"

        print(f"  BDDL path:  {bddl_file_path} -> {'FOUND' if os.path.exists(bddl_file_path) else 'NOT FOUND'}")
        print(f"  Init path:  {init_file_path} -> {'FOUND' if os.path.exists(init_file_path) else 'NOT FOUND'}")

        if not os.path.exists(bddl_file_path) or not os.path.exists(init_file_path):
            os.makedirs(init_file_path, exist_ok=True)
            os.makedirs(bddl_file_path, exist_ok=True)

            log_content = f"{use_swap},{use_object},{use_language},{use_task},{use_environment}"
            with open(os.path.join(bddl_file_path, "log.txt"), "w") as lf:
                lf.write(log_content)

            perturbation.create_env(configs=evaluation_cfg)
        else:
            with open(os.path.join(bddl_file_path, "log.txt"), "r") as lf:
                log_contents = lf.read().strip()

            expected_log = f"{use_swap},{use_object},{use_language},{use_task},{use_environment}"

            if log_contents != expected_log:
                for folder in [bddl_file_path, init_file_path]:
                    for root, dirs, files in os.walk(folder, topdown=False):
                        for name in files:
                            os.remove(os.path.join(root, name))
                        for name in dirs:
                            os.rmdir(os.path.join(root, name))
                os.makedirs(init_file_path, exist_ok=True)
                os.makedirs(bddl_file_path, exist_ok=True)

                with open(os.path.join(bddl_file_path, "log.txt"), "w") as lf:
                    lf.write(expected_log)

                perturbation.create_env(configs=evaluation_cfg)

        task_suite_name = task_suite_name + "_temp"

    # Step 2: Handle the case when only one use_xxx flag is True
    elif sum([use_swap, use_object, use_language, use_task, use_environment]) == 1:
        assert "perturbation_mapping" in evaluation_cfg, (
            "evaluation_config.yaml is missing 'perturbation_mapping'! "
            "Cannot determine perturbation suffix."
        )
        if use_swap:
            perturb_key = "use_swap"
        elif use_object:
            perturb_key = "use_object"
        elif use_language:
            perturb_key = "use_language"
        elif use_task:
            perturb_key = "use_task"
        elif use_environment:
            perturb_key = "use_environment"

        perturb_suffix = evaluation_cfg["perturbation_mapping"].get(perturb_key, "")
        assert perturb_suffix, (
            f"perturbation_mapping['{perturb_key}'] is empty or missing in evaluation_config.yaml!"
        )

        init_file_path = evaluation_cfg.get("init_file_dir", "").rstrip("/") + "/" + task_suite_name + "_" + perturb_suffix
        bddl_file_path = evaluation_cfg["bddl_files_path"]

        print(f"  Perturbation: {perturb_key} -> suffix '{perturb_suffix}'")
        print(f"  BDDL path:  {bddl_file_path} -> {'FOUND' if os.path.exists(bddl_file_path) else 'NOT FOUND'}")
        print(f"  Init path:  {init_file_path} -> {'FOUND' if os.path.exists(init_file_path) else 'NOT FOUND'}")

        if not os.path.exists(init_file_path):
            print("  -> Generating perturbation files via perturbation.create_env()...")
            perturbation.create_env(configs=evaluation_cfg)
        else:
            print("  -> Perturbation files already exist, reusing.")

        task_suite_name = task_suite_name + "_" + perturb_suffix

    else:
        print("  No perturbation flags set -- using original suite as-is.")

    print(f"  Final task_suite_name: {task_suite_name}")

    # =========================================================================
    # End LIBERO-PRO perturbation handling
    # =========================================================================

    # Load model
    print("=" * 60)
    print(f"Loading ReasonVLA Multi-Layer model")
    print(f"  Base model:     {args.base_model}")
    print(f"  Checkpoint:     {args.checkpoint_path}")
    print(f"  Stage:          {args.stage}")
    print(f"  Inject layers:  {args.inject_layers or 'default [0, 8, 16]'}")
    print("=" * 60)

    if args.stage == 2 and args.lora_dir is None:
        raise ValueError("--lora_dir is required for stage 2 (directory with LoRA adapter files)")

    model = ReasonVLA.from_finetuned(
        model_name=args.base_model,
        checkpoint_path=args.checkpoint_path,
        stage=args.stage,
        lora_dir=args.lora_dir,
        hidden_layer=args.hidden_layer,
        inject_layers=args.inject_layers,
    )

    print(f"  Model type:     {type(model.vla)}")
    print(f"  Inject layers:  {model.inject_layers}")

    # Set and validate action un-normalization key
    unnorm_key = check_unnorm_key(task_suite_name, model.vla)

    # Get expected image dimensions
    resize_size = 224

    # Initialize local logging
    os.makedirs(args.local_log_dir, exist_ok=True)
    run_id = f"EVAL-{task_suite_name}-reasonvla-multilayer-stage{args.stage}-{DATE_TIME}"
    if args.run_id_note is not None:
        run_id += f"--{args.run_id_note}"
    local_log_filepath = os.path.join(args.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    atexit.register(log_file.close)
    print(f"Output directory: {args.local_log_dir}")
    print(f"Logging to: {local_log_filepath}")

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    assert task_suite_name in benchmark_dict, (
        f"Task suite '{task_suite_name}' not found in benchmark! "
        f"Make sure PYTHONPATH includes LIBERO-PRO/libero for perturbed suites."
    )
    task_suite = benchmark_dict[task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    print(f"Task suite: {task_suite_name}")
    print(f"Unnorm key: {unnorm_key}")
    print(f"Num tasks in suite: {num_tasks_in_suite}")
    log_file.write(f"Task suite: {task_suite_name}\n")
    log_file.write(f"Unnorm key: {unnorm_key}\n")
    log_file.write(f"Num tasks in suite: {num_tasks_in_suite}\n")

    # Verify expected number of tasks
    EXPECTED_NUM_TASKS = {
        "libero_spatial": 10, "libero_object": 10, "libero_goal": 10,
        "libero_10": 10, "libero_90": 90,
    }
    base_suite = get_base_suite(task_suite_name)
    if base_suite in EXPECTED_NUM_TASKS:
        expected = EXPECTED_NUM_TASKS[base_suite]
        assert num_tasks_in_suite == expected, (
            f"Expected {expected} tasks for {task_suite_name} (base={base_suite}), "
            f"got {num_tasks_in_suite}!"
        )

    # =========================================================================
    # SANITY CHECKS
    # =========================================================================
    sanity_lines = [
        "=" * 60,
        "SANITY CHECKS",
        f"  Task suite:     {task_suite_name}",
        f"  Base suite:     {get_base_suite(task_suite_name)}",
        f"  Unnorm key:     {unnorm_key}",
        f"  Norm stats keys: {list(model.vla.norm_stats.keys())}",
        f"  Center crop:    {center_crop}",
        f"  Stage:          {args.stage}",
        f"  Hidden layer:   {args.hidden_layer}",
        f"  Inject layers:  {model.inject_layers}",
        f"  Checkpoint:     {args.checkpoint_path}",
        f"  Resize size:    {resize_size}",
        f"  Num trials:     {args.num_trials_per_task}",
        f"  Seed:           {args.seed}",
        f"  Eval config:    {args.evaluation_config_path}",
    ]
    if "image_aug" in args.base_model or "image_aug" in args.checkpoint_path:
        if not center_crop:
            sanity_lines.append("  WARNING: checkpoint path contains 'image_aug' but center_crop=False!")
    if center_crop and "image_aug" not in args.base_model and "image_aug" not in args.checkpoint_path:
        sanity_lines.append("  NOTE: center_crop=True but no 'image_aug' in model/checkpoint path (may be fine)")
    norm_stats = model.vla.norm_stats[unnorm_key]
    action_stats = norm_stats.get("action", norm_stats)
    sanity_lines.append(f"  Action norm q01: {action_stats.get('q01', 'N/A')}")
    sanity_lines.append(f"  Action norm q99: {action_stats.get('q99', 'N/A')}")
    sanity_lines.append("=" * 60)
    for line in sanity_lines:
        print(line)
        log_file.write(line + "\n")
    log_file.flush()

    # Start evaluation
    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)

        # Log init states hash for cross-run reproducibility verification
        state_hash = hashlib.md5(initial_states.tobytes()).hexdigest()
        print(f"  Task {task_id} init_states hash: {state_hash} (shape={initial_states.shape})")
        log_file.write(f"  Task {task_id} init_states hash: {state_hash} (shape={initial_states.shape})\n")

        # Log BDDL file path for this task
        bddl_file = task.bddl_file if hasattr(task, 'bddl_file') else "N/A"
        problem_folder = task.problem_folder if hasattr(task, 'problem_folder') else "N/A"
        print(f"  Task {task_id} BDDL: {problem_folder}/{bddl_file}")
        log_file.write(f"  Task {task_id} BDDL: {problem_folder}/{bddl_file}\n")

        env, task_description = get_libero_env(task, "openvla", resolution=256)

        # Use language from BDDL file (via env) instead of filename-based description.
        bddl_language = getattr(env, "language_instruction", None)
        if isinstance(bddl_language, (list, tuple)):
            bddl_language = " ".join(str(w) for w in bddl_language)
        if bddl_language and bddl_language.strip():
            if bddl_language.strip().lower() != task_description.strip().lower():
                print(f"\n[Filename instruction]: {task_description}")
                print(f"[BDDL instruction]:    {bddl_language}")
                log_file.write(f"\n[Filename instruction]: {task_description}\n")
                log_file.write(f"[BDDL instruction]:    {bddl_language}\n")
                task_description = bddl_language.strip()
        elif bddl_language is None and task_id == 0:
            print("  WARNING: env.language_instruction not found -- using filename-based description.")
            print("  If running LIBERO-PRO language perturbation, the model will get the WRONG instruction!")
            log_file.write("  WARNING: env.language_instruction not found\n")

        # Log the exact prompt the model will receive for this task
        print(f"  Task {task_id} model prompt: \"{task_description}\"")
        log_file.write(f"  Task {task_id} model prompt: \"{task_description}\"\n")

        # Check num_trials vs available init states
        num_init_states = len(initial_states)
        assert args.num_trials_per_task <= num_init_states, (
            f"num_trials_per_task ({args.num_trials_per_task}) > available init states "
            f"({num_init_states}) for task {task_id}! Would crash with IndexError."
        )

        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            print(f"\nTask: {task_description}")
            log_file.write(f"\nTask: {task_description}\n")

            env.reset()
            obs = env.set_init_state(initial_states[episode_idx])

            t = 0
            done = False
            replay_images = []
            first_action_logged = False
            max_steps = TASK_MAX_STEPS.get(task_suite_name) or TASK_MAX_STEPS.get(get_base_suite(task_suite_name), 400)

            print(f"Starting episode {task_episodes+1}...")
            log_file.write(f"Starting episode {task_episodes+1}...\n")
            while t < max_steps + args.num_steps_wait:
                try:
                    if t < args.num_steps_wait:
                        obs, reward, _done, info = env.step(get_libero_dummy_action("openvla"))
                        t += 1
                        continue

                    img = get_libero_image(obs, resize_size)
                    replay_images.append(img)

                    observation = {
                        "full_image": img,
                        "state": np.concatenate(
                            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
                        ),
                    }

                    action = get_action_from_model(
                        model, observation, task_description, unnorm_key,
                        center_crop=center_crop
                    )
                    assert action.shape == (ACTION_DIM,), f"Expected action shape ({ACTION_DIM},), got {action.shape}"

                    if not first_action_logged and total_episodes == 0:
                        print(f"  [SANITY] First raw action: {action}")
                        print(f"  [SANITY] Gripper (raw): {action[-1]:.4f} (expect ~[0,1])")
                        log_file.write(f"  [SANITY] First raw action: {action}\n")
                        log_file.write(f"  [SANITY] Gripper (raw): {action[-1]:.4f}\n")
                        first_action_logged = True

                    action = normalize_gripper_action(action, binarize=True)
                    action = invert_gripper_action(action)

                    action_list = action.tolist() if hasattr(action, "tolist") else list(action)
                    obs, reward, done, info = env.step(action_list)
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception as e:
                    tb = traceback.format_exc()
                    print(f"Caught exception: {e}\n{tb}")
                    log_file.write(f"Caught exception: {e}\n{tb}\n")
                    break

            task_episodes += 1
            total_episodes += 1

            if replay_images:
                save_rollout_video(
                    replay_images, total_episodes, success=done, task_description=task_description,
                    log_file=log_file, output_dir=args.local_log_dir
                )
            else:
                print(f"  No images captured for episode {total_episodes}, skipping video.")
                log_file.write(f"  No images captured for episode {total_episodes}, skipping video.\n")

            print(f"Success: {done}")
            print(f"# episodes completed so far: {total_episodes}")
            print(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")
            log_file.write(f"Success: {done}\n")
            log_file.write(f"# episodes completed so far: {total_episodes}\n")
            log_file.write(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)\n")
            log_file.flush()

        env.close()

        print(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        print(f"Current total success rate: {float(total_successes) / float(total_episodes)}")
        log_file.write(f"Current task success rate: {float(task_successes) / float(task_episodes)}\n")
        log_file.write(f"Current total success rate: {float(total_successes) / float(total_episodes)}\n")
        log_file.flush()

    log_file.close()

    print(f"\n{'=' * 60}")
    if total_episodes > 0:
        print(f"FINAL RESULTS: {total_successes}/{total_episodes} = {total_successes/total_episodes*100:.1f}%")
    else:
        print("FINAL RESULTS: No episodes completed!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
