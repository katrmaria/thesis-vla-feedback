"""
run_libero_eval_reason_vla_projector_crossattn.py

Usage:
    python run_libero_eval_reason_vla_projector_crossattn.py \
        --base_model openvla/openvla-7b-finetuned-libero-spatial \
        --checkpoint_path /path/to/checkpoint-4950.pth \
        --stage 1 \
        --task_suite_name libero_spatial \
        --center_crop True \
        --num_trials_per_task 50 \
        --seed 7 \
        --hidden_layer 7
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
from PIL import Image
from libero.libero import benchmark

# ---- Path setup ----
ALVIS_HOME = os.environ.get("ALVIS_HOME", "/cephyr/users/mariakat/Alvis")
WORK_DIR = os.path.join(ALVIS_HOME, "openvla")
OPENVLA_REPO = os.path.join(WORK_DIR, "openvla_repo")

sys.path.insert(0, OPENVLA_REPO)
sys.path.insert(0, WORK_DIR)

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

from reason_vla_projector_crossattn import ReasonVLAProjectorCrossAttn

import experiments.robot.libero.libero_utils as _lu
import libero.libero.benchmark as _bm
import reason_vla_projector_crossattn as _rv
print("=" * 60)
print("IMPORT PATH CHECK")
print(f"  libero.libero.benchmark:         {_bm.__file__}")
print(f"  libero_utils:                    {_lu.__file__}")
print(f"  reason_vla_projector_crossattn:  {_rv.__file__}")
print("=" * 60)
del _lu, _bm, _rv

ACTION_DIM = 7

TASK_MAX_STEPS = {
    "libero_spatial": 220, "libero_object": 280, "libero_goal": 300,
    "libero_10": 520, "libero_90": 400,
}


def save_rollout_video(rollout_images, idx, success, task_description, log_file=None, output_dir="."):
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


def get_action_from_model(model, obs, task_label, unnorm_key, center_crop=False):
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
    parser = argparse.ArgumentParser(description="LIBERO eval for ReasonVLAProjectorCrossAttn")
    parser.add_argument("--base_model", type=str, default="openvla/openvla-7b-finetuned-libero-spatial")
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--task_suite_name", type=str, default="libero_spatial")
    parser.add_argument("--center_crop", type=str, default="True")
    parser.add_argument("--num_trials_per_task", type=int, default=50)
    parser.add_argument("--num_steps_wait", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--local_log_dir", type=str, default="./experiments/logs")
    parser.add_argument("--run_id_note", type=str, default=None)
    parser.add_argument("--stage", type=int, required=True, choices=[1, 2])
    parser.add_argument("--lora_dir", type=str, default=None)
    parser.add_argument("--hidden_layer", type=int, default=7,
                        help="LLM hidden layer for text-token extraction (training default: 7)")
    args = parser.parse_args()

    center_crop = args.center_crop.lower() in ("true", "1", "yes")
    if "image_aug" in args.checkpoint_path:
        assert center_crop, "Expecting center_crop==True because model was trained with image augmentations!"

    set_seed_everywhere(args.seed)

    task_suite_name = args.task_suite_name

    print("=" * 60)
    print(f"Loading ReasonVLAProjectorCrossAttn model")
    print(f"  Base model:   {args.base_model}")
    print(f"  Checkpoint:   {args.checkpoint_path}")
    print(f"  Stage:        {args.stage}")
    print(f"  Hidden layer: {args.hidden_layer}")
    print("=" * 60)

    if args.stage == 2 and args.lora_dir is None:
        raise ValueError("--lora_dir is required for stage 2 (directory with LoRA adapter files)")

    model = ReasonVLAProjectorCrossAttn.from_finetuned(
        model_name=args.base_model,
        checkpoint_path=args.checkpoint_path,
        stage=args.stage,
        lora_dir=args.lora_dir,
        hidden_layer=args.hidden_layer,
    )

    print(f"  Model type:   {type(model.vla)}")

    unnorm_key = task_suite_name
    if unnorm_key not in model.vla.norm_stats and f"{unnorm_key}_no_noops" in model.vla.norm_stats:
        unnorm_key = f"{unnorm_key}_no_noops"
    assert unnorm_key in model.vla.norm_stats, (
        f"Action un-norm key '{unnorm_key}' not found in model norm_stats"
        f"Available keys: {list(model.vla.norm_stats.keys())}"
    )

    resize_size = 224

    print("=" * 60)
    print("SANITY CHECKS")
    print(f"  Task suite:     {task_suite_name}")
    print(f"  Unnorm key:     {unnorm_key}")
    print(f"  Norm stats keys: {list(model.vla.norm_stats.keys())}")
    print(f"  Center crop:    {center_crop}")
    print(f"  Stage:          {args.stage}")
    print(f"  Hidden layer:   {args.hidden_layer}")
    print(f"  Resize size:    {resize_size}")
    print(f"  Num trials:     {args.num_trials_per_task}")
    print(f"  Seed:           {args.seed}")

    if "image_aug" in args.base_model or "image_aug" in args.checkpoint_path:
        if not center_crop:
            print("  WARNING: checkpoint path contains 'image_aug' but center_crop=False!")
    if center_crop and "image_aug" not in args.base_model and "image_aug" not in args.checkpoint_path:
        print("  NOTE: center_crop=True but no 'image_aug' in model/checkpoint path (may be fine)")

    norm_stats = model.vla.norm_stats[unnorm_key]
    action_stats = norm_stats.get("action", norm_stats)
    print(f"  Action norm q01: {action_stats.get('q01', 'N/A')}")
    print(f"  Action norm q99: {action_stats.get('q99', 'N/A')}")
    print("=" * 60)

    run_id = f"EVAL-{task_suite_name}-projcrossattn-stage{args.stage}-{DATE_TIME}"
    if args.run_id_note:
        run_id += f"--{args.run_id_note}"
    os.makedirs(args.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(args.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    atexit.register(log_file.close)
    print(f"Logging to local log file: {local_log_filepath}")

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    print(f"Task suite: {task_suite_name}")
    print(f"Num tasks in suite: {num_tasks_in_suite}")
    log_file.write(f"Task suite: {task_suite_name}\n")
    log_file.write(f"Num tasks in suite: {num_tasks_in_suite}\n")

    EXPECTED_NUM_TASKS = {
        "libero_spatial": 10, "libero_object": 10, "libero_goal": 10,
        "libero_10": 10, "libero_90": 90,
    }
    if task_suite_name in EXPECTED_NUM_TASKS:
        expected = EXPECTED_NUM_TASKS[task_suite_name]
        assert num_tasks_in_suite == expected, (
            f"Expected {expected} tasks for {task_suite_name}, got {num_tasks_in_suite}!"
        )

    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)

        state_hash = hashlib.md5(initial_states.tobytes()).hexdigest()
        print(f"  Task {task_id} init_states hash: {state_hash} (shape={initial_states.shape})")
        log_file.write(f"  Task {task_id} init_states hash: {state_hash} (shape={initial_states.shape})\n")

        bddl_file = task.bddl_file if hasattr(task, 'bddl_file') else "N/A"
        problem_folder = task.problem_folder if hasattr(task, 'problem_folder') else "N/A"
        print(f"  Task {task_id} BDDL: {problem_folder}/{bddl_file}")
        log_file.write(f"  Task {task_id} BDDL: {problem_folder}/{bddl_file}\n")

        env, task_description = get_libero_env(task, "openvla", resolution=256)

        print(f"  Task {task_id} model prompt: \"{task_description}\"")
        log_file.write(f"  Task {task_id} model prompt: \"{task_description}\"\n")

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
            max_steps = TASK_MAX_STEPS.get(task_suite_name, 400)

            print(f"Starting episode {task_episodes+1}...")
            log_file.write(f"Starting episode {task_episodes+1}...\n")

            while t < max_steps + args.num_steps_wait:
                try:
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(get_libero_dummy_action("openvla"))
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
    print(f"FINAL RESULTS: {total_successes}/{total_episodes} = {total_successes/total_episodes*100:.1f}%")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
