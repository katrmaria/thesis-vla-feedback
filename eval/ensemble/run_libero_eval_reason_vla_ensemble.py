"""
run_libero_eval_reason_vla_ensemble.py

Evaluates an ensemble of ReasonVLA models by averaging their predictions.
Loads multiple checkpoints (with different hl / mechanism / stage) and at each
simulation step runs all models on the same image, then averages their actions.

Usage:
    python run_libero_eval_reason_vla_ensemble.py \
        --base_model openvla/openvla-7b-finetuned-libero-spatial \
        --ensemble_configs "ckpt1.pth,1,none,-1,additive" "ckpt2.pth,2,lora_dir,24,additive" \
        --task_suite_name libero_spatial \
        --center_crop True \
        --num_trials_per_task 50 \
        --seed 7

Each ensemble_config is a comma-separated 5-tuple:
    <checkpoint_path>,<stage>,<lora_dir_or_"none">,<hidden_layer>,<feedback_mode>
"""

import atexit
import os
import sys
import argparse
import traceback

import imageio
import numpy as np
import tensorflow as tf
import tqdm
from PIL import Image
from libero.libero import benchmark

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

from reason_vla import ReasonVLA

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
    if log_file is not None:
        log_file.write(f"Saved rollout MP4 at path {mp4_path}\n")
    return mp4_path


def get_ensemble_action(models, obs, task_label, unnorm_key, center_crop=False, weights=None):
    """Run all models, average (optionally weighted) their action predictions."""
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

    actions = []
    for model in models:
        a = model.generate(image, task_label, unnorm_key=unnorm_key)
        actions.append(np.asarray(a))

    actions = np.stack(actions, axis=0)  # [num_models, 7]
    if weights is not None:
        weights = np.asarray(weights, dtype=np.float32)
        weights = weights / weights.sum()
        return (actions * weights[:, None]).sum(axis=0)
    return actions.mean(axis=0)


def parse_ensemble_config(config_str):
    """Parse 'ckpt,stage,lora_or_none,hl,feedback' into a dict."""
    parts = config_str.split(",")
    if len(parts) != 5:
        raise ValueError(f"Each --ensemble_configs entry must have 5 comma-separated fields: {config_str}")
    ckpt, stage, lora, hl, fb = parts
    stage = int(stage)
    lora_dir = None if lora.lower() in ("none", "") else lora
    hl = int(hl)
    return {"checkpoint_path": ckpt, "stage": stage, "lora_dir": lora_dir,
            "hidden_layer": hl, "feedback_mode": fb}


def main():
    parser = argparse.ArgumentParser(description="LIBERO ensemble eval for ReasonVLA")
    parser.add_argument("--base_model", type=str, default="openvla/openvla-7b-finetuned-libero-spatial")
    parser.add_argument("--ensemble_configs", type=str, nargs="+", required=True,
                        help="Each config: 'ckpt.pth,stage,lora_or_none,hl,feedback'")
    parser.add_argument("--weights", type=float, nargs="*", default=None,
                        help="Optional weights for each model (same length as ensemble_configs)")
    parser.add_argument("--task_suite_name", type=str, default="libero_spatial")
    parser.add_argument("--center_crop", type=str, default="True")
    parser.add_argument("--num_trials_per_task", type=int, default=50)
    parser.add_argument("--num_steps_wait", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--local_log_dir", type=str, default="./experiments/logs")
    parser.add_argument("--run_id_note", type=str, default=None)
    args = parser.parse_args()

    center_crop = args.center_crop.lower() in ("true", "1", "yes")
    set_seed_everywhere(args.seed)

    configs = [parse_ensemble_config(c) for c in args.ensemble_configs]
    if args.weights is not None and len(args.weights) != len(configs):
        raise ValueError("--weights length must match number of ensemble_configs")

    print("=" * 60)
    print(f"Loading ensemble of {len(configs)} models")
    for i, c in enumerate(configs):
        print(f"  [{i}] hl={c['hidden_layer']} stage={c['stage']} fb={c['feedback_mode']}")
        print(f"       ckpt={c['checkpoint_path']}")
        print(f"       lora={c['lora_dir']}")
    print("=" * 60)

    models = []
    for c in configs:
        if c["stage"] == 2 and c["lora_dir"] is None:
            raise ValueError(f"--lora_dir required for stage 2 model: {c['checkpoint_path']}")
        m = ReasonVLA.from_finetuned(
            model_name=args.base_model,
            checkpoint_path=c["checkpoint_path"],
            stage=c["stage"],
            lora_dir=c["lora_dir"],
            hidden_layer=c["hidden_layer"],
            feedback_mode=c["feedback_mode"],
        )
        models.append(m)
    print(f"Loaded {len(models)} models")

    unnorm_key = args.task_suite_name
    m0 = models[0]
    if unnorm_key not in m0.vla.norm_stats and f"{unnorm_key}_no_noops" in m0.vla.norm_stats:
        unnorm_key = f"{unnorm_key}_no_noops"
    resize_size = 224

    run_id = f"EVAL-{args.task_suite_name}-ensemble-{DATE_TIME}"
    if args.run_id_note:
        run_id += f"--{args.run_id_note}"
    os.makedirs(args.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(args.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    atexit.register(log_file.close)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks

    total_episodes = 0
    total_successes = 0

    for task_id in range(num_tasks_in_suite):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = get_libero_env(task, args.base_model, resolution=256)
        task_successes = 0
        task_episodes = 0

        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task), desc=f"Task {task_id}"):
            env.reset()
            obs = env.set_init_state(initial_states[episode_idx])
            task_max_steps = TASK_MAX_STEPS[args.task_suite_name]
            t = 0
            rollout_images = []
            replay_images = []
            success = False

            while t < task_max_steps + args.num_steps_wait:
                try:
                    if t < args.num_steps_wait:
                        obs, _, _, _ = env.step(get_libero_dummy_action(args.base_model))
                        t += 1
                        continue

                    img = get_libero_image(obs, resize_size)
                    obs["full_image"] = img
                    rollout_images.append(img)

                    action = get_ensemble_action(
                        models, obs, task_description, unnorm_key,
                        center_crop=center_crop, weights=args.weights,
                    )
                    action = normalize_gripper_action(action, binarize=True)
                    action = invert_gripper_action(action)
                    obs, reward, done, info = env.step(action.tolist())
                    if done:
                        task_successes += 1
                        total_successes += 1
                        success = True
                        break
                    t += 1
                except Exception as e:
                    print(f"Error at step {t}: {e}")
                    traceback.print_exc()
                    break

            task_episodes += 1
            total_episodes += 1
            save_rollout_video(rollout_images, total_episodes, success, task_description,
                               log_file=log_file, output_dir=args.local_log_dir)
            log_file.write(f"Success: {success}\n")
            log_file.write(f"# episodes completed so far: {total_episodes}\n")
            log_file.write(f"# successes: {total_successes} ({total_successes/total_episodes*100:.1f}%)\n")
            log_file.write(f"Current task success rate: {task_successes/task_episodes:.4f}\n")
            log_file.write(f"Current total success rate: {total_successes/total_episodes:.4f}\n")
            log_file.flush()

    print(f"\nOverall success rate: {total_successes/total_episodes:.4f} ({total_successes/total_episodes*100:.1f}%)")
    log_file.write(f"\nOverall success rate: {total_successes/total_episodes:.4f} ({total_successes/total_episodes*100:.1f}%)\n")


if __name__ == "__main__":
    main()
