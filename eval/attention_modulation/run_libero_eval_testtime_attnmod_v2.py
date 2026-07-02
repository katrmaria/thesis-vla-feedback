"""
run_libero_eval_testtime_attnmod_v2.py

LIBERO eval for TestTimeAttentionModulationV2 (no training — pure test-time).
Supports three methods: song, logit_bias, attn_gate.

Usage:
    # Song et al. faithful (suppress bottom 20% at L28)
    python run_libero_eval_testtime_attnmod_v2.py \
        --base_model openvla/openvla-7b-finetuned-libero-spatial \
        --task_suite_name libero_spatial \
        --method song --rho 0.2 --lam 0.1

    # Attention logit bias at L10-15
    python run_libero_eval_testtime_attnmod_v2.py \
        --base_model openvla/openvla-7b-finetuned-libero-spatial \
        --task_suite_name libero_spatial \
        --method logit_bias --beta 1.0

    # Attention weight gating at L10-15
    python run_libero_eval_testtime_attnmod_v2.py \
        --base_model openvla/openvla-7b-finetuned-libero-spatial \
        --task_suite_name libero_spatial \
        --method attn_gate
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
    get_libero_dummy_action, get_libero_env, get_libero_image, quat2axisangle,
)
from experiments.robot.openvla_utils import crop_and_resize
from experiments.robot.robot_utils import (
    DATE_TIME, invert_gripper_action, normalize_gripper_action, set_seed_everywhere,
)

from attention_modulation_testtime_v2 import TestTimeAttentionModulationV2

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
    return model.generate(image, task_label, unnorm_key=unnorm_key)


def main():
    parser = argparse.ArgumentParser(description="LIBERO eval for test-time attention modulation V2")
    parser.add_argument("--base_model", type=str, default="openvla/openvla-7b-finetuned-libero-spatial")
    parser.add_argument("--task_suite_name", type=str, default="libero_spatial")
    parser.add_argument("--center_crop", type=str, default="True")
    parser.add_argument("--num_trials_per_task", type=int, default=50)
    parser.add_argument("--num_steps_wait", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--local_log_dir", type=str, default="./experiments/logs")
    parser.add_argument("--run_id_note", type=str, default=None)

    # Method selection
    parser.add_argument("--method", type=str, required=True, choices=["song", "logit_bias", "attn_gate", "vit_bias"])

    # Song params
    parser.add_argument("--post_layer", type=int, default=27)
    parser.add_argument("--pre_layer", type=int, default=15)
    parser.add_argument("--review_layer", type=int, default=28)
    parser.add_argument("--rho", type=float, default=0.2, help="Percentile threshold for Song (suppress bottom rho%%)")
    parser.add_argument("--lam", type=float, default=0.1, help="Suppression factor for Song")

    # Logit bias params
    parser.add_argument("--bias_layers", type=int, nargs="+", default=None, help="Layers for logit bias (default: 10-15)")
    parser.add_argument("--beta", type=float, default=1.0, help="Scaling factor for logit bias")

    # Attn gate params
    parser.add_argument("--gate_layers", type=int, nargs="+", default=None, help="Layers for attn gating (default: 10-15)")

    # Shared
    parser.add_argument("--teacher_layer", type=int, default=11)
    parser.add_argument("--teacher_head", type=int, default=-1, help="-1 for auto sharpest head")
    parser.add_argument("--mask_source_layers", type=int, nargs="+", default=None)

    args = parser.parse_args()

    center_crop = args.center_crop.lower() in ("true", "1", "yes")
    set_seed_everywhere(args.seed)
    task_suite_name = args.task_suite_name

    # Build kwargs and load model based on method
    if args.method == "vit_bias":
        from attention_modulation_testtime_vit import ViTAttentionBiasVLA

        model_kwargs = dict(
            teacher_layer=args.teacher_layer,
            teacher_head=args.teacher_head,
            beta=args.beta,
        )

        print("=" * 60)
        print(f"Loading ViTAttentionBiasVLA — method: vit_bias")
        print(f"  Base model:      {args.base_model}")
        print(f"  Beta:            {args.beta}")
        print(f"  Teacher:         L{args.teacher_layer} H{args.teacher_head}")
        print("=" * 60)

        model = ViTAttentionBiasVLA.from_pretrained(
            args.base_model, **model_kwargs
        )
    else:
        model_kwargs = dict(
            method=args.method,
            post_layer=args.post_layer,
            pre_layer=args.pre_layer,
            review_layer=args.review_layer,
            rho=args.rho,
            lam=args.lam,
            beta=args.beta,
            teacher_layer=args.teacher_layer,
            teacher_head=args.teacher_head,
        )
        if args.bias_layers is not None:
            model_kwargs["bias_layers"] = args.bias_layers
        if args.gate_layers is not None:
            model_kwargs["gate_layers"] = args.gate_layers
        if args.mask_source_layers is not None:
            model_kwargs["mask_source_layers"] = args.mask_source_layers

        print("=" * 60)
        print(f"Loading TestTimeAttentionModulationV2 — method: {args.method}")
        print(f"  Base model:      {args.base_model}")
        if args.method == "song":
            print(f"  Post layer:      L{args.post_layer}")
            print(f"  Pre layer:       L{args.pre_layer}")
            print(f"  Review layer:    L{args.review_layer}")
            print(f"  Rho:             {args.rho}")
            print(f"  Lambda:          {args.lam}")
        elif args.method == "logit_bias":
            print(f"  Bias layers:     {args.bias_layers or list(range(10, 16))}")
            print(f"  Beta:            {args.beta}")
            print(f"  Teacher:         L{args.teacher_layer} H{args.teacher_head}")
        elif args.method == "attn_gate":
            print(f"  Gate layers:     {args.gate_layers or list(range(10, 16))}")
            print(f"  Teacher:         L{args.teacher_layer} H{args.teacher_head}")
        print("=" * 60)

        model = TestTimeAttentionModulationV2.from_pretrained(
            args.base_model, **model_kwargs
        )
    print(f"  Model type:      {type(model.vla)}")

    unnorm_key = task_suite_name
    if unnorm_key not in model.vla.norm_stats and f"{unnorm_key}_no_noops" in model.vla.norm_stats:
        unnorm_key = f"{unnorm_key}_no_noops"
    assert unnorm_key in model.vla.norm_stats, (
        f"Action un-norm key '{unnorm_key}' not found. Available: {list(model.vla.norm_stats.keys())}"
    )

    resize_size = 224

    print("=" * 60)
    print("SANITY CHECKS")
    print(f"  Task suite:      {task_suite_name}")
    print(f"  Unnorm key:      {unnorm_key}")
    print(f"  Center crop:     {center_crop}")
    print(f"  Num trials:      {args.num_trials_per_task}")
    print(f"  Seed:            {args.seed}")
    norm_stats = model.vla.norm_stats[unnorm_key]
    action_stats = norm_stats.get("action", norm_stats)
    print(f"  Action norm q01: {action_stats.get('q01', 'N/A')}")
    print(f"  Action norm q99: {action_stats.get('q99', 'N/A')}")
    print("=" * 60)

    # Run ID
    if args.method == "song":
        method_str = f"song_rho{args.rho}_lam{args.lam}_L{args.post_layer}vs{args.pre_layer}_rev{args.review_layer}"
    elif args.method == "logit_bias":
        layers_str = "_".join(str(l) for l in (args.bias_layers or list(range(10, 16))))
        method_str = f"logitbias_L{layers_str}_beta{args.beta}"
    elif args.method == "vit_bias":
        method_str = f"vitbias_beta{args.beta}_teacher{args.teacher_layer}"
    else:
        layers_str = "_".join(str(l) for l in (args.gate_layers or list(range(10, 16))))
        method_str = f"attngate_L{layers_str}"

    run_id = f"EVAL-{task_suite_name}-v2-{method_str}-{DATE_TIME}"
    if args.run_id_note:
        run_id += f"--{args.run_id_note}"
    os.makedirs(args.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(args.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    atexit.register(log_file.close)
    print(f"Logging to: {local_log_filepath}")

    # Write config
    log_file.write(f"Method: {args.method}\n")
    log_file.write(f"Task suite: {task_suite_name}\n")
    log_file.write(f"Base model: {args.base_model}\n")
    log_file.write(f"Model kwargs: {model_kwargs}\n")
    log_file.write(f"Seed: {args.seed}\n")

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    print(f"Task suite: {task_suite_name}  ({num_tasks_in_suite} tasks)")
    log_file.write(f"Num tasks: {num_tasks_in_suite}\n")

    EXPECTED_NUM_TASKS = {
        "libero_spatial": 10, "libero_object": 10, "libero_goal": 10,
        "libero_10": 10, "libero_90": 90,
    }
    if task_suite_name in EXPECTED_NUM_TASKS:
        assert num_tasks_in_suite == EXPECTED_NUM_TASKS[task_suite_name]

    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)

        state_hash = hashlib.md5(initial_states.tobytes()).hexdigest()
        print(f"  Task {task_id} init_states hash: {state_hash} (shape={initial_states.shape})")
        log_file.write(f"  Task {task_id} init_states hash: {state_hash} (shape={initial_states.shape})\n")

        bddl_file = task.bddl_file if hasattr(task, "bddl_file") else "N/A"
        problem_folder = task.problem_folder if hasattr(task, "problem_folder") else "N/A"
        print(f"  Task {task_id} BDDL: {problem_folder}/{bddl_file}")
        log_file.write(f"  Task {task_id} BDDL: {problem_folder}/{bddl_file}\n")

        env, task_description = get_libero_env(task, "openvla", resolution=256)
        print(f"  Task {task_id} model prompt: \"{task_description}\"")
        log_file.write(f"  Task {task_id} model prompt: \"{task_description}\"\n")

        num_init_states = len(initial_states)
        assert args.num_trials_per_task <= num_init_states

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

            print(f"Starting episode {task_episodes + 1}...")
            log_file.write(f"Starting episode {task_episodes + 1}...\n")

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
                        "state": np.concatenate((
                            obs["robot0_eef_pos"],
                            quat2axisangle(obs["robot0_eef_quat"]),
                            obs["robot0_gripper_qpos"],
                        )),
                    }

                    action = get_action_from_model(
                        model, observation, task_description, unnorm_key, center_crop=center_crop
                    )
                    assert action.shape == (ACTION_DIM,)

                    if not first_action_logged and total_episodes == 0:
                        print(f"  [SANITY] First raw action: {action}")
                        print(f"  [SANITY] Gripper (raw): {action[-1]:.4f}")
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
                    log_file=log_file, output_dir=args.local_log_dir,
                )

            print(f"Success: {done}")
            print(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")
            log_file.write(f"Success: {done}\n")
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
    print(f"FINAL: {total_successes}/{total_episodes} = {total_successes / total_episodes * 100:.1f}%")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
