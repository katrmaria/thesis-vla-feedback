"""
diagnostic_projector_xattn.py

Per-layer instruction-dependence diagnostic for the projector cross-attention
model. Mirrors the baseline OpenVLA notebook diagnostic byte-for-byte
(same sampling, same preprocessing, same gripper post-processing, same metrics,
same plot). Only difference: pass 2 routes through the cross-attention.

Outputs:
  image_instruction_similarity_projcrossattn.pdf
  diagnostic_projector_xattn_results.npz
"""

import os
import sys
import random
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image as PILImage
import tensorflow as tf
import tensorflow_datasets as tfds

ALVIS_HOME = os.environ.get("ALVIS_HOME", "/cephyr/users/mariakat/Alvis")
WORK_DIR = os.path.join(ALVIS_HOME, "openvla")
OPENVLA_REPO = os.path.join(WORK_DIR, "openvla_repo")
sys.path.insert(0, OPENVLA_REPO)
sys.path.insert(0, WORK_DIR)

from reason_vla_projector_crossattn import (
    ReasonVLAProjectorCrossAttn, override_vision_backbone, override_projector,
)

CKPT = "/mimer/NOBACKUP/groups/robot_unforseen/mariakat/runs/reason_vla/rvla-projcrossattn-6484205/stage1/checkpoint-6600.pth"
BASE_MODEL = "openvla/openvla-7b-finetuned-libero-spatial"
HIDDEN_LAYER = 7
NUM_LAYERS = 32
NUM_PATCHES = 256
SAMPLES_PER_TASK = 100
N_PER_TASK = 10
N_IMAGES_PER_TASK_BLOCKB = 50
UNNORM_KEY = "libero_spatial"
RLDS_DIR = "/mimer/NOBACKUP/groups/robot_unforseen/mariakat/data/modified_libero_rlds"
DEVICE = "cuda"


# ---------------------------------------------------------------------------
# Verbatim helpers from openvla/experiments/robot/*
# ---------------------------------------------------------------------------
def crop_and_resize(image, crop_scale, batch_size):
    assert image.shape.ndims == 3 or image.shape.ndims == 4
    expanded_dims = False
    if image.shape.ndims == 3:
        image = tf.expand_dims(image, axis=0)
        expanded_dims = True
    new_heights = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,))
    new_widths = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,))
    height_offsets = (1 - new_heights) / 2
    width_offsets = (1 - new_widths) / 2
    bounding_boxes = tf.stack(
        [height_offsets, width_offsets, height_offsets + new_heights, width_offsets + new_widths], axis=1
    )
    image = tf.image.crop_and_resize(image, bounding_boxes, tf.range(batch_size), (224, 224))
    if expanded_dims:
        image = image[0]
    return image


def resize_image(img, resize_size):
    assert isinstance(resize_size, tuple)
    img = tf.image.encode_jpeg(img)
    img = tf.io.decode_image(img, expand_animations=False, dtype=tf.uint8)
    img = tf.image.resize(img, resize_size, method="lanczos3", antialias=True)
    img = tf.cast(tf.clip_by_value(tf.round(img), 0, 255), tf.uint8)
    return img.numpy()


def normalize_gripper_action(action, binarize=True):
    action[..., -1] = 2 * (action[..., -1] - 0.0) / (1.0 - 0.0) - 1
    if binarize:
        action[..., -1] = np.sign(action[..., -1])
    return action


def invert_gripper_action(action):
    action[..., -1] = action[..., -1] * -1.0
    return action


def get_eval_obs(image_np):
    img_224 = resize_image(image_np, (224, 224))
    return {"full_image": img_224}


def prepare_inputs_like_eval(vla, processor, base_vla_name, obs, task_label, center_crop=True):
    image = PILImage.fromarray(obs["full_image"]).convert("RGB")
    if center_crop:
        image = tf.convert_to_tensor(np.array(image))
        orig_dtype = image.dtype
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = crop_and_resize(image, 0.9, batch_size=1)
        image = tf.clip_by_value(image, 0, 1)
        image = tf.image.convert_image_dtype(image, orig_dtype, saturate=True)
        image = PILImage.fromarray(image.numpy()).convert("RGB")
    if "openvla-v01" in base_vla_name:
        prompt = (
            f"A chat between a curious user and an artificial intelligence assistant. "
            f"The assistant gives helpful, detailed, and polite answers to the user's questions. "
            f"USER: What action should the robot take to {task_label.lower()}? ASSISTANT:"
        )
    else:
        prompt = f"In: What action should the robot take to {task_label.lower()}?\nOut:"
    inputs = processor(prompt, image).to(DEVICE, dtype=torch.bfloat16)
    if inputs["input_ids"].dim() == 1:
        inputs["input_ids"] = inputs["input_ids"].unsqueeze(0)
    if inputs["pixel_values"].dim() == 3:
        inputs["pixel_values"] = inputs["pixel_values"].unsqueeze(0)
    if "attention_mask" in inputs and inputs["attention_mask"].dim() == 1:
        inputs["attention_mask"] = inputs["attention_mask"].unsqueeze(0)
    if not torch.all(inputs["input_ids"][:, -1] == 29871):
        inputs["input_ids"] = torch.cat(
            (inputs["input_ids"],
             torch.unsqueeze(torch.Tensor([29871]).long(), dim=0).to(inputs["input_ids"].device)), dim=1
        )
        if "attention_mask" in inputs:
            inputs["attention_mask"] = torch.cat(
                [inputs["attention_mask"],
                 torch.ones(1, 1, device=inputs["attention_mask"].device,
                            dtype=inputs["attention_mask"].dtype)], dim=1
            )
    return inputs


# ---------------------------------------------------------------------------
# Two-pass forward helpers (projector cross-attention)
# ---------------------------------------------------------------------------
@torch.no_grad()
def two_pass_forward(model, inputs):
    """Returns the pass-2 vla output object (return_dict=True) with hidden_states."""
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask")
    pixel_values = inputs["pixel_values"]

    # Pass 1: extract text-token hidden states at model.hidden_layer
    model.reset_image_reasoning()
    out1 = model.vla(
        input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values,
        labels=None, output_hidden_states=True, return_dict=True,
    )
    text_hidden = out1.hidden_states[model.hidden_layer][:, 1 + NUM_PATCHES:, :]

    # Pass 2 inline (mirrors library second_forward + the bf16 cast it forgets)
    vb = model.vla.vision_backbone
    patch_features = vb(pixel_values)
    projected = model.vla.projector(patch_features)
    refined = model.cross_attn(projected, text_hidden).to(dtype=projected.dtype)
    with override_vision_backbone(vb, patch_features):
        with override_projector(model.vla, refined):
            out2 = model.vla(
                input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values,
                labels=None, output_hidden_states=True, return_dict=True,
            )
    return out2


@torch.no_grad()
def predict_action_two_pass(model, inputs):
    """Mirrors get_vla_action -> normalize+invert gripper, routed through pass 2."""
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask")
    pixel_values = inputs["pixel_values"]

    # Pass 1
    model.reset_image_reasoning()
    out1 = model.vla(
        input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values,
        labels=None, output_hidden_states=True, return_dict=True,
    )
    text_hidden = out1.hidden_states[model.hidden_layer][:, 1 + NUM_PATCHES:, :]

    # Pass 2: cross-attention + predict_action
    vb = model.vla.vision_backbone
    patch_features = vb(pixel_values)
    projected = model.vla.projector(patch_features)
    refined = model.cross_attn(projected, text_hidden).to(dtype=projected.dtype)
    with override_vision_backbone(vb, patch_features):
        with override_projector(model.vla, refined):
            action = model.vla.predict_action(
                input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values,
                unnorm_key=UNNORM_KEY, do_sample=False,
            )
    if hasattr(action, "cpu"):
        action = action.cpu().numpy()
    elif not isinstance(action, np.ndarray):
        action = np.array(action)
    action = normalize_gripper_action(action.copy(), binarize=True)
    action = invert_gripper_action(action)
    return action


def predict_action_from_sample(model, sample, instruction):
    obs = get_eval_obs(sample["image"])
    inputs = prepare_inputs_like_eval(model.vla, model.processor, BASE_MODEL, obs,
                                      instruction, center_crop=True)
    return predict_action_two_pass(model, inputs)


def get_eval_inputs(model, sample, instruction):
    obs = get_eval_obs(sample["image"])
    return prepare_inputs_like_eval(model.vla, model.processor, BASE_MODEL, obs,
                                    instruction, center_crop=True)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def build_samples_by_task():
    ds = tfds.load("libero_spatial_no_noops", split="train", data_dir=RLDS_DIR)
    by_task = {}
    for ep in ds:
        for step in ep["steps"]:
            instr = step["language_instruction"].numpy().decode()
            img = step["observation"]["image"].numpy()
            act = step["action"].numpy().astype(np.float32)
            by_task.setdefault(instr, []).append({
                "task": instr, "image": img, "action_gt": act,
            })
        if len(by_task) >= 10 and min(len(v) for v in by_task.values()) >= 300:
            break
    print(f"Loaded {sum(len(v) for v in by_task.values())} samples across {len(by_task)} tasks")
    for k, v in by_task.items():
        print(f"  [{len(v):4d}]  {k}")
    return by_task


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    random.seed(42); np.random.seed(42); torch.manual_seed(42)

    print("Loading projector cross-attention model")
    model = ReasonVLAProjectorCrossAttn.from_finetuned(
        model_name=BASE_MODEL, checkpoint_path=CKPT,
        stage=1, hidden_layer=HIDDEN_LAYER,
    )
    model.eval()

    print("Building samples_by_task from libero_spatial RLDS")
    samples_by_task = build_samples_by_task()

    balanced_samples = []
    for task, samples in samples_by_task.items():
        if len(samples) >= SAMPLES_PER_TASK:
            selected = random.sample(samples, SAMPLES_PER_TASK)
        else:
            selected = samples
        balanced_samples.extend(selected)
    random.shuffle(balanced_samples)
    print(f"Using {len(balanced_samples)} balanced samples")

    task_keys = list(samples_by_task.keys())
    image_pool = {task: [s for s in balanced_samples if s["task"] == task][:N_PER_TASK]
                  for task in task_keys}

    # =================================================================
    # Block A: per-layer cosine
    # =================================================================
    image_effect, instr_effect = [], []
    for task_idx, task in enumerate(task_keys):
        # Same instruction, different images
        img_cos_per_layer = [[] for _ in range(NUM_LAYERS + 1)]
        for slot_idx in range(N_PER_TASK):
            hs_for_slot = []
            for img_task in task_keys:
                sample = image_pool[img_task][slot_idx]
                inputs = get_eval_inputs(model, sample, task)
                out = two_pass_forward(model, inputs)
                hs_for_slot.append([out.hidden_states[l][0, -1, :].float().cpu()
                                    for l in range(NUM_LAYERS + 1)])
                del out; torch.cuda.empty_cache()
            for l in range(NUM_LAYERS + 1):
                for i in range(len(hs_for_slot)):
                    for j in range(i + 1, len(hs_for_slot)):
                        img_cos_per_layer[l].append(F.cosine_similarity(
                            hs_for_slot[i][l].unsqueeze(0),
                            hs_for_slot[j][l].unsqueeze(0)).item())

        # Same image, different instructions
        instr_cos_per_layer = [[] for _ in range(NUM_LAYERS + 1)]
        for test_sample in image_pool[task]:
            hs_for_anchor = []
            for other_task in task_keys:
                inputs = get_eval_inputs(model, test_sample, other_task)
                out = two_pass_forward(model, inputs)
                hs_for_anchor.append([out.hidden_states[l][0, -1, :].float().cpu()
                                      for l in range(NUM_LAYERS + 1)])
                del out; torch.cuda.empty_cache()
            for l in range(NUM_LAYERS + 1):
                for i in range(len(hs_for_anchor)):
                    for j in range(i + 1, len(hs_for_anchor)):
                        instr_cos_per_layer[l].append(F.cosine_similarity(
                            hs_for_anchor[i][l].unsqueeze(0),
                            hs_for_anchor[j][l].unsqueeze(0)).item())

        img_cos_mean = [float(np.mean(img_cos_per_layer[l])) for l in range(NUM_LAYERS + 1)]
        instr_cos_mean = [float(np.mean(instr_cos_per_layer[l])) for l in range(NUM_LAYERS + 1)]
        image_effect.append(img_cos_mean); instr_effect.append(instr_cos_mean)
        print(f"  Task {task_idx} ({task[:50]})")
        print(f"    L{NUM_LAYERS}: diff images cos={img_cos_mean[-1]:.4f}, "
              f"diff instrs cos={instr_cos_mean[-1]:.4f}")

    avg_image = np.mean(image_effect, axis=0)
    avg_instr = np.mean(instr_effect, axis=0)
    print(f"\n{'Layer':>6}  {'Diff images (same instr)':>25}  "
          f"{'Diff instrs (same image)':>25}  {'Diff':>8}")
    for l in range(NUM_LAYERS + 1):
        diff = avg_image[l] - avg_instr[l]
        print(f"  L{l:2d}:  {avg_image[l]:>25.4f}  {avg_instr[l]:>25.4f}  {diff:>+8.4f}")

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(range(NUM_LAYERS + 1), avg_image, "g-o", linewidth=3, markersize=7,
            label="Different images, same instruction")
    ax.plot(range(NUM_LAYERS + 1), avg_instr, "b-s", linewidth=3, markersize=7,
            label="Different instructions, same image")
    ax.set_xlabel("Layer", fontsize=22, labelpad=12)
    ax.set_ylabel("Pairwise cosine similarity", fontsize=22, labelpad=12)
    ax.set_xticks(range(0, NUM_LAYERS + 1, 4))
    ax.tick_params(axis="x", labelsize=18)
    ax.tick_params(axis="y", labelsize=18)
    ax.legend(fontsize=17, frameon=True, framealpha=1, edgecolor="black", loc="best")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_pdf = os.path.join(WORK_DIR, "image_instruction_similarity_projcrossattn.pdf")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()

    # =================================================================
    # Block B: per-task action L2
    # =================================================================
    print("\n=== Block B: per-task action comparison ===")
    all_results = []
    for img_task_idx, img_task in enumerate(task_keys):
        short = img_task.replace("pick up the black bowl ", "").replace(" and place it on the plate", "")
        task_samples = samples_by_task[img_task]
        step_size = max(1, len(task_samples) // N_IMAGES_PER_TASK_BLOCKB)
        test_indices = list(range(0, len(task_samples), step_size))[:N_IMAGES_PER_TASK_BLOCKB]

        correct_l2s, wrong_l2s, wrong_l2_to_correct = [], [], []
        for sample_idx in test_indices:
            test_sample = task_samples[sample_idx]
            gt = test_sample["action_gt"]
            actions = [predict_action_from_sample(model, test_sample, instr_task)
                       for instr_task in task_keys]
            correct_action = actions[img_task_idx]
            l2_correct = float(np.linalg.norm(correct_action[:6] - gt[:6]))
            l2_wrongs = [float(np.linalg.norm(actions[j][:6] - gt[:6]))
                         for j in range(len(task_keys)) if j != img_task_idx]
            l2_to_correct = [float(np.linalg.norm(correct_action - actions[j]))
                             for j in range(len(task_keys)) if j != img_task_idx]
            correct_l2s.append(l2_correct)
            wrong_l2s.append(float(np.mean(l2_wrongs)))
            wrong_l2_to_correct.append(float(np.mean(l2_to_correct)))

        all_results.append({
            "task": short,
            "correct_l2_mean": float(np.mean(correct_l2s)),
            "correct_l2_std":  float(np.std(correct_l2s)),
            "wrong_l2_mean":   float(np.mean(wrong_l2s)),
            "wrong_l2_std":    float(np.std(wrong_l2s)),
            "gap_mean":        float(np.mean(wrong_l2s) - np.mean(correct_l2s)),
            "l2_to_correct_mean": float(np.mean(wrong_l2_to_correct)),
            "l2_to_correct_std":  float(np.std(wrong_l2_to_correct)),
        })
        print(f"  Task {img_task_idx} ({short}): "
              f"correct L2={np.mean(correct_l2s):.4f}+/-{np.std(correct_l2s):.4f}, "
              f"wrong L2={np.mean(wrong_l2s):.4f}+/-{np.std(wrong_l2s):.4f}, "
              f"gap={np.mean(wrong_l2s) - np.mean(correct_l2s):.4f}, "
              f"L2 correct->wrong={np.mean(wrong_l2_to_correct):.4f}+/-{np.std(wrong_l2_to_correct):.4f}")

    print(f"\n{'=' * 70}")
    print(f"Action comparison ({N_IMAGES_PER_TASK_BLOCKB} images per task)")
    print(f"{'=' * 70}")
    print(f"{'Task':>30}  {'Correct L2':>12}  {'Wrong L2':>12}  {'Gap':>8}  {'L2 corr->wrong':>14}")
    for r in all_results:
        print(f"  {r['task']:>28}  "
              f"{r['correct_l2_mean']:>10.4f}+/-{r['correct_l2_std']:.3f}  "
              f"{r['wrong_l2_mean']:>10.4f}+/-{r['wrong_l2_std']:.3f}  "
              f"{r['gap_mean']:>8.4f}  "
              f"{r['l2_to_correct_mean']:>12.4f}+/-{r['l2_to_correct_std']:.3f}")

    sorted_results = sorted(all_results, key=lambda x: x["gap_mean"])
    print(f"\nRanked by instruction importance (gap = wrong L2 - correct L2):")
    for r in sorted_results:
        print(f"  {r['task']:>28}: gap={r['gap_mean']:.4f}")

    out_npz = os.path.join(WORK_DIR, "diagnostic_projector_xattn_results.npz")
    np.savez(out_npz,
             avg_image=avg_image, avg_instr=avg_instr,
             image_effect=np.array(image_effect),
             instr_effect=np.array(instr_effect),
             task_keys=np.array(task_keys, dtype=object),
             action_results=np.array(all_results, dtype=object))
    print(f"\nSaved: {out_pdf}")
    print(f"Saved: {out_npz}")


if __name__ == "__main__":
    main()
