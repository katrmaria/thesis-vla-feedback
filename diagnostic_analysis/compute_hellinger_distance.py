"""
compute_hellinger_distance.py

Compute the Hellinger distance between each LLM layer's attention distribution
and the post-integrated layer (L27), following Song et al.'s methodology.

For each image, captures last-token-to-image-patch attention at all 32 layers,
averaged across heads. Then computes Hellinger distance to L27 for each layer.

This identifies the pre-integrated layer: the one whose attention is most
different from the post-integrated layer, meaning visual information has not
yet been integrated there.

Usage:
    python compute_hellinger_distance.py \
        --base_model openvla/openvla-7b-finetuned-libero-spatial \
        --task_suite_name libero_spatial \
        --num_images_per_task 10 \
        --post_layer 27 \
        --output_dir ./hellinger_results
"""

import argparse
import json
import logging
import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from libero.libero import benchmark

# ---- Path setup ----
ALVIS_HOME = os.environ.get("ALVIS_HOME", "/cephyr/users/mariakat/Alvis")
WORK_DIR = os.path.join(ALVIS_HOME, "openvla")
OPENVLA_REPO = os.path.join(WORK_DIR, "openvla_repo")
sys.path.insert(0, OPENVLA_REPO)
sys.path.insert(0, WORK_DIR)

from experiments.robot.libero.libero_utils import (
    get_libero_env, get_libero_image, get_libero_dummy_action,
)
from experiments.robot.robot_utils import set_seed_everywhere

from transformers import AutoModelForVision2Seq, AutoProcessor, AutoConfig, AutoImageProcessor
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.models.backbones.llm.prompting import PurePromptBuilder

logger = logging.getLogger(__name__)

NUM_PATCHES = 256
NUM_LAYERS = 32


def hellinger_distance(p, q):
    """
    Compute Hellinger distance between two probability distributions.
    H(P, Q) = (1/sqrt(2)) * sqrt(sum((sqrt(p_j) - sqrt(q_j))^2))
    """
    p = p.float().clamp(min=1e-10)
    q = q.float().clamp(min=1e-10)
    # Normalize to probability distributions
    p = p / p.sum(dim=-1, keepdim=True)
    q = q / q.sum(dim=-1, keepdim=True)
    return (1.0 / np.sqrt(2)) * torch.sqrt(((torch.sqrt(p) - torch.sqrt(q)) ** 2).sum(dim=-1))


def main():
    parser = argparse.ArgumentParser(description="Compute Hellinger distance for pre-integrated layer selection")
    parser.add_argument("--base_model", type=str, default="openvla/openvla-7b-finetuned-libero-spatial")
    parser.add_argument("--task_suite_name", type=str, default="libero_spatial")
    parser.add_argument("--num_images_per_task", type=int, default=10)
    parser.add_argument("--post_layer", type=int, default=27)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output_dir", type=str, default="./hellinger_results")
    parser.add_argument("--num_steps_wait", type=int, default=10)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    set_seed_everywhere(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Load model ----
    logger.info(f"Loading model from {args.base_model}")
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    vla = AutoModelForVision2Seq.from_pretrained(
        args.base_model,
        attn_implementation="eager",
        device_map="auto",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    # Force eager attention
    if hasattr(vla, "language_model") and hasattr(vla.language_model, "config"):
        vla.language_model.config._attn_implementation = "eager"
        for layer in vla.language_model.model.layers:
            if hasattr(layer, "self_attn"):
                layer.self_attn._attn_implementation = "eager"

    processor = AutoProcessor.from_pretrained(args.base_model, trust_remote_code=True)
    vla.eval()

    # ---- Get LLM layers ----
    if hasattr(vla, "language_model"):
        llm_layers = vla.language_model.model.layers
    else:
        llm_layers = vla.model.layers

    # ---- Set up attention capture for all layers ----
    captured_attentions = {}  # layer_idx -> [B, 256] per sample

    def make_capture_hook(layer_idx):
        original_forward = llm_layers[layer_idx].self_attn.forward

        def patched_forward(*args, **kwargs):
            kwargs["output_attentions"] = True
            outputs = original_forward(*args, **kwargs)
            attn_weights = outputs[1]
            if attn_weights is not None:
                # last-token -> image patches, averaged across heads
                patch_attn = attn_weights[:, :, -1, 1:1 + NUM_PATCHES].float().mean(dim=1)  # [B, 256]
                captured_attentions[layer_idx] = patch_attn.detach().cpu()
            return (outputs[0], None) + tuple(outputs[2:])

        return original_forward, patched_forward

    # ---- Initialize LIBERO ----
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks = task_suite.n_tasks
    logger.info(f"Task suite: {args.task_suite_name} ({num_tasks} tasks)")

    resize_size = 224

    # ---- Collect attention from images ----
    # all_attentions[layer_idx] = list of [256] tensors, one per image
    all_attentions = {l: [] for l in range(NUM_LAYERS)}
    total_images = 0

    for task_id in range(num_tasks):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = get_libero_env(task, "openvla", resolution=256)

        num_images = min(args.num_images_per_task, len(initial_states))
        logger.info(f"Task {task_id}: '{task_description}' — collecting {num_images} images")

        for ep_idx in range(num_images):
            env.reset()
            obs = env.set_init_state(initial_states[ep_idx])

            # Wait steps to get a meaningful observation
            for _ in range(args.num_steps_wait):
                obs, _, _, _ = env.step(get_libero_dummy_action("openvla"))

            img = get_libero_image(obs, resize_size)
            image = Image.fromarray(img).convert("RGB")

            # Build prompt
            prompt_builder = PurePromptBuilder("openvla")
            prompt_builder.add_turn("human", f"What action should the robot take to {task_description.lower()}?")
            prompt_text = prompt_builder.get_prompt()

            inputs = processor(prompt_text, image).to(vla.device, dtype=torch.bfloat16)

            # Patch all layers to capture attention
            originals = {}
            for l in range(NUM_LAYERS):
                orig, patched = make_capture_hook(l)
                originals[l] = orig
                llm_layers[l].self_attn.forward = patched

            captured_attentions.clear()

            # Run forward pass
            with torch.inference_mode():
                _ = vla(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                    pixel_values=inputs["pixel_values"],
                    labels=None,
                    output_attentions=False,
                    return_dict=True,
                )

            # Restore all layers
            for l in range(NUM_LAYERS):
                llm_layers[l].self_attn.forward = originals[l]

            # Store captured attentions
            for l in range(NUM_LAYERS):
                if l in captured_attentions:
                    all_attentions[l].append(captured_attentions[l].squeeze(0))  # [256]

            total_images += 1
            if total_images % 10 == 0:
                logger.info(f"  Processed {total_images} images")

        env.close()

    logger.info(f"Total images collected: {total_images}")

    # ---- Compute Hellinger distances to post-integrated layer ----
    post_layer = args.post_layer
    post_attn = torch.stack(all_attentions[post_layer])  # [N, 256]

    distances = {}
    per_sample_distances = {}

    for l in range(NUM_LAYERS):
        if l == post_layer:
            distances[l] = 0.0
            continue
        layer_attn = torch.stack(all_attentions[l])  # [N, 256]
        # Compute Hellinger distance per sample, then average
        h_dists = hellinger_distance(layer_attn, post_attn)  # [N]
        distances[l] = h_dists.mean().item()
        per_sample_distances[l] = h_dists.tolist()

    # ---- Print results ----
    print("\n" + "=" * 60)
    print(f"Hellinger distance to post-integrated layer L{post_layer}")
    print(f"Computed over {total_images} images from {args.task_suite_name}")
    print("=" * 60)

    sorted_layers = sorted(distances.items(), key=lambda x: x[1], reverse=True)
    for l, d in sorted_layers:
        marker = ""
        if l == post_layer:
            marker = " (post-integrated)"
        elif d == sorted_layers[0][1]:
            marker = " ← BEST PRE-INTEGRATED"
        print(f"  Layer {l:2d}: H = {d:.4f}{marker}")

    best_layer = sorted_layers[0][0]
    print(f"\nRecommended pre-integrated layer: L{best_layer} (H = {sorted_layers[0][1]:.4f})")

    # ---- Also show by phase ----
    print("\n--- By phase ---")
    phases = {
        "Fusion (0-1)": range(0, 2),
        "Language (2-7)": range(2, 8),
        "Transition (8-9)": range(8, 10),
        "Grounding (10-15)": range(10, 16),
        "Post-grounding (16-26)": range(16, 27),
        "Post-integrated (27)": [27],
        "Review (28)": [28],
        "Final (29-31)": range(29, 32),
    }
    for phase_name, layer_range in phases.items():
        phase_dists = [distances.get(l, 0.0) for l in layer_range]
        avg_d = np.mean(phase_dists)
        max_d = max(phase_dists)
        max_l = list(layer_range)[np.argmax(phase_dists)]
        print(f"  {phase_name:25s}  avg H = {avg_d:.4f}  max H = {max_d:.4f} (L{max_l})")

    # ---- Save results ----
    results = {
        "post_layer": post_layer,
        "task_suite": args.task_suite_name,
        "num_images": total_images,
        "distances": {str(l): d for l, d in distances.items()},
        "recommended_pre_layer": best_layer,
        "recommended_distance": sorted_layers[0][1],
    }
    results_path = os.path.join(args.output_dir, f"hellinger_{args.task_suite_name}_post{post_layer}.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {results_path}")

    # ---- Plot ----
    fig, ax = plt.subplots(figsize=(12, 5))
    layers = list(range(NUM_LAYERS))
    dists = [distances.get(l, 0.0) for l in layers]

    # Color by phase
    colors = []
    for l in layers:
        if l <= 1:
            colors.append("#2196F3")  # fusion - blue
        elif l <= 7:
            colors.append("#9E9E9E")  # language - gray
        elif l <= 9:
            colors.append("#FF9800")  # transition - orange
        elif l <= 15:
            colors.append("#4CAF50")  # grounding - green
        elif l <= 26:
            colors.append("#BDBDBD")  # post-grounding - light gray
        elif l == 27:
            colors.append("#F44336")  # post-integrated - red
        elif l == 28:
            colors.append("#9C27B0")  # review - purple
        else:
            colors.append("#795548")  # final - brown

    ax.bar(layers, dists, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Layer index", fontsize=12)
    ax.set_ylabel(f"Hellinger distance to L{post_layer}", fontsize=12)
    ax.set_title(f"Hellinger distance to post-integrated layer (L{post_layer}) — {args.task_suite_name}\n"
                 f"Recommended pre-integrated layer: L{best_layer} (H = {sorted_layers[0][1]:.4f})",
                 fontsize=12)
    ax.set_xticks(layers)
    ax.axhline(y=0, color="black", linewidth=0.5)

    # Mark the best
    ax.annotate(f"L{best_layer}", xy=(best_layer, sorted_layers[0][1]),
                xytext=(best_layer + 1.5, sorted_layers[0][1] + 0.02),
                arrowprops=dict(arrowstyle="->", color="red"),
                fontsize=11, color="red", fontweight="bold")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2196F3", label="Fusion (0-1)"),
        Patch(facecolor="#9E9E9E", label="Language (2-7)"),
        Patch(facecolor="#FF9800", label="Transition (8-9)"),
        Patch(facecolor="#4CAF50", label="Grounding (10-15)"),
        Patch(facecolor="#BDBDBD", label="Post-grounding (16-26)"),
        Patch(facecolor="#F44336", label="Post-integrated (27)"),
        Patch(facecolor="#9C27B0", label="Review (28)"),
        Patch(facecolor="#795548", label="Final (29-31)"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=8)

    plt.tight_layout()
    plot_path = os.path.join(args.output_dir, f"hellinger_{args.task_suite_name}_post{post_layer}.png")
    plt.savefig(plot_path, dpi=150)
    logger.info(f"Plot saved to {plot_path}")
    plt.close()


if __name__ == "__main__":
    main()
