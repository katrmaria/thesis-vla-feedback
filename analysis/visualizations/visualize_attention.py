"""
visualize_attention.py

Show which image regions matter for action prediction.
Works for vanilla OpenVLA and ReasonVLA (where attention reflects the feedback-modified features).

For vanilla:  image → ViT → projector → LLM → extract attention to image tokens
For ReasonVLA: image → ViT(+hint) → projector → LLM → extract attention to image tokens

Usage:
    # Vanilla OpenVLA
    python visualize_attention.py \
        --base_model openvla/openvla-7b-finetuned-libero-spatial \
        --task_suite_name libero_spatial --task_index 0 \
        --output_dir $HOME/openvla/attention_viz

    # ReasonVLA
    python visualize_attention.py \
        --base_model openvla/openvla-7b-finetuned-libero-spatial \
        --checkpoint_path /path/to/checkpoint.pth \
        --stage 1 --hidden_layer -1 --feedback_mode additive \
        --task_suite_name libero_spatial --task_index 0 \
        --output_dir $HOME/openvla/attention_viz

    # Multiple tasks (same scene, same model)
    python visualize_attention.py \
        --base_model openvla/openvla-7b-finetuned-libero-spatial \
        --task_suite_name libero_spatial --all_tasks \
        --output_dir $HOME/openvla/attention_viz
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import torch
from PIL import Image as PILImage

# ---- Path setup ----
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ALVIS_HOME = os.environ.get("ALVIS_HOME", SCRIPT_DIR)
WORK_DIR = os.path.join(ALVIS_HOME, "openvla") if os.path.isdir(os.path.join(ALVIS_HOME, "openvla")) else SCRIPT_DIR
OPENVLA_REPO = os.path.join(WORK_DIR, "openvla_repo") if os.path.isdir(os.path.join(WORK_DIR, "openvla_repo")) else WORK_DIR

for p in [OPENVLA_REPO, WORK_DIR, SCRIPT_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

NUM_IMAGE_TOKENS = 256
GRID_SIZE = 16
IMAGE_START = 1  # after BOS


# ---- Image preprocessing ----

def crop_and_resize_tf(image, crop_scale, batch_size):
    expanded_dims = len(image.shape) == 3
    if expanded_dims:
        image = tf.expand_dims(image, axis=0)
    s = tf.sqrt(crop_scale)
    new_h = tf.reshape(tf.clip_by_value(s, 0, 1), shape=(batch_size,))
    new_w = tf.reshape(tf.clip_by_value(s, 0, 1), shape=(batch_size,))
    h_off = (1 - new_h) / 2
    w_off = (1 - new_w) / 2
    boxes = tf.stack([h_off, w_off, h_off + new_h, w_off + new_w], axis=1)
    image = tf.image.crop_and_resize(image, boxes, tf.range(batch_size), (224, 224))
    if expanded_dims:
        image = image[0]
    return image


def preprocess_image(image_np, center_crop=True):
    image_pil = PILImage.fromarray(image_np).convert("RGB")
    if center_crop:
        image_tf = tf.convert_to_tensor(np.array(image_pil))
        orig_dtype = image_tf.dtype
        image_tf = tf.image.convert_image_dtype(image_tf, tf.float32)
        image_tf = crop_and_resize_tf(image_tf, crop_scale=0.9, batch_size=1)
        image_tf = tf.clip_by_value(image_tf, 0, 1)
        image_tf = tf.image.convert_image_dtype(image_tf, orig_dtype, saturate=True)
        image_pil = PILImage.fromarray(image_tf.numpy()).convert("RGB")
    return image_pil


# ---- Get LIBERO frame ----

def get_libero_frame(task_suite_name, task_index):
    from libero.libero import benchmark
    from experiments.robot.libero.libero_utils import (
        get_libero_env as _get_env, get_libero_image, get_libero_dummy_action,
    )
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[task_suite_name]()
    task = task_suite.get_task(task_index)
    env, task_desc = _get_env(task, "openvla", resolution=256)
    env.seed(42)
    env.reset()
    obs, _, _, _ = env.step(get_libero_dummy_action("openvla"))
    image_np = get_libero_image(obs, resize_size=256)
    env.close()
    return image_np, task_desc


# ---- Tokenize ----

def tokenize(processor, image_pil, task_description):
    prompt = f"In: What action should the robot take to {task_description.lower()}?\nOut:"
    inputs = processor(prompt, image_pil)
    inputs_cuda = {
        k: v.to(device=DEVICE, dtype=torch.bfloat16) if k == 'pixel_values' else v.to(DEVICE)
        for k, v in inputs.items()
    }
    # Add empty token 29871 if needed
    if inputs_cuda['input_ids'][0, -1].item() != 29871:
        inputs_cuda['input_ids'] = torch.cat(
            (inputs_cuda['input_ids'], torch.tensor([[29871]]).to(DEVICE)), dim=1
        )
        if 'attention_mask' in inputs_cuda:
            inputs_cuda['attention_mask'] = torch.cat(
                (inputs_cuda['attention_mask'],
                 torch.ones((1, 1), dtype=inputs_cuda['attention_mask'].dtype).to(DEVICE)),
                dim=1
            )
    return inputs_cuda


# ---- Extract attention: vanilla ----

@torch.inference_mode()
def get_attention_vanilla(vla, processor, image_pil, task_description, layer=24):
    """
    Vanilla forward: image → ViT → projector → LLM with output_attentions.
    Returns attention map [16, 16] = last token attending to image patches.
    """
    inputs = tokenize(processor, image_pil, task_description)
    outputs = vla(
        input_ids=inputs['input_ids'],
        attention_mask=inputs.get('attention_mask'),
        pixel_values=inputs['pixel_values'],
        output_attentions=True,
        return_dict=True,
    )
    attn = outputs.attentions[layer]  # [batch, heads, seq, seq]
    # Last token attending to image patches, averaged over all heads
    attn_to_image = attn[0, :, -1, IMAGE_START:IMAGE_START + NUM_IMAGE_TOKENS]
    attn_map = attn_to_image.float().mean(dim=0).reshape(GRID_SIZE, GRID_SIZE).cpu().numpy()

    del outputs
    torch.cuda.empty_cache()
    return attn_map


# ---- Extract attention: ReasonVLA Pass 2 ----

@torch.inference_mode()
def get_attention_reasonvla(model, image_pil, task_description, layer=24):
    """
    ReasonVLA two-pass:
      Pass 1: extract hidden states → compute hint
      Pass 2: modified features → LLM with output_attentions
    Returns attention map [16, 16] from Pass 2.
    """
    from reason_vla import disable_patch_embeds, override_vision_backbone

    model.reset_image_reasoning()

    inputs = tokenize(model.processor, image_pil, task_description)
    input_ids = inputs['input_ids']
    attention_mask = inputs.get('attention_mask')
    pixel_values = inputs['pixel_values']

    # === Pass 1: get hidden states ===
    output1 = model.vla(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pixel_values=pixel_values,
        labels=None,
        output_hidden_states=True,
        return_dict=True,
    )
    hidden_state = output1.hidden_states[model.hidden_layer]
    image_reasoning = hidden_state[:, 1:1 + model.num_patches, :]
    model.set_image_reasoning(image_reasoning)

    del output1, hidden_state, image_reasoning
    torch.cuda.empty_cache()

    # === Pass 2: modified features + attention ===
    vb = model.vla.vision_backbone

    if model.is_fused:
        img_main, img_fused = torch.split(pixel_values, [3, 3], dim=1)
        main_patches = vb.featurizer.patch_embed(img_main)
        fused_patches = vb.fused_featurizer.patch_embed(img_fused)

        # Apply feedback
        if model.feedback_mode == "additive":
            main_patches = main_patches + model._hint_main.to(main_patches)
            fused_patches = fused_patches + model._hint_fused.to(fused_patches)
        elif model.feedback_mode == "film":
            main_patches = (1 + model._gamma_main.to(main_patches)) * main_patches + model._beta_main.to(main_patches)
            fused_patches = (1 + model._gamma_fused.to(fused_patches)) * fused_patches + model._beta_fused.to(fused_patches)
        elif model.feedback_mode == "gated":
            main_patches = main_patches + model._gate_main.to(main_patches) * model._hint_main.to(main_patches)
            fused_patches = fused_patches + model._gate_fused.to(fused_patches) * model._hint_fused.to(fused_patches)
        elif model.feedback_mode == "adaln":
            main_patches = model._adaln(main_patches, model._gamma_main, model._beta_main)
            fused_patches = model._adaln(fused_patches, model._gamma_fused, model._beta_fused)
        elif model.feedback_mode == "scaled":
            main_patches = model._scaled_hint(main_patches, model._hint_main, model.hint_alpha)
            fused_patches = model._scaled_hint(fused_patches, model._hint_fused, model.hint_alpha_fused)

        with disable_patch_embeds(vb):
            main_features = vb.featurizer(main_patches)
            fused_features = vb.fused_featurizer(fused_patches)
        patch_features = torch.cat([main_features, fused_features], dim=2)
    else:
        patches = vb.featurizer.patch_embed(pixel_values)
        if model.feedback_mode == "additive":
            patches = patches + model._hint_main.to(patches)
        elif model.feedback_mode == "film":
            patches = (1 + model._gamma_main.to(patches)) * patches + model._beta_main.to(patches)
        elif model.feedback_mode == "gated":
            patches = patches + model._gate_main.to(patches) * model._hint_main.to(patches)
        elif model.feedback_mode == "adaln":
            patches = model._adaln(patches, model._gamma_main, model._beta_main)
        elif model.feedback_mode == "scaled":
            patches = model._scaled_hint(patches, model._hint_main, model.hint_alpha)
        with disable_patch_embeds(vb):
            patch_features = vb.featurizer(patches)

    # Forward with modified features + attention extraction
    with override_vision_backbone(vb, patch_features):
        output2 = model.vla(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,  # ignored by override
            output_attentions=True,
            return_dict=True,
        )

    attn = output2.attentions[layer]
    attn_to_image = attn[0, :, -1, IMAGE_START:IMAGE_START + NUM_IMAGE_TOKENS]
    attn_map = attn_to_image.float().mean(dim=0).reshape(GRID_SIZE, GRID_SIZE).cpu().numpy()

    del output2
    torch.cuda.empty_cache()
    return attn_map


# ---- Visualization ----

def overlay_attention(ax, image_pil, attn_map, title=""):
    attn_norm = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
    attn_resized = np.array(PILImage.fromarray(
        (attn_norm * 255).astype(np.uint8)
    ).resize(image_pil.size, PILImage.BILINEAR)) / 255

    ax.imshow(image_pil)
    ax.imshow(attn_resized, cmap='hot', alpha=0.5)
    ax.set_title(title, fontsize=9)
    ax.axis('off')


def visualize_single_task(image_pil, attn_map, task_desc, layers_maps, output_path):
    """
    Show image + attention overlay for multiple layers.
    layers_maps: dict {layer_idx: attn_map}
    """
    n = len(layers_maps) + 1
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))

    axes[0].imshow(image_pil)
    axes[0].set_title(f"Task: {task_desc[:40]}...", fontsize=8)
    axes[0].axis('off')

    for i, (layer_idx, amap) in enumerate(sorted(layers_maps.items())):
        overlay_attention(axes[i + 1], image_pil, amap, title=f"Layer {layer_idx}")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def visualize_all_tasks(images_and_maps, output_path, model_name=""):
    """
    Grid: rows = tasks, columns = [image, layer1, layer2, ...]
    images_and_maps: list of (image_pil, task_desc, {layer: attn_map})
    """
    n_tasks = len(images_and_maps)
    layers = sorted(images_and_maps[0][2].keys())
    n_cols = 1 + len(layers)

    fig, axes = plt.subplots(n_tasks, n_cols, figsize=(4 * n_cols, 3.5 * n_tasks))
    if n_tasks == 1:
        axes = axes.reshape(1, -1)

    for row, (image_pil, task_desc, layers_maps) in enumerate(images_and_maps):
        axes[row, 0].imshow(image_pil)
        short = task_desc[:35] + "..." if len(task_desc) > 35 else task_desc
        axes[row, 0].set_title(f"T{row}: {short}", fontsize=7)
        axes[row, 0].axis('off')

        for col, layer_idx in enumerate(layers):
            overlay_attention(axes[row, col + 1], image_pil, layers_maps[layer_idx],
                            title=f"L{layer_idx}" if row == 0 else "")

    fig.suptitle(f"Attention to Image Regions — {model_name}", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---- Model loading ----

def load_model(args):
    from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
    from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
    from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
    from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    if args.checkpoint_path is not None:
        # ReasonVLA — load with eager attention
        from reason_vla import ReasonVLA
        from prismatic.vla.action_tokenizer import ActionTokenizer
        print(f"Loading ReasonVLA (stage {args.stage}, hl={args.hidden_layer}, {args.feedback_mode})...")
        print("  Using eager attention to extract attention weights")

        vla = AutoModelForVision2Seq.from_pretrained(
            args.base_model,
            attn_implementation="eager",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).to(DEVICE)

        model = ReasonVLA(vla, hidden_layer=args.hidden_layer, feedback_mode=args.feedback_mode)
        model.stage = args.stage
        model.model_name = args.base_model
        model.processor = AutoProcessor.from_pretrained(args.base_model, trust_remote_code=True)
        model.action_tokenizer = ActionTokenizer(model.processor.tokenizer)

        if args.stage == 2 and args.lora_dir:
            model.vla.load_adapter(args.lora_dir)

        model.load_reasoning_modules(args.checkpoint_path)
        for p in model.get_reasoning_parameters():
            p.data = p.data.to(device=DEVICE, dtype=torch.bfloat16)

        model.vla.eval()
        model.eval()
        return "reasonvla", model

    else:
        # Vanilla OpenVLA — eager attention
        print(f"Loading vanilla OpenVLA: {args.base_model}")
        vla = AutoModelForVision2Seq.from_pretrained(
            args.base_model,
            attn_implementation="eager",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).to(DEVICE)
        vla.eval()
        processor = AutoProcessor.from_pretrained(args.base_model, trust_remote_code=True)
        return "vanilla", (vla, processor)


def get_attention(model_type, model_or_tuple, image_pil, task_desc, layer):
    """Dispatch to vanilla or reasonvla attention extraction."""
    if model_type == "vanilla":
        vla, processor = model_or_tuple
        return get_attention_vanilla(vla, processor, image_pil, task_desc, layer)
    else:
        return get_attention_reasonvla(model_or_tuple, image_pil, task_desc, layer)


# ---- Main ----

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--stage", type=int, default=1)
    parser.add_argument("--lora_dir", type=str, default=None)
    parser.add_argument("--hidden_layer", type=int, default=-1)
    parser.add_argument("--feedback_mode", type=str, default="additive")
    parser.add_argument("--task_suite_name", type=str, default="libero_spatial")
    parser.add_argument("--task_index", type=int, default=0)
    parser.add_argument("--all_tasks", action="store_true", help="Run all tasks in the suite")
    parser.add_argument("--layers", type=int, nargs="+", default=[14, 24],
                        help="LLM layers to extract attention from")
    parser.add_argument("--center_crop", action="store_true", default=True)
    parser.add_argument("--no_center_crop", action="store_true")
    parser.add_argument("--output_dir", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.no_center_crop:
        args.center_crop = False
    os.makedirs(args.output_dir, exist_ok=True)

    model_type, model = load_model(args)
    model_name = "ReasonVLA" if model_type == "reasonvla" else "Vanilla OpenVLA"

    if args.all_tasks:
        from libero.libero import benchmark
        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite = benchmark_dict[args.task_suite_name]()
        n_tasks = task_suite.n_tasks

        results = []
        for ti in range(n_tasks):
            image_np, task_desc = get_libero_frame(args.task_suite_name, ti)
            image_pil = preprocess_image(image_np, center_crop=args.center_crop)
            print(f"Task {ti}: {task_desc}")

            layers_maps = {}
            for layer in args.layers:
                layers_maps[layer] = get_attention(model_type, model, image_pil, task_desc, layer)

            results.append((image_pil, task_desc, layers_maps))

        tag = "reasonvla" if model_type == "reasonvla" else "vanilla"
        output_path = os.path.join(args.output_dir, f"all_tasks_{args.task_suite_name}_{tag}.png")
        visualize_all_tasks(results, output_path, model_name)

    else:
        image_np, task_desc = get_libero_frame(args.task_suite_name, args.task_index)
        image_pil = preprocess_image(image_np, center_crop=args.center_crop)
        print(f"Task: {task_desc}")

        layers_maps = {}
        for layer in args.layers:
            print(f"  Extracting attention at layer {layer}...")
            layers_maps[layer] = get_attention(model_type, model, image_pil, task_desc, layer)

        tag = "reasonvla" if model_type == "reasonvla" else "vanilla"
        output_path = os.path.join(
            args.output_dir,
            f"attention_{args.task_suite_name}_t{args.task_index}_{tag}.png"
        )
        visualize_single_task(image_pil, None, task_desc, layers_maps, output_path)

    print("Done!")


if __name__ == "__main__":
    main()
