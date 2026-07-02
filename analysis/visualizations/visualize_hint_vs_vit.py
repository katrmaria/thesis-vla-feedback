"""
visualize_hint_vs_vit.py    (F3)

Compare the per-patch magnitude of the additive feedback (||hint||) to the
per-patch magnitude of the ViT's own output features (||ViT(image)||).
If these heatmaps differ in spatial structure, the LLM is adding something
the ViT alone doesn't produce -- i.e., the LLM-processed image tokens
carry richer information than raw vision encoder output.

Needs GPU (loads the additive ReasonVLA model).

Usage:
  python visualize_hint_vs_vit.py \\
      --image_path /path/to/frame_step0.png \\
      --instruction "pick up the black bowl ..." \\
      --out_dir hint_vs_vit_t4
"""

import argparse
import os
import sys
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import cm
import tensorflow as tf

ALVIS_HOME = os.environ.get("ALVIS_HOME", "/cephyr/users/mariakat/Alvis")
WORK_DIR = os.path.join(ALVIS_HOME, "openvla")
OPENVLA_REPO = os.path.join(WORK_DIR, "openvla_repo")
sys.path.insert(0, OPENVLA_REPO)
sys.path.insert(0, WORK_DIR)

from experiments.robot.openvla_utils import crop_and_resize
from reason_vla import ReasonVLA
from prismatic.vla.action_tokenizer import ActionTokenizer  # noqa: F401

# Additive checkpoint (stage 2 with LoRA)
DEFAULT_ADDITIVE_CKPT = "/mimer/NOBACKUP/groups/robot_unforseen/mariakat/runs/reason_vla/reason-vla-6175887/stage2/checkpoint-3300.pth"
DEFAULT_ADDITIVE_LORA = "/mimer/NOBACKUP/groups/robot_unforseen/mariakat/runs/reason_vla/reason-vla-6175887/stage2/lora-3300"
DEFAULT_BASE_MODEL = "openvla/openvla-7b-finetuned-libero-spatial"
HIDDEN_LAYER = -1
NUM_PATCHES = 256
GRID = 16
IMAGE_SIZE = 224


def preprocess_image(image_path, center_crop=True):
    image = Image.open(image_path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
    if center_crop:
        image_tf = tf.convert_to_tensor(np.array(image))
        orig_dtype = image_tf.dtype
        image_tf = tf.image.convert_image_dtype(image_tf, tf.float32)
        image_tf = crop_and_resize(image_tf, 0.9, batch_size=1)
        image_tf = tf.clip_by_value(image_tf, 0, 1)
        image_tf = tf.image.convert_image_dtype(image_tf, orig_dtype, saturate=True)
        image = Image.fromarray(image_tf.numpy()).convert("RGB")
    return image


def grid_heatmap(vec256, size=IMAGE_SIZE):
    g = vec256.reshape(GRID, GRID).astype(np.float32)
    return np.array(Image.fromarray(g).resize((size, size), Image.BILINEAR))


def overlay(ax, frame, heatmap, title, vmin, vmax, alpha=0.55, cmap_name="viridis",
            title_fontsize=16):
    ax.imshow(frame)
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    rgba = plt.colormaps[cmap_name](norm(heatmap)); rgba[..., 3] = alpha
    ax.imshow(rgba)
    ax.set_title(title, fontsize=title_fontsize, fontweight="bold", color="black")
    ax.axis("off")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--instruction", type=str, required=True)
    parser.add_argument("--base_model", type=str, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--additive_ckpt", type=str, default=DEFAULT_ADDITIVE_CKPT)
    parser.add_argument("--additive_lora", type=str, default=DEFAULT_ADDITIVE_LORA)
    parser.add_argument("--additive_stage", type=int, default=2)
    parser.add_argument("--additive_hidden_layer", type=int, default=HIDDEN_LAYER)
    parser.add_argument("--out_dir", type=str, default="./hint_vs_vit")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading additive ReasonVLA (stage {args.additive_stage}, hl={args.additive_hidden_layer})")
    model = ReasonVLA.from_finetuned(
        model_name=args.base_model,
        checkpoint_path=args.additive_ckpt,
        stage=args.additive_stage,
        lora_dir=args.additive_lora,
        hidden_layer=args.additive_hidden_layer,
        feedback_mode="additive",
    )
    model.eval()

    image = preprocess_image(args.image_path, center_crop=True)
    img_np = np.array(image)

    from prismatic.models.backbones.llm.prompting import PurePromptBuilder, VicunaV15ChatPromptBuilder
    model_name = getattr(model, "model_name", "")
    pb_fn = VicunaV15ChatPromptBuilder if "v01" in model_name else PurePromptBuilder
    pb = pb_fn("openvla")
    pb.add_turn("human", f"What action should the robot take to {args.instruction.lower()}?")
    prompt_text = pb.get_prompt()

    inputs = model.processor(prompt_text, image).to(model.vla.device, dtype=torch.bfloat16)
    pixel_values = inputs["pixel_values"]

    # ---- (1) Hint: run pass 1, populate _hint_main ----
    model.reset_image_reasoning()
    with torch.inference_mode():
        out = model.vla(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            pixel_values=pixel_values,
            labels=None, output_hidden_states=True, return_dict=True,
        )
        h = out.hidden_states[model.hidden_layer]
        image_reasoning = h[:, 1:1 + NUM_PATCHES, :]
        model.set_image_reasoning(image_reasoning)

    hint_main = model._hint_main.float()             # [1, 256, 1024]
    hint_main_norm = hint_main.norm(dim=-1).squeeze(0).cpu().numpy()  # [256]
    hint_fused = model._hint_fused.float() if model.is_fused else None
    hint_fused_norm = (hint_fused.norm(dim=-1).squeeze(0).cpu().numpy()
                       if hint_fused is not None else None)

    # ---- (2) ViT-only patch features (baseline view of the image) ----
    vb = model.vla.vision_backbone
    with torch.inference_mode():
        if model.is_fused:
            img_main, img_fused = torch.split(pixel_values, [3, 3], dim=1)
            vit_main  = vb.featurizer(img_main).float()           # [1, 256, 1024]
            vit_fused = vb.fused_featurizer(img_fused).float()    # [1, 256, 1152]
        else:
            vit_main  = vb.featurizer(pixel_values).float()
            vit_fused = None

    vit_main_norm  = vit_main.norm(dim=-1).squeeze(0).cpu().numpy()
    vit_fused_norm = vit_fused.norm(dim=-1).squeeze(0).cpu().numpy() if vit_fused is not None else None

    # ---- Comparison stats ----
    def cos(a, b):
        a = a.flatten(); b = b.flatten()
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

    print("\nMagnitude comparison:")
    print(f"  ||hint||   main   min/max/mean: {hint_main_norm.min():.2f} / {hint_main_norm.max():.2f} / {hint_main_norm.mean():.2f}")
    print(f"  ||ViT||    main   min/max/mean: {vit_main_norm.min():.2f}  / {vit_main_norm.max():.2f}  / {vit_main_norm.mean():.2f}")
    print(f"  cos(hint_norm, vit_norm) main:  {cos(hint_main_norm, vit_main_norm):.4f}")
    print(f"  spatial similarity? lower = ViT and hint differ in WHERE they put magnitude")
    if vit_fused_norm is not None:
        print(f"  ||hint||  fused   min/max/mean: {hint_fused_norm.min():.2f} / {hint_fused_norm.max():.2f} / {hint_fused_norm.mean():.2f}")
        print(f"  ||ViT||   fused   min/max/mean: {vit_fused_norm.min():.2f}  / {vit_fused_norm.max():.2f}  / {vit_fused_norm.mean():.2f}")
        print(f"  cos(hint_norm, vit_norm) fused: {cos(hint_fused_norm, vit_fused_norm):.4f}")

    # ---- Figure: side-by-side  ||ViT|| vs ||hint||  for each stream ----
    rows = 2 if vit_fused_norm is not None else 1
    fig, axes = plt.subplots(rows, 4, figsize=(26, 7 * rows))
    if rows == 1: axes = axes[None, :]

    for ri, (label, vit_n, hint_n) in enumerate([
        ("DINOv2", vit_main_norm,  hint_main_norm),
        ("SigLIP", vit_fused_norm, hint_fused_norm),
    ][:rows]):
        axes[ri, 0].imshow(img_np)
        # 3-line title (blank middle line) so it aligns vertically with the other panels
        axes[ri, 0].set_title(f"{label}\nInput\n ", fontsize=18, fontweight="bold",
                              color="black")
        axes[ri, 0].axis("off")
        overlay(axes[ri, 1], img_np, grid_heatmap(vit_n),
                f"{label}\n||ViT(image)||\nmax = {vit_n.max():.1f}",
                vmin=float(vit_n.min()), vmax=float(vit_n.max()), title_fontsize=18)
        overlay(axes[ri, 2], img_np, grid_heatmap(hint_n),
                f"{label}\n||hint||\nmax = {hint_n.max():.1f}",
                vmin=float(hint_n.min()), vmax=float(hint_n.max()), title_fontsize=18)
        # Difference of normalized maps (purely spatial pattern, not magnitude)
        vit_norm  = (vit_n  - vit_n.min())  / (vit_n.max()  - vit_n.min()  + 1e-8)
        hint_norm = (hint_n - hint_n.min()) / (hint_n.max() - hint_n.min() + 1e-8)
        diff = hint_norm - vit_norm
        dmax = float(np.abs(diff).max())
        overlay(axes[ri, 3], img_np, grid_heatmap(diff),
                f"{label}\nNormalized difference\nred: hint>ViT,  blue: ViT>hint",
                vmin=-dmax, vmax=dmax, cmap_name="bwr", alpha=0.5, title_fontsize=18)

    fig.suptitle(f"LLM-derived feedback vs ViT patch features\n"
                 f"Task: {args.instruction}", fontsize=20, fontweight="bold",
                 color="black", y=0.97)
    fig.subplots_adjust(wspace=-0.12, hspace=0.28, top=0.82, bottom=0.05,
                        left=0.03, right=0.97)
    out = os.path.join(args.out_dir, "hint_vs_vit.png")
    fig.savefig(out, dpi=140)
    fig.savefig(out.replace(".png", ".pdf"))
    plt.close(fig)
    print(f"\nSaved: {out}  (+ .pdf)")

    # Save arrays for later analysis
    np.savez(os.path.join(args.out_dir, "hint_vs_vit.npz"),
             hint_main_norm=hint_main_norm, vit_main_norm=vit_main_norm,
             hint_fused_norm=hint_fused_norm if hint_fused_norm is not None else np.zeros(0),
             vit_fused_norm=vit_fused_norm if vit_fused_norm is not None else np.zeros(0))


if __name__ == "__main__":
    main()
