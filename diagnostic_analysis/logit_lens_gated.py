"""
logit_lens_gated.py

Decode what the gated ReasonVLA "sees" at each image patch by projecting the
LLM hidden states at image-patch positions (layer 24, the gated model's
hidden_layer) through the LM head. Top-k vocab tokens per patch.

Usage:
  python logit_lens_gated.py --image_path /path/to/frame.png \\
      --instruction "pick up the black bowl in the top drawer..." \\
      [--out_dir ./logit_lens_out]
"""

import argparse
import os
import sys
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf

ALVIS_HOME = os.environ.get("ALVIS_HOME", "/cephyr/users/mariakat/Alvis")
WORK_DIR = os.path.join(ALVIS_HOME, "openvla")
OPENVLA_REPO = os.path.join(WORK_DIR, "openvla_repo")
sys.path.insert(0, OPENVLA_REPO)
sys.path.insert(0, WORK_DIR)

from experiments.robot.openvla_utils import crop_and_resize
from reason_vla import ReasonVLA, disable_patch_embeds, override_vision_backbone
from prismatic.vla.action_tokenizer import ActionTokenizer  # noqa: F401  (registers module)

# Gated member config (from memory)
DEFAULT_GATED_CKPT = "/mimer/NOBACKUP/groups/robot_unforseen/mariakat/runs/reason_vla/reason-vla-6495121/stage1/checkpoint-4950.pth"
DEFAULT_BASE_MODEL = "openvla/openvla-7b-finetuned-libero-spatial"
HIDDEN_LAYER = 24
NUM_PATCHES = 256
GRID = 16  # 16 x 16


def preprocess_image(image_path, center_crop=True):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((224, 224), Image.LANCZOS)
    if center_crop:
        image_tf = tf.convert_to_tensor(np.array(image))
        orig_dtype = image_tf.dtype
        image_tf = tf.image.convert_image_dtype(image_tf, tf.float32)
        image_tf = crop_and_resize(image_tf, 0.9, batch_size=1)
        image_tf = tf.clip_by_value(image_tf, 0, 1)
        image_tf = tf.image.convert_image_dtype(image_tf, orig_dtype, saturate=True)
        image = Image.fromarray(image_tf.numpy()).convert("RGB")
    return image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, required=True,
                        help="Path to image file (PNG/JPG) to analyze")
    parser.add_argument("--instruction", type=str, required=True,
                        help="Task instruction (e.g. 'pick up the black bowl...')")
    parser.add_argument("--base_model", type=str, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--gated_ckpt", type=str, default=DEFAULT_GATED_CKPT)
    parser.add_argument("--hidden_layer", type=int, default=HIDDEN_LAYER)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--out_dir", type=str, default="./logit_lens_out")
    parser.add_argument("--include_pass2", action="store_true",
                        help="Also decode LLM hidden states AFTER feedback is applied (Pass 2)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading gated ReasonVLA from {args.gated_ckpt} (hl={args.hidden_layer})")
    model = ReasonVLA.from_finetuned(
        model_name=args.base_model,
        checkpoint_path=args.gated_ckpt,
        stage=1, lora_dir=None,
        hidden_layer=args.hidden_layer,
        feedback_mode="gated",
    )
    model.eval()
    tokenizer = model.processor.tokenizer
    lm_head = model.vla.language_model.get_output_embeddings()

    print(f"Preprocessing image: {args.image_path}")
    image = preprocess_image(args.image_path, center_crop=True)
    image.save(os.path.join(args.out_dir, "input.png"))

    # Build prompt the same way ReasonVLA.generate does
    from prismatic.models.backbones.llm.prompting import PurePromptBuilder, VicunaV15ChatPromptBuilder
    model_name = getattr(model, "model_name", "")
    pb_fn = VicunaV15ChatPromptBuilder if "v01" in model_name else PurePromptBuilder
    pb = pb_fn("openvla")
    pb.add_turn("human", f"What action should the robot take to {args.instruction.lower()}?")
    prompt_text = pb.get_prompt()

    inputs = model.processor(prompt_text, image).to(model.vla.device, dtype=torch.bfloat16)
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask", None)
    pixel_values = inputs["pixel_values"]

    # ============================================================
    #  PASS 1: vanilla forward, extract hidden states at image positions
    # ============================================================
    print("Pass 1: vanilla forward with output_hidden_states=True ...")
    with torch.inference_mode():
        out = model.vla(
            input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values,
            labels=None, output_hidden_states=True, return_dict=True,
        )
    h_pass1 = out.hidden_states[args.hidden_layer]      # [1, seq, D]
    image_reasoning_pass1 = h_pass1[:, 1:1 + NUM_PATCHES, :].clone()
    print(f"  image_reasoning (Pass 1) shape: {tuple(image_reasoning_pass1.shape)}")

    decode_and_save(image_reasoning_pass1, lm_head, tokenizer,
                    image, args, suffix="pass1",
                    title_note="Pass 1 (pre-feedback LLM view)")

    # ============================================================
    #  PASS 2: re-run with feedback applied, decode hidden states again
    # ============================================================
    if args.include_pass2:
        print("\nPass 2: applying gated feedback and re-running ...")

        # Trigger the visual_reasoner+unmerger to populate hints from Pass 1's image_reasoning
        model.set_image_reasoning(image_reasoning_pass1)
        if not hasattr(model, "_hint_main") or model._hint_main is None:
            raise RuntimeError("Hints were not populated by set_image_reasoning")

        vb = model.vla.vision_backbone
        # Replicate generate()'s Pass 2 patch_features computation
        if model.is_fused:
            img_main, img_fused = torch.split(pixel_values, [3, 3], dim=1)
            main_patches  = vb.featurizer.patch_embed(img_main)
            fused_patches = vb.fused_featurizer.patch_embed(img_fused)
            # gated feedback: patches + gate * hint
            main_patches  = main_patches  + model._gate_main.to(main_patches)   * model._hint_main.to(main_patches)
            fused_patches = fused_patches + model._gate_fused.to(fused_patches) * model._hint_fused.to(fused_patches)
            with disable_patch_embeds(vb):
                main_features  = vb.featurizer(main_patches)
                fused_features = vb.fused_featurizer(fused_patches)
            patch_features = torch.cat([main_features, fused_features], dim=2)
        else:
            patches = vb.featurizer.patch_embed(pixel_values)
            patches = patches + model._gate_main.to(patches) * model._hint_main.to(patches)
            with disable_patch_embeds(vb):
                patch_features = vb.featurizer(patches)

        # Re-run vla with the feedback-augmented ViT output and capture hidden states
        with torch.inference_mode():
            with override_vision_backbone(vb, patch_features):
                out2 = model.vla(
                    input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values,
                    labels=None, output_hidden_states=True, return_dict=True,
                )

        h_pass2 = out2.hidden_states[args.hidden_layer]
        image_reasoning_pass2 = h_pass2[:, 1:1 + NUM_PATCHES, :].clone()
        print(f"  image_reasoning (Pass 2) shape: {tuple(image_reasoning_pass2.shape)}")

        decode_and_save(image_reasoning_pass2, lm_head, tokenizer,
                        image, args, suffix="pass2",
                        title_note="Pass 2 (post-feedback LLM view)")

        # Side-by-side comparison figure
        _save_side_by_side(args.out_dir, image, args.hidden_layer, args.instruction)


def decode_and_save(image_reasoning, lm_head, tokenizer, image, args,
                    suffix, title_note=""):
    """Run logit lens on a [1, 256, D] hidden-state slice; save grid + text + keyword heatmaps."""
    with torch.inference_mode():
        logits = lm_head(image_reasoning)                       # [1, 256, V] (bf16)
    logits = logits.float()
    topk_idx = logits[0].topk(args.topk, dim=-1).indices.cpu().numpy()
    topk_val = logits[0].topk(args.topk, dim=-1).values.cpu().numpy()

    grid_top1 = np.empty((GRID, GRID), dtype=object)
    for pid in range(NUM_PATCHES):
        r, c = pid // GRID, pid % GRID
        token_str = tokenizer.decode([int(topk_idx[pid, 0])]).strip() or "_"
        grid_top1[r, c] = token_str

    print(f"\n[{suffix}] Top-1 token per patch (16x16 grid):")
    for r in range(GRID):
        print(" | ".join(f"{grid_top1[r, c]:>10s}" for c in range(GRID)))

    txt_path = os.path.join(args.out_dir, f"topk_per_patch_{suffix}.txt")
    with open(txt_path, "w") as f:
        f.write(f"Image: {args.image_path}\nInstruction: {args.instruction}\n")
        f.write(f"Hidden layer: {args.hidden_layer}    Suffix: {suffix}\n\n")
        for pid in range(NUM_PATCHES):
            r, c = pid // GRID, pid % GRID
            tokens = [tokenizer.decode([int(t)]).strip() for t in topk_idx[pid]]
            f.write(f"patch ({r:2d},{c:2d}) top{args.topk}: {tokens}\n")

    img_np = np.array(image)
    H, W = img_np.shape[:2]
    cell_h, cell_w = H / GRID, W / GRID
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(img_np)
    for pid in range(NUM_PATCHES):
        r, c = pid // GRID, pid % GRID
        ax.text(c * cell_w + cell_w / 2, r * cell_h + cell_h / 2, grid_top1[r, c],
                ha="center", va="center", fontsize=7, color="yellow",
                bbox=dict(facecolor="black", alpha=0.4, pad=0.5, edgecolor="none"))
    ax.set_title(f"Logit lens [{suffix}] {title_note}\nlayer {args.hidden_layer}\n{args.instruction}",
                 fontsize=10)
    ax.axis("off")
    out_png = os.path.join(args.out_dir, f"logit_lens_grid_{suffix}.png")
    fig.tight_layout(); fig.savefig(out_png, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {out_png}")

    # Keyword heatmaps
    keywords = ["bowl", "plate", "drawer", "table", "ramekin"]
    keyword_token_ids = {}
    for kw in keywords:
        ids = tokenizer.encode(" " + kw, add_special_tokens=False)
        if ids:
            keyword_token_ids[kw] = ids[0]
    if keyword_token_ids:
        fig, axes = plt.subplots(1, len(keyword_token_ids),
                                 figsize=(4 * len(keyword_token_ids), 5))
        if len(keyword_token_ids) == 1:
            axes = [axes]
        for ax, (kw, tid) in zip(axes, keyword_token_ids.items()):
            heat = logits[0, :, tid].float().cpu().numpy().reshape(GRID, GRID)
            im = ax.imshow(heat, cmap="viridis")
            ax.set_title(f"'{kw}'"); ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.046)
        fig.suptitle(f"Keyword logits [{suffix}] {title_note}", fontsize=10)
        fig.tight_layout()
        kw_png = os.path.join(args.out_dir, f"logit_lens_keywords_{suffix}.png")
        fig.savefig(kw_png, dpi=140, bbox_inches="tight"); plt.close(fig)
        print(f"  Saved: {kw_png}")


def _save_side_by_side(out_dir, image, hidden_layer, instruction):
    """Stitch pass1 and pass2 grids into one comparison PNG."""
    p1 = os.path.join(out_dir, "logit_lens_grid_pass1.png")
    p2 = os.path.join(out_dir, "logit_lens_grid_pass2.png")
    if not (os.path.exists(p1) and os.path.exists(p2)):
        return
    im1 = plt.imread(p1); im2 = plt.imread(p2)
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    axes[0].imshow(im1); axes[0].axis("off"); axes[0].set_title("Pass 1: pre-feedback")
    axes[1].imshow(im2); axes[1].axis("off"); axes[1].set_title("Pass 2: post-feedback")
    fig.suptitle(f"Gated ReasonVLA logit lens, layer {hidden_layer}\n{instruction}", fontsize=11)
    fig.tight_layout()
    out_png = os.path.join(out_dir, "logit_lens_compare.png")
    fig.savefig(out_png, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {out_png}")


if __name__ == "__main__":
    main()
