"""
visualize_hints_episode.py

For one episode of the ensemble logged eval, overlay the per-patch hint
magnitudes (hint_additive[*,0] = DINOv2-stream, hint_additive[*,1] = SigLIP-stream,
same for gated) on selected keyframes of the rollout video.

This is the direct visualization of what each ensemble member is feeding back
to the ViT at each timestep -- no logit lens needed.

Usage:
  python visualize_hints_episode.py \\
      --npz /path/to/ensemble_log_ep=1--success=True--task=...npz \\
      --mp4 /path/to/episode=1--success=True--task=...mp4 \\
      --keyframes 0,30,60,100 \\
      --out_dir hint_overlays_ep1
"""

import argparse
import os
import numpy as np
import imageio.v2 as imageio
import matplotlib.pyplot as plt
from matplotlib import cm
from PIL import Image

GRID = 16            # 16 * 16 = 256 patches
NUM_PATCHES = 256
IMAGE_SIZE = 224     # what the model sees


def load_frame(mp4_path, step):
    reader = imageio.get_reader(mp4_path)
    frame = reader.get_data(step)
    reader.close()
    return frame  # HxWx3 uint8


def hint_to_heatmap(hint_vec, size=IMAGE_SIZE):
    """[256] -> [size, size] upsampled."""
    grid = hint_vec.reshape(GRID, GRID)
    img = Image.fromarray(grid.astype(np.float32))
    img = img.resize((size, size), Image.BILINEAR)
    return np.array(img)


def overlay(ax, frame, heatmap, title, vmin=None, vmax=None):
    # Show base image
    ax.imshow(frame)
    # Heatmap with alpha
    norm = plt.Normalize(vmin=vmin if vmin is not None else heatmap.min(),
                         vmax=vmax if vmax is not None else heatmap.max())
    rgba = cm.viridis(norm(heatmap))
    rgba[..., 3] = 0.55  # alpha
    ax.imshow(rgba)
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=str, required=True)
    parser.add_argument("--mp4", type=str, required=True)
    parser.add_argument("--keyframes", type=str, default="0,30,60,100",
                        help="Comma-separated step indices to visualize")
    parser.add_argument("--out_dir", type=str, default="hint_overlays")
    parser.add_argument("--per_member_norm", action="store_true",
                        help="Normalize heatmap per-member instead of global (sharper)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    d = np.load(args.npz, allow_pickle=True)
    print(f"Episode T={int(d['T'])}, success={bool(d['success'])}")
    print(f"Task: {d['task']}")

    hint_add = d["hint_additive"].astype(np.float32)  # [T, 2, 256]
    hint_gat = d["hint_gated"].astype(np.float32)     # [T, 2, 256]

    keyframes = [int(x) for x in args.keyframes.split(",") if x.strip()]
    keyframes = [k for k in keyframes if k < int(d["T"])]

    # Choose color scaling
    if args.per_member_norm:
        scale = {
            "add_main":  (hint_add[:, 0, :].min(),  hint_add[:, 0, :].max()),
            "add_fused": (hint_add[:, 1, :].min(),  hint_add[:, 1, :].max()),
            "gat_main":  (hint_gat[:, 0, :].min(),  hint_gat[:, 0, :].max()),
            "gat_fused": (hint_gat[:, 1, :].min(),  hint_gat[:, 1, :].max()),
        }
    else:
        # Single global range so the four columns are directly comparable
        all_vals = np.concatenate([hint_add.flatten(), hint_gat.flatten()])
        gmin, gmax = float(all_vals.min()), float(all_vals.max())
        scale = {k: (gmin, gmax) for k in ["add_main", "add_fused", "gat_main", "gat_fused"]}

    n_rows = len(keyframes)
    n_cols = 5  # frame, add_main, add_fused, gat_main, gat_fused
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    if n_rows == 1:
        axes = axes[None, :]

    for ri, step in enumerate(keyframes):
        frame = load_frame(args.mp4, step)
        if frame.shape[:2] != (IMAGE_SIZE, IMAGE_SIZE):
            frame = np.array(Image.fromarray(frame).resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR))

        # Column 0: original frame
        axes[ri, 0].imshow(frame)
        axes[ri, 0].set_title(f"Step {step}", fontsize=10)
        axes[ri, 0].axis("off")

        # Column 1-4: hint overlays
        for ci, (name, hint_arr, idx) in enumerate([
            ("additive  hint_main",  hint_add, 0),
            ("additive  hint_fused", hint_add, 1),
            ("gated  hint_main",     hint_gat, 0),
            ("gated  hint_fused",    hint_gat, 1),
        ]):
            heat = hint_to_heatmap(hint_arr[step, idx, :])
            vmin, vmax = scale[["add_main", "add_fused", "gat_main", "gat_fused"][ci]]
            overlay(axes[ri, 1 + ci], frame, heat, name, vmin=vmin, vmax=vmax)

    fig.suptitle(f"Per-patch hint magnitude overlays\n"
                 f"task: {d['task']}     success={bool(d['success'])}     T={int(d['T'])}",
                 fontsize=11)
    fig.tight_layout()
    out = os.path.join(args.out_dir, "hint_overlay.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")

    # Also save a "gated only" simpler figure (focused thesis fig)
    fig, axes = plt.subplots(n_rows, 3, figsize=(12, 4 * n_rows))
    if n_rows == 1:
        axes = axes[None, :]
    for ri, step in enumerate(keyframes):
        frame = load_frame(args.mp4, step)
        if frame.shape[:2] != (IMAGE_SIZE, IMAGE_SIZE):
            frame = np.array(Image.fromarray(frame).resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR))
        axes[ri, 0].imshow(frame); axes[ri, 0].set_title(f"Step {step}"); axes[ri, 0].axis("off")
        for ci, idx in enumerate([0, 1]):
            heat = hint_to_heatmap(hint_gat[step, idx, :])
            vmin, vmax = scale[("gat_main", "gat_fused")[idx]]
            overlay(axes[ri, 1 + ci], frame, heat,
                    f"gated  {'hint_main (DINOv2)' if idx == 0 else 'hint_fused (SigLIP)'}",
                    vmin=vmin, vmax=vmax)
    fig.suptitle(f"Gated feedback over time -- where the LLM tells the ViT to look\n"
                 f"task: {d['task']}", fontsize=10)
    fig.tight_layout()
    out2 = os.path.join(args.out_dir, "hint_overlay_gated_only.png")
    fig.savefig(out2, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out2}")


if __name__ == "__main__":
    main()
