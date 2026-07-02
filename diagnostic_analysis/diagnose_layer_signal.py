"""
diagnose_layer_signal.py

Diagnostic: does layer X of the finetuned OpenVLA give an instruction-discriminative signal?

For a fixed image, run K different instructions through the full multimodal forward.
At each LLM layer, extract instruction-token hidden states, mean-pool across tokens,
and compute pairwise cosine similarity between the K instruction vectors.

If all layers show cosine similarity ~1.0, the checkpoint is instruction-blind at every
layer and the projector cross-attention cannot help. If some layers show lower similarity,
those are the candidates for `--hidden-layer`.

Usage:
    python diagnose_layer_signal.py \
        --vla-path openvla/openvla-7b-finetuned-libero-spatial \
        --image-path /path/to/sample_image.png
"""

import os

# Force HF cache to the shared group cache on Mimer (must be set before importing transformers)
# This mirrors what the sbatch training/eval scripts do, so interactive runs stay consistent.
if "HF_HOME" not in os.environ:
    DEFAULT_HF_HOME = "/mimer/NOBACKUP/groups/robot_unforseen/.cache/huggingface"
    if os.path.isdir(DEFAULT_HF_HOME):
        os.environ["HF_HOME"] = DEFAULT_HF_HOME
        os.environ.setdefault("TRANSFORMERS_CACHE", f"{DEFAULT_HF_HOME}/hub")

# Force offline mode so Alvis compute nodes don't hang on HF Hub network calls.
# The model is already in the group cache; no reason to talk to the Hub.
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import argparse
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor, AutoConfig, AutoImageProcessor

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.models.backbones.llm.prompting import PurePromptBuilder, VicunaV15ChatPromptBuilder


# Test instructions on a LIBERO-like scene. Edit to match your image content.
DEFAULT_INSTRUCTIONS = [
    "pick up the black bowl",
    "pick up the red block",
    "pick up the milk carton",
    "put the bowl on the plate",
    "put the plate on the bowl",
]


def load_model(vla_path):
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    # FlashAttention requires Ampere+ GPUs (A100, A40, RTX 30xx). T4 (Turing) is NOT supported.
    # Default to SDPA which works on all modern GPUs including T4.
    attn_impl = "sdpa"
    if torch.cuda.is_available():
        capability = torch.cuda.get_device_capability(0)
        if capability[0] >= 8:  # Ampere+ has compute capability 8.0+
            try:
                import flash_attn  # noqa: F401
                attn_impl = "flash_attention_2"
            except ImportError:
                pass
    print(f"Using attn_implementation={attn_impl}")

    vla = AutoModelForVision2Seq.from_pretrained(
        vla_path,
        attn_implementation=attn_impl,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(vla_path, trust_remote_code=True)
    vla.eval()
    return vla, processor


@torch.inference_mode()
def extract_all_layers(vla, processor, image, instruction, model_name):
    """
    Run full multimodal forward. Return hidden states at instruction token positions
    for all layers. Shape: [num_layers+1, T_instruction, d_llm].
    """
    prompt_builder_fn = VicunaV15ChatPromptBuilder if "v01" in model_name else PurePromptBuilder
    prompt_builder = prompt_builder_fn("openvla")
    prompt_builder.add_turn("human", f"What action should the robot take to {instruction.lower()}?")
    prompt_text = prompt_builder.get_prompt()

    inputs = processor(prompt_text, image).to(vla.device, dtype=torch.bfloat16)

    output = vla(
        input_ids=inputs["input_ids"],
        attention_mask=inputs.get("attention_mask", None),
        pixel_values=inputs["pixel_values"],
        labels=None,
        output_hidden_states=True,
        return_dict=True,
    )

    # Sequence layout: [BOS, patch_1..patch_256, instruction_tokens...]
    # output.hidden_states is a tuple of (num_layers + 1) tensors
    num_patches = 256
    all_layers = torch.stack(output.hidden_states, dim=0)  # [L+1, 1, 1+256+T, d]
    instruction_hidden = all_layers[:, 0, 1 + num_patches:, :]  # [L+1, T, d]

    return instruction_hidden.float().cpu()


def pairwise_cosine(vectors):
    """vectors: [K, d]. Returns [K, K] matrix of pairwise cosine similarities."""
    normed = F.normalize(vectors, dim=-1)
    return normed @ normed.T


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vla-path", type=str, required=True,
                        help="HF model path (e.g., openvla/openvla-7b-finetuned-libero-spatial)")
    parser.add_argument("--image-path", type=str, default=None,
                        help="Path to image. If omitted, uses a blank 224x224 image.")
    parser.add_argument("--instructions", type=str, nargs="+", default=DEFAULT_INSTRUCTIONS,
                        help="Instructions to test")
    parser.add_argument("--layers-to-show", type=str, default="0,4,8,12,16,20,24,28,31",
                        help="Comma-separated list of layers to print")
    args = parser.parse_args()

    print(f"HF_HOME = {os.environ.get('HF_HOME', '(default)')}")
    print(f"Loading model from {args.vla_path}...")
    vla, processor = load_model(args.vla_path)

    if args.image_path:
        image = Image.open(args.image_path).convert("RGB")
    else:
        print("No image given; using blank gray image.")
        image = Image.new("RGB", (224, 224), (128, 128, 128))

    print(f"\nInstructions ({len(args.instructions)}):")
    for i, instr in enumerate(args.instructions):
        print(f"  [{i}] {instr}")

    # Extract hidden states for each instruction
    print("\nRunning full multimodal forward for each instruction...")
    all_instruction_hiddens = []  # list of [L+1, T_i, d] tensors (T_i may differ per instruction)
    for instr in args.instructions:
        hiddens = extract_all_layers(vla, processor, image, instr, args.vla_path)
        all_instruction_hiddens.append(hiddens)
        print(f"  '{instr}' -> T={hiddens.shape[1]} tokens")

    num_layers = all_instruction_hiddens[0].shape[0]
    d = all_instruction_hiddens[0].shape[2]
    K = len(args.instructions)

    # For each layer, mean-pool over instruction tokens to get [K, d], then pairwise cosine
    layers_to_show = [int(x) for x in args.layers_to_show.split(",")]

    print("\n" + "=" * 80)
    print("LAYER-WISE PAIRWISE COSINE SIMILARITY (mean-pooled across instruction tokens)")
    print("=" * 80)
    print("Interpretation: lower = more instruction-discriminative (what we want).")
    print("                ~1.0 = instruction-blind at this layer (bad signal source).\n")

    # Header for the pairwise table
    header = "  Layer | " + " ".join([f"sim_{i}{j}" for i in range(K) for j in range(i+1, K)]) + " | mean_off_diag"
    print(header)
    print("-" * len(header))

    results = {}
    for layer_idx in range(num_layers):
        pooled = torch.stack([h[layer_idx].mean(dim=0) for h in all_instruction_hiddens], dim=0)  # [K, d]
        sim = pairwise_cosine(pooled)  # [K, K]

        off_diag = []
        for i in range(K):
            for j in range(i + 1, K):
                off_diag.append(sim[i, j].item())

        mean_sim = float(np.mean(off_diag))
        results[layer_idx] = (off_diag, mean_sim)

        if layer_idx in layers_to_show:
            pairs_str = " ".join([f"{s:+.3f}" for s in off_diag])
            print(f"  {layer_idx:5d} | {pairs_str} | {mean_sim:+.4f}")

    # Summary: best and worst layers
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    sorted_layers = sorted(results.items(), key=lambda kv: kv[1][1])
    print("\nMost instruction-discriminative (lowest mean cosine similarity):")
    for layer_idx, (_, mean_sim) in sorted_layers[:5]:
        print(f"  layer {layer_idx:3d}: mean_sim = {mean_sim:+.4f}")

    print("\nLeast instruction-discriminative (highest mean cosine similarity):")
    for layer_idx, (_, mean_sim) in sorted_layers[-5:]:
        print(f"  layer {layer_idx:3d}: mean_sim = {mean_sim:+.4f}")

    # Verdict
    print("\n" + "=" * 80)
    print("VERDICT")
    print("=" * 80)
    _, mean_12 = results[12]
    _, mean_best = sorted_layers[0][1], sorted_layers[0][1][1]
    best_layer = sorted_layers[0][0]

    if mean_12 > 0.99:
        print(f"Layer 12 is INSTRUCTION-BLIND (mean_sim = {mean_12:+.4f}).")
        print(f"Best layer is {best_layer} with mean_sim = {mean_best:+.4f}.")
        if mean_best > 0.99:
            print("ALL layers are instruction-blind. Rethink signal source (e.g., use raw embeddings).")
        else:
            print(f"Recommend --hidden-layer {best_layer} for the training script.")
    elif mean_12 > 0.95:
        print(f"Layer 12 is WEAKLY DISCRIMINATIVE (mean_sim = {mean_12:+.4f}).")
        print(f"Consider layer {best_layer} (mean_sim = {mean_best:+.4f}) instead.")
    else:
        print(f"Layer 12 is USABLE (mean_sim = {mean_12:+.4f}).")
        print(f"Proceed with --hidden-layer 12 for the training script.")


if __name__ == "__main__":
    main()
