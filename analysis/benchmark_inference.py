"""
benchmark_inference.py

Time a single inference call per method, matching the per-step call used in
the LIBERO eval scripts. Each timed function takes a fresh numpy image plus
a task label and returns one action, including the center-crop and processor
preprocessing the eval scripts apply.

Usage:
    python benchmark_inference.py --method baseline
    python benchmark_inference.py --method reasonvla --checkpoint <path.pth> \
        --stage 1 --hidden_layer 24 --feedback_mode gated
    python benchmark_inference.py --method projcrossattn --checkpoint <path.pth> \
        --stage 1 --hidden_layer 7
    python benchmark_inference.py --method attnmod --attnmod_method song
"""
import argparse
import time

import numpy as np
import tensorflow as tf
import torch
from PIL import Image
from transformers import (
    AutoModelForVision2Seq, AutoProcessor, AutoConfig, AutoImageProcessor,
)

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import (
    PrismaticImageProcessor, PrismaticProcessor,
)

NUM_WARMUP = 5
NUM_ITERS = 50


def register_openvla():
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)


def crop_and_resize(image, crop_scale, batch_size):
    """Center crop with crop_scale area, resize back to original size.

    Matches `crop_and_resize` in openvla.experiments.robot.openvla_utils.
    """
    assert image.shape.ndims == 3 or image.shape.ndims == 4
    expanded_dims = False
    if image.shape.ndims == 3:
        image = tf.expand_dims(image, axis=0)
        expanded_dims = True

    new_h_w = tf.sqrt(tf.cast(crop_scale, tf.float32))
    new_offset = (1.0 - new_h_w) / 2.0
    boxes = tf.tile(
        tf.expand_dims(tf.stack([new_offset, new_offset, new_offset + new_h_w, new_offset + new_h_w], axis=0), 0),
        [batch_size, 1],
    )
    image = tf.image.crop_and_resize(
        image, boxes, box_indices=tf.range(batch_size), crop_size=image.shape[1:3]
    )
    if expanded_dims:
        image = image[0]
    return image


def apply_center_crop(image_pil):
    """Apply the LIBERO eval center crop (crop scale 0.9) to a PIL image."""
    image_tf = tf.convert_to_tensor(np.array(image_pil))
    orig_dtype = image_tf.dtype
    image_tf = tf.image.convert_image_dtype(image_tf, tf.float32)
    image_tf = crop_and_resize(image_tf, crop_scale=0.9, batch_size=1)
    image_tf = tf.clip_by_value(image_tf, 0, 1)
    image_tf = tf.image.convert_image_dtype(image_tf, orig_dtype, saturate=True)
    return Image.fromarray(image_tf.numpy()).convert("RGB")


def time_one_call(fn, *args, **kwargs):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    torch.cuda.synchronize()
    return time.perf_counter() - t0, result


def benchmark(name, infer_fn, infer_args):
    print(f"\n--- {name} ---")
    print(f"Warmup ({NUM_WARMUP} iters)...")
    for _ in range(NUM_WARMUP):
        _ = infer_fn(*infer_args)

    print(f"Timing ({NUM_ITERS} iters)...")
    times = []
    for _ in range(NUM_ITERS):
        dt, _ = time_one_call(infer_fn, *infer_args)
        times.append(dt)

    times_ms = np.array(times) * 1000.0
    print(f"  mean   {times_ms.mean():8.2f} ms")
    print(f"  median {np.median(times_ms):8.2f} ms")
    print(f"  std    {times_ms.std():8.2f} ms")
    print(f"  min    {times_ms.min():8.2f} ms")
    print(f"  max    {times_ms.max():8.2f} ms")
    return times_ms


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--method",
        choices=["baseline", "reasonvla", "projcrossattn", "attnmod"],
        required=True,
    )
    p.add_argument(
        "--base_model",
        default="openvla/openvla-7b-finetuned-libero-spatial",
    )
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--lora_dir", default=None)
    p.add_argument("--stage", type=int, default=1)
    p.add_argument("--hidden_layer", type=int, default=-1)
    p.add_argument("--feedback_mode", default="additive")
    p.add_argument("--attnmod_method", default="song",
                   choices=["song", "logit_bias", "attn_gate"])
    p.add_argument("--unnorm_key", default="libero_spatial")
    p.add_argument("--center_crop", type=int, default=1,
                   help="1 to include center crop (matches eval), 0 to skip")
    args = p.parse_args()

    register_openvla()

    # Synthetic 256x256 RGB numpy image, same dtype as obs["full_image"] in eval.
    image_np = (np.random.rand(256, 256, 3) * 255).astype(np.uint8)
    task = "pick up the black bowl and place it on the plate"
    use_center_crop = bool(args.center_crop)

    if args.method == "baseline":
        try:
            import flash_attn  # noqa
            attn_impl = "flash_attention_2"
        except ImportError:
            attn_impl = None
        vla = AutoModelForVision2Seq.from_pretrained(
            args.base_model,
            **({"attn_implementation": attn_impl} if attn_impl else {}),
            torch_dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        vla.eval()
        proc = AutoProcessor.from_pretrained(args.base_model, trust_remote_code=True)
        prompt = f"In: What action should the robot take to {task.lower()}?\nOut:"

        def step(image_np_in):
            image = Image.fromarray(image_np_in).convert("RGB")
            if use_center_crop:
                image = apply_center_crop(image)
            inputs = proc(prompt, image).to(vla.device, dtype=torch.bfloat16)
            action = vla.predict_action(
                **inputs,
                unnorm_key=args.unnorm_key,
                do_sample=False,
            )
            return action

        label = "Baseline OpenVLA (per-step incl. preproc"
        label += " + center crop)" if use_center_crop else ")"
        benchmark(label, step, (image_np,))

    elif args.method == "reasonvla":
        # For gated mode, the model was trained with ReasonVLAGatedFixed (identity-init
        # gate + bounded hint magnitude). The eval pipeline monkey-patches the class so
        # `from_finetuned` instantiates the fixed one. We mirror that here.
        import reason_vla as _rvla_mod
        if args.feedback_mode == "gated":
            from reason_vla_gated_fixed import ReasonVLAGatedFixed
            _rvla_mod.ReasonVLA = ReasonVLAGatedFixed
            ModelClass = ReasonVLAGatedFixed
        else:
            from reason_vla import ReasonVLA
            ModelClass = ReasonVLA
        model = ModelClass.from_finetuned(
            args.base_model,
            args.checkpoint,
            stage=args.stage,
            lora_dir=args.lora_dir,
            hidden_layer=args.hidden_layer,
            feedback_mode=args.feedback_mode,
        )

        def step(image_np_in):
            image = Image.fromarray(image_np_in).convert("RGB")
            if use_center_crop:
                image = apply_center_crop(image)
            return model.generate(image, task, unnorm_key=args.unnorm_key)

        label = (
            f"ReasonVLA ({args.feedback_mode}, hl={args.hidden_layer}, "
            f"stage={args.stage}) per-step"
        )
        label += " + center crop" if use_center_crop else ""
        benchmark(label, step, (image_np,))

    elif args.method == "projcrossattn":
        from reason_vla_projector_crossattn import ReasonVLAProjectorCrossAttn
        model = ReasonVLAProjectorCrossAttn.from_finetuned(
            args.base_model,
            args.checkpoint,
            stage=args.stage,
            lora_dir=args.lora_dir,
            hidden_layer=args.hidden_layer,
        )

        def step(image_np_in):
            image = Image.fromarray(image_np_in).convert("RGB")
            if use_center_crop:
                image = apply_center_crop(image)
            return model.generate(image, task, unnorm_key=args.unnorm_key)

        label = (
            f"Projector Cross-Attention (hl={args.hidden_layer}, "
            f"stage={args.stage}) per-step"
        )
        label += " + center crop" if use_center_crop else ""
        benchmark(label, step, (image_np,))

    elif args.method == "attnmod":
        from attention_modulation_testtime_v2 import TestTimeAttentionModulationV2
        model = TestTimeAttentionModulationV2.from_pretrained(
            args.base_model, method=args.attnmod_method,
        )

        def step(image_np_in):
            image = Image.fromarray(image_np_in).convert("RGB")
            if use_center_crop:
                image = apply_center_crop(image)
            return model.generate(image, task, unnorm_key=args.unnorm_key)

        label = f"Inference Attention Modulation ({args.attnmod_method}) per-step"
        label += " + center crop" if use_center_crop else ""
        benchmark(label, step, (image_np,))


if __name__ == "__main__":
    main()
