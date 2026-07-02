"""
count_params.py

Load each method's wrapper around OpenVLA and print parameter counts:
  - total params (entire wrapped model including frozen VLA)
  - trainable params (what the optimiser actually updates)
  - per-submodule breakdown (visual reasoner, unmergers, gate, alpha, LoRA, etc.)

Usage:
    python count_params.py --method baseline
    python count_params.py --method reasonvla --feedback_mode additive
    python count_params.py --method reasonvla --feedback_mode gated
    python count_params.py --method reasonvla --feedback_mode film
    python count_params.py --method reasonvla --feedback_mode adaln
    python count_params.py --method reasonvla --feedback_mode scaled
    python count_params.py --method projcrossattn
    python count_params.py --method attnmod
    python count_params.py --method reasonvla --stage 2 --feedback_mode additive
"""
import argparse
from collections import OrderedDict

import torch
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForVision2Seq, AutoConfig, AutoImageProcessor, AutoProcessor,
)

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import (
    PrismaticImageProcessor, PrismaticProcessor,
)


def register_openvla():
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)


def count_params(module, only_trainable=False):
    return sum(
        p.numel() for p in module.parameters()
        if (not only_trainable) or p.requires_grad
    )


def fmt(n):
    if n >= 1e9:
        return f"{n/1e9:8.3f} B"
    if n >= 1e6:
        return f"{n/1e6:8.2f} M"
    if n >= 1e3:
        return f"{n/1e3:8.2f} K"
    return f"{n:10d}  "


def print_breakdown(name, by_submodule, totals):
    print(f"\n=== {name} ===")
    print(f"{'submodule':<40} {'total':>14} {'trainable':>14}")
    print("-" * 70)
    for k, (tot, tr) in by_submodule.items():
        print(f"{k:<40} {fmt(tot):>14} {fmt(tr):>14}")
    print("-" * 70)
    grand_total, grand_train = totals
    print(f"{'TOTAL':<40} {fmt(grand_total):>14} {fmt(grand_train):>14}")
    print(f"trainable / total: {100*grand_train/grand_total:.3f} %")


def load_baseline_vla(base_model):
    return AutoModelForVision2Seq.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )


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
    p.add_argument("--feedback_mode", default="additive",
                   choices=["additive", "film", "gated", "adaln", "scaled"])
    p.add_argument("--stage", type=int, default=1, help="1 or 2")
    p.add_argument("--lora_rank", type=int, default=None,
                   help="Override default LoRA rank (32 for ReasonVLA, 16 for projcrossattn)")
    p.add_argument("--hidden_layer", type=int, default=-1)
    args = p.parse_args()

    register_openvla()
    print(f"Loading base VLA, {args.base_model}")
    vla = load_baseline_vla(args.base_model)

    if args.method == "baseline":
        by_sub = OrderedDict()
        by_sub["vision_backbone"] = (
            count_params(vla.vision_backbone),
            count_params(vla.vision_backbone, only_trainable=True),
        )
        by_sub["projector"] = (
            count_params(vla.projector),
            count_params(vla.projector, only_trainable=True),
        )
        by_sub["language_model"] = (
            count_params(vla.language_model),
            count_params(vla.language_model, only_trainable=True),
        )
        totals = (count_params(vla), count_params(vla, only_trainable=True))
        print_breakdown("Baseline OpenVLA", by_sub, totals)
        return

    if args.method == "reasonvla":
        # Pick the correct class for the gated feedback mode
        if args.feedback_mode == "gated":
            from reason_vla_gated_fixed import ReasonVLAGatedFixed as ReasonVLA
        else:
            from reason_vla import ReasonVLA

        model = ReasonVLA(vla, hidden_layer=args.hidden_layer, feedback_mode=args.feedback_mode)
        model.freeze_vla()
        model.unfreeze_reasoning()

        by_sub = OrderedDict()
        by_sub["vla.vision_backbone (frozen)"] = (
            count_params(vla.vision_backbone),
            count_params(vla.vision_backbone, only_trainable=True),
        )
        by_sub["vla.projector (frozen)"] = (
            count_params(vla.projector),
            count_params(vla.projector, only_trainable=True),
        )
        by_sub["vla.language_model (frozen)"] = (
            count_params(vla.language_model),
            count_params(vla.language_model, only_trainable=True),
        )
        by_sub["visual_reasoner"] = (
            count_params(model.visual_reasoner),
            count_params(model.visual_reasoner, only_trainable=True),
        )
        # Per-mode auxiliaries
        if args.feedback_mode == "additive":
            by_sub["main_unmerger"] = (count_params(model.main_unmerger), count_params(model.main_unmerger, only_trainable=True))
            if model.is_fused:
                by_sub["fused_unmerger"] = (count_params(model.fused_unmerger), count_params(model.fused_unmerger, only_trainable=True))
        elif args.feedback_mode == "film":
            by_sub["main_unmerger_gamma"] = (count_params(model.main_unmerger_gamma), count_params(model.main_unmerger_gamma, only_trainable=True))
            by_sub["main_unmerger_beta"]  = (count_params(model.main_unmerger_beta),  count_params(model.main_unmerger_beta,  only_trainable=True))
            if model.is_fused:
                by_sub["fused_unmerger_gamma"] = (count_params(model.fused_unmerger_gamma), count_params(model.fused_unmerger_gamma, only_trainable=True))
                by_sub["fused_unmerger_beta"]  = (count_params(model.fused_unmerger_beta),  count_params(model.fused_unmerger_beta,  only_trainable=True))
        elif args.feedback_mode == "gated":
            by_sub["main_unmerger"] = (count_params(model.main_unmerger), count_params(model.main_unmerger, only_trainable=True))
            by_sub["main_gate"] = (count_params(model.main_gate), count_params(model.main_gate, only_trainable=True))
            if model.is_fused:
                by_sub["fused_unmerger"] = (count_params(model.fused_unmerger), count_params(model.fused_unmerger, only_trainable=True))
                by_sub["fused_gate"] = (count_params(model.fused_gate), count_params(model.fused_gate, only_trainable=True))
        elif args.feedback_mode == "adaln":
            by_sub["main_unmerger_gamma"] = (count_params(model.main_unmerger_gamma), count_params(model.main_unmerger_gamma, only_trainable=True))
            by_sub["main_unmerger_beta"]  = (count_params(model.main_unmerger_beta),  count_params(model.main_unmerger_beta,  only_trainable=True))
            if model.is_fused:
                by_sub["fused_unmerger_gamma"] = (count_params(model.fused_unmerger_gamma), count_params(model.fused_unmerger_gamma, only_trainable=True))
                by_sub["fused_unmerger_beta"]  = (count_params(model.fused_unmerger_beta),  count_params(model.fused_unmerger_beta,  only_trainable=True))
        elif args.feedback_mode == "scaled":
            by_sub["main_unmerger"] = (count_params(model.main_unmerger), count_params(model.main_unmerger, only_trainable=True))
            by_sub["hint_alpha (scalar)"] = (1, 1)
            if model.is_fused:
                by_sub["fused_unmerger"] = (count_params(model.fused_unmerger), count_params(model.fused_unmerger, only_trainable=True))
                by_sub["hint_alpha_fused (scalar)"] = (1, 1)

        if args.stage == 2:
            lora_rank = args.lora_rank if args.lora_rank is not None else 32
            target_linear = []
            for name, module in model.vla.named_modules():
                if not name.startswith("language_model"):
                    continue
                if isinstance(module, torch.nn.Linear):
                    target_linear.append(name)
            target_linear = [n for n in target_linear if "embed_tokens" not in n and "lm_head" not in n]
            lora_config = LoraConfig(
                r=lora_rank, lora_alpha=min(lora_rank, 16),
                lora_dropout=0.0, bias="none",
                target_modules=target_linear, task_type="CAUSAL_LM",
            )
            model.vla = get_peft_model(model.vla, lora_config)
            lora_total    = sum(p.numel() for n, p in model.vla.named_parameters() if "lora_" in n)
            lora_train    = sum(p.numel() for n, p in model.vla.named_parameters() if "lora_" in n and p.requires_grad)
            by_sub[f"LoRA r={lora_rank} (LLM linears)"] = (lora_total, lora_train)

        totals = (count_params(model), count_params(model, only_trainable=True))
        print_breakdown(f"ReasonVLA ({args.feedback_mode}, stage {args.stage})", by_sub, totals)
        return

    if args.method == "projcrossattn":
        from reason_vla_projector_crossattn import ReasonVLAProjectorCrossAttn
        model = ReasonVLAProjectorCrossAttn(vla, hidden_layer=args.hidden_layer)
        model.freeze_vla()
        model.unfreeze_reasoning()

        by_sub = OrderedDict()
        by_sub["vla.vision_backbone (frozen)"] = (
            count_params(vla.vision_backbone),
            count_params(vla.vision_backbone, only_trainable=True),
        )
        by_sub["vla.projector (frozen)"] = (
            count_params(vla.projector),
            count_params(vla.projector, only_trainable=True),
        )
        by_sub["vla.language_model (frozen)"] = (
            count_params(vla.language_model),
            count_params(vla.language_model, only_trainable=True),
        )
        by_sub["cross_attn"] = (
            count_params(model.cross_attn),
            count_params(model.cross_attn, only_trainable=True),
        )

        if args.stage == 2:
            lora_rank = args.lora_rank if args.lora_rank is not None else 16
            target_linear = []
            for name, module in model.vla.named_modules():
                if not name.startswith("language_model"):
                    continue
                if isinstance(module, torch.nn.Linear):
                    target_linear.append(name)
            target_linear = [n for n in target_linear if "embed_tokens" not in n and "lm_head" not in n]
            lora_config = LoraConfig(
                r=lora_rank, lora_alpha=min(lora_rank, 16),
                lora_dropout=0.0, bias="none",
                target_modules=target_linear, task_type="CAUSAL_LM",
            )
            model.vla = get_peft_model(model.vla, lora_config)
            lora_total    = sum(p.numel() for n, p in model.vla.named_parameters() if "lora_" in n)
            lora_train    = sum(p.numel() for n, p in model.vla.named_parameters() if "lora_" in n and p.requires_grad)
            by_sub[f"LoRA r={lora_rank} (LLM linears)"] = (lora_total, lora_train)

        totals = (count_params(model), count_params(model, only_trainable=True))
        print_breakdown(f"Projector Cross-Attention (stage {args.stage})", by_sub, totals)
        return

    if args.method == "attnmod":
        from attention_modulation_testtime_v2 import TestTimeAttentionModulationV2
        model = TestTimeAttentionModulationV2(vla, method="song")
        # Nothing trainable, just confirm
        by_sub = OrderedDict()
        by_sub["vla (entire model, frozen)"] = (
            count_params(model.vla),
            count_params(model.vla, only_trainable=True),
        )
        totals = (count_params(model), count_params(model, only_trainable=True))
        print_breakdown("Inference Attention Modulation (no training)", by_sub, totals)
        return


if __name__ == "__main__":
    main()
