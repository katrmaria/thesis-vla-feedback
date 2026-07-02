"""
attention_modulation_testtime.py

Test-time-only attention modulation — NO TRAINING.

Two-pass inference:
  PASS 1: run base OpenVLA. Capture attention weights from a "teacher" head
          at a sharp-grounding layer (default: L11, head 26 — max attention
          ~0.98-0.99 per cross-suite analysis).
  PASS 2: install a pre-hook at an earlier "inject" layer (default: L9) that
          multiplies patch magnitudes by (1 + alpha * captured_attention).

Hyperparameters:
  --teacher-layer  (default 11)
  --teacher-head   (default 26, set -1 for auto: per-sample argmax over heads)
  --inject-layer   (default 9)
  --alpha          (default 1.0, modulation strength)

No learned parameters, no optimizer, no loss. Pure diagnostic.
"""

import logging
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor, AutoConfig, AutoImageProcessor

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.models.backbones.llm.prompting import PurePromptBuilder

logger = logging.getLogger(__name__)

NUM_PATCHES = 256


class TestTimeAttentionModulation(nn.Module):
    """Wrap a frozen OpenVLA with two-pass test-time attention modulation."""

    def __init__(
        self,
        vla,
        teacher_layer: int = 11,
        teacher_head: int = 26,   # -1 → auto per-sample
        inject_layer: int = 9,
        alpha: float = 1.0,
    ):
        super().__init__()
        self.vla = vla
        self.teacher_layer = teacher_layer
        self.teacher_head = teacher_head
        self.inject_layer = inject_layer
        self.alpha = alpha

        self._captured_patch_weights = None
        self._teacher_original_forward = None
        self._inject_handle = None

        for p in self.vla.parameters():
            p.requires_grad = False
        self.vla.eval()

    # ---- Layer access ----

    def _get_llm_layers(self):
        vla = self.vla
        if hasattr(vla, "base_model"):
            vla = vla.base_model
        if hasattr(vla, "language_model"):
            return vla.language_model.model.layers
        if hasattr(vla, "model") and hasattr(vla.model, "layers"):
            return vla.model.layers
        raise RuntimeError(f"Cannot find transformer layers in {type(vla)}")

    # ---- PASS 1: teacher attention capture ----

    def _register_teacher_capture(self):
        layers = self._get_llm_layers()
        self_attn = layers[self.teacher_layer].self_attn
        self._teacher_original_forward = self_attn.forward
        parent = self

        def patched_forward(*args, **kwargs):
            kwargs["output_attentions"] = True
            outputs = parent._teacher_original_forward(*args, **kwargs)
            attn_weights = outputs[1]  # [B, H, T, T] or None (flash-attn)
            if attn_weights is not None:
                # last-token → image-patch attention
                last_tok = attn_weights[:, :, -1, 1:1 + NUM_PATCHES]  # [B, H, 256]
                if parent.teacher_head >= 0:
                    patch_attn = last_tok[:, parent.teacher_head, :]  # [B, 256]
                else:
                    # auto: per-sample, pick head with sharpest peak
                    max_per_head = last_tok.max(dim=-1).values  # [B, H]
                    best = max_per_head.argmax(dim=-1)          # [B]
                    patch_attn = torch.stack(
                        [last_tok[b, best[b], :] for b in range(last_tok.shape[0])]
                    )
                # re-normalize over the 256 patches
                patch_attn = patch_attn / patch_attn.sum(dim=-1, keepdim=True).clamp(min=1e-8)
                parent._captured_patch_weights = patch_attn.detach()
            # strip attentions from output so downstream doesn't see them
            return (outputs[0], None) + tuple(outputs[2:])

        self_attn.forward = patched_forward

    def _unregister_teacher_capture(self):
        if self._teacher_original_forward is not None:
            layers = self._get_llm_layers()
            layers[self.teacher_layer].self_attn.forward = self._teacher_original_forward
            self._teacher_original_forward = None

    # ---- PASS 2: injection pre-hook ----

    def _register_injection(self):
        layers = self._get_llm_layers()
        parent = self

        def hook_fn(module, args, kwargs):
            if len(args) > 0:
                hidden_states = args[0]
            elif "hidden_states" in kwargs:
                hidden_states = kwargs["hidden_states"]
            else:
                return args, kwargs

            # Only modulate when image patches are in the sequence (first-prompt pass).
            # Autoregressive token-by-token passes with KV cache have seq_len=1.
            if hidden_states.shape[1] < 1 + NUM_PATCHES:
                return args, kwargs
            if parent._captured_patch_weights is None:
                return args, kwargs

            weights = parent._captured_patch_weights  # [B, 256]
            # boost factor per patch: 1 + alpha * weight
            boost = (1.0 + parent.alpha * weights).unsqueeze(-1).to(hidden_states.dtype)

            modified = hidden_states.clone()
            modified[:, 1:1 + NUM_PATCHES, :] = (
                hidden_states[:, 1:1 + NUM_PATCHES, :] * boost
            )

            if len(args) > 0:
                return (modified,) + args[1:], kwargs
            kwargs["hidden_states"] = modified
            return args, kwargs

        self._inject_handle = layers[self.inject_layer].register_forward_pre_hook(
            hook_fn, with_kwargs=True
        )

    def _unregister_injection(self):
        if self._inject_handle is not None:
            self._inject_handle.remove()
            self._inject_handle = None

    # ---- Loader ----

    @classmethod
    def from_pretrained(cls, model_name, teacher_layer=11, teacher_head=26,
                        inject_layer=9, alpha=1.0, device_map="auto"):
        AutoConfig.register("openvla", OpenVLAConfig)
        AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
        AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
        AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

        # Need eager attention (not flash) so output_attentions works
        vla = AutoModelForVision2Seq.from_pretrained(
            model_name,
            attn_implementation="eager",
            device_map=device_map,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )

        model = cls(
            vla,
            teacher_layer=teacher_layer,
            teacher_head=teacher_head,
            inject_layer=inject_layer,
            alpha=alpha,
        )
        model.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        model.eval()
        logger.info(
            f"TestTimeAttentionModulation loaded: teacher=L{teacher_layer}H{teacher_head}, "
            f"inject=L{inject_layer}, alpha={alpha}"
        )
        return model

    # ---- Inference ----

    @torch.inference_mode()
    def generate(self, image, task_description, unnorm_key=None):
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        prompt_builder = PurePromptBuilder("openvla")
        prompt_builder.add_turn(
            "human",
            f"What action should the robot take to {task_description.lower()}?",
        )
        prompt_text = prompt_builder.get_prompt()

        inputs = self.processor(prompt_text, image).to(
            self.vla.device, dtype=torch.bfloat16
        )
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")
        pixel_values = inputs["pixel_values"]

        # PASS 1: capture teacher attention
        self._captured_patch_weights = None
        self._register_teacher_capture()
        try:
            _ = self.vla(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                labels=None,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
            )
        finally:
            self._unregister_teacher_capture()

        if self._captured_patch_weights is None:
            logger.warning(
                "No teacher attention captured — running vanilla predict_action "
                "(check that attn_implementation='eager')"
            )
            return self.vla.predict_action(
                input_ids, unnorm_key=unnorm_key, pixel_values=pixel_values, do_sample=False
            )

        # PASS 2: modulate at inject layer, predict action
        self._register_injection()
        try:
            action = self.vla.predict_action(
                input_ids,
                unnorm_key=unnorm_key,
                pixel_values=pixel_values,
                do_sample=False,
            )
        finally:
            self._unregister_injection()
            self._captured_patch_weights = None

        return action
