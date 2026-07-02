"""
attention_modulation_testtime_v2.py

Three new inference-time attention modulation methods for OpenVLA.
All training-free, two-pass, inference-only.

Methods:
  "song"         — Faithful Song et al.: suppress bottom-ρ% patches by λ at review layer L28.
                    Contrastive mask |A^(post) - A^(pre)| from L27 and L15.
  "logit_bias"   — Add mask as bias to attention logits at grounding layers (10-15) before softmax.
                    Operates in attention space, not hidden state space.
  "attn_gate"    — Multiply post-softmax attention weights by mask at grounding layers (10-15),
                    then re-normalize. Sharpens grounding heads' focus.

All share the same two-pass structure:
  Pass 1: Run model with eager attention, capture attention weights.
  Pass 2: Apply intervention via hooks, predict action.
"""

import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor, AutoConfig, AutoImageProcessor

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.vla.action_tokenizer import ActionTokenizer

logger = logging.getLogger(__name__)

NUM_PATCHES = 256


class TestTimeAttentionModulationV2(nn.Module):
    """
    Unified wrapper for three inference-time attention modulation methods.

    Args:
        vla: OpenVLAForActionPrediction (loaded with attn_implementation="eager")
        method: "song", "logit_bias", or "attn_gate"
        --- Song et al. params ---
        post_layer: post-integrated layer for contrastive mask (default: 27)
        pre_layer: pre-integrated layer for contrastive mask (default: 15)
        review_layer: where to apply suppression (default: 28)
        rho: percentile threshold, suppress bottom rho% (default: 0.2)
        lam: suppression factor for low-scoring patches (default: 0.1)
        --- Logit bias params ---
        bias_layers: layers where bias is added to attention logits (default: 10-15)
        beta: scaling factor for the logit bias (default: 1.0)
        mask_source_layers: layers to extract mask from (default: 10-15, averaged)
        --- Attn gate params ---
        gate_layers: layers where attention weights are gated (default: 10-15)
        --- Shared ---
        teacher_layer: layer to extract attention from in Pass 1 (default: 11)
        teacher_head: head index, -1 for auto sharpest (default: -1)
    """

    def __init__(
        self,
        vla,
        method="song",
        # Song params
        post_layer=27,
        pre_layer=15,
        review_layer=28,
        rho=0.2,
        lam=0.1,
        # Logit bias params
        bias_layers=None,
        beta=1.0,
        mask_source_layers=None,
        # Attn gate params
        gate_layers=None,
        # Shared
        teacher_layer=11,
        teacher_head=-1,
    ):
        super().__init__()
        self.vla = vla
        self.method = method

        # Song params
        self.post_layer = post_layer
        self.pre_layer = pre_layer
        self.review_layer = review_layer
        self.rho = rho
        self.lam = lam

        # Logit bias params
        self.bias_layers = bias_layers if bias_layers is not None else list(range(10, 16))
        self.beta = beta
        self.mask_source_layers = mask_source_layers if mask_source_layers is not None else list(range(10, 16))

        # Attn gate params
        self.gate_layers = gate_layers if gate_layers is not None else list(range(10, 16))

        # Shared
        self.teacher_layer = teacher_layer
        self.teacher_head = teacher_head

        # State
        self._captured_mask = None  # [B, 256] mask for logit_bias and attn_gate
        self._captured_post_attn = None  # [B, 256] for Song
        self._captured_pre_attn = None   # [B, 256] for Song
        self._hook_handles = []
        self._patched_forwards = []  # (module, original_forward) pairs

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

    # ---- Cleanup ----

    def _cleanup(self):
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles = []
        for module, orig_fwd in self._patched_forwards:
            module.forward = orig_fwd
        self._patched_forwards = []

    # ================================================================
    # PASS 1: Capture attention
    # ================================================================

    def _capture_attention_pass1(self, input_ids, attention_mask, pixel_values):
        """Run Pass 1 and capture the attention we need for the chosen method."""

        if self.method == "song":
            # Need attention at post_layer and pre_layer
            self._captured_post_attn = None
            self._captured_pre_attn = None

            layers_to_capture = {self.post_layer: "post", self.pre_layer: "pre"}
            llm_layers = self._get_llm_layers()

            for layer_idx, label in layers_to_capture.items():
                self_attn = llm_layers[layer_idx].self_attn
                orig_fwd = self_attn.forward
                self._patched_forwards.append((self_attn, orig_fwd))
                parent = self

                def make_patched(original, layer_label):
                    def patched_forward(*args, **kwargs):
                        kwargs["output_attentions"] = True
                        outputs = original(*args, **kwargs)
                        attn_weights = outputs[1]
                        if attn_weights is not None:
                            # last-token -> image patches, averaged across heads
                            patch_attn = attn_weights[:, :, -1, 1:1 + NUM_PATCHES].float().mean(dim=1)  # [B, 256]
                            if layer_label == "post":
                                parent._captured_post_attn = patch_attn.detach()
                            else:
                                parent._captured_pre_attn = patch_attn.detach()
                        return (outputs[0], None) + tuple(outputs[2:])
                    return patched_forward

                self_attn.forward = make_patched(orig_fwd, label)

        else:
            # logit_bias and attn_gate: capture from teacher layer (sharpest head or specific head)
            self._captured_mask = None
            llm_layers = self._get_llm_layers()
            self_attn = llm_layers[self.teacher_layer].self_attn
            orig_fwd = self_attn.forward
            self._patched_forwards.append((self_attn, orig_fwd))
            parent = self

            def patched_forward(*args, **kwargs):
                kwargs["output_attentions"] = True
                outputs = orig_fwd(*args, **kwargs)
                attn_weights = outputs[1]
                if attn_weights is not None:
                    last_tok = attn_weights[:, :, -1, 1:1 + NUM_PATCHES]  # [B, H, 256]
                    if parent.teacher_head >= 0:
                        patch_attn = last_tok[:, parent.teacher_head, :]  # [B, 256]
                    else:
                        # auto: pick head with sharpest peak per sample
                        max_per_head = last_tok.max(dim=-1).values  # [B, H]
                        best = max_per_head.argmax(dim=-1)  # [B]
                        patch_attn = torch.stack(
                            [last_tok[b, best[b], :] for b in range(last_tok.shape[0])]
                        )
                    # min-max normalize
                    mins = patch_attn.min(dim=-1, keepdim=True).values
                    maxs = patch_attn.max(dim=-1, keepdim=True).values
                    patch_attn = (patch_attn - mins) / (maxs - mins + 1e-8)
                    parent._captured_mask = patch_attn.detach()
                return (outputs[0], None) + tuple(outputs[2:])

            self_attn.forward = patched_forward

        # Run forward pass
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
            self._cleanup()

        # For Song: compute contrastive mask
        if self.method == "song":
            if self._captured_post_attn is not None and self._captured_pre_attn is not None:
                contrastive = torch.abs(self._captured_post_attn - self._captured_pre_attn)
                self._captured_mask = contrastive.detach()
            else:
                logger.warning("Failed to capture attention for Song method")

    # ================================================================
    # PASS 2: Apply intervention
    # ================================================================

    def _setup_pass2_hooks(self):
        """Install hooks for the chosen method."""
        llm_layers = self._get_llm_layers()

        if self.method == "song":
            self._setup_song_hook(llm_layers)
        elif self.method == "logit_bias":
            self._setup_logit_bias_hooks(llm_layers)
        elif self.method == "attn_gate":
            self._setup_attn_gate_hooks(llm_layers)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    # ---- Song: suppress bottom-rho% at review layer ----

    def _setup_song_hook(self, llm_layers):
        parent = self

        def hook_fn(module, args, kwargs):
            if len(args) > 0:
                hidden_states = args[0]
            elif "hidden_states" in kwargs:
                hidden_states = kwargs["hidden_states"]
            else:
                return args, kwargs

            if hidden_states.shape[1] < 1 + NUM_PATCHES:
                return args, kwargs
            if parent._captured_mask is None:
                return args, kwargs

            mask = parent._captured_mask  # [B, 256] contrastive scores

            # Find threshold: bottom rho% get suppressed
            threshold = torch.quantile(mask.float(), parent.rho, dim=-1, keepdim=True)  # [B, 1]

            # Build suppression mask: 1.0 for patches above threshold, lam for below
            suppress = torch.where(mask < threshold, parent.lam, 1.0)  # [B, 256]
            suppress = suppress.unsqueeze(-1).to(hidden_states.dtype)  # [B, 256, 1]

            modified = hidden_states.clone()
            modified[:, 1:1 + NUM_PATCHES, :] = (
                hidden_states[:, 1:1 + NUM_PATCHES, :] * suppress
            )

            if len(args) > 0:
                return (modified,) + args[1:], kwargs
            kwargs["hidden_states"] = modified
            return args, kwargs

        handle = llm_layers[self.review_layer].register_forward_pre_hook(hook_fn, with_kwargs=True)
        self._hook_handles.append(handle)

    # ---- Logit bias: add mask to attention logits before softmax ----

    def _setup_logit_bias_hooks(self, llm_layers):
        parent = self

        for layer_idx in self.bias_layers:
            self_attn = llm_layers[layer_idx].self_attn
            orig_fwd = self_attn.forward
            self._patched_forwards.append((self_attn, orig_fwd))

            def make_patched(original):
                def patched_forward(*args, **kwargs):
                    captured_bias = parent._captured_mask  # [B, 256]
                    if captured_bias is None:
                        return original(*args, **kwargs)

                    # Monkey-patch F.softmax temporarily to inject bias before softmax.
                    # In eager attention, the flow is:
                    #   attn_weights = Q @ K^T * scaling + causal_mask
                    #   attn_weights = F.softmax(attn_weights)
                    #   output = attn_weights @ V
                    # We add bias to image patch positions (1:257) before softmax.
                    # This fires on both the initial prompt pass and autoregressive steps
                    # (where KV cache means last dim >= 257), which is intentional.

                    original_softmax = F.softmax
                    bias = captured_bias  # [B, 256]

                    def biased_softmax(input, dim=-1, **sm_kwargs):
                        if input.dim() == 4 and input.shape[-1] >= 1 + NUM_PATCHES:
                            b = bias.to(input.device, input.dtype)
                            bias_expanded = parent.beta * b.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, 256]
                            input = input.clone()
                            input[:, :, :, 1:1 + NUM_PATCHES] = (
                                input[:, :, :, 1:1 + NUM_PATCHES] + bias_expanded
                            )
                        return original_softmax(input, dim=dim, **sm_kwargs)

                    F.softmax = biased_softmax
                    try:
                        outputs = original(*args, **kwargs)
                    finally:
                        F.softmax = original_softmax

                    return outputs

                return patched_forward

            self_attn.forward = make_patched(orig_fwd)

    # ---- Attn gate: multiply post-softmax weights by mask, re-normalize ----

    def _setup_attn_gate_hooks(self, llm_layers):
        parent = self

        for layer_idx in self.gate_layers:
            self_attn = llm_layers[layer_idx].self_attn
            orig_fwd = self_attn.forward
            self._patched_forwards.append((self_attn, orig_fwd))

            def make_patched(original):
                def patched_forward(*args, **kwargs):
                    gate_mask = parent._captured_mask  # [B, 256]
                    if gate_mask is None:
                        return original(*args, **kwargs)

                    # Similar approach: wrap F.softmax to apply gating after softmax
                    original_softmax = F.softmax

                    def gated_softmax(input, dim=-1, **sm_kwargs):
                        # Compute normal softmax first
                        attn_weights = original_softmax(input, dim=dim, **sm_kwargs)
                        # Gate image patch positions
                        if attn_weights.dim() == 4 and attn_weights.shape[-1] >= 1 + NUM_PATCHES:
                            g = gate_mask.to(attn_weights.device, attn_weights.dtype)
                            # [B, 256] -> [B, 1, 1, 256]
                            g_expanded = g.unsqueeze(1).unsqueeze(1)
                            attn_weights = attn_weights.clone()
                            attn_weights[:, :, :, 1:1 + NUM_PATCHES] = (
                                attn_weights[:, :, :, 1:1 + NUM_PATCHES] * g_expanded
                            )
                            # Re-normalize so weights sum to 1
                            attn_weights = attn_weights / attn_weights.sum(dim=-1, keepdim=True).clamp(min=1e-8)
                        return attn_weights

                    F.softmax = gated_softmax
                    try:
                        outputs = original(*args, **kwargs)
                    finally:
                        F.softmax = original_softmax

                    return outputs

                return patched_forward

            self_attn.forward = make_patched(orig_fwd)

    # ================================================================
    # Inference
    # ================================================================

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

        # PASS 1: capture attention
        self._captured_mask = None
        self._captured_post_attn = None
        self._captured_pre_attn = None
        self._capture_attention_pass1(input_ids, attention_mask, pixel_values)

        if self._captured_mask is None:
            logger.warning(
                "No attention captured — running vanilla predict_action "
                "(check that attn_implementation='eager')"
            )
            return self.vla.predict_action(
                input_ids, unnorm_key=unnorm_key, pixel_values=pixel_values, do_sample=False
            )

        # PASS 2: apply intervention and predict
        self._setup_pass2_hooks()
        try:
            action = self.vla.predict_action(
                input_ids,
                unnorm_key=unnorm_key,
                pixel_values=pixel_values,
                do_sample=False,
            )
        finally:
            self._cleanup()
            self._captured_mask = None
            self._captured_post_attn = None
            self._captured_pre_attn = None

        return action

    # ================================================================
    # Loader
    # ================================================================

    @classmethod
    def from_pretrained(cls, model_name, method="song", device_map="auto", **kwargs):
        AutoConfig.register("openvla", OpenVLAConfig)
        AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
        AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
        AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

        vla = AutoModelForVision2Seq.from_pretrained(
            model_name,
            attn_implementation="eager",
            device_map=device_map,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )

        # Force eager attention on all layers
        if hasattr(vla, "language_model") and hasattr(vla.language_model, "config"):
            vla.language_model.config._attn_implementation = "eager"
            for layer in vla.language_model.model.layers:
                if hasattr(layer, "self_attn"):
                    layer.self_attn._attn_implementation = "eager"

        model = cls(vla, method=method, **kwargs)
        model.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        model.action_tokenizer = ActionTokenizer(model.processor.tokenizer)
        model.eval()

        logger.info(f"TestTimeAttentionModulationV2 loaded: method={method}, kwargs={kwargs}")
        return model
