"""
attention_modulation_testtime_vit.py

Vision-side attention bias: bias the ViT's self-attention toward task-relevant
patches using the LLM's grounding signal.

Training-free, two-pass, inference-only.

Method:
  Pass 1: Run the full model with eager attention. Capture the LLM's grounding
          signal (sharpest head at L11) as a 256-dim importance mask.
  Pass 2: Hook into the DINOv2 and SigLIP ViT self-attention layers. Before
          softmax, add β * mask[j] to the attention logits for each key patch j.
          This makes all patches attend more to the important ones, so the ViT
          extracts richer features for task-relevant regions.

  The ViT's patch embeddings and layer norms are untouched. Only the attention
  allocation changes.

  attn_logits[i, j] += β * mask[j]   for all query patches i, key patch j
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


class ViTAttentionBiasVLA(nn.Module):
    """
    Two-pass inference wrapper: uses LLM grounding attention to bias
    ViT self-attention toward task-relevant patches.
    """

    def __init__(
        self,
        vla,
        teacher_layer=11,
        teacher_head=-1,
        beta=1.0,
        vit_layers=None,
    ):
        """
        Args:
            vla: OpenVLAForActionPrediction (loaded with attn_implementation="eager")
            teacher_layer: LLM layer to extract mask from (default: 11)
            teacher_head: head index, -1 for auto sharpest (default: -1)
            beta: scaling factor for the attention logit bias (default: 1.0)
            vit_layers: which ViT block indices to bias (default: all blocks)
        """
        super().__init__()
        self.vla = vla
        self.teacher_layer = teacher_layer
        self.teacher_head = teacher_head
        self.beta = beta
        self.vit_layers = vit_layers  # None = all

        self._captured_mask = None  # [B, 256]
        self._patched_forwards = []  # (module, original_forward)

        for p in self.vla.parameters():
            p.requires_grad = False
        self.vla.eval()

    def _get_llm_layers(self):
        vla = self.vla
        if hasattr(vla, "base_model"):
            vla = vla.base_model
        if hasattr(vla, "language_model"):
            return vla.language_model.model.layers
        if hasattr(vla, "model") and hasattr(vla.model, "layers"):
            return vla.model.layers
        raise RuntimeError(f"Cannot find transformer layers in {type(vla)}")

    def _get_vit_featurizers(self):
        """Return list of (name, featurizer) for DINOv2 and SigLIP."""
        vb = self.vla.vision_backbone
        featurizers = []
        if hasattr(vb, "dino_featurizer"):
            featurizers.append(("dino", vb.dino_featurizer))
        if hasattr(vb, "siglip_featurizer"):
            featurizers.append(("siglip", vb.siglip_featurizer))
        if not featurizers and hasattr(vb, "featurizer"):
            featurizers.append(("main", vb.featurizer))
        return featurizers

    def _cleanup(self):
        for module, orig_fwd in self._patched_forwards:
            module.forward = orig_fwd
        self._patched_forwards = []

    # ================================================================
    # PASS 1: Capture LLM grounding attention
    # ================================================================

    def _capture_attention_pass1(self, input_ids, attention_mask, pixel_values):
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
                    patch_attn = last_tok[:, parent.teacher_head, :]
                else:
                    max_per_head = last_tok.max(dim=-1).values
                    best = max_per_head.argmax(dim=-1)
                    patch_attn = torch.stack(
                        [last_tok[b, best[b], :] for b in range(last_tok.shape[0])]
                    )
                # min-max normalize to [0, 1]
                mins = patch_attn.min(dim=-1, keepdim=True).values
                maxs = patch_attn.max(dim=-1, keepdim=True).values
                patch_attn = (patch_attn - mins) / (maxs - mins + 1e-8)
                parent._captured_mask = patch_attn.detach()
            return (outputs[0], None) + tuple(outputs[2:])

        self_attn.forward = patched_forward

        try:
            _ = self.vla(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                labels=None,
                output_attentions=False,
                return_dict=True,
            )
        finally:
            self._cleanup()

    # ================================================================
    # PASS 2: Bias ViT self-attention
    # ================================================================

    def _setup_vit_hooks(self):
        """Patch ViT attention blocks to add logit bias toward important patches."""
        parent = self
        featurizers = self._get_vit_featurizers()

        for name, featurizer in featurizers:
            blocks = featurizer.blocks
            num_blocks = len(blocks)

            # Determine which blocks to bias
            if self.vit_layers is not None:
                target_blocks = self.vit_layers
            else:
                target_blocks = list(range(num_blocks))

            for block_idx in target_blocks:
                if block_idx >= num_blocks:
                    continue
                attn_module = blocks[block_idx].attn
                orig_fwd = attn_module.forward
                self._patched_forwards.append((attn_module, orig_fwd))

                def make_patched(original, attn_mod):
                    def patched_forward(x):
                        mask = parent._captured_mask
                        if mask is None:
                            return original(x)

                        # Manually compute attention with bias
                        # x shape: [B, N, C] where N = num_patches (256) + register tokens
                        B, N, C = x.shape
                        num_heads = attn_mod.num_heads
                        head_dim = attn_mod.head_dim if hasattr(attn_mod, 'head_dim') else C // num_heads

                        qkv = attn_mod.qkv(x).reshape(B, N, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
                        q, k, v = qkv.unbind(0)  # each [B, H, N, head_dim]

                        # Apply q_norm and k_norm if they exist
                        if hasattr(attn_mod, 'q_norm') and attn_mod.q_norm is not None:
                            q = attn_mod.q_norm(q)
                        if hasattr(attn_mod, 'k_norm') and attn_mod.k_norm is not None:
                            k = attn_mod.k_norm(k)

                        # Compute attention scores
                        scale = head_dim ** -0.5
                        attn = (q * scale) @ k.transpose(-2, -1)  # [B, H, N, N]

                        # Add bias to the first 256 key positions (image patches)
                        # DINOv2 has register tokens, so patches might not start at 0.
                        # DINOv2 with registers: [CLS, reg1, reg2, reg3, reg4, patch_0, ..., patch_255]
                        # So patches start after CLS + register tokens.
                        # SigLIP: no CLS token, patches are [patch_0, ..., patch_255]
                        #
                        # We need to figure out where the 256 image patches are.
                        # N could be 256 (SigLIP), 261 (DINOv2: 1 CLS + 4 registers + 256 patches),
                        # or other sizes.
                        num_image_patches = NUM_PATCHES
                        patch_start = N - num_image_patches  # patches are at the end

                        b = mask.to(attn.device, attn.dtype)  # [B, 256]
                        bias = parent.beta * b.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, 256]

                        attn[:, :, :, patch_start:patch_start + num_image_patches] = (
                            attn[:, :, :, patch_start:patch_start + num_image_patches] + bias
                        )

                        # Softmax + dropout
                        attn = attn.softmax(dim=-1)
                        if hasattr(attn_mod, 'attn_drop'):
                            attn = attn_mod.attn_drop(attn)

                        # Compute output
                        x_out = (attn @ v).transpose(1, 2).reshape(B, N, C)
                        x_out = attn_mod.proj(x_out)
                        if hasattr(attn_mod, 'proj_drop'):
                            x_out = attn_mod.proj_drop(x_out)

                        return x_out

                    return patched_forward

                attn_module.forward = make_patched(orig_fwd, attn_module)

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

        # PASS 1: capture LLM grounding attention
        self._captured_mask = None
        self._capture_attention_pass1(input_ids, attention_mask, pixel_values)

        if self._captured_mask is None:
            logger.warning("No attention captured — running vanilla predict_action")
            return self.vla.predict_action(
                input_ids, unnorm_key=unnorm_key, pixel_values=pixel_values, do_sample=False
            )

        # PASS 2: bias ViT attention and predict
        self._setup_vit_hooks()
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

        return action

    # ================================================================
    # Loader
    # ================================================================

    @classmethod
    def from_pretrained(cls, model_name, device_map="auto", **kwargs):
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

        # Force eager attention on LLM (needed for Pass 1)
        if hasattr(vla, "language_model") and hasattr(vla.language_model, "config"):
            vla.language_model.config._attn_implementation = "eager"
            for layer in vla.language_model.model.layers:
                if hasattr(layer, "self_attn"):
                    layer.self_attn._attn_implementation = "eager"

        model = cls(vla, **kwargs)
        model.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        model.action_tokenizer = ActionTokenizer(model.processor.tokenizer)
        model.eval()

        logger.info(f"ViTAttentionBiasVLA loaded: kwargs={kwargs}")
        return model
