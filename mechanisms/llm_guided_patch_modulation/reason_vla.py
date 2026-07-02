"""
reason_vla.py

Two-pass visual reasoning wrapper for OpenVLA (HF extern version).
Pass 1 calls the full vla.forward(output_hidden_states=True),
getting hidden states from output.hidden_states[-1].

Architecture:
    ReasonVLA wraps OpenVLAForActionPrediction / PrismaticForConditionalGeneration and adds:
      - VisualReasoner: gated FFN that processes LLM hidden states at image positions
      - Two unmergers: project reasoning hint back to DINOv2 / SigLIP patch-embed space

    Pass 1: Full vla.forward(output_hidden_states=True) → extract hidden_states[-1]
            at image token positions → VisualReasoner → unmergers → reasoning hints
    Pass 2: Run patch_embed manually, add hints, identity trick, call vla.forward()
            → vla.forward() handles projector → LLM → loss

    Stage 1: Freeze everything, train only VisualReasoner + unmergers
    Stage 2: Add LoRA to LLM, train LoRA + VisualReasoner + unmergers

Designed to work with HF AutoClasses:
    vla = AutoModelForVision2Seq.from_pretrained("openvla/openvla-7b", ...)
    model = ReasonVLA(vla)
"""

import argparse
from copy import deepcopy
from contextlib import contextmanager, nullcontext
import itertools
import logging
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from accelerate import Accelerator
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict, PeftModel
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor, AutoConfig, AutoImageProcessor
from transformers.optimization import get_cosine_schedule_with_warmup

import wandb

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.models.backbones.llm.prompting import PurePromptBuilder, VicunaV15ChatPromptBuilder
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics

logger = logging.getLogger(__name__)

IGNORE_INDEX = -100

# ---- Model Definitions ----

class VisualReasoner(nn.Module):
    """Gated feed-forward network that produces a reasoning signal from LLM hidden states."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        hidden_dim = in_dim * 2
        self.visual_reasoner = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(0.1),
        )
        self.gate = nn.Linear(in_dim, out_dim)
        self.proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.visual_reasoner(x)
        gate_values = torch.sigmoid(self.gate(x))
        return gate_values * self.proj(features)


class PatchUnmerger(nn.Module):
    """
    Projects from LLM hidden-state space back to a vision encoder's patch-embed space.
    No spatial merge to reverse (unlike Qwen) — just LayerNorm + Linear.
    """

    def __init__(self, llm_dim: int, vision_dim: int):
        super().__init__()
        self.ln = nn.LayerNorm(llm_dim)
        self.proj = nn.Linear(llm_dim, vision_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.ln(x))


# ---- Identity trick utilities ----

def identity_forward(x, *args, **kwargs):
    return x


@contextmanager
def disable_patch_embeds(vision_backbone):
    """
    Replace patch_embed.forward with identity on both featurizers
    so that pre-computed modified embeddings pass through unchanged.
    """
    is_fused = hasattr(vision_backbone, "fused_featurizer") and vision_backbone.use_fused_vision_backbone

    # featurizer = DINOv2 ViT-L (primary backbone)
    orig_main = vision_backbone.featurizer.patch_embed.forward
    vision_backbone.featurizer.patch_embed.forward = identity_forward
    orig_fused = None
    if is_fused:
        # fused_featurizer = SigLIP ViT-SO (secondary backbone)
        orig_fused = vision_backbone.fused_featurizer.patch_embed.forward
        vision_backbone.fused_featurizer.patch_embed.forward = identity_forward

    try:
        yield
    finally:
        vision_backbone.featurizer.patch_embed.forward = orig_main          # restore DINOv2
        if is_fused and orig_fused is not None:
            vision_backbone.fused_featurizer.patch_embed.forward = orig_fused  # restore SigLIP


@contextmanager
def override_vision_backbone(vision_backbone, precomputed_features):
    """
    Replace vision_backbone.forward to return precomputed features.
    This lets vla.forward() call vision_backbone(pixel_values) internally
    and get our hint-augmented features instead.
    """
    orig_forward = vision_backbone.forward
    vision_backbone.forward = lambda *args, **kwargs: precomputed_features
    try:
        yield
    finally:
        vision_backbone.forward = orig_forward


# ---- Main wrapper ----

class ReasonVLA(nn.Module):
    """
    Two-pass visual reasoning wrapper around OpenVLAForActionPrediction.
      - forward() dispatches to second_forward based on reasoning_hint state
      - second_forward: patch_embed → add hint → identity trick → vla.forward()
      - vla.forward() handles projector → LLM → loss
    """

    def __init__(self, vla, hidden_layer=-1, feedback_mode="additive"):
        """
        Args:
            vla: an OpenVLAForActionPrediction or PrismaticForConditionalGeneration loaded
                 via AutoModelForVision2Seq.from_pretrained(...)
            hidden_layer: which LLM hidden layer to extract for reasoning (default: -1 = last layer).
                          Based on logit lens analysis, layers 23-24 are where action decisions form.
            feedback_mode: how to inject the hint into patches:
                  "additive" — patches = patches + hint (default, preserves original features)
                  "film"     — patches = (1 + gamma) * patches + beta (scale + shift per patch per dim)
                  "gated"    — patches = patches + gate * hint (learned gate controls how much hint per patch)
                  "adaln"    — patches = gamma * norm(patches) + beta (adaptive layer norm, full distribution control)
                  "scaled"   — patches = patches + alpha * normalize(hint) * ||patches|| (hint normalized to match patch scale)
        """
        super().__init__()
        self.vla = vla
        self.hidden_layer = hidden_layer
        self.feedback_mode = feedback_mode

        llm_dim = vla.config.text_config.hidden_size  # e.g. 4096
        vb = vla.vision_backbone

        # Detect fused backbone (DINOv2 + SigLIP)
        self.is_fused = hasattr(vb, "fused_featurizer") and vb.use_fused_vision_backbone

        # Get vision dimensions and num_patches
        # featurizer = DINOv2 ViT-L, embed_dim=1024
        self.main_vision_dim = vb.featurizer.embed_dim
        self.num_patches = vb.featurizer.patch_embed.num_patches  # 256 for 224px / patch_size=14
        if self.is_fused:
            # fused_featurizer = SigLIP ViT-SO, embed_dim=1152
            self.fused_vision_dim = vb.fused_featurizer.embed_dim

        # Reasoning modules
        self.visual_reasoner = VisualReasoner(llm_dim, llm_dim)

        if feedback_mode == "additive":
            # Single unmerger → hint (patches = patches + hint)
            self.main_unmerger = PatchUnmerger(llm_dim, self.main_vision_dim)
            if self.is_fused:
                self.fused_unmerger = PatchUnmerger(llm_dim, self.fused_vision_dim)

        elif feedback_mode == "film":
            # Two unmergers per encoder → gamma + beta
            # patches = (1 + gamma) * patches + beta
            self.main_unmerger_gamma = PatchUnmerger(llm_dim, self.main_vision_dim)
            self.main_unmerger_beta = PatchUnmerger(llm_dim, self.main_vision_dim)
            if self.is_fused:
                self.fused_unmerger_gamma = PatchUnmerger(llm_dim, self.fused_vision_dim)
                self.fused_unmerger_beta = PatchUnmerger(llm_dim, self.fused_vision_dim)

        elif feedback_mode == "gated":
            # Unmerger for hint + gate network
            # patches = patches + gate * hint
            self.main_unmerger = PatchUnmerger(llm_dim, self.main_vision_dim)
            self.main_gate = nn.Sequential(
                nn.LayerNorm(llm_dim),
                nn.Linear(llm_dim, 1),
                nn.Sigmoid(),
            )
            if self.is_fused:
                self.fused_unmerger = PatchUnmerger(llm_dim, self.fused_vision_dim)
                self.fused_gate = nn.Sequential(
                    nn.LayerNorm(llm_dim),
                    nn.Linear(llm_dim, 1),
                    nn.Sigmoid(),
                )

        elif feedback_mode == "adaln":
            # Two unmergers per encoder → gamma + beta (same modules as FiLM)
            # patches = gamma * normalize(patches) + beta
            self.main_unmerger_gamma = PatchUnmerger(llm_dim, self.main_vision_dim)
            self.main_unmerger_beta = PatchUnmerger(llm_dim, self.main_vision_dim)
            if self.is_fused:
                self.fused_unmerger_gamma = PatchUnmerger(llm_dim, self.fused_vision_dim)
                self.fused_unmerger_beta = PatchUnmerger(llm_dim, self.fused_vision_dim)

        elif feedback_mode == "scaled":
            # Same unmerger as additive, but hint is L2-normalized and scaled to match patch magnitude
            # patches = patches + alpha * (hint / ||hint||) * ||patches||
            self.main_unmerger = PatchUnmerger(llm_dim, self.main_vision_dim)
            self.hint_alpha = nn.Parameter(torch.tensor(5.0))  # strong perturbation to create gradient signal
            if self.is_fused:
                self.fused_unmerger = PatchUnmerger(llm_dim, self.fused_vision_dim)
                self.hint_alpha_fused = nn.Parameter(torch.tensor(5.0))

        # Reasoning hint state (set during forward, between pass 1 and pass 2)
        self.reasoning_hint = None

    def set_image_reasoning(self, image_hidden: torch.Tensor) -> None:
        """
        Compute reasoning hints from LLM hidden states at image positions.
        """
        reasoning_out = self.visual_reasoner(image_hidden)       # [bsz, num_patches, llm_dim]

        if self.feedback_mode == "additive":
            self._hint_main = self.main_unmerger(reasoning_out)      # [bsz, 256, 1024]
            if self.is_fused:
                self._hint_fused = self.fused_unmerger(reasoning_out)  # [bsz, 256, 1152]

        elif self.feedback_mode == "film":
            self._gamma_main = self.main_unmerger_gamma(reasoning_out)  # [bsz, 256, 1024]
            self._beta_main = self.main_unmerger_beta(reasoning_out)    # [bsz, 256, 1024]
            if self.is_fused:
                self._gamma_fused = self.fused_unmerger_gamma(reasoning_out)  # [bsz, 256, 1152]
                self._beta_fused = self.fused_unmerger_beta(reasoning_out)    # [bsz, 256, 1152]

        elif self.feedback_mode == "gated":
            self._hint_main = self.main_unmerger(reasoning_out)        # [bsz, 256, 1024]
            self._gate_main = self.main_gate(reasoning_out)            # [bsz, 256, 1]
            if self.is_fused:
                self._hint_fused = self.fused_unmerger(reasoning_out)  # [bsz, 256, 1152]
                self._gate_fused = self.fused_gate(reasoning_out)      # [bsz, 256, 1]

        elif self.feedback_mode == "adaln":
            self._gamma_main = self.main_unmerger_gamma(reasoning_out)  # [bsz, 256, 1024]
            self._beta_main = self.main_unmerger_beta(reasoning_out)    # [bsz, 256, 1024]
            if self.is_fused:
                self._gamma_fused = self.fused_unmerger_gamma(reasoning_out)
                self._beta_fused = self.fused_unmerger_beta(reasoning_out)

        elif self.feedback_mode == "scaled":
            self._hint_main = self.main_unmerger(reasoning_out)      # [bsz, 256, 1024]
            if self.is_fused:
                self._hint_fused = self.fused_unmerger(reasoning_out)  # [bsz, 256, 1152]

        self.reasoning_hint = True  # flag that hint is ready

    def reset_image_reasoning(self) -> None:
        self.reasoning_hint = None
        self._hint_main = None
        self._hint_fused = None
        self._gamma_main = None
        self._beta_main = None
        self._gamma_fused = None
        self._beta_fused = None
        self._gate_main = None
        self._gate_fused = None

    def _adaln(self, patches, gamma, beta):
        """Adaptive Layer Normalization: gamma * normalize(patches) + beta."""
        mean = patches.mean(dim=-1, keepdim=True)
        std = patches.std(dim=-1, keepdim=True)
        patches_normed = (patches - mean) / (std + 1e-6)
        return gamma.to(patches) * patches_normed + beta.to(patches)

    def _scaled_hint(self, patches, hint, alpha):
        """Normalize hint to match patch scale: patches + alpha * normalize(hint) * ||patches||."""
        hint = hint.to(patches)
        hint_normed = hint / (hint.norm(dim=-1, keepdim=True) + 1e-6)
        patch_scale = patches.norm(dim=-1, keepdim=True)
        return patches + alpha * hint_normed * patch_scale

    def second_forward(self, input_ids, attention_mask, pixel_values, *args, **kwargs):
        """
        Second forward pass using the identity trick.
          1. Run patch_embed manually → get embeddings
          2. Add reasoning hints to embeddings
          3. Identity trick on patch_embed, run through ViT blocks
          4. Override vision_backbone.forward → call vla.forward()
             vla.forward() handles: projector → embeddings → LLM → loss
        """
        vb = self.vla.vision_backbone

        if self.is_fused:
            # Split channel-stacked pixel values: [bsz, 6, H, W] → 2 × [bsz, 3, H, W]
            # img_main = DINOv2 input, img_fused = SigLIP input
            img_main, img_fused = torch.split(pixel_values, [3, 3], dim=1)

            # Run patch_embed manually for each ViT
            main_patches = vb.featurizer.patch_embed(img_main)          # DINOv2: [bsz, 256, 1024]
            fused_patches = vb.fused_featurizer.patch_embed(img_fused)  # SigLIP: [bsz, 256, 1152]

            # Apply reasoning feedback
            if self.feedback_mode == "additive":
                main_patches = main_patches + self._hint_main.to(main_patches)
                fused_patches = fused_patches + self._hint_fused.to(fused_patches)
            elif self.feedback_mode == "film":
                main_patches = (1 + self._gamma_main.to(main_patches)) * main_patches + self._beta_main.to(main_patches)
                fused_patches = (1 + self._gamma_fused.to(fused_patches)) * fused_patches + self._beta_fused.to(fused_patches)
            elif self.feedback_mode == "gated":
                main_patches = main_patches + self._gate_main.to(main_patches) * self._hint_main.to(main_patches)
                fused_patches = fused_patches + self._gate_fused.to(fused_patches) * self._hint_fused.to(fused_patches)
            elif self.feedback_mode == "adaln":
                main_patches = self._adaln(main_patches, self._gamma_main, self._beta_main)
                fused_patches = self._adaln(fused_patches, self._gamma_fused, self._beta_fused)
            elif self.feedback_mode == "scaled":
                main_patches = self._scaled_hint(main_patches, self._hint_main, self.hint_alpha)
                fused_patches = self._scaled_hint(fused_patches, self._hint_fused, self.hint_alpha_fused)

            # Identity trick on patch_embeds, run modified patches through ViT blocks
            with disable_patch_embeds(vb):
                main_features = vb.featurizer(main_patches)          # DINOv2: [bsz, 256, 1024]
                fused_features = vb.fused_featurizer(fused_patches)  # SigLIP: [bsz, 256, 1152]
            patch_features = torch.cat([main_features, fused_features], dim=2)  # [bsz, 256, 2176]

        else:
            # Single backbone
            patches = vb.featurizer.patch_embed(pixel_values)
            if self.feedback_mode == "additive":
                patches = patches + self._hint_main.to(patches)
            elif self.feedback_mode == "film":
                patches = (1 + self._gamma_main.to(patches)) * patches + self._beta_main.to(patches)
            elif self.feedback_mode == "gated":
                patches = patches + self._gate_main.to(patches) * self._hint_main.to(patches)
            elif self.feedback_mode == "adaln":
                patches = self._adaln(patches, self._gamma_main, self._beta_main)
            elif self.feedback_mode == "scaled":
                patches = self._scaled_hint(patches, self._hint_main, self.hint_alpha)
            with disable_patch_embeds(vb):
                patch_features = vb.featurizer(patches)

        # Override vision_backbone.forward to return our hint-augmented features,
        # then call vla.forward() — it handles projector → LLM → loss.
        with override_vision_backbone(vb, patch_features):
            output = self.vla(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,  # ignored by overridden vb.forward
                *args,
                **kwargs,
            )

        return output

    def forward(self, stage, input_ids, attention_mask, pixel_values, *args, **kwargs):
        """
          - If reasoning_hint is set → second_forward (pass 2 with hint)
          - Otherwise → normal vla forward (pass 1 to extract hidden states)
        """
        if self.reasoning_hint is not None:
            return self.second_forward(input_ids, attention_mask, pixel_values, *args, **kwargs)

        # Pass 1: full vla.forward() to extract hidden states at image positions.
        # Unlike model_wrapper.py which calls the inner model (no head) and gets .last_hidden_state,
        # here we call the full model because OpenVLA has no single inner module that does
        # vision + projector + LLM without the head.
        # The training loop should use output.hidden_states[-1] to get the last layer hidden states.
        output = self.vla(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            labels=None,  # no loss for pass 1
            output_hidden_states=True,
            return_dict=True,
        )

        return output

    # ---- Checkpoint utilities ----

    def save_reasoning_modules(self, path: str) -> None:
        """Save only the reasoning-specific modules (not the VLA)."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        save_dict = {
            "visual_reasoner": self.visual_reasoner.state_dict(),
            "feedback_mode": self.feedback_mode,
        }
        if self.feedback_mode == "additive":
            save_dict["main_unmerger"] = self.main_unmerger.state_dict()
            if self.is_fused:
                save_dict["fused_unmerger"] = self.fused_unmerger.state_dict()
        elif self.feedback_mode == "film":
            save_dict["main_unmerger_gamma"] = self.main_unmerger_gamma.state_dict()
            save_dict["main_unmerger_beta"] = self.main_unmerger_beta.state_dict()
            if self.is_fused:
                save_dict["fused_unmerger_gamma"] = self.fused_unmerger_gamma.state_dict()
                save_dict["fused_unmerger_beta"] = self.fused_unmerger_beta.state_dict()
        elif self.feedback_mode == "gated":
            save_dict["main_unmerger"] = self.main_unmerger.state_dict()
            save_dict["main_gate"] = self.main_gate.state_dict()
            if self.is_fused:
                save_dict["fused_unmerger"] = self.fused_unmerger.state_dict()
                save_dict["fused_gate"] = self.fused_gate.state_dict()
        elif self.feedback_mode == "adaln":
            save_dict["main_unmerger_gamma"] = self.main_unmerger_gamma.state_dict()
            save_dict["main_unmerger_beta"] = self.main_unmerger_beta.state_dict()
            if self.is_fused:
                save_dict["fused_unmerger_gamma"] = self.fused_unmerger_gamma.state_dict()
                save_dict["fused_unmerger_beta"] = self.fused_unmerger_beta.state_dict()
        elif self.feedback_mode == "scaled":
            save_dict["main_unmerger"] = self.main_unmerger.state_dict()
            save_dict["hint_alpha"] = self.hint_alpha.data
            if self.is_fused:
                save_dict["fused_unmerger"] = self.fused_unmerger.state_dict()
                save_dict["hint_alpha_fused"] = self.hint_alpha_fused.data
        torch.save(save_dict, path)
        logger.info(f"Reasoning modules saved ({self.feedback_mode}): {path}")

    def load_reasoning_modules(self, path: str) -> None:
        """Load reasoning-specific modules from checkpoint."""
        state_dict = torch.load(path, map_location="cpu")
        self.visual_reasoner.load_state_dict(state_dict["visual_reasoner"])

        # Handle both additive and film checkpoints
        saved_mode = state_dict.get("feedback_mode", "additive")  # backwards compatible
        if saved_mode == "additive" and self.feedback_mode == "additive":
            self.main_unmerger.load_state_dict(state_dict["main_unmerger"])
            if self.is_fused and "fused_unmerger" in state_dict:
                self.fused_unmerger.load_state_dict(state_dict["fused_unmerger"])
        elif saved_mode == "film" and self.feedback_mode == "film":
            self.main_unmerger_gamma.load_state_dict(state_dict["main_unmerger_gamma"])
            self.main_unmerger_beta.load_state_dict(state_dict["main_unmerger_beta"])
            if self.is_fused and "fused_unmerger_gamma" in state_dict:
                self.fused_unmerger_gamma.load_state_dict(state_dict["fused_unmerger_gamma"])
                self.fused_unmerger_beta.load_state_dict(state_dict["fused_unmerger_beta"])
        elif saved_mode == "gated" and self.feedback_mode == "gated":
            self.main_unmerger.load_state_dict(state_dict["main_unmerger"])
            self.main_gate.load_state_dict(state_dict["main_gate"])
            if self.is_fused and "fused_unmerger" in state_dict:
                self.fused_unmerger.load_state_dict(state_dict["fused_unmerger"])
                self.fused_gate.load_state_dict(state_dict["fused_gate"])
        elif saved_mode == "adaln" and self.feedback_mode == "adaln":
            self.main_unmerger_gamma.load_state_dict(state_dict["main_unmerger_gamma"])
            self.main_unmerger_beta.load_state_dict(state_dict["main_unmerger_beta"])
            if self.is_fused and "fused_unmerger_gamma" in state_dict:
                self.fused_unmerger_gamma.load_state_dict(state_dict["fused_unmerger_gamma"])
                self.fused_unmerger_beta.load_state_dict(state_dict["fused_unmerger_beta"])
        elif saved_mode == "scaled" and self.feedback_mode == "scaled":
            self.main_unmerger.load_state_dict(state_dict["main_unmerger"])
            self.hint_alpha.data = state_dict["hint_alpha"]
            if self.is_fused and "fused_unmerger" in state_dict:
                self.fused_unmerger.load_state_dict(state_dict["fused_unmerger"])
                self.hint_alpha_fused.data = state_dict["hint_alpha_fused"]
        elif saved_mode != self.feedback_mode:
            raise ValueError(f"Checkpoint was saved with feedback_mode='{saved_mode}' "
                           f"but model uses feedback_mode='{self.feedback_mode}'")
        logger.info(f"Reasoning modules loaded ({self.feedback_mode}): {path}")

    # ---- Inference methods ----

    @classmethod
    def from_finetuned(cls, model_name, checkpoint_path, stage=1, lora_dir=None,
                        merge_path=None, merge_lora_dir=None,
                        device_map="auto", hidden_layer=-1, feedback_mode="additive"):
        """
        Load a trained ReasonVLA for inference, with optional checkpoint merging.

        Args:
            model_name: HuggingFace model name for the base OpenVLA (e.g. "openvla/openvla-7b-finetuned-libero-spatial")
            checkpoint_path: Path to .pth checkpoint (e.g. final_checkpoint_stage1.pth)
            stage: 1 or 2
            lora_dir: Directory with LoRA adapter files (required for stage 2)
            merge_path: Path to second .pth checkpoint to average with (optional)
            merge_lora_dir: Directory with second LoRA adapter to average with (optional, stage 2 only)
            device_map: Device map for model loading (default: "auto")
            hidden_layer: which LLM hidden layer to extract for reasoning (default: -1 = last)
        """
        # Register OpenVLA model to HF Auto Classes
        AutoConfig.register("openvla", OpenVLAConfig)
        AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
        AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
        AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

        # Load base VLA
        try:
            import flash_attn  # noqa: F401
            attn_impl = "flash_attention_2"
        except ImportError:
            attn_impl = None

        vla = AutoModelForVision2Seq.from_pretrained(
            model_name,
            **({"attn_implementation": attn_impl} if attn_impl else {}),
            device_map=device_map,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )

        # Wrap in ReasonVLA
        model = cls(vla, hidden_layer=hidden_layer, feedback_mode=feedback_mode)
        model.stage = stage
        model.model_name = model_name

        # Load processor and action tokenizer for inference
        model.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        model.action_tokenizer = ActionTokenizer(model.processor.tokenizer)

        # For stage 2, apply LoRA before loading checkpoint
        if stage == 2:
            if lora_dir is None:
                raise ValueError("--lora_dir is required for stage 2")
            model.vla.load_adapter(lora_dir)

        # Load reasoning module weights
        if not os.path.exists(checkpoint_path):
            raise ValueError(f"Checkpoint not found: {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location="cpu")

        # Merge with second checkpoint if provided
        if merge_path is not None:
            if not os.path.exists(merge_path):
                raise ValueError(f"Merge checkpoint not found: {merge_path}")

            # Merge LoRA weights (stage 2 only)
            # Note: model.vla here is NOT a PeftModel (load_adapter uses inject_adapter_in_model),
            # so get_peft_model_state_dict / set_peft_model_state_dict won't work.
            # Instead, we directly average the LoRA parameters in-place.
            if stage == 2 and merge_lora_dir is not None:
                # Save first adapter's LoRA weights
                lora_state_a = {
                    n: p.data.clone() for n, p in model.vla.named_parameters()
                    if "lora_" in n
                }
                # Load second adapter (overwrites first adapter's weights in-place)
                model.vla.load_adapter(merge_lora_dir)
                # Average first and second adapter weights
                for n, p in model.vla.named_parameters():
                    if n in lora_state_a:
                        p.data.copy_((lora_state_a[n] + p.data) / 2)
                del lora_state_a
                logger.info(f"Merged LoRA weights from {lora_dir} and {merge_lora_dir}")

            # Merge reasoning modules
            merge_state_dict = torch.load(merge_path, map_location="cpu")
            state_dict["visual_reasoner"] = {
                k: (state_dict["visual_reasoner"][k] + merge_state_dict["visual_reasoner"][k]) / 2
                for k in state_dict["visual_reasoner"].keys()
                if k in merge_state_dict["visual_reasoner"]
            }
            # Average all non-visual_reasoner keys present in both checkpoints
            for key in state_dict:
                if key == "visual_reasoner":
                    continue
                if key in merge_state_dict:
                    state_dict[key] = {
                        k: (state_dict[key][k] + merge_state_dict[key][k]) / 2
                        for k in state_dict[key].keys()
                        if k in merge_state_dict[key]
                    }
            logger.info(f"Merged reasoning modules from {checkpoint_path} and {merge_path}")

        # Load (possibly merged) reasoning modules
        model.visual_reasoner.load_state_dict(state_dict["visual_reasoner"])
        saved_mode = state_dict.get("feedback_mode", "additive")
        if saved_mode != model.feedback_mode:
            raise ValueError(f"Checkpoint feedback_mode='{saved_mode}' != model feedback_mode='{model.feedback_mode}'")
        if model.feedback_mode in ("additive", "gated"):
            model.main_unmerger.load_state_dict(state_dict["main_unmerger"])
            if model.is_fused and "fused_unmerger" in state_dict:
                model.fused_unmerger.load_state_dict(state_dict["fused_unmerger"])
            if model.feedback_mode == "gated":
                model.main_gate.load_state_dict(state_dict["main_gate"])
                if model.is_fused and "fused_gate" in state_dict:
                    model.fused_gate.load_state_dict(state_dict["fused_gate"])
        elif model.feedback_mode in ("film", "adaln"):
            model.main_unmerger_gamma.load_state_dict(state_dict["main_unmerger_gamma"])
            model.main_unmerger_beta.load_state_dict(state_dict["main_unmerger_beta"])
            if model.is_fused and "fused_unmerger_gamma" in state_dict:
                model.fused_unmerger_gamma.load_state_dict(state_dict["fused_unmerger_gamma"])
                model.fused_unmerger_beta.load_state_dict(state_dict["fused_unmerger_beta"])
        elif model.feedback_mode == "scaled":
            model.main_unmerger.load_state_dict(state_dict["main_unmerger"])
            model.hint_alpha.data = state_dict["hint_alpha"]
            if model.is_fused and "fused_unmerger" in state_dict:
                model.fused_unmerger.load_state_dict(state_dict["fused_unmerger"])
                model.hint_alpha_fused.data = state_dict["hint_alpha_fused"]

        # Move reasoning modules to same device and dtype as VLA
        device = next(model.vla.parameters()).device
        dtype = next(model.vla.parameters()).dtype
        for p in model.get_reasoning_parameters():
            p.data = p.data.to(device=device, dtype=dtype)

        # Set eval mode
        model.vla.eval()
        model.eval()
        logger.info(f"ReasonVLA loaded (stage {stage}) from {checkpoint_path}")
        return model

    @torch.inference_mode()
    def generate(self, image, task_description, unnorm_key=None):
        """
        Two-pass action prediction.

        Pass 1: self.vla(input_ids, pixel_values, output_hidden_states=True)
                → hidden_states[-1][:, 1:1+num_patches, :] → set_image_reasoning()
        Pass 2: second_forward logic (split → patch_embed → add hints → identity trick
                → override vision_backbone) but calls predict_action instead of vla()

        Args:
            image: PIL Image or numpy array
            task_description: Task string (e.g. "pick up the red block")
            unnorm_key: Key for action unnormalization stats

        Returns:
            np.ndarray of continuous (unnormalized) actions, shape (action_dim,)
        """
        self.reset_image_reasoning()

        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        # Build prompt (match training: VicunaV15 for v01 models, Pure otherwise)
        model_name = getattr(self, "model_name", "")
        prompt_builder_fn = VicunaV15ChatPromptBuilder if "v01" in model_name else PurePromptBuilder
        prompt_builder = prompt_builder_fn("openvla")
        prompt_builder.add_turn("human", f"What action should the robot take to {task_description.lower()}?")
        prompt_text = prompt_builder.get_prompt()

        # Tokenize with HF processor (match baseline: cast everything to device + bf16)
        inputs = self.processor(prompt_text, image).to(self.vla.device, dtype=torch.bfloat16)
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask", None)
        pixel_values = inputs["pixel_values"]

        # === PASS 1: Extract hidden states (LoRA ON for stage 2) ===
        if getattr(self, "stage", 1) == 2 and hasattr(self.vla, "enable_adapters"):
            self.vla.enable_adapters()
        output = self.vla(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            labels=None,
            output_hidden_states=True,
            return_dict=True,
        )
        if getattr(self, "stage", 1) == 2 and hasattr(self.vla, "disable_adapters"):
            self.vla.disable_adapters()

        # Extract hidden states at image patch positions
        # Layout: [BOS, patch_1, ..., patch_256, text_tokens...]
        hidden_state = output.hidden_states[self.hidden_layer]
        image_reasoning = hidden_state[:, 1:1 + self.num_patches, :]

        # visual_reasoner → unmergers → store hints
        self.set_image_reasoning(image_reasoning)

        del output, hidden_state, image_reasoning
        torch.cuda.empty_cache()

        # === PASS 2: Predict action with hints (LoRA OFF) ===
        vb = self.vla.vision_backbone

        if self.is_fused:
            img_main, img_fused = torch.split(pixel_values, [3, 3], dim=1)
            main_patches = vb.featurizer.patch_embed(img_main)
            fused_patches = vb.fused_featurizer.patch_embed(img_fused)
            if self.feedback_mode == "additive":
                main_patches = main_patches + self._hint_main.to(main_patches)
                fused_patches = fused_patches + self._hint_fused.to(fused_patches)
            elif self.feedback_mode == "film":
                main_patches = (1 + self._gamma_main.to(main_patches)) * main_patches + self._beta_main.to(main_patches)
                fused_patches = (1 + self._gamma_fused.to(fused_patches)) * fused_patches + self._beta_fused.to(fused_patches)
            elif self.feedback_mode == "gated":
                main_patches = main_patches + self._gate_main.to(main_patches) * self._hint_main.to(main_patches)
                fused_patches = fused_patches + self._gate_fused.to(fused_patches) * self._hint_fused.to(fused_patches)
            elif self.feedback_mode == "adaln":
                main_patches = self._adaln(main_patches, self._gamma_main, self._beta_main)
                fused_patches = self._adaln(fused_patches, self._gamma_fused, self._beta_fused)
            elif self.feedback_mode == "scaled":
                main_patches = self._scaled_hint(main_patches, self._hint_main, self.hint_alpha)
                fused_patches = self._scaled_hint(fused_patches, self._hint_fused, self.hint_alpha_fused)
            with disable_patch_embeds(vb):
                main_features = vb.featurizer(main_patches)
                fused_features = vb.fused_featurizer(fused_patches)
            patch_features = torch.cat([main_features, fused_features], dim=2)

        else:
            patches = vb.featurizer.patch_embed(pixel_values)
            if self.feedback_mode == "additive":
                patches = patches + self._hint_main.to(patches)
            elif self.feedback_mode == "film":
                patches = (1 + self._gamma_main.to(patches)) * patches + self._beta_main.to(patches)
            elif self.feedback_mode == "gated":
                patches = patches + self._gate_main.to(patches) * self._hint_main.to(patches)
            elif self.feedback_mode == "adaln":
                patches = self._adaln(patches, self._gamma_main, self._beta_main)
            elif self.feedback_mode == "scaled":
                patches = self._scaled_hint(patches, self._hint_main, self.hint_alpha)
            with disable_patch_embeds(vb):
                patch_features = vb.featurizer(patches)

        # Override vision_backbone and predict action
        # LoRA already disabled above — predict_action runs with original LLM weights
        with override_vision_backbone(vb, patch_features):
            actions = self.vla.predict_action(
                input_ids,
                unnorm_key=unnorm_key,
                pixel_values=pixel_values,  # ignored by overridden vb.forward
                attention_mask=attention_mask,
                do_sample=False,
            )

        # Re-enable LoRA so the model is left in a consistent state
        if getattr(self, "stage", 1) == 2 and hasattr(self.vla, "enable_adapters"):
            self.vla.enable_adapters()

        return actions

    def get_reasoning_parameters(self):
        """Iterator over all trainable reasoning parameters (for optimizer)."""
        params = [self.visual_reasoner.parameters()]
        if self.feedback_mode == "additive":
            params.append(self.main_unmerger.parameters())
            if self.is_fused:
                params.append(self.fused_unmerger.parameters())
        elif self.feedback_mode == "film":
            params.append(self.main_unmerger_gamma.parameters())
            params.append(self.main_unmerger_beta.parameters())
            if self.is_fused:
                params.append(self.fused_unmerger_gamma.parameters())
                params.append(self.fused_unmerger_beta.parameters())
        elif self.feedback_mode == "gated":
            params.append(self.main_unmerger.parameters())
            params.append(self.main_gate.parameters())
            if self.is_fused:
                params.append(self.fused_unmerger.parameters())
                params.append(self.fused_gate.parameters())
        elif self.feedback_mode == "adaln":
            params.append(self.main_unmerger_gamma.parameters())
            params.append(self.main_unmerger_beta.parameters())
            if self.is_fused:
                params.append(self.fused_unmerger_gamma.parameters())
                params.append(self.fused_unmerger_beta.parameters())
        elif self.feedback_mode == "scaled":
            params.append(self.main_unmerger.parameters())
            params.append(iter([self.hint_alpha]))
            if self.is_fused:
                params.append(self.fused_unmerger.parameters())
                params.append(iter([self.hint_alpha_fused]))
        return itertools.chain(*params)

    def freeze_vla(self) -> None:
        """Freeze all VLA parameters (for stage 1)."""
        for param in self.vla.parameters():
            param.requires_grad = False
        self.vla.eval()

    def unfreeze_reasoning(self) -> None:
        """Unfreeze reasoning modules (for both stages)."""
        modules = [self.visual_reasoner]
        if self.feedback_mode == "additive":
            modules.append(self.main_unmerger)
            if self.is_fused:
                modules.append(self.fused_unmerger)
        elif self.feedback_mode == "film":
            modules.extend([self.main_unmerger_gamma, self.main_unmerger_beta])
            if self.is_fused:
                modules.extend([self.fused_unmerger_gamma, self.fused_unmerger_beta])
        elif self.feedback_mode == "gated":
            modules.extend([self.main_unmerger, self.main_gate])
            if self.is_fused:
                modules.extend([self.fused_unmerger, self.fused_gate])
        elif self.feedback_mode == "adaln":
            modules.extend([self.main_unmerger_gamma, self.main_unmerger_beta])
            if self.is_fused:
                modules.extend([self.fused_unmerger_gamma, self.fused_unmerger_beta])
        elif self.feedback_mode == "scaled":
            modules.append(self.main_unmerger)
            if self.is_fused:
                modules.append(self.fused_unmerger)
        for module in modules:
            for param in module.parameters():
                param.requires_grad = True
            module.train()
        # Standalone parameters (not in modules)
        if self.feedback_mode == "scaled":
            self.hint_alpha.requires_grad = True
            if self.is_fused:
                self.hint_alpha_fused.requires_grad = True

def load_training_state(checkpoint_path):
    """
    Load training state (optimizer, scheduler, global_step) from a checkpoint.

    Args:
        checkpoint_path: Path to a training_state-*.pth file, or a checkpoint-*.pth / directory.
                         If a checkpoint .pth is given, looks for the matching training_state file.
                         If a directory is given, finds the latest training_state file.
    Returns:
        dict with keys: global_step, epoch, stage, optimizer, lr_scheduler
        or None if no training state found.
    """
    if os.path.isdir(checkpoint_path):
        # Find latest training_state file in directory
        import glob
        state_files = sorted(glob.glob(os.path.join(checkpoint_path, "training_state-*.pth")))
        if not state_files:
            # Check subdirectories (stage1/, stage2/)
            state_files = sorted(glob.glob(os.path.join(checkpoint_path, "*/training_state-*.pth")))
        if not state_files:
            # Check for final training state
            for stage_num in [2, 1]:
                final_state = os.path.join(checkpoint_path, f"final_training_state_stage{stage_num}.pth")
                if os.path.exists(final_state):
                    state_files = [final_state]
                    break
        if not state_files:
            logger.warning(f"No training_state files found in {checkpoint_path}")
            return None
        state_path = state_files[-1]
    elif checkpoint_path.endswith("training_state"):
        state_path = checkpoint_path
    elif "checkpoint-" in checkpoint_path:
        # Derive training_state path from checkpoint path
        # checkpoint-5000.pth → training_state-5000.pth
        state_path = checkpoint_path.replace("checkpoint-", "training_state-")
    elif "final_checkpoint_stage" in checkpoint_path:
        # final_checkpoint_stage1.pth → final_training_state_stage1.pth
        state_path = checkpoint_path.replace("final_checkpoint_stage", "final_training_state_stage")
    else:
        state_path = checkpoint_path

    if not os.path.exists(state_path):
        logger.warning(f"Training state not found: {state_path}")
        return None

    state = torch.load(state_path, map_location="cpu")
    logger.info(f"Loaded training state from {state_path} (step {state['global_step']}, stage {state['stage']})")
    return state


# ---- Training Loop ----

def train_loop(
    model: ReasonVLA,
    dataloader: DataLoader,
    optimizer,
    lr_scheduler,
    accelerator: "Accelerator",
    action_tokenizer,
    pad_token_id: int,
    stage: int,
    max_steps: int,
    save_steps: int = 5000,
    output_dir: str = "runs",
    run_name: str = "debug",
    wandb_step_offset: int = 0,
    start_step: int = 0,
):
    """
    Stage 1: only VisualReasoner + unmergers are trainable (VLA frozen)
    Stage 2: LoRA + VisualReasoner + unmergers are trainable

    Each step:
      Pass 1: forward (no labels) → extract hidden_states[-1] at image positions
              → set_image_reasoning (visual_reasoner → unmergers → hints)
      Pass 2: forward with hints + labels → loss → backward

    Metrics (matching OpenVLA's base_strategy.py):
      - loss, action_accuracy, l1_loss, step_time, lr

    Uses HF Accelerator for gradient accumulation, mixed precision, DDP, etc.
    """
    import time

    run_name = run_name or "debug"
    # gets ReasonVLA from DDP wrapper
    unwrapped_model = accelerator.unwrap_model(model)
    num_patches = unwrapped_model.num_patches  # 256

    logger.info(f"Starting Stage {stage} training, max_steps={max_steps}"
                + (f", resuming from step {start_step}" if start_step > 0 else ""))

    global_step = start_step
    loss_cum = 0.0
    acc_cum = 0.0
    l1_cum = 0.0
    micro_count = 0

    # Save initial parameters to verify updates are happening (checked once at first checkpoint)
    old_params = None if start_step > 0 else {
        name: param.clone().detach()
        for name, param in unwrapped_model.visual_reasoner.named_parameters()
    }

    # For resume: count steps to skip in the dataloader (RLDS streams from start)
    steps_to_skip = start_step * accelerator.gradient_accumulation_steps if start_step > 0 else 0
    if steps_to_skip > 0:
        logger.info(f"Skipping {steps_to_skip} micro-batches to reach step {start_step}...")

    step_start_time = time.time()

    for epoch in range(999):  # effectively infinite; we stop at max_steps
        for batch in dataloader:
            # Skip already-completed batches on resume
            if steps_to_skip > 0:
                steps_to_skip -= 1
                if steps_to_skip % 1000 == 0 and steps_to_skip > 0:
                    logger.info(f"  ...{steps_to_skip} micro-batches remaining to skip")
                continue
            with accelerator.accumulate(model):
                # Reset hints
                unwrapped_model.reset_image_reasoning()
                # Move batch to device 
                # [1, seq_len] — [BOS, patch_placeholder×256, prompt_tokens..., action_token×7, EOS]
                input_ids = batch["input_ids"].to(accelerator.device)
                # [1, seq_len] — all 1s (no padding with batch_size=1)
                attention_mask = batch["attention_mask"].to(accelerator.device)
                # [1, 6, 224, 224] — channels 0-2 = DINOv2 input, channels 3-5 = SigLIP input
                pixel_values = batch["pixel_values"].to(accelerator.device)
                # [1, seq_len] — [IGNORE×(1+256+prompt_len), action_1, action_2, ..., action_7, EOS]
                labels = batch["labels"].to(accelerator.device)

                # ---------- Derive first-pass input_ids (no action tokens) ----------
                # Labels layout: [IGNORE, ..., IGNORE, action_1, ..., action_7, EOS]
                # Truncate at first action token so Pass 1 sees only the prompt.
                # Safe with batch_size=1 (gradient accumulation doesn't affect this).
                first_action_pos = (labels != IGNORE_INDEX).long().argmax(dim=1)
                cut = first_action_pos.min().item()
                first_pass_ids = input_ids[:, :cut]
                first_pass_attn = attention_mask[:, :cut]

                # ---------- Pass 1: Extract hidden states (no labels, no loss) ----------
                # Stage 1: no grad for pass 1 (VLA is frozen, just extracting features)
                # Stage 2: need grad for pass 1 (LoRA is learning)
                maybe_no_grad = torch.no_grad if stage == 1 else nullcontext

                with maybe_no_grad(): # ← no_grad block starts
                    output = model(      # VLA forward, no gradients
                        stage=stage,
                        input_ids=first_pass_ids,
                        attention_mask=first_pass_attn,
                        pixel_values=pixel_values,
                        return_dict=True,
                    )
                # Outside no_grad
                # Extract hidden states at image token positions.
                # Multimodal embeddings layout: [BOS, patch_1, ..., patch_256, text_tokens...]
                hidden_state = output.hidden_states[unwrapped_model.hidden_layer]
                image_reasoning = hidden_state[:, 1:1 + num_patches, :]  # [bsz, 256, 4096]

                # OUTSIDE no_grad, gradients tracked through VR weights
                # visual_reasoner → unmergers → store hints
                unwrapped_model.set_image_reasoning(image_reasoning)

                # Capture feedback signal norms (detached, no grad overhead)
                fb_mode = unwrapped_model.feedback_mode
                if fb_mode in ("additive", "gated", "scaled"):
                    _hint_main = unwrapped_model._hint_main
                    _hint_fused = unwrapped_model._hint_fused if unwrapped_model.is_fused else None
                    hint_norm_main = _hint_main.detach().float().norm().item() if _hint_main is not None else 0.0
                    hint_norm_fused = _hint_fused.detach().float().norm().item() if _hint_fused is not None else 0.0
                elif fb_mode in ("film", "adaln"):
                    _g_main = unwrapped_model._gamma_main
                    _b_main = unwrapped_model._beta_main
                    _g_fused = unwrapped_model._gamma_fused if unwrapped_model.is_fused else None
                    _b_fused = unwrapped_model._beta_fused if unwrapped_model.is_fused else None
                    hint_norm_main = (_g_main.detach().float().norm().item() + _b_main.detach().float().norm().item()) if _g_main is not None else 0.0
                    hint_norm_fused = (_g_fused.detach().float().norm().item() + _b_fused.detach().float().norm().item()) if _g_fused is not None else 0.0
                else:
                    hint_norm_main = 0.0
                    hint_norm_fused = 0.0
                hint_norm = (hint_norm_main**2 + hint_norm_fused**2) ** 0.5

                # Free Pass 1 activations to save GPU memory before Pass 2
                del output, hidden_state, image_reasoning
                torch.cuda.empty_cache()

                # ---------- Pass 2: Forward with hints + full input_ids → loss ----------
                # In stage 2, disable LoRA for pass 2 — only the reasoning hint modifies output.
                disable_lora_ctx = nullcontext # (no LoRA to disable)
                if stage == 2:
                    disable_lora_ctx = unwrapped_model.vla.disable_adapter

                with accelerator.autocast():
                    with disable_lora_ctx():
                        output2 = model(                     # VLA forward WITH hints
                            stage=stage,
                            input_ids=input_ids,             # full sequence (prompt + actions)
                            attention_mask=attention_mask,
                            pixel_values=pixel_values,
                            labels=labels,                   # computes loss
                            return_dict=True,
                        )
                        loss = output2.loss

                # ---------- Action metrics (matching OpenVLA base_strategy.py) ----------
                # Compute action token accuracy and L1 loss on continuous actions.
                # logits layout: [BOS, patch_1..patch_256, prompt..., action_1..action_7, EOS]
                # labels layout: [IGNORE, ..., IGNORE, action_1, ..., action_7, EOS]
                # Same slicing as OpenVLA: logits[:, num_patches:-1] aligned with labels[:, 1:]
                action_preds = output2.logits[:, num_patches:-1].argmax(dim=2)
                action_gt = labels[:, 1:].to(action_preds.device)
                mask = action_gt > action_tokenizer.action_token_begin_idx # only action tokens (not IGNORE, not text)

                action_accuracy = torch.tensor(0.0)
                action_l1_loss = torch.tensor(0.0)
                if mask.any():
                    correct_preds = (action_preds == action_gt) & mask
                    action_accuracy = correct_preds.sum().float() / mask.sum().float()

                    continuous_actions_pred = torch.tensor(
                        action_tokenizer.decode_token_ids_to_actions(action_preds[mask].cpu().numpy())
                    )
                    continuous_actions_gt = torch.tensor(
                        action_tokenizer.decode_token_ids_to_actions(action_gt[mask].cpu().numpy())
                    )
                    action_l1_loss = torch.nn.functional.l1_loss(continuous_actions_pred, continuous_actions_gt)

                if torch.isnan(loss) or torch.isinf(loss):
                    raise ValueError(f"Loss is {loss}, terminating training.")

                accelerator.backward(loss)        # gradients flow back to VR weights

                if accelerator.sync_gradients: # (for example: True only every 16th micro-batch.)
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)   # prevents gradient explosion

                    # Compute gradient norms before zero_grad 
                    if accelerator.is_main_process:
                        vr_grad_norm = sum(
                            p.grad.detach().float().norm().item() ** 2
                            for p in unwrapped_model.visual_reasoner.parameters()
                            if p.grad is not None
                        ) ** 0.5
                        # Unmerger grad norm — use all reasoning params minus visual_reasoner
                        um_params = list(unwrapped_model.get_reasoning_parameters())
                        vr_params_set = set(id(p) for p in unwrapped_model.visual_reasoner.parameters())
                        um_grad_norm = sum(
                            p.grad.detach().float().norm().item() ** 2
                            for p in um_params
                            if p.grad is not None and id(p) not in vr_params_set
                        ) ** 0.5
                        lora_grad_norm = 0.0
                        if stage == 2:
                            lora_grad_norm = sum(
                                p.grad.detach().float().norm().item() ** 2
                                for n, p in unwrapped_model.vla.named_parameters()
                                if p.grad is not None and "lora" in n.lower()
                            ) ** 0.5

                optimizer.step()       # AdamW updates VR + unmerger weights using accumulated gradients
                lr_scheduler.step()
                optimizer.zero_grad()   # clears gradients for next accumulation cycle

                loss_cum += loss.item()
                acc_cum += action_accuracy.item()
                l1_cum += action_l1_loss.item()
                micro_count += 1

                # Only count optimizer steps (not micro-batches) so max_steps is meaningful
                if accelerator.sync_gradients:
                    global_step += 1

                # ---------- Per-step wandb logging (matching OpenVLA VLAMetrics) ----------
                if accelerator.sync_gradients and accelerator.is_main_process:
                    step_time = time.time() - step_start_time
                    step_start_time = time.time()
                    current_lr = lr_scheduler.get_last_lr()[0]

                    gpu_mem_gb = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0

                    # Average metrics over all micro-batches in this optimizer step
                    avg_loss = loss_cum / micro_count if micro_count > 0 else 0.0
                    avg_acc = acc_cum / micro_count if micro_count > 0 else 0.0
                    avg_l1 = l1_cum / micro_count if micro_count > 0 else 0.0

                    wandb.log({
                        f"train/loss_stage{stage}": avg_loss,
                        f"train/action_accuracy_stage{stage}": avg_acc,
                        f"train/l1_loss_stage{stage}": avg_l1,
                        f"train/lr_stage{stage}": current_lr,
                        "train/global_step": global_step,
                        "train/epoch": epoch,
                        "train/step_time": step_time,
                        # Feedback module diagnostics
                        "feedback/hint_norm": hint_norm,
                        "feedback/hint_norm_main": hint_norm_main,
                        "feedback/hint_norm_fused": hint_norm_fused,
                        "feedback/grad_norm_visual_reasoner": vr_grad_norm,
                        "feedback/grad_norm_unmerge": um_grad_norm,
                        "feedback/grad_norm_lora": lora_grad_norm,
                        **({"feedback/hint_alpha": unwrapped_model.hint_alpha.item()} if fb_mode == "scaled" else {}),
                        **({"feedback/hint_alpha_fused": unwrapped_model.hint_alpha_fused.item()} if fb_mode == "scaled" and unwrapped_model.is_fused else {}),
                        # System
                        "system/gpu_memory_gb": gpu_mem_gb,
                    }, step=wandb_step_offset + global_step)

                    # Reset accumulators after logging
                    loss_cum = 0.0
                    acc_cum = 0.0
                    l1_cum = 0.0
                    micro_count = 0

                    # Console log every 10 optimizer steps
                    if global_step % 10 == 0:
                        logger.info(
                            f"Stage {stage} | Epoch {epoch} | Step {global_step} | "
                            f"loss: {avg_loss:.4f} | acc: {avg_acc:.4f} | "
                            f"l1: {avg_l1:.4f} | lr: {current_lr:.2e} | "
                            f"hint: {hint_norm:.4f} | vr_grad: {vr_grad_norm:.4f} | "
                            f"um_grad: {um_grad_norm:.4f}"
                        )

                    # Warn if no gradients at step 2 (something is wrong)
                    if global_step == 2 and vr_grad_norm == 0.0 and um_grad_norm == 0.0:
                        logger.warning("No gradients on visual_reasoner or unmerge at step 2! "
                                       "Check computation graph.")

                # ---------- Checkpointing ----------
                # Save frequently early (every save_steps), less often later (every 5× save_steps)
                # Transition at 10× save_steps (e.g., ~10 epochs)
                early_threshold = save_steps * 10
                current_save_interval = save_steps if global_step <= early_threshold else save_steps * 5
                should_save = global_step > 0 and global_step % current_save_interval == 0
                if should_save and accelerator.is_main_process:
                    # Verify visual_reasoner params updated (check once, then clear)
                    if old_params is not None:
                        unw_vr = accelerator.unwrap_model(model).visual_reasoner
                        updated = any(
                            not torch.equal(param.data.cpu(), old_params[name].cpu())
                            for name, param in unw_vr.named_parameters()
                        )
                        if not updated:
                            raise RuntimeError(
                                "[ERROR] visual_reasoner parameters not updating! "
                                "Check checkpoint_interval vs accumulated batch size."
                            )
                        old_params = None

                    logger.info(f"Stage {stage} Epoch {epoch} Step {global_step}, saving checkpoint...")

                    checkpoint_dir = os.path.join(output_dir, run_name, f"stage{stage}")
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    accelerator.unwrap_model(model).save_reasoning_modules(
                        os.path.join(checkpoint_dir, f"checkpoint-{global_step}.pth")
                    )
                    if stage == 2:
                        accelerator.unwrap_model(model).vla.save_pretrained(
                            os.path.join(checkpoint_dir, f"lora-{global_step}")
                        )
                    # Save optimizer, scheduler, and step for resume
                    training_state = {
                        "global_step": global_step,
                        "epoch": epoch,
                        "stage": stage,
                        "optimizer": optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                    }
                    torch.save(training_state, os.path.join(checkpoint_dir, f"training_state-{global_step}.pth"))
                    logger.info(f"Checkpoint + training state saved at step {global_step}")

                # Stop at max_steps
                if global_step >= max_steps:
                    logger.info(f"Stage {stage} reached max_steps={max_steps}, stopping.")
                    break

        if global_step >= max_steps:
            break

    # Save final checkpoint
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        final_dir = os.path.join(output_dir, run_name)
        os.makedirs(final_dir, exist_ok=True)
        accelerator.unwrap_model(model).save_reasoning_modules(
            os.path.join(final_dir, f"final_checkpoint_stage{stage}.pth")
        )
        if stage == 2:
            accelerator.unwrap_model(model).vla.save_pretrained(
                os.path.join(final_dir, "lora-final")
            )
        # Save final training state for resume
        training_state = {
            "global_step": global_step,
            "epoch": epoch,
            "stage": stage,
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
        }
        torch.save(training_state, os.path.join(final_dir, f"final_training_state_stage{stage}.pth"))
        logger.info(f"Stage {stage} final checkpoint saved.")


# ---- Main Routine ----

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vla-path", type=str, default="openvla/openvla-7b", help="Path to OpenVLA model")
    parser.add_argument("--data-root-dir", type=str, required=True, help="Path to RLDS dataset directory")
    parser.add_argument("--dataset-name", type=str, required=True, help="Name of fine-tuning dataset")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory for saving checkpoints")
    parser.add_argument("--training-stage", type=int, default=None,
                        help="Training stage (1 or 2, default: None = both stages)")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--max-steps", type=int, default=200_000, help="Max training steps per stage")
    parser.add_argument("--save-steps", type=int, default=5000, help="Steps between checkpoints")
    parser.add_argument("--image-aug", action="store_true", help="Enable image augmentations")
    parser.add_argument("--shuffle-buffer-size", type=int, default=100_000, help="RLDS shuffle buffer size")
    parser.add_argument("--lora-rank", type=int, default=32, help="LoRA rank for stage 2")
    parser.add_argument("--lora-dropout", type=float, default=0.0, help="LoRA dropout")
    parser.add_argument("--resume", type=str, default=None, help="Path to run dir or specific .pth checkpoint file")
    parser.add_argument("--resume-step", type=int, default=0,
                        help="Step to resume from (auto-detected from checkpoint if 0)")
    parser.add_argument("--grad-accum-steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--hidden-layer", type=int, default=-1,
                        help="LLM hidden layer to extract for reasoning (-1=last, 23=action decision layer)")
    parser.add_argument("--feedback-mode", type=str, default="additive",
                        choices=["additive", "film", "gated", "adaln", "scaled"],
                        help="How to inject: additive (patches+hint), film ((1+gamma)*patches+beta), gated (patches+gate*hint), adaln (gamma*norm(patches)+beta), scaled (patches+alpha*norm(hint)*||patches||)")
    args = parser.parse_args()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    run_name = os.environ.get("RUN_NAME", None)
    output_path = os.path.join(args.output_dir, run_name if run_name else "debug")
    os.makedirs(output_path, exist_ok=True)

    # Register OpenVLA model to HF Auto Classes
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    # Load OpenVLA Processor and Model
    processor = AutoProcessor.from_pretrained(args.vla_path, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        args.vla_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    # Wrap VLA with ReasonVLA
    model = ReasonVLA(vla, hidden_layer=args.hidden_layer, feedback_mode=args.feedback_mode)

    # Setup Accelerator
    accelerator = Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_steps=args.grad_accum_steps,
    )

    if accelerator.is_main_process:
        wandb.init(project="openvla_fixed", config=vars(args), name=run_name,
                   mode="disabled" if run_name is None else None)

    # Resume from checkpoint if provided
    resume_training_state = None
    if args.resume is not None:
        if args.resume.endswith(".pth"):
            # Direct path to specific checkpoint file
            checkpoint_path = args.resume
        else:
            # Directory — look for final_checkpoint_stage1.pth
            checkpoint_path = os.path.join(args.resume, "final_checkpoint_stage1.pth")
        if os.path.exists(checkpoint_path):
            logger.info(f"Loading checkpoint from {checkpoint_path}")
            model.load_reasoning_modules(checkpoint_path)
            # Try to load training state for full resume (optimizer, scheduler, step)
            resume_training_state = load_training_state(checkpoint_path)
            if resume_training_state and args.resume_step == 0:
                args.resume_step = resume_training_state["global_step"]
                logger.info(f"Auto-detected resume step: {args.resume_step}")
        else:
            logger.warning(f"No checkpoint found at {checkpoint_path}, starting from scratch.")

    # Build RLDS data pipeline (reusing OpenVLA's existing classes)
    # maps continuous actions ↔ discrete token IDs (256 bins per action dimension)
    action_tokenizer = ActionTokenizer(processor.tokenizer)
    # for each sample: takes image → applies image_processor transforms (resize to 224×224, normalize), takes action 
    # → tokenizes to 7 token IDs, takes task description 
    # → builds prompt "What action should the robot take to {task}?", tokenizes everything
    batch_transform = RLDSBatchTransform(
        action_tokenizer,
        processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder if "v01" not in args.vla_path else VicunaV15ChatPromptBuilder,
    )
    # streams from TFDS, applies transform, shuffles with buffer of 100k
    vla_dataset = RLDSDataset(
        Path(args.data_root_dir),
        args.dataset_name,
        batch_transform,
        resize_resolution=tuple(vla.config.image_sizes),
        shuffle_buffer_size=args.shuffle_buffer_size,
        image_aug=args.image_aug,
    )

    # Save dataset statistics (needed for de-normalization at inference)
    if accelerator.is_main_process:
        save_dataset_statistics(vla_dataset.dataset_statistics, Path(output_path))
    #pads sequences to same length (right-side padding), creates input_ids, attention_mask, 
    # labels (with IGNORE_INDEX=-100 for non-action positions), pixel_values
    base_collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right"
    )

    def collator(instances):
        batch = base_collator(instances)
        batch.pop("dataset_names", None)
        return batch

    dataloader = DataLoader(
        vla_dataset,
        batch_size=args.batch_size,
        sampler=None,
        collate_fn=collator,
        num_workers=0,  # RLDS/TFDS handles its own parallelism
    )

    # Freeze VLA and unfreeze reasoning modules (needed for both stages)
    model.freeze_vla()
    model.unfreeze_reasoning()

    # --- Stage 1: Train only reasoning modules ---
    if args.training_stage is None or args.training_stage == 1:
        s1_start_step = args.resume_step if (resume_training_state and resume_training_state.get("stage") == 1) else 0

        optimizer_stage1 = torch.optim.AdamW(model.get_reasoning_parameters(), lr=args.lr)
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer_stage1, num_warmup_steps=100, num_training_steps=args.max_steps,
        )

        # Restore optimizer/scheduler state if resuming stage 1
        if s1_start_step > 0 and resume_training_state:
            if "optimizer" in resume_training_state:
                optimizer_stage1.load_state_dict(resume_training_state["optimizer"])
                logger.info("Restored optimizer state for stage 1 resume")
            if "lr_scheduler" in resume_training_state:
                lr_scheduler.load_state_dict(resume_training_state["lr_scheduler"])
                logger.info("Restored scheduler state for stage 1 resume")

        # Wraps model (DDP if multi-GPU), optimizer, scheduler, dataloader. Moves everything to GPU.
        model, optimizer_stage1, lr_scheduler, dataloader = accelerator.prepare(
            model, optimizer_stage1, lr_scheduler, dataloader,
        )

        if accelerator.is_main_process:
            wandb.watch(accelerator.unwrap_model(model).visual_reasoner, log="all")

        train_loop(model, dataloader, optimizer_stage1, lr_scheduler, accelerator,
                   action_tokenizer=action_tokenizer,
                   pad_token_id=processor.tokenizer.pad_token_id,
                   stage=1, max_steps=args.max_steps, save_steps=args.save_steps,
                   output_dir=args.output_dir, run_name=run_name,
                   start_step=s1_start_step)

        model = accelerator.unwrap_model(model)

    # --- Stage 2: Add LoRA to LLM, train LoRA + reasoning modules ---
    if args.training_stage is None or args.training_stage == 2:
        s2_start_step = args.resume_step if (resume_training_state and resume_training_state.get("stage") == 2) else 0

        # Collect LLM linear layers for LoRA
        target_linear = []
        for name, module in model.vla.named_modules():
            if not name.startswith("language_model"):
                continue
            if isinstance(module, nn.Linear):
                target_linear.append(name)
        target_linear = [n for n in target_linear if "embed_tokens" not in n and "lm_head" not in n]

        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=min(args.lora_rank, 16),
            lora_dropout=args.lora_dropout,
            bias="none",
            target_modules=target_linear,
            task_type="CAUSAL_LM",
        )
        model.vla = get_peft_model(model.vla, lora_config)
        if args.resume is not None and not args.resume.endswith(".pth"):
            # Load existing LoRA adapter from directory (not from a .pth checkpoint)
            if os.path.isdir(args.resume):
                model.vla.load_adapter(args.resume, "default")
                model.vla.set_adapter("default")
        model.vla.print_trainable_parameters()

        total_trainable = 0
        lora_grads = 0
        for n, p in model.named_parameters():
            if p.requires_grad:
                total_trainable += p.numel()
                if "lora" in n.lower():
                    lora_grads += (p.grad is not None)
        print("trainable:", total_trainable, "  lora grads present:", lora_grads)

        if accelerator.is_main_process:
            wandb.watch(model.vla, log="all")

        # Trainable params: LoRA + reasoning modules
        trainable_params = itertools.chain(
            model.get_reasoning_parameters(),
            (p for p in model.vla.parameters() if p.requires_grad),  # only LoRA params
        )
        optimizer_stage2 = torch.optim.AdamW(trainable_params, lr=args.lr)
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer_stage2, num_warmup_steps=100, num_training_steps=args.max_steps,
        )

        # Restore optimizer/scheduler state if resuming stage 2
        if s2_start_step > 0 and resume_training_state:
            if "optimizer" in resume_training_state:
                optimizer_stage2.load_state_dict(resume_training_state["optimizer"])
                logger.info("Restored optimizer state for stage 2 resume")
            if "lr_scheduler" in resume_training_state:
                lr_scheduler.load_state_dict(resume_training_state["lr_scheduler"])
                logger.info("Restored scheduler state for stage 2 resume")

        # Re-create dataset + dataloader so the RLDS stream resets from the beginning
        vla_dataset_s2 = RLDSDataset(
            Path(args.data_root_dir),
            args.dataset_name,
            batch_transform,
            resize_resolution=tuple(vla.config.image_sizes),
            shuffle_buffer_size=args.shuffle_buffer_size,
            image_aug=args.image_aug,
        )
        dataloader = DataLoader(
            vla_dataset_s2,
            batch_size=args.batch_size,
            sampler=None,
            collate_fn=collator,
            num_workers=0,
        )

        model, optimizer_stage2, lr_scheduler, dataloader = accelerator.prepare(
            model, optimizer_stage2, lr_scheduler, dataloader,
        )
        # Offset wandb steps so Stage 2 continues from where Stage 1 left off
        stage2_wandb_offset = args.max_steps if (args.training_stage is None) else 0
        train_loop(model, dataloader, optimizer_stage2, lr_scheduler, accelerator,
                   action_tokenizer=action_tokenizer,
                   pad_token_id=processor.tokenizer.pad_token_id,
                   stage=2, max_steps=args.max_steps, save_steps=args.save_steps,
                   output_dir=args.output_dir, run_name=run_name,
                   wandb_step_offset=stage2_wandb_offset,
                   start_step=s2_start_step)


if __name__ == "__main__":
    main()
