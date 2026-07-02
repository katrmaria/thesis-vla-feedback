"""
reason_vla_multilayer.py

Multi-layer visual reasoning wrapper for OpenVLA.

Instead of injecting reasoning hints only at the patch-embedding level (layer 0),
this variant injects hints at multiple depths through the ViT blocks.
This keeps the reasoning signal strong throughout the vision encoder, informing
early (edges/textures), middle (shapes), and late (objects) layers of abstraction.

Architecture:
    Same two-pass approach as reason_vla.py, but with multi-layer injection:
      - VisualReasoner: gated FFN that processes LLM hidden states at image positions
      - MultiLayerUnmerger: one PatchUnmerger per injection layer per encoder
      - Layer 0: hint added to patch embeddings (before ViT blocks)
      - Layers > 0: hints added via forward hooks on ViT blocks

    Pass 1: Full vla.forward(output_hidden_states=True) -> extract hidden_states[-1]
            at image token positions -> VisualReasoner -> multi-layer unmergers -> hints
    Pass 2: Inject hints at multiple ViT depths via hooks, identity trick, vla.forward()

    Stage 1: Freeze everything, train only VisualReasoner + multi-layer unmergers
    Stage 2: Add LoRA to LLM, train LoRA + VisualReasoner + multi-layer unmergers

Designed to work with HF AutoClasses:
    vla = AutoModelForVision2Seq.from_pretrained("openvla/openvla-7b", ...)
    model = ReasonVLA(vla, inject_layers=[0, 8, 16])
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

# Default injection layers: patch-embed (0), early (8), middle (16)
# DINOv2 ViT-L has 24 blocks, extracts from block 22. SigLIP ViT-SO has 27 blocks.
# We inject well before the extraction point for maximum effect.
DEFAULT_INJECT_LAYERS = [0, 8, 16]

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
    No spatial merge to reverse (unlike Qwen) -- just LayerNorm + Linear.
    """

    def __init__(self, llm_dim: int, vision_dim: int):
        super().__init__()
        self.ln = nn.LayerNorm(llm_dim)
        self.proj = nn.Linear(llm_dim, vision_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.ln(x))


class MultiLayerUnmerger(nn.Module):
    """
    One PatchUnmerger per injection layer.
    Each layer has its own learned projection because feature distributions
    differ at each depth of the ViT, even though the dimension stays the same.
    """

    def __init__(self, llm_dim: int, vision_dim: int, inject_layers: list):
        super().__init__()
        self.unmergers = nn.ModuleDict({
            str(layer): PatchUnmerger(llm_dim, vision_dim)
            for layer in inject_layers
        })

    def forward(self, reasoning_out: torch.Tensor, layer_idx: int) -> torch.Tensor:
        return self.unmergers[str(layer_idx)](reasoning_out)


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

    orig_main = vision_backbone.featurizer.patch_embed.forward
    vision_backbone.featurizer.patch_embed.forward = identity_forward
    orig_fused = None
    if is_fused:
        orig_fused = vision_backbone.fused_featurizer.patch_embed.forward
        vision_backbone.fused_featurizer.patch_embed.forward = identity_forward

    try:
        yield
    finally:
        vision_backbone.featurizer.patch_embed.forward = orig_main
        if is_fused and orig_fused is not None:
            vision_backbone.fused_featurizer.patch_embed.forward = orig_fused


@contextmanager
def override_vision_backbone(vision_backbone, precomputed_features):
    """
    Replace vision_backbone.forward to return precomputed features.
    """
    orig_forward = vision_backbone.forward
    vision_backbone.forward = lambda *args, **kwargs: precomputed_features
    try:
        yield
    finally:
        vision_backbone.forward = orig_forward


def _make_inject_hook(hint, num_patches):
    """
    Create a forward hook that adds a reasoning hint to the patch positions
    of a ViT block's output.

    The block output may include CLS + register tokens before the patches:
      DINOv2 reg4: [bsz, 1+4+256, 1024] = [bsz, 261, 1024]
      SigLIP:      [bsz, 256, 1152]      (no CLS)

    The hint has shape [bsz, num_patches, vision_dim]. We add it only
    to the last num_patches positions (skipping CLS/register prefix).
    """
    def hook_fn(module, input, output):
        h = hint.to(output)
        offset = output.shape[1] - num_patches
        modified = output.clone()
        modified[:, offset:, :] = modified[:, offset:, :] + h
        return modified
    return hook_fn


# ---- Main wrapper ----

class ReasonVLA(nn.Module):
    """
    Multi-layer visual reasoning wrapper around OpenVLAForActionPrediction.

    Instead of injecting hints only at the patch embedding level,
    this variant injects at multiple ViT block depths for stronger
    reasoning signal throughout the vision encoder.
    """

    def __init__(self, vla, hidden_layer=-1, inject_layers=None):
        """
        Args:
            vla: an OpenVLAForActionPrediction loaded via AutoModelForVision2Seq.from_pretrained(...)
            hidden_layer: which LLM hidden layer to extract for reasoning (default: -1 = last layer)
            inject_layers: list of ViT layer indices to inject hints at.
                           0 = patch embedding level, >0 = after that block index.
                           Default: [0, 8, 16]
        """
        super().__init__()
        self.vla = vla
        self.hidden_layer = hidden_layer
        self.inject_layers = inject_layers or DEFAULT_INJECT_LAYERS

        llm_dim = vla.config.text_config.hidden_size  # e.g. 4096
        vb = vla.vision_backbone

        # Detect fused backbone (DINOv2 + SigLIP)
        self.is_fused = hasattr(vb, "fused_featurizer") and vb.use_fused_vision_backbone

        # Get vision dimensions and num_patches
        self.main_vision_dim = vb.featurizer.embed_dim  # DINOv2: 1024
        self.num_patches = vb.featurizer.patch_embed.num_patches  # 256
        self.num_main_blocks = len(vb.featurizer.blocks)  # 24 for DINOv2 ViT-L
        if self.is_fused:
            self.fused_vision_dim = vb.fused_featurizer.embed_dim  # SigLIP: 1152
            self.num_fused_blocks = len(vb.fused_featurizer.blocks)  # 27 for SigLIP ViT-SO

        # Validate inject_layers against ViT depth
        non_zero_layers = [l for l in self.inject_layers if l > 0]
        if non_zero_layers:
            max_main_block = self.num_main_blocks - 2  # extraction from second-to-last
            for l in non_zero_layers:
                if l > max_main_block:
                    logger.warning(
                        f"Inject layer {l} >= extraction layer {max_main_block} for DINOv2. "
                        f"Hint at this depth may not affect extracted features."
                    )

        # Reasoning modules
        self.visual_reasoner = VisualReasoner(llm_dim, llm_dim)
        self.main_multi_unmerger = MultiLayerUnmerger(llm_dim, self.main_vision_dim, self.inject_layers)
        if self.is_fused:
            self.fused_multi_unmerger = MultiLayerUnmerger(llm_dim, self.fused_vision_dim, self.inject_layers)

        # Reasoning hint state
        self.reasoning_hint = None

        logger.info(
            f"MultiLayer ReasonVLA: inject_layers={self.inject_layers}, "
            f"main_blocks={self.num_main_blocks}, "
            f"fused_blocks={getattr(self, 'num_fused_blocks', 'N/A')}"
        )

    def set_image_reasoning(self, image_hidden: torch.Tensor) -> None:
        """
        Compute reasoning hints from LLM hidden states for all injection layers.
        """
        reasoning_out = self.visual_reasoner(image_hidden)  # [bsz, num_patches, llm_dim]

        # Compute per-layer hints for DINOv2
        self._hints_main = {}
        for layer in self.inject_layers:
            self._hints_main[layer] = self.main_multi_unmerger(reasoning_out, layer)

        # Compute per-layer hints for SigLIP
        if self.is_fused:
            self._hints_fused = {}
            for layer in self.inject_layers:
                self._hints_fused[layer] = self.fused_multi_unmerger(reasoning_out, layer)

        self.reasoning_hint = True

    def reset_image_reasoning(self) -> None:
        self.reasoning_hint = None
        self._hints_main = None
        self._hints_fused = None

    def _register_block_hooks(self, featurizer, hints_dict, num_patches):
        """
        Register forward hooks on ViT blocks for layers > 0.
        Returns list of hook handles (caller must remove them).
        """
        hooks = []
        for layer in self.inject_layers:
            if layer == 0:
                continue  # layer 0 is handled at patch_embed level
            hint = hints_dict[layer]
            handle = featurizer.blocks[layer].register_forward_hook(
                _make_inject_hook(hint, num_patches)
            )
            hooks.append(handle)
        return hooks

    def second_forward(self, input_ids, attention_mask, pixel_values, *args, **kwargs):
        """
        Second forward pass with multi-layer hint injection.
          1. Run patch_embed manually -> get embeddings
          2. Add layer-0 hint to embeddings (if 0 in inject_layers)
          3. Register forward hooks on ViT blocks for deeper layers
          4. Identity trick on patch_embed, run through ViT blocks (hooks fire)
          5. Override vision_backbone.forward -> call vla.forward()
        """
        vb = self.vla.vision_backbone

        if self.is_fused:
            img_main, img_fused = torch.split(pixel_values, [3, 3], dim=1)

            # Run patch_embed manually
            main_patches = vb.featurizer.patch_embed(img_main)       # [bsz, 256, 1024]
            fused_patches = vb.fused_featurizer.patch_embed(img_fused)  # [bsz, 256, 1152]

            # Layer 0: add hint at patch embedding level
            if 0 in self.inject_layers:
                main_patches = main_patches + self._hints_main[0].to(main_patches)
                fused_patches = fused_patches + self._hints_fused[0].to(fused_patches)

            # Register hooks for deeper layers
            main_hooks = self._register_block_hooks(vb.featurizer, self._hints_main, self.num_patches)
            fused_hooks = self._register_block_hooks(vb.fused_featurizer, self._hints_fused, self.num_patches)

            try:
                with disable_patch_embeds(vb):
                    main_features = vb.featurizer(main_patches)
                    fused_features = vb.fused_featurizer(fused_patches)
            finally:
                for h in main_hooks + fused_hooks:
                    h.remove()

            patch_features = torch.cat([main_features, fused_features], dim=2)

        else:
            patches = vb.featurizer.patch_embed(pixel_values)

            if 0 in self.inject_layers:
                patches = patches + self._hints_main[0].to(patches)

            main_hooks = self._register_block_hooks(vb.featurizer, self._hints_main, self.num_patches)
            try:
                with disable_patch_embeds(vb):
                    patch_features = vb.featurizer(patches)
            finally:
                for h in main_hooks:
                    h.remove()

        with override_vision_backbone(vb, patch_features):
            output = self.vla(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                *args,
                **kwargs,
            )

        return output

    def forward(self, stage, input_ids, attention_mask, pixel_values, *args, **kwargs):
        """
          - If reasoning_hint is set -> second_forward (pass 2 with multi-layer hints)
          - Otherwise -> normal vla forward (pass 1 to extract hidden states)
        """
        if self.reasoning_hint is not None:
            return self.second_forward(input_ids, attention_mask, pixel_values, *args, **kwargs)

        output = self.vla(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            labels=None,
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
            "main_multi_unmerger": self.main_multi_unmerger.state_dict(),
            "inject_layers": self.inject_layers,
        }
        if self.is_fused:
            save_dict["fused_multi_unmerger"] = self.fused_multi_unmerger.state_dict()
        torch.save(save_dict, path)
        logger.info(f"Reasoning modules saved: {path}")

    def load_reasoning_modules(self, path: str) -> None:
        """Load reasoning-specific modules from checkpoint."""
        state_dict = torch.load(path, map_location="cpu")
        # Validate inject_layers match
        saved_layers = state_dict.get("inject_layers", None)
        if saved_layers is not None and saved_layers != self.inject_layers:
            logger.warning(
                f"Checkpoint inject_layers={saved_layers} != current {self.inject_layers}. "
                f"Loading may fail if layer counts differ."
            )
        self.visual_reasoner.load_state_dict(state_dict["visual_reasoner"])
        self.main_multi_unmerger.load_state_dict(state_dict["main_multi_unmerger"])
        if self.is_fused and "fused_multi_unmerger" in state_dict:
            self.fused_multi_unmerger.load_state_dict(state_dict["fused_multi_unmerger"])
        logger.info(f"Reasoning modules loaded: {path}")

    # ---- Inference methods ----

    @classmethod
    def from_finetuned(cls, model_name, checkpoint_path, stage=1, lora_dir=None,
                        merge_path=None, merge_lora_dir=None,
                        device_map="auto", hidden_layer=-1, inject_layers=None):
        """
        Load a trained ReasonVLA for inference, with optional checkpoint merging.
        """
        AutoConfig.register("openvla", OpenVLAConfig)
        AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
        AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
        AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

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

        model = cls(vla, hidden_layer=hidden_layer, inject_layers=inject_layers)
        model.stage = stage

        model.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        model.action_tokenizer = ActionTokenizer(model.processor.tokenizer)

        if stage == 2:
            if lora_dir is None:
                raise ValueError("--lora_dir is required for stage 2")
            model.vla.load_adapter(lora_dir)

        if not os.path.exists(checkpoint_path):
            raise ValueError(f"Checkpoint not found: {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location="cpu")

        # Merge with second checkpoint if provided
        if merge_path is not None:
            if not os.path.exists(merge_path):
                raise ValueError(f"Merge checkpoint not found: {merge_path}")

            if stage == 2 and merge_lora_dir is not None:
                other_lora_state = deepcopy(get_peft_model_state_dict(model.vla))
                model.vla.load_adapter(merge_lora_dir, "default2")
                current_lora_state = deepcopy(get_peft_model_state_dict(model.vla, "default2"))
                merged_lora = {
                    k: (current_lora_state[k] + other_lora_state[k]) / 2
                    for k in current_lora_state.keys()
                    if k in other_lora_state
                }
                set_peft_model_state_dict(model.vla, merged_lora)
                logger.info(f"Merged LoRA weights from {lora_dir} and {merge_lora_dir}")

            merge_state_dict = torch.load(merge_path, map_location="cpu")
            for key in ["visual_reasoner", "main_multi_unmerger"]:
                if key in state_dict and key in merge_state_dict:
                    state_dict[key] = {
                        k: (state_dict[key][k] + merge_state_dict[key][k]) / 2
                        for k in state_dict[key].keys()
                        if k in merge_state_dict[key]
                    }
            if model.is_fused and "fused_multi_unmerger" in state_dict and "fused_multi_unmerger" in merge_state_dict:
                state_dict["fused_multi_unmerger"] = {
                    k: (state_dict["fused_multi_unmerger"][k] + merge_state_dict["fused_multi_unmerger"][k]) / 2
                    for k in state_dict["fused_multi_unmerger"].keys()
                    if k in merge_state_dict["fused_multi_unmerger"]
                }
            logger.info(f"Merged reasoning modules from {checkpoint_path} and {merge_path}")

        model.visual_reasoner.load_state_dict(state_dict["visual_reasoner"])
        model.main_multi_unmerger.load_state_dict(state_dict["main_multi_unmerger"])
        if model.is_fused and "fused_multi_unmerger" in state_dict:
            model.fused_multi_unmerger.load_state_dict(state_dict["fused_multi_unmerger"])

        device = next(model.vla.parameters()).device
        dtype = next(model.vla.parameters()).dtype
        model.visual_reasoner.to(device=device, dtype=dtype)
        model.main_multi_unmerger.to(device=device, dtype=dtype)
        if model.is_fused:
            model.fused_multi_unmerger.to(device=device, dtype=dtype)

        model.visual_reasoner.eval()
        model.main_multi_unmerger.eval()
        if model.is_fused:
            model.fused_multi_unmerger.eval()
        model.vla.eval()
        model.eval()
        logger.info(f"ReasonVLA (multi-layer) loaded (stage {stage}) from {checkpoint_path}")
        return model

    @torch.inference_mode()
    def generate(self, image, task_description, unnorm_key=None):
        """
        Two-pass action prediction with multi-layer hint injection.
        """
        self.reset_image_reasoning()

        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        prompt_builder = PurePromptBuilder("openvla")
        prompt_builder.add_turn("human", f"What action should the robot take to {task_description.lower()}?")
        prompt_text = prompt_builder.get_prompt()

        inputs = self.processor(prompt_text, image).to(self.vla.device, dtype=torch.bfloat16)
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask", None)
        pixel_values = inputs["pixel_values"]

        # === PASS 1: Extract hidden states ===
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

        hidden_state = output.hidden_states[self.hidden_layer]
        image_reasoning = hidden_state[:, 1:1 + self.num_patches, :]

        self.set_image_reasoning(image_reasoning)

        del output, hidden_state, image_reasoning
        torch.cuda.empty_cache()

        # === PASS 2: Predict action with multi-layer hints ===
        vb = self.vla.vision_backbone

        if self.is_fused:
            img_main, img_fused = torch.split(pixel_values, [3, 3], dim=1)
            main_patches = vb.featurizer.patch_embed(img_main)
            fused_patches = vb.fused_featurizer.patch_embed(img_fused)

            if 0 in self.inject_layers:
                main_patches = main_patches + self._hints_main[0].to(main_patches)
                fused_patches = fused_patches + self._hints_fused[0].to(fused_patches)

            main_hooks = self._register_block_hooks(vb.featurizer, self._hints_main, self.num_patches)
            fused_hooks = self._register_block_hooks(vb.fused_featurizer, self._hints_fused, self.num_patches)

            try:
                with disable_patch_embeds(vb):
                    main_features = vb.featurizer(main_patches)
                    fused_features = vb.fused_featurizer(fused_patches)
            finally:
                for h in main_hooks + fused_hooks:
                    h.remove()

            patch_features = torch.cat([main_features, fused_features], dim=2)

        else:
            patches = vb.featurizer.patch_embed(pixel_values)
            if 0 in self.inject_layers:
                patches = patches + self._hints_main[0].to(patches)

            main_hooks = self._register_block_hooks(vb.featurizer, self._hints_main, self.num_patches)
            try:
                with disable_patch_embeds(vb):
                    patch_features = vb.featurizer(patches)
            finally:
                for h in main_hooks:
                    h.remove()

        with override_vision_backbone(vb, patch_features):
            actions = self.vla.predict_action(
                input_ids,
                unnorm_key=unnorm_key,
                pixel_values=pixel_values,
                do_sample=False,
            )

        return actions

    def get_reasoning_parameters(self):
        """Iterator over all trainable reasoning parameters (for optimizer)."""
        params = [self.visual_reasoner.parameters(), self.main_multi_unmerger.parameters()]
        if self.is_fused:
            params.append(self.fused_multi_unmerger.parameters())
        return itertools.chain(*params)

    def freeze_vla(self) -> None:
        """Freeze all VLA parameters (for stage 1)."""
        for param in self.vla.parameters():
            param.requires_grad = False
        self.vla.eval()

    def unfreeze_reasoning(self) -> None:
        """Unfreeze reasoning modules (for both stages)."""
        modules = [self.visual_reasoner, self.main_multi_unmerger]
        if self.is_fused:
            modules.append(self.fused_multi_unmerger)
        for module in modules:
            for param in module.parameters():
                param.requires_grad = True
            module.train()


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
):
    """
    Training loop for multi-layer ReasonVLA.
    Same two-pass structure as reason_vla.py but with multi-layer hint injection.
    """
    import time

    run_name = run_name or "debug"
    unwrapped_model = accelerator.unwrap_model(model)
    num_patches = unwrapped_model.num_patches
    inject_layers = unwrapped_model.inject_layers

    logger.info(f"Starting Stage {stage} training, max_steps={max_steps}, inject_layers={inject_layers}")

    global_step = 0
    loss_cum = 0.0
    acc_cum = 0.0
    l1_cum = 0.0
    micro_count = 0

    old_params = {
        name: param.clone().detach()
        for name, param in unwrapped_model.visual_reasoner.named_parameters()
    }

    step_start_time = time.time()

    for epoch in range(999):
        for batch in dataloader:
            with accelerator.accumulate(model):
                unwrapped_model.reset_image_reasoning()

                input_ids = batch["input_ids"].to(accelerator.device)
                attention_mask = batch["attention_mask"].to(accelerator.device)
                pixel_values = batch["pixel_values"].to(accelerator.device)
                labels = batch["labels"].to(accelerator.device)

                # Derive first-pass input_ids (no action tokens)
                first_action_pos = (labels != IGNORE_INDEX).long().argmax(dim=1)
                cut = first_action_pos.min().item()
                first_pass_ids = input_ids[:, :cut]
                first_pass_attn = attention_mask[:, :cut]

                # ---------- Pass 1: Extract hidden states ----------
                maybe_no_grad = torch.no_grad if stage == 1 else nullcontext

                with maybe_no_grad():
                    output = model(
                        stage=stage,
                        input_ids=first_pass_ids,
                        attention_mask=first_pass_attn,
                        pixel_values=pixel_values,
                        return_dict=True,
                    )

                hidden_state = output.hidden_states[unwrapped_model.hidden_layer]
                image_reasoning = hidden_state[:, 1:1 + num_patches, :]

                unwrapped_model.set_image_reasoning(image_reasoning)

                # Capture per-layer hint norms for diagnostics
                hint_norms = {}
                total_norm_sq = 0.0
                for layer in inject_layers:
                    h = unwrapped_model._hints_main[layer]
                    n = h.detach().float().norm().item() if h is not None else 0.0
                    hint_norms[f"main_L{layer}"] = n
                    total_norm_sq += n ** 2
                if unwrapped_model.is_fused:
                    for layer in inject_layers:
                        h = unwrapped_model._hints_fused[layer]
                        n = h.detach().float().norm().item() if h is not None else 0.0
                        hint_norms[f"fused_L{layer}"] = n
                        total_norm_sq += n ** 2
                hint_norm_total = total_norm_sq ** 0.5

                del output, hidden_state, image_reasoning
                torch.cuda.empty_cache()

                # ---------- Pass 2: Forward with multi-layer hints + labels ----------
                disable_lora_ctx = nullcontext
                if stage == 2:
                    disable_lora_ctx = unwrapped_model.vla.disable_adapter

                with accelerator.autocast():
                    with disable_lora_ctx():
                        output2 = model(
                            stage=stage,
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            pixel_values=pixel_values,
                            labels=labels,
                            return_dict=True,
                        )
                        loss = output2.loss

                # ---------- Action metrics ----------
                action_preds = output2.logits[:, num_patches:-1].argmax(dim=2)
                action_gt = labels[:, 1:].to(action_preds.device)
                mask = action_gt > action_tokenizer.action_token_begin_idx

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

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)

                    if accelerator.is_main_process:
                        vr_grad_norm = sum(
                            p.grad.detach().float().norm().item() ** 2
                            for p in unwrapped_model.visual_reasoner.parameters()
                            if p.grad is not None
                        ) ** 0.5
                        um_grad_norm = sum(
                            p.grad.detach().float().norm().item() ** 2
                            for p in itertools.chain(
                                unwrapped_model.main_multi_unmerger.parameters(),
                                unwrapped_model.fused_multi_unmerger.parameters() if unwrapped_model.is_fused else []
                            )
                            if p.grad is not None
                        ) ** 0.5
                        lora_grad_norm = 0.0
                        if stage == 2:
                            lora_grad_norm = sum(
                                p.grad.detach().float().norm().item() ** 2
                                for n, p in unwrapped_model.vla.named_parameters()
                                if p.grad is not None and "lora" in n.lower()
                            ) ** 0.5

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                loss_cum += loss.item()
                acc_cum += action_accuracy.item()
                l1_cum += action_l1_loss.item()
                micro_count += 1

                if accelerator.sync_gradients:
                    global_step += 1

                # ---------- Per-step wandb logging ----------
                if accelerator.sync_gradients and accelerator.is_main_process:
                    step_time = time.time() - step_start_time
                    step_start_time = time.time()
                    current_lr = lr_scheduler.get_last_lr()[0]

                    gpu_mem_gb = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0

                    avg_loss = loss_cum / micro_count if micro_count > 0 else 0.0
                    avg_acc = acc_cum / micro_count if micro_count > 0 else 0.0
                    avg_l1 = l1_cum / micro_count if micro_count > 0 else 0.0

                    log_dict = {
                        f"train/loss_stage{stage}": avg_loss,
                        f"train/action_accuracy_stage{stage}": avg_acc,
                        f"train/l1_loss_stage{stage}": avg_l1,
                        f"train/lr_stage{stage}": current_lr,
                        "train/global_step": global_step,
                        "train/epoch": epoch,
                        "train/step_time": step_time,
                        # Aggregate feedback diagnostics
                        "feedback/hint_norm_total": hint_norm_total,
                        "feedback/grad_norm_visual_reasoner": vr_grad_norm,
                        "feedback/grad_norm_unmerge": um_grad_norm,
                        "feedback/grad_norm_lora": lora_grad_norm,
                        "system/gpu_memory_gb": gpu_mem_gb,
                    }
                    # Per-layer hint norms
                    for key, val in hint_norms.items():
                        log_dict[f"feedback/hint_norm_{key}"] = val

                    wandb.log(log_dict, step=wandb_step_offset + global_step)

                    loss_cum = 0.0
                    acc_cum = 0.0
                    l1_cum = 0.0
                    micro_count = 0

                    if global_step % 10 == 0:
                        layer_norms_str = " | ".join(f"L{l}={hint_norms.get(f'main_L{l}', 0):.3f}" for l in inject_layers)
                        logger.info(
                            f"Stage {stage} | Epoch {epoch} | Step {global_step} | "
                            f"loss: {avg_loss:.4f} | acc: {avg_acc:.4f} | "
                            f"l1: {avg_l1:.4f} | lr: {current_lr:.2e} | "
                            f"hints: [{layer_norms_str}] | vr_grad: {vr_grad_norm:.4f}"
                        )

                    if global_step == 2 and vr_grad_norm == 0.0 and um_grad_norm == 0.0:
                        logger.warning("No gradients on visual_reasoner or unmerge at step 2! "
                                       "Check computation graph.")

                # ---------- Checkpointing ----------
                if global_step > 0 and global_step % save_steps == 0 and accelerator.is_main_process:
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
                    logger.info(f"Checkpoint saved at step {global_step}")

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
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--grad-accum-steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--hidden-layer", type=int, default=-1,
                        help="LLM hidden layer to extract for reasoning (-1=last)")
    parser.add_argument("--inject-layers", type=int, nargs="+", default=None,
                        help=f"ViT layer indices to inject hints at (default: {DEFAULT_INJECT_LAYERS}). "
                             f"0=patch-embed, >0=after that block index.")
    args = parser.parse_args()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    run_name = os.environ.get("RUN_NAME", None)
    output_path = os.path.join(args.output_dir, run_name if run_name else "debug")
    os.makedirs(output_path, exist_ok=True)

    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    processor = AutoProcessor.from_pretrained(args.vla_path, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        args.vla_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    # Wrap VLA with multi-layer ReasonVLA
    inject_layers = args.inject_layers or DEFAULT_INJECT_LAYERS
    model = ReasonVLA(vla, hidden_layer=args.hidden_layer, inject_layers=inject_layers)

    accelerator = Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_steps=args.grad_accum_steps,
    )

    if accelerator.is_main_process:
        config_dict = vars(args)
        config_dict["inject_layers"] = inject_layers
        wandb.init(project="openvla_fixed", config=config_dict, name=run_name,
                   mode="disabled" if run_name is None else None)

    if args.resume is not None:
        checkpoint_path = os.path.join(args.resume, "final_checkpoint_stage1.pth")
        if os.path.exists(checkpoint_path):
            logger.info(f"Loading checkpoint from {checkpoint_path}")
            model.load_reasoning_modules(checkpoint_path)
        else:
            logger.warning(f"No checkpoint found at {checkpoint_path}, starting from scratch.")

    action_tokenizer = ActionTokenizer(processor.tokenizer)
    batch_transform = RLDSBatchTransform(
        action_tokenizer,
        processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder if "v01" not in args.vla_path else VicunaV15ChatPromptBuilder,
    )
    vla_dataset = RLDSDataset(
        Path(args.data_root_dir),
        args.dataset_name,
        batch_transform,
        resize_resolution=tuple(vla.config.image_sizes),
        shuffle_buffer_size=args.shuffle_buffer_size,
        image_aug=args.image_aug,
    )

    if accelerator.is_main_process:
        save_dataset_statistics(vla_dataset.dataset_statistics, Path(output_path))

    base_collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right"
    )
    # Wrap collator to drop non-tensor fields (dataset_names is bytes from RLDS,
    # which Accelerate can't concatenate during gradient accumulation)
    def collator(instances):
        batch = base_collator(instances)
        batch.pop("dataset_names", None)
        return batch

    dataloader = DataLoader(
        vla_dataset,
        batch_size=args.batch_size,
        sampler=None,
        collate_fn=collator,
        num_workers=0,
    )

    model.freeze_vla()
    model.unfreeze_reasoning()

    # --- Stage 1 ---
    if args.training_stage is None or args.training_stage == 1:
        optimizer_stage1 = torch.optim.AdamW(model.get_reasoning_parameters(), lr=args.lr)
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer_stage1, num_warmup_steps=100, num_training_steps=args.max_steps,
        )

        model, optimizer_stage1, lr_scheduler, dataloader = accelerator.prepare(
            model, optimizer_stage1, lr_scheduler, dataloader,
        )
        if accelerator.is_main_process:
            wandb.watch(accelerator.unwrap_model(model).visual_reasoner, log="all")

        train_loop(model, dataloader, optimizer_stage1, lr_scheduler, accelerator,
                   action_tokenizer=action_tokenizer,
                   pad_token_id=processor.tokenizer.pad_token_id,
                   stage=1, max_steps=args.max_steps, save_steps=args.save_steps,
                   output_dir=args.output_dir, run_name=run_name)

        model = accelerator.unwrap_model(model)

    # --- Stage 2 ---
    if args.training_stage is None or args.training_stage == 2:
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
        if args.resume is not None:
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

        trainable_params = itertools.chain(
            model.visual_reasoner.parameters(),
            model.vla.parameters(),
            model.main_multi_unmerger.parameters(),
            *(
                [model.fused_multi_unmerger.parameters()] if model.is_fused else []
            ),
        )
        optimizer_stage2 = torch.optim.AdamW(trainable_params, lr=args.lr)
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer_stage2, num_warmup_steps=100, num_training_steps=args.max_steps,
        )

        model, optimizer_stage2, lr_scheduler, dataloader = accelerator.prepare(
            model, optimizer_stage2, lr_scheduler, dataloader,
        )
        stage2_wandb_offset = args.max_steps if (args.training_stage is None) else 0
        train_loop(model, dataloader, optimizer_stage2, lr_scheduler, accelerator,
                   action_tokenizer=action_tokenizer,
                   pad_token_id=processor.tokenizer.pad_token_id,
                   stage=2, max_steps=args.max_steps, save_steps=args.save_steps,
                   output_dir=args.output_dir, run_name=run_name,
                   wandb_step_offset=stage2_wandb_offset)


if __name__ == "__main__":
    main()
