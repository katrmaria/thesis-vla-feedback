"""
reason_vla_projector_crossattn_contrastive.py

Same architecture as reason_vla_projector_crossattn.py (cross-attention at the projector
output, two-pass forward) plus a contrastive loss term that forces the model to be
sensitive to the instruction.

For each batch we run TWO complete two-pass forwards:
  - Correct: standard forward with the true instruction -> loss_correct
  - Wrong:   forward with the same image + GT action but a randomly-sampled wrong
             instruction (from the libero_spatial pool) -> loss_wrong

We add a hinge term: max(0, loss_correct - loss_wrong + margin), weighted by lambda.
The hinge is positive (penalizing) when the model predicts the GT action regardless
of which instruction it gets, i.e. when it has learned a template-bound offset.

This directly attacks the failure mode observed in the original projcrossattn:
all single-model variants are template-blind and don't transfer to task perturbation.
"""

import argparse
from contextlib import contextmanager, nullcontext
import itertools
import logging
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from accelerate import Accelerator
from peft import LoraConfig, get_peft_model
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


# ---- Wrong-instruction pool for contrastive training ----
# All 10 libero_spatial task instructions. At each training step we sample one
# at random as the "wrong" instruction. If sampled equals the true instruction
# we sample again.
LIBERO_SPATIAL_INSTRUCTIONS = [
    "pick up the black bowl between the plate and the ramekin and place it on the plate",
    "pick up the black bowl next to the ramekin and place it on the plate",
    "pick up the black bowl from table center and place it on the plate",
    "pick up the black bowl on the cookie box and place it on the plate",
    "pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate",
    "pick up the black bowl on the ramekin and place it on the plate",
    "pick up the black bowl next to the cookie box and place it on the plate",
    "pick up the black bowl on the stove and place it on the plate",
    "pick up the black bowl next to the plate and place it on the plate",
    "pick up the black bowl on the wooden cabinet and place it on the plate",
]


def precompute_wrong_prompt_token_ids(tokenizer, instructions, prompt_builder_fn):
    """
    Pre-tokenize the wrong-instruction prompt prefixes (text only, no image).
    Each entry is the token sequence for "<BOS> In: What action ... to <instr>?\nOut: ".
    Action tokens are appended per-batch from the original labels.
    """
    prefixes = []
    for instr in instructions:
        prompt = prompt_builder_fn("openvla")
        prompt.add_turn("human", f"What action should the robot take to {instr.lower()}?")
        prompt_text = prompt.get_prompt()
        encoded = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=True)
        prefixes.append(encoded["input_ids"][0])  # [L_prefix]
    return prefixes


def build_wrong_batch(orig_input_ids, orig_attention_mask, orig_labels,
                      wrong_prefix_ids, pad_token_id, device):
    """
    Construct a batch with the same image (caller passes pixel_values separately) and
    same GT action tokens, but with a different (wrong) instruction prefix.

    Returns: dict with input_ids, attention_mask, labels (all padded to same length).
    """
    batch_size = orig_input_ids.shape[0]
    wrong_prefix_len = wrong_prefix_ids.shape[0]
    wrong_prefix_ids = wrong_prefix_ids.to(device)

    new_inputs, new_labels, new_attn, lengths = [], [], [], []
    for b in range(batch_size):
        action_mask = orig_labels[b] != IGNORE_INDEX
        action_token_ids = orig_labels[b][action_mask]  # [n_action_tokens]
        new_ids = torch.cat([wrong_prefix_ids, action_token_ids], dim=0)
        new_lbl = torch.full_like(new_ids, IGNORE_INDEX)
        new_lbl[wrong_prefix_len:] = action_token_ids
        new_inputs.append(new_ids)
        new_labels.append(new_lbl)
        lengths.append(new_ids.shape[0])

    max_len = max(lengths)
    out_ids = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long, device=device)
    out_lbl = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=torch.long, device=device)
    out_attn = torch.zeros((batch_size, max_len), dtype=torch.long, device=device)
    for b in range(batch_size):
        L = lengths[b]
        out_ids[b, :L] = new_inputs[b]
        out_lbl[b, :L] = new_labels[b]
        out_attn[b, :L] = 1
    return {"input_ids": out_ids, "attention_mask": out_attn, "labels": out_lbl}


# ---- Identity trick utilities ----

@contextmanager
def override_vision_backbone(vision_backbone, precomputed_features):
    """Replace vision_backbone.forward to return precomputed patch_features."""
    orig_forward = vision_backbone.forward
    vision_backbone.forward = lambda *args, **kwargs: precomputed_features
    try:
        yield
    finally:
        vision_backbone.forward = orig_forward


@contextmanager
def override_projector(vla, precomputed_projected):
    """Replace projector.forward to return precomputed refined patches."""
    orig_forward = vla.projector.forward
    vla.projector.forward = lambda *args, **kwargs: precomputed_projected
    try:
        yield
    finally:
        vla.projector.forward = orig_forward


# ---- Cross-attention module ----

class ProjectorCrossAttention(nn.Module):
    """
    Cross-attention at the projector output.
    Projected patches (Q) attend to text token hidden states (K, V). Both in d_llm (4096).
    Identity at init: gate = 0 so output = projected + tanh(0) * out = projected.
    """
    def __init__(self, d_llm, num_heads=4, attn_dropout=0.1):
        super().__init__()
        assert d_llm % num_heads == 0, f"d_llm ({d_llm}) must be divisible by num_heads ({num_heads})"

        self.ln_patches = nn.LayerNorm(d_llm)
        self.ln_text = nn.LayerNorm(d_llm)
        self.q_proj = nn.Linear(d_llm, d_llm)
        self.k_proj = nn.Linear(d_llm, d_llm)
        self.v_proj = nn.Linear(d_llm, d_llm)
        self.out_proj = nn.Linear(d_llm, d_llm)
        self.num_heads = num_heads
        self.head_dim = d_llm // num_heads
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.gate = nn.Parameter(torch.zeros(1))

        # Small init for out_proj so gate gets a non-zero gradient at step zero
        nn.init.normal_(self.out_proj.weight, std=0.01)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, projected_patches, text_hidden):
        """
        Args:
            projected_patches: [B, N, d_llm] projector output (N=256 for OpenVLA)
            text_hidden: [B, T, d_llm] instruction hidden states from pass 1
        Returns:
            refined_patches: [B, N, d_llm] = projected + tanh(gate) * cross_attn(projected, text_hidden)
        """
        B, N, D = projected_patches.shape
        T = text_hidden.shape[1]

        patches_normed = self.ln_patches(projected_patches)
        text_normed = self.ln_text(text_hidden)

        Q = self.q_proj(patches_normed)   # [B, N, D]
        K = self.k_proj(text_normed)      # [B, T, D]
        V = self.v_proj(text_normed)      # [B, T, D]

        Q = Q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, N, d_head]
        K = K.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, T, d_head]
        V = V.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, T, d_head]

        attn_logits = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)  # [B, H, N, T]
        attn_probs = torch.softmax(attn_logits, dim=-1)

        with torch.no_grad():
            eps = 1e-9
            entropy = -(attn_probs * (attn_probs + eps).log()).sum(dim=-1)  # [B, H, N]
            self._last_entropy = entropy.mean().item()
            self._last_entropy_norm = self._last_entropy / float(np.log(max(T, 2)))
            # Max attention weight per query, averaged over heads and patches.
            # High max + low entropy = concentrated (good grounding).
            # Low max + high entropy = uniform (useless).
            self._last_attn_max = attn_probs.max(dim=-1).values.mean().item()

        attn = self.attn_dropout(attn_probs)
        out = torch.matmul(attn, V)  # [B, H, N, d_head]

        out = out.transpose(1, 2).contiguous().view(B, N, D)  # [B, N, D]
        out = self.out_proj(out)

        gated_residual = torch.tanh(self.gate) * out

        # Diagnostic: magnitude of the residual relative to the input.
        # residual_ratio near 0 => module is contributing nothing.
        # residual_ratio near 1 => module is comparable in size to input (could dominate).
        with torch.no_grad():
            proj_norm = projected_patches.float().norm().item()
            res_norm = gated_residual.float().norm().item()
            self._last_residual_norm = res_norm
            self._last_residual_ratio = res_norm / (proj_norm + 1e-9)

        return projected_patches + gated_residual


# ---- Main wrapper ----

class ReasonVLAProjectorCrossAttn(nn.Module):
    """
    Two-pass grounding wrapper.
    Pass 1: extract text hidden states from layer `hidden_layer` of the full multimodal forward.
    Pass 2: cross-attention at projector output, feed refined patches to LLM.
    """

    def __init__(self, vla, hidden_layer=12):
        super().__init__()
        self.vla = vla
        self.hidden_layer = hidden_layer

        llm_dim = vla.config.text_config.hidden_size  # 4096 for LLaMA-2-7B
        vb = vla.vision_backbone
        self.is_fused = hasattr(vb, "fused_featurizer") and vb.use_fused_vision_backbone
        self.num_patches = vb.featurizer.patch_embed.num_patches  # 256 for 224px

        # Single cross-attention module operating in d_llm space
        self.cross_attn = ProjectorCrossAttention(llm_dim)

        # State (populated between pass 1 and pass 2)
        self._text_hidden = None
        self.reasoning_hint = None

    def set_text_hidden(self, text_hidden):
        """Store detached text hidden states for use in pass 2."""
        self._text_hidden = text_hidden
        self.reasoning_hint = True

    def reset_image_reasoning(self):
        self._text_hidden = None
        self.reasoning_hint = None

    def second_forward(self, input_ids, attention_mask, pixel_values, *args, **kwargs):
        """
        Pass 2 with cross-attention at projector output.
          1. Run vision_backbone + projector manually.
          2. Cross-attention with stored text_hidden.
          3. Override both to return cached tensors.
          4. Call vla.forward() with labels -> loss.
        """
        vb = self.vla.vision_backbone

        # Step 1: Run frozen ViT + projector manually
        with torch.no_grad():
            patch_features = vb(pixel_values)  # [B, 256, 2176]

        # Projector is frozen too, but we want gradients to flow through the cross-attn
        # input. Re-enable grad here; the projector weights themselves won't update.
        projected = self.vla.projector(patch_features)  # [B, 256, 4096]

        # Step 2: Cross-attention
        refined = self.cross_attn(projected, self._text_hidden)  # [B, 256, 4096]

        # Step 3 + 4: Override both and call vla.forward()
        with override_vision_backbone(vb, patch_features):
            with override_projector(self.vla, refined):
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
        Dispatcher: pass 1 (no text_hidden set) or pass 2 (text_hidden set).
        """
        if self.reasoning_hint is not None:
            return self.second_forward(input_ids, attention_mask, pixel_values, *args, **kwargs)

        # Pass 1: full vla.forward to extract hidden states
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

    def save_reasoning_modules(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        save_dict = {"cross_attn": self.cross_attn.state_dict()}
        torch.save(save_dict, path)
        logger.info(f"Projector cross-attention saved: {path}")

    def load_reasoning_modules(self, path):
        state_dict = torch.load(path, map_location="cpu")
        self.cross_attn.load_state_dict(state_dict["cross_attn"])
        logger.info(f"Projector cross-attention loaded: {path}")

    # ---- Inference ----

    @classmethod
    def from_finetuned(cls, model_name, checkpoint_path, stage=1, lora_dir=None,
                       device_map="auto", hidden_layer=12):
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

        model = cls(vla, hidden_layer=hidden_layer)
        model.stage = stage
        model.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        model.action_tokenizer = ActionTokenizer(model.processor.tokenizer)

        if stage == 2:
            if lora_dir is None:
                raise ValueError("--lora_dir is required for stage 2")
            model.vla.load_adapter(lora_dir)

        if not os.path.exists(checkpoint_path):
            raise ValueError(f"Checkpoint not found: {checkpoint_path}")
        model.load_reasoning_modules(checkpoint_path)

        device = next(model.vla.parameters()).device
        dtype = next(model.vla.parameters()).dtype
        model.cross_attn.to(device=device, dtype=dtype)
        # Keep gate in fp32 to avoid bfloat16 precision trap near zero
        model.cross_attn.gate = nn.Parameter(model.cross_attn.gate.data.float())

        model.cross_attn.eval()
        model.vla.eval()
        model.eval()
        logger.info(f"ReasonVLAProjectorCrossAttn loaded (stage {stage}) from {checkpoint_path}")
        return model

    @torch.inference_mode()
    def generate(self, image, task_description, unnorm_key=None):
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

        # === PASS 1: Extract text token hidden states ===
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

        # Layout: [BOS, patch_1..patch_256, instruction_tokens...]
        hidden_state = output.hidden_states[self.hidden_layer]
        text_hidden = hidden_state[:, 1 + self.num_patches:, :]  # [B, T, 4096]
        self.set_text_hidden(text_hidden)

        del output, hidden_state
        torch.cuda.empty_cache()

        # === PASS 2: Cross-attention + predict action ===
        vb = self.vla.vision_backbone
        patch_features = vb(pixel_values)
        projected = self.vla.projector(patch_features)
        refined = self.cross_attn(projected, self._text_hidden)
        # Gate is fp32 (precision-near-zero trap) so refined promotes to fp32.
        # LLM weights are bfloat16; cast back before the override.
        refined = refined.to(dtype=projected.dtype)

        with override_vision_backbone(vb, patch_features):
            with override_projector(self.vla, refined):
                actions = self.vla.predict_action(
                    input_ids,
                    unnorm_key=unnorm_key,
                    pixel_values=pixel_values,
                    attention_mask=attention_mask,
                    do_sample=False,
                )

        if getattr(self, "stage", 1) == 2 and hasattr(self.vla, "enable_adapters"):
            self.vla.enable_adapters()

        return actions

    # ---- Parameter management ----

    def get_reasoning_parameters(self):
        return self.cross_attn.parameters()

    def freeze_vla(self):
        for p in self.vla.parameters():
            p.requires_grad = False
        self.vla.eval()

    def unfreeze_reasoning(self):
        for p in self.cross_attn.parameters():
            p.requires_grad = True
        self.cross_attn.train()


def load_training_state(checkpoint_path):
    """
    Load training state (optimizer, scheduler, global_step) from a checkpoint.
    Looks for training_state-{step}.pth files alongside checkpoint-{step}.pth,
    or final_training_state_stage{N}.pth alongside final_checkpoint_stage{N}.pth.
    """
    if os.path.isdir(checkpoint_path):
        import glob
        state_files = sorted(glob.glob(os.path.join(checkpoint_path, "training_state-*.pth")))
        if not state_files:
            state_files = sorted(glob.glob(os.path.join(checkpoint_path, "*/training_state-*.pth")))
        if not state_files:
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
        state_path = checkpoint_path.replace("checkpoint-", "training_state-")
    elif "final_checkpoint_stage" in checkpoint_path:
        state_path = checkpoint_path.replace("final_checkpoint_stage", "final_training_state_stage")
    else:
        state_path = checkpoint_path

    if not os.path.exists(state_path):
        logger.warning(f"Training state not found: {state_path}")
        return None

    state = torch.load(state_path, map_location="cpu")
    logger.info(f"Loaded training state from {state_path} "
                f"(step {state['global_step']}, stage {state['stage']})")
    return state


# ---- Training Loop ----

def train_loop(
    model: ReasonVLAProjectorCrossAttn,
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
    keep_lora_pass2: bool = False,
    start_step: int = 0,
    wrong_prefix_token_ids=None,
    contrastive_lambda: float = 0.0,
    contrastive_margin: float = 0.0,
):
    import time

    run_name = run_name or "debug"
    unwrapped_model = accelerator.unwrap_model(model)
    num_patches = unwrapped_model.num_patches

    logger.info(f"Starting Stage {stage} training (projector-crossattn), "
                f"hidden_layer={unwrapped_model.hidden_layer}, max_steps={max_steps}"
                + (f", resuming from step {start_step}" if start_step > 0 else ""))

    global_step = start_step
    loss_cum = 0.0
    acc_cum = 0.0
    l1_cum = 0.0
    loss_correct_cum = 0.0
    loss_wrong_cum = 0.0
    hinge_cum = 0.0
    contrast_count = 0
    micro_count = 0
    # Accumulate pooled text_hidden across micro-batches (for pairwise cosine sim)
    text_hidden_cache = []

    # Skip params-changed verification when resuming (params already different from init)
    old_params = None if start_step > 0 else {
        name: param.clone().detach()
        for name, param in unwrapped_model.cross_attn.named_parameters()
    }

    # On resume: skip already-processed micro-batches in the RLDS stream
    steps_to_skip = start_step * accelerator.gradient_accumulation_steps if start_step > 0 else 0
    if steps_to_skip > 0:
        logger.info(f"Skipping {steps_to_skip} micro-batches to reach step {start_step}...")

    step_start_time = time.time()

    for epoch in range(999):
        for batch in dataloader:
            if steps_to_skip > 0:
                steps_to_skip -= 1
                if steps_to_skip % 1000 == 0 and steps_to_skip > 0:
                    logger.info(f"  ...{steps_to_skip} micro-batches remaining to skip")
                continue
            with accelerator.accumulate(model):
                unwrapped_model.reset_image_reasoning()

                input_ids = batch["input_ids"].to(accelerator.device)
                attention_mask = batch["attention_mask"].to(accelerator.device)
                pixel_values = batch["pixel_values"].to(accelerator.device)
                labels = batch["labels"].to(accelerator.device)

                # Truncate at first action token so pass 1 sees only BOS + patches + instruction
                first_action_pos = (labels != IGNORE_INDEX).long().argmax(dim=1)
                cut = first_action_pos.min().item()
                first_pass_ids = input_ids[:, :cut]
                first_pass_attn = attention_mask[:, :cut]

                # ---------- Pass 1: Extract text hidden states at layer `hidden_layer` ----------
                # Stage 1: no grad (VLA fully frozen).
                # Stage 2: grad tracked so LoRA can learn through pass 1.
                maybe_no_grad = torch.no_grad if stage == 1 else nullcontext

                with maybe_no_grad():
                    output = model(
                        stage=stage,
                        input_ids=first_pass_ids,
                        attention_mask=first_pass_attn,
                        pixel_values=pixel_values,
                        return_dict=True,
                    )

                # Sequence layout: [BOS, patch_1..patch_256, instruction_tokens...]
                hidden_state = output.hidden_states[unwrapped_model.hidden_layer]
                text_hidden = hidden_state[:, 1 + num_patches:, :]  # [B, T, 4096]
                unwrapped_model.set_text_hidden(text_hidden)

                # Diagnostic: cache pooled text_hidden per micro-batch.
                # At sync_gradients we compute pairwise cosine across all 32 cached vectors.
                # Near 1.0 => text_hidden is instruction-blind (bad K/V signal for cross-attn).
                # Below 0.9 => healthy instruction-discriminative signal.
                with torch.no_grad():
                    text_hidden_cache.append(
                        text_hidden.float().mean(dim=1).detach().cpu()  # [B, D], mean over tokens
                    )

                del output, hidden_state
                torch.cuda.empty_cache()

                # ---------- Pass 2: Cross-attention + loss ----------
                # Disable LoRA in pass 2 so cross-attention is the only output-modifying path.
                disable_lora_ctx = nullcontext
                if stage == 2 and not keep_lora_pass2:
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
                        loss_correct = output2.loss
                        loss = loss_correct

                # ---------- Contrastive forward (wrong-instruction asymmetry) ----------
                loss_wrong_val = float("nan")
                hinge_val = float("nan")
                if contrastive_lambda > 0.0 and wrong_prefix_token_ids is not None:
                    # Sample wrong prefix; reject if its token sequence matches the
                    # current batch's instruction (which would make loss_wrong == loss_correct).
                    n_wrong = len(wrong_prefix_token_ids)
                    cur_instr_tokens = first_pass_ids[0].tolist() if first_pass_ids.numel() else []
                    for _ in range(n_wrong):
                        wrong_idx = random.randint(0, n_wrong - 1)
                        wrong_prefix = wrong_prefix_token_ids[wrong_idx]
                        if wrong_prefix.tolist() != cur_instr_tokens:
                            break
                    wrong_batch = build_wrong_batch(
                        input_ids, attention_mask, labels,
                        wrong_prefix, pad_token_id, accelerator.device,
                    )
                    w_input_ids = wrong_batch["input_ids"]
                    w_attn = wrong_batch["attention_mask"]
                    w_labels = wrong_batch["labels"]

                    # Wrong-instruction Pass 1 + Pass 2 (full two-pass forward)
                    unwrapped_model.reset_image_reasoning()
                    w_first_action_pos = (w_labels != IGNORE_INDEX).long().argmax(dim=1)
                    w_cut = w_first_action_pos.min().item()
                    w_first_pass_ids = w_input_ids[:, :w_cut]
                    w_first_pass_attn = w_attn[:, :w_cut]
                    with maybe_no_grad():
                        w_out1 = model(
                            stage=stage,
                            input_ids=w_first_pass_ids,
                            attention_mask=w_first_pass_attn,
                            pixel_values=pixel_values,
                            return_dict=True,
                        )
                    w_hidden = w_out1.hidden_states[unwrapped_model.hidden_layer]
                    w_text_hidden = w_hidden[:, 1 + num_patches:, :]
                    unwrapped_model.set_text_hidden(w_text_hidden)
                    del w_out1, w_hidden

                    with accelerator.autocast():
                        with disable_lora_ctx():
                            w_out2 = model(
                                stage=stage,
                                input_ids=w_input_ids,
                                attention_mask=w_attn,
                                pixel_values=pixel_values,
                                labels=w_labels,
                                return_dict=True,
                            )
                            loss_wrong = w_out2.loss

                    loss_wrong_val = loss_wrong.item()
                    # Hinge: penalize when loss_correct >= loss_wrong (model is instruction-blind)
                    hinge = torch.clamp(loss_correct - loss_wrong + contrastive_margin, min=0.0)
                    hinge_val = hinge.item()
                    loss = loss_correct + contrastive_lambda * hinge
                    # Drop intermediates not needed for backward (autograd graph still holds them)
                    del w_out2, w_text_hidden, wrong_batch

                # ---------- Action metrics (matches base_strategy.py) ----------
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
                        ca_grad_norm = sum(
                            p.grad.detach().float().norm().item() ** 2
                            for p in unwrapped_model.cross_attn.parameters()
                            if p.grad is not None
                        ) ** 0.5
                        # Per-component gradient norms (diagnose which part is stuck)
                        gate_param = unwrapped_model.cross_attn.gate
                        gate_grad_norm = (
                            gate_param.grad.detach().float().norm().item()
                            if gate_param.grad is not None else 0.0
                        )
                        q_grad_norm = (
                            unwrapped_model.cross_attn.q_proj.weight.grad.detach().float().norm().item()
                            if unwrapped_model.cross_attn.q_proj.weight.grad is not None else 0.0
                        )
                        out_grad_norm = (
                            unwrapped_model.cross_attn.out_proj.weight.grad.detach().float().norm().item()
                            if unwrapped_model.cross_attn.out_proj.weight.grad is not None else 0.0
                        )
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
                if not math.isnan(loss_wrong_val):
                    loss_correct_cum += loss_correct.item()
                    loss_wrong_cum += loss_wrong_val
                    hinge_cum += hinge_val
                    contrast_count += 1
                micro_count += 1

                if accelerator.sync_gradients:
                    global_step += 1

                if accelerator.sync_gradients and accelerator.is_main_process:
                    step_time = time.time() - step_start_time
                    step_start_time = time.time()
                    current_lr = lr_scheduler.get_last_lr()[0]
                    gpu_mem_gb = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0

                    avg_loss = loss_cum / micro_count if micro_count > 0 else 0.0
                    avg_acc = acc_cum / micro_count if micro_count > 0 else 0.0
                    avg_l1 = l1_cum / micro_count if micro_count > 0 else 0.0

                    # Compute pairwise cosine sim across all micro-batches in this optimizer step.
                    # Each entry in text_hidden_cache is [B, D]; concat and measure divergence.
                    text_cos_sim = float("nan")
                    if len(text_hidden_cache) >= 2:
                        pooled = torch.cat(text_hidden_cache, dim=0)  # [N, D]
                        if pooled.shape[0] >= 2:
                            normed = torch.nn.functional.normalize(pooled, dim=-1)
                            cos_matrix = normed @ normed.T
                            off_diag_mask = ~torch.eye(
                                cos_matrix.shape[0], dtype=torch.bool
                            )
                            text_cos_sim = cos_matrix[off_diag_mask].mean().item()

                    gate_raw = unwrapped_model.cross_attn.gate.item()
                    gate_val = torch.tanh(unwrapped_model.cross_attn.gate).item()
                    ent_norm = getattr(unwrapped_model.cross_attn, "_last_entropy_norm", float("nan"))
                    attn_max = getattr(unwrapped_model.cross_attn, "_last_attn_max", float("nan"))
                    residual_ratio = getattr(unwrapped_model.cross_attn, "_last_residual_ratio", float("nan"))
                    residual_norm = getattr(unwrapped_model.cross_attn, "_last_residual_norm", float("nan"))

                    avg_loss_correct = loss_correct_cum / contrast_count if contrast_count > 0 else float("nan")
                    avg_loss_wrong = loss_wrong_cum / contrast_count if contrast_count > 0 else float("nan")
                    avg_hinge = hinge_cum / contrast_count if contrast_count > 0 else float("nan")

                    log_dict = {
                        f"train/loss_stage{stage}": avg_loss,
                        f"train/action_accuracy_stage{stage}": avg_acc,
                        f"train/l1_loss_stage{stage}": avg_l1,
                        f"train/lr_stage{stage}": current_lr,
                        "train/global_step": global_step,
                        "train/epoch": epoch,
                        "train/step_time": step_time,
                        # Contrastive diagnostics
                        f"contrast/loss_correct_stage{stage}": avg_loss_correct,
                        f"contrast/loss_wrong_stage{stage}": avg_loss_wrong,
                        f"contrast/hinge_stage{stage}": avg_hinge,
                        f"contrast/loss_gap_stage{stage}": avg_loss_wrong - avg_loss_correct,
                        # Cross-attention contribution diagnostics
                        "feedback/residual_ratio": residual_ratio,       # ||gate*out|| / ||projected||
                        "feedback/residual_norm": residual_norm,
                        "feedback/gate": gate_val,                        # tanh(gate) in [-1, 1]
                        "feedback/gate_raw": gate_raw,                    # unbounded, watch for explosion
                        "feedback/attn_entropy_norm": ent_norm,           # 1.0 = uniform, 0 = one-hot
                        "feedback/attn_max": attn_max,                    # 1/T = uniform, 1.0 = one-hot
                        # Signal quality (is text_hidden instruction-discriminative?)
                        "feedback/text_hidden_cos_sim": text_cos_sim,     # NaN if BS=1
                        # Per-component gradient norms
                        "feedback/grad_norm_crossattn": ca_grad_norm,
                        "feedback/grad_gate": gate_grad_norm,
                        "feedback/grad_q_proj": q_grad_norm,
                        "feedback/grad_out_proj": out_grad_norm,
                        "feedback/grad_norm_lora": lora_grad_norm,
                        "system/gpu_memory_gb": gpu_mem_gb,
                    }
                    wandb.log(log_dict, step=wandb_step_offset + global_step)

                    loss_cum = 0.0
                    acc_cum = 0.0
                    l1_cum = 0.0
                    loss_correct_cum = 0.0
                    loss_wrong_cum = 0.0
                    hinge_cum = 0.0
                    contrast_count = 0
                    micro_count = 0
                    text_hidden_cache = []  # reset for next accumulation window

                    if global_step % 10 == 0:
                        logger.info(
                            f"Stage {stage} | Epoch {epoch} | Step {global_step} | "
                            f"loss: {avg_loss:.4f} | acc: {avg_acc:.4f} | "
                            f"l1: {avg_l1:.4f} | lr: {current_lr:.2e} | "
                            f"gate: {gate_val:+.4f} (raw={gate_raw:+.3f}) | "
                            f"res_ratio: {residual_ratio:.4f} | "
                            f"attn_max: {attn_max:.3f} | ent_norm: {ent_norm:.3f} | "
                            f"txt_cos: {text_cos_sim:.3f} | "
                            f"ca_grad: {ca_grad_norm:.4f} (gate={gate_grad_norm:.2e})"
                        )

                    if global_step == 2 and ca_grad_norm == 0.0:
                        logger.warning("No gradients on cross-attention at step 2! "
                                       "Check that out_proj is not zero-initialized.")
                    if global_step == 1 and residual_ratio > 1e-3:
                        logger.warning(
                            f"Step 1 residual_ratio={residual_ratio:.2e} is not ~0. "
                            f"Identity-at-init may be broken (gate not at zero?)."
                        )

                # Checkpointing
                early_threshold = save_steps * 10
                current_save_interval = save_steps if global_step <= early_threshold else save_steps * 5
                should_save = global_step > 0 and global_step % current_save_interval == 0
                if should_save and accelerator.is_main_process:
                    if old_params is not None:
                        updated = any(
                            not torch.equal(param.data.cpu(), old_params[name].cpu())
                            for name, param in accelerator.unwrap_model(model).cross_attn.named_parameters()
                        )
                        if not updated:
                            raise RuntimeError("[ERROR] Cross-attention parameters not updating!")
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
                    # Save optimizer + scheduler + step for resume
                    training_state = {
                        "global_step": global_step,
                        "epoch": epoch,
                        "stage": stage,
                        "optimizer": optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                    }
                    torch.save(
                        training_state,
                        os.path.join(checkpoint_dir, f"training_state-{global_step}.pth"),
                    )
                    logger.info(f"Checkpoint + training state saved at step {global_step}")

                if global_step >= max_steps:
                    logger.info(f"Stage {stage} reached max_steps={max_steps}, stopping.")
                    break

        if global_step >= max_steps:
            break

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
        # Save final training state so stage 2 can resume if needed
        training_state = {
            "global_step": global_step,
            "epoch": epoch,
            "stage": stage,
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
        }
        torch.save(
            training_state,
            os.path.join(final_dir, f"final_training_state_stage{stage}.pth"),
        )
        logger.info(f"Stage {stage} final checkpoint + training state saved.")


# ---- Main ----

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vla-path", type=str, default="openvla/openvla-7b")
    parser.add_argument("--data-root-dir", type=str, required=True)
    parser.add_argument("--dataset-name", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--training-stage", type=int, default=None,
                        help="1, 2, or None (both stages)")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=200_000)
    parser.add_argument("--save-steps", type=int, default=5000)
    parser.add_argument("--image-aug", action="store_true")
    parser.add_argument("--shuffle-buffer-size", type=int, default=100_000)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to run dir or specific checkpoint .pth to resume from")
    parser.add_argument("--resume-step", type=int, default=0,
                        help="Step to resume from (auto-detected from training_state if 0)")
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--hidden-layer", type=int, default=12,
                        help="LLM hidden layer to extract text hidden states from (default: 12, peak fusion)")
    parser.add_argument("--keep-lora-pass2", action="store_true",
                        help="Keep LoRA adapters ON during pass 2 (default: disable)")
    parser.add_argument("--contrastive-lambda", type=float, default=0.1,
                        help="Weight on the hinge term: loss = loss_correct + lambda * max(0, loss_correct - loss_wrong + margin)")
    parser.add_argument("--contrastive-margin", type=float, default=0.0,
                        help="Margin in the hinge term (default 0.0; any positive gap satisfies)")
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

    model = ReasonVLAProjectorCrossAttn(vla, hidden_layer=args.hidden_layer)

    # Keep gate in fp32 to avoid bfloat16 precision trap near zero
    model.cross_attn.gate = nn.Parameter(model.cross_attn.gate.data.float())
    logger.info("Gate parameter kept in fp32 (bfloat16 precision fix)")

    accelerator = Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_steps=args.grad_accum_steps,
    )

    if accelerator.is_main_process:
        wandb.init(project="openvla_projector_crossattn_contrastive", config=vars(args), name=run_name,
                   mode="disabled" if run_name is None else None)

    # Resume: load cross-attn weights + detect training state (optimizer, scheduler, step)
    resume_training_state = None
    if args.resume is not None:
        if args.resume.endswith(".pth"):
            checkpoint_path = args.resume
        else:
            # Directory - try intermediate checkpoint first (highest step), then final
            import glob
            ckpt_files = sorted(
                glob.glob(os.path.join(args.resume, "**", "checkpoint-*.pth"), recursive=True),
                key=lambda p: int(os.path.basename(p).replace("checkpoint-", "").replace(".pth", "")),
            )
            if ckpt_files:
                checkpoint_path = ckpt_files[-1]
            else:
                checkpoint_path = os.path.join(args.resume, "final_checkpoint_stage1.pth")
        if os.path.exists(checkpoint_path):
            logger.info(f"Loading cross-attn weights from {checkpoint_path}")
            model.load_reasoning_modules(checkpoint_path)
            resume_training_state = load_training_state(checkpoint_path)
            if resume_training_state and args.resume_step == 0:
                args.resume_step = resume_training_state["global_step"]
                logger.info(f"Auto-detected resume step: {args.resume_step}")
        else:
            logger.warning(f"No checkpoint found at {checkpoint_path}, starting from scratch.")

    action_tokenizer = ActionTokenizer(processor.tokenizer)
    prompt_builder_fn = PurePromptBuilder if "v01" not in args.vla_path else VicunaV15ChatPromptBuilder

    # Pre-tokenize the wrong-instruction prompt prefixes (one per task)
    wrong_prefix_token_ids = precompute_wrong_prompt_token_ids(
        processor.tokenizer, LIBERO_SPATIAL_INSTRUCTIONS, prompt_builder_fn,
    )
    if accelerator.is_main_process:
        logger.info(f"Pre-tokenized {len(wrong_prefix_token_ids)} wrong-instruction prefixes "
                    f"(lambda={args.contrastive_lambda}, margin={args.contrastive_margin})")

    batch_transform = RLDSBatchTransform(
        action_tokenizer,
        processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=prompt_builder_fn,
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
    def collator(instances):
        batch = base_collator(instances)
        batch.pop("dataset_names", None)
        return batch

    dataloader = DataLoader(
        vla_dataset, batch_size=args.batch_size, sampler=None, collate_fn=collator, num_workers=0,
    )

    model.freeze_vla()
    model.unfreeze_reasoning()

    # --- Stage 1: train only cross-attention ---
    if args.training_stage is None or args.training_stage == 1:
        s1_start_step = (
            args.resume_step
            if (resume_training_state and resume_training_state.get("stage") == 1)
            else 0
        )

        optimizer = torch.optim.AdamW(model.get_reasoning_parameters(), lr=args.lr)
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=100, num_training_steps=args.max_steps,
        )

        # Restore optimizer/scheduler state if resuming stage 1
        if s1_start_step > 0 and resume_training_state:
            if "optimizer" in resume_training_state:
                optimizer.load_state_dict(resume_training_state["optimizer"])
                logger.info("Restored optimizer state for stage 1 resume")
            if "lr_scheduler" in resume_training_state:
                lr_scheduler.load_state_dict(resume_training_state["lr_scheduler"])
                logger.info("Restored scheduler state for stage 1 resume")

        model, optimizer, lr_scheduler, dataloader = accelerator.prepare(
            model, optimizer, lr_scheduler, dataloader,
        )

        train_loop(model, dataloader, optimizer, lr_scheduler, accelerator,
                   action_tokenizer=action_tokenizer,
                   pad_token_id=processor.tokenizer.pad_token_id,
                   stage=1, max_steps=args.max_steps, save_steps=args.save_steps,
                   output_dir=args.output_dir, run_name=run_name,
                   start_step=s1_start_step,
                   wrong_prefix_token_ids=wrong_prefix_token_ids,
                   contrastive_lambda=args.contrastive_lambda,
                   contrastive_margin=args.contrastive_margin)

        model = accelerator.unwrap_model(model)

    # --- Stage 2: add LoRA to LLM, train LoRA + cross-attention ---
    if args.training_stage is None or args.training_stage == 2:
        s2_start_step = (
            args.resume_step
            if (resume_training_state and resume_training_state.get("stage") == 2)
            else 0
        )

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
            if os.path.isdir(args.resume):
                model.vla.load_adapter(args.resume, "default")
                model.vla.set_adapter("default")
        model.vla.print_trainable_parameters()

        trainable_params = itertools.chain(
            model.get_reasoning_parameters(),
            (p for p in model.vla.parameters() if p.requires_grad),
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

        vla_dataset_s2 = RLDSDataset(
            Path(args.data_root_dir),
            args.dataset_name,
            batch_transform,
            resize_resolution=tuple(vla.config.image_sizes),
            shuffle_buffer_size=args.shuffle_buffer_size,
            image_aug=args.image_aug,
        )
        dataloader = DataLoader(
            vla_dataset_s2, batch_size=args.batch_size, sampler=None,
            collate_fn=collator, num_workers=0,
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
                   wandb_step_offset=stage2_wandb_offset,
                   keep_lora_pass2=args.keep_lora_pass2,
                   start_step=s2_start_step,
                   wrong_prefix_token_ids=wrong_prefix_token_ids,
                   contrastive_lambda=args.contrastive_lambda,
                   contrastive_margin=args.contrastive_margin)


if __name__ == "__main__":
    main()
