"""ReasonVLA variant with two architectural fixes for the gated feedback mode.

Diagnosis from training run 6559914 (gated, hl=24, libero_goal):
  - hint magnitude ~436 throughout training — the unmerger output is unbounded
    and dwarfs the O(1)–O(10) patch features it's added to.
  - With gate sigmoid centered at 0.5 at init, ~half of that 436-norm hint
    lands on every patch from step 0, overwhelming the frozen base.
  - Loss collapses 9.7 -> 0.001 in 30 steps via shortcut memorization, gradients
    vanish, eval generalizes catastrophically (12% on libero_goal).

Two fixes applied here, both targeting only the `gated` feedback path:

  Fix 1: bias the gate logit toward identity at init
    - main_gate's final Linear has zero weight + bias = GATE_INIT_BIAS (= -5),
      so sigmoid(bias) = 0.0067 at init. Frozen base is preserved; the gate
      learns to open up only where reasoning helps.

  Fix 2: bound the unmerger output magnitude
    - Apply F.layer_norm to the unmerger output (per-patch, over vision_dim)
      then multiply by HINT_SCALE (= 0.1). Hint norm is now ~0.1 * sqrt(d_v)
      ~ a few units, comparable to patch features instead of 50-400x larger.

These together give init-eval ~= baseline (since gate ~ 0 and even at gate = 1
the bounded hint is small), so training has a useful starting point and learns
gradually instead of memorizing.

Usage (point your existing scripts at this file's class):

    from reason_vla_gated_fixed import ReasonVLAGatedFixed as ReasonVLA

Or in train/eval scripts, swap the import line:

    # was:  from reason_vla import ReasonVLA
    from reason_vla_gated_fixed import ReasonVLAGatedFixed as ReasonVLA
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from reason_vla import ReasonVLA


GATE_INIT_BIAS: float = -5.0   # sigmoid(-5) ~ 0.0067 -> near-identity at init
HINT_SCALE: float = 0.1        # post-layernorm scale on the unmerger output


def _identity_init_gate(gate_seq: nn.Sequential) -> None:
    """Zero the final Linear's weight and set its bias to a large negative value
    so the sigmoid output starts ~0 and the gated feedback is near-identity."""
    final_linear = gate_seq[1]
    assert isinstance(final_linear, nn.Linear), (
        f"Expected gate[1] to be nn.Linear, got {type(final_linear)}"
    )
    nn.init.zeros_(final_linear.weight)
    nn.init.constant_(final_linear.bias, GATE_INIT_BIAS)


class ReasonVLAGatedFixed(ReasonVLA):
    """ReasonVLA with identity-init gate + bounded hint magnitude.

    Only the `gated` feedback_mode is affected. Other modes fall through to the
    parent class unchanged. State-dict layout is identical to the parent class
    (same module names), so save/load is fully compatible — the only difference
    is initial parameter values for `main_gate` / `fused_gate`.
    """

    def __init__(self, vla, hidden_layer: int = -1, feedback_mode: str = "additive"):
        super().__init__(vla, hidden_layer=hidden_layer, feedback_mode=feedback_mode)
        if self.feedback_mode == "gated":
            _identity_init_gate(self.main_gate)
            if self.is_fused:
                _identity_init_gate(self.fused_gate)

    def set_image_reasoning(self, image_hidden: torch.Tensor) -> None:
        """Override only the gated branch to bound the hint magnitude.

        Other feedback modes delegate to the parent implementation.
        """
        if self.feedback_mode != "gated":
            return super().set_image_reasoning(image_hidden)

        reasoning_out = self.visual_reasoner(image_hidden)         # [bsz, num_patches, llm_dim]

        raw_hint_main = self.main_unmerger(reasoning_out)          # [bsz, 256, vision_dim]
        self._hint_main = HINT_SCALE * F.layer_norm(
            raw_hint_main, normalized_shape=(raw_hint_main.shape[-1],)
        )
        self._gate_main = self.main_gate(reasoning_out)            # [bsz, 256, 1]

        if self.is_fused:
            raw_hint_fused = self.fused_unmerger(reasoning_out)    # [bsz, 256, vision_dim_fused]
            self._hint_fused = HINT_SCALE * F.layer_norm(
                raw_hint_fused, normalized_shape=(raw_hint_fused.shape[-1],)
            )
            self._gate_fused = self.fused_gate(reasoning_out)      # [bsz, 256, 1]

        self.reasoning_hint = True
