"""Training entry point that runs reason_vla.main() with the fixed gated class.

Use this in place of `python reason_vla.py ...` when you want the
identity-init gate + bounded hint magnitude (only affects feedback_mode=gated).

All argparse flags from reason_vla.py are forwarded as-is — the only difference
is that the model instantiated inside main() is ReasonVLAGatedFixed.

Usage:
    python reason_vla_gated_fixed_train.py \
        --vla-path openvla/openvla-7b-finetuned-libero-goal \
        --dataset-name libero_goal_no_noops \
        --feedback-mode gated --hidden-layer 24 \
        --lr 2e-5 --max-steps 4950 --training-stage 1 \
        ...
"""
import reason_vla
from reason_vla_gated_fixed import ReasonVLAGatedFixed

# Swap the class reference inside reason_vla so main() picks up the fixed one
# at the line `model = ReasonVLA(vla, hidden_layer=..., feedback_mode=...)`.
reason_vla.ReasonVLA = ReasonVLAGatedFixed


if __name__ == "__main__":
    reason_vla.main()
