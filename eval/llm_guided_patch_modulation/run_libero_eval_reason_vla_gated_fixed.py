"""LIBERO eval entry point that uses ReasonVLAGatedFixed for the gated path.

Wraps run_libero_eval_reason_vla.main() after monkey-patching the ReasonVLA
symbol the eval script uses, so `from_finetuned`'s `cls(...)` instantiates the
fixed subclass. Same CLI args as the original eval script.
"""
import reason_vla
from reason_vla_gated_fixed import ReasonVLAGatedFixed

# Patch BEFORE importing the eval module so its top-level
# `from reason_vla import ReasonVLA` picks up the fixed class.
reason_vla.ReasonVLA = ReasonVLAGatedFixed

import run_libero_eval_reason_vla  # noqa: E402

# Belt-and-suspenders: also overwrite the symbol the eval module already bound.
run_libero_eval_reason_vla.ReasonVLA = ReasonVLAGatedFixed


if __name__ == "__main__":
    run_libero_eval_reason_vla.main()
