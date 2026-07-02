# Mechanism 3, Projector Cross-Attention

Instruction-token hidden states from a chosen LLM layer L* are used as keys
and values for a cross-attention block that queries the projected patches,
gated by a single tanh scalar.

    P' = P + tanh(g) * CrossAttn(Q=P, K=H_txt, V=H_txt)

## Results on LIBERO-Spatial

| Configuration | Layer | Success rate |
|---|---:|---:|
| OpenVLA baseline | — | 83.6 |
| **Projector Cross-Attention** | 7 | **86.2** |
| Projector Cross-Attention | 24 | 85.4 |

Layer 7 is strongest, before the main fusion, when the instruction is
still a pure encoding of the task.

## Contrastive variant

`reason_vla_projector_crossattn_contrastive.py` adds a wrong-instruction
forward and a hinge term penalising when L_correct >= L_wrong. The hinge
gradient was zero throughout training in our runs, so the variant did not
improve the primary result.

## Files

- `reason_vla_projector_crossattn.py` main class, 4 heads dim 1024
- `reason_vla_projector_crossattn_contrastive.py` contrastive variant

## Parameters

67 M in stage 1. Optional 80 M LoRA adapter in stage 2.
