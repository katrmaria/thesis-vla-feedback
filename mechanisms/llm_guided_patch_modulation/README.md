# Mechanism 2, LLM-Guided Patch Modulation

Image-token hidden states from a chosen LLM layer L* are passed through a
gated visual reasoner and two per-encoder unmergers, then applied to the
patch embeddings before the ViT blocks.

## Feedback modes

| Mode | Formula |
|---|---|
| Additive | p' = p + r |
| Gated | p' = p + sigmoid(gate) * layernorm(r) * 0.1 |
| FiLM | p' = p * gamma + beta |

Gated is initialised so the model matches the baseline exactly until the
gate opens during training.

## Results on LIBERO-Spatial

| Configuration | Success rate |
|---|---:|
| OpenVLA baseline | 83.6 |
| Best additive, hl=-1, lr=2e-5 | 87.4 |
| Best gated, hl=24, lr=2e-5 | **88.0** |
| Ensemble of the above | **89.2** |

## Files

- `reason_vla.py` main class (additive, FiLM, AdaLN, scaled modes)
- `reason_vla_gated_fixed.py` gated variant, produces the 88.0 result
- `reason_vla_gated_fixed_train.py` training loop for gated
- `reason_vla_multilayer.py` multi-point variant injecting at 3 ViT layers

## Parameters

160 M in stage 1 (visual reasoner plus unmergers). Optional 80 M LoRA
adapter in stage 2.
