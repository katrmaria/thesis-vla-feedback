# Diagnostic Analysis

Utilities used to analyse how OpenVLA processes vision and language across
its LLM layers, and to identify the extraction layers used by the three
mechanisms.

## Layers identified

| Layer | Role | Used by |
|---|---|---|
| L2 | Pre-integrated reference (selected by Hellinger distance from L27) | Mechanism 1, contrastive mask |
| L7 | Instruction encoded before fusion | Mechanism 3 |
| L9 | Upstream of grounding phase | Mechanism 1, residual scaling |
| L10 to L15 | Main vision-language fusion | Mechanism 1, patch-embedding mask |
| L11 | Sharpest grounding head | Mechanism 1, ViT bias and residual scaling |
| L23 to L26 | Still affect action despite low attention | Mechanism 2, hyperparameter search |
| L24 | Selected by Mechanism 2 gated | Mechanism 2 gated winner |
| L27 | Post-integrated reference | Mechanism 1, contrastive mask |
| L28 | Late visual review | Mechanism 1, Song adaptation |
| L31 | Predicts the action | Baseline output |

## Files

- `compute_hellinger_distance.py` per-layer Hellinger distance for
  pre-integrated layer selection (Mechanism 1)
- `diagnose_layer_signal.py` layer-wise removal of image or instruction tokens
- `diagnostic_projector_xattn.py` instruction-dependence diagnostic (Mechanism 3)
- `visualize_attention.py` per-layer attention heatmaps
- `logit_lens_gated.py` layer-wise inspection of the gated hint vector
