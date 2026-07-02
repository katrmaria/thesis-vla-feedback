# Mechanism 1, Inference-Time Attention Modulation

Training-free. Uses attention weights from Pass 1 as a per-patch importance
mask in Pass 2.

## Four variants

| Variant | Side | Mask source | Where applied |
|---|---|---|---|
| Patch-embedding modulation | Vision | Avg attention L10 to L15 | Before ViT blocks |
| ViT attention bias | Vision | Sharpest head at L11 | ViT self-attention logits |
| Residual scaling at L9 | Language | Sharpest head at L11 | h * (1 + alpha * mask) at L9 |
| Contrastive suppression at L28 | Language | \|A(L27) - A(L2)\| | Suppress bottom 20% at L28 |

## Results on LIBERO-Spatial

| Variant | Success rate |
|---|---:|
| OpenVLA baseline | 83.6 |
| Patch-embedding | 85.6 |
| ViT attention bias | 84.8 |
| **Residual scaling at L9** | **85.8** |
| Contrastive suppression at L28 | 84.6 |

## Files

- `attention_modulation_testtime_v2.py` language-side variants
- `attention_modulation_testtime_vit.py` vision-side variants
- `attention_modulation_testtime.py` earlier residual-at-L9 implementation

## Parameters

Zero trainable.
