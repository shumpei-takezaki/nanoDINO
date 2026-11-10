# Implementation notes

Where this implementation follows, and where it departs from, the official [DINO](https://github.com/facebookresearch/dino) and
[DINOv2](https://github.com/facebookresearch/dinov2) code.

- **iBOT masking** follows DINOv2 `prepare_tokens_with_masks`. The mask token replaces patch embeddings before the positional embedding and the Transformer. Registers are inserted after CLS and get no positional embedding.
- **Mask sampling** follows the official DINOv2 `MaskingGenerator` and `collate_data_and_cast`. Exactly `int(0.5·2B)` global student samples are masked, with ratios stratified over [0.1, 0.5], and the masks are shuffled across the batch. The per-sample marginal is Bernoulli(0.5) × Uniform(0.1, 0.5), but the number of masked samples per batch is fixed.
- **Centering** uses DINO-style EMA centering for both losses, with separate centers for CLS and iBOT. The iBOT center is updated from masked positions only. Official DINOv2 defaults to Sinkhorn-Knopp centering; this repo does not.
- **KoLeo** is computed per global view on the student backbone CLS token and then averaged over views. Official DINOv2 sums over views.
- **DINO head:** the last layer is `F.linear(x, normalize(W))`. This is exactly weight norm with a frozen unit gain. The last layer is frozen for epoch 1.
- **Backbone outputs:** all outputs of `forward_features` are taken after the final LayerNorm. Register tokens are never included in `patch_tokens`.
- **Not implemented** (all raise an error): AMP, a separate iBOT head, and augmentation profiles other than `mrnghia90`.
