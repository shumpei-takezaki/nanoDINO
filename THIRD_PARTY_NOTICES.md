# Third-party code

nanoDINO is released under the MIT License ([LICENSE](LICENSE)). One file is adapted from another project and keeps its license:

| File | Source | License |
|---|---|---|
| `src/nanodino/masking.py` | [DINOv2](https://github.com/facebookresearch/dinov2) (`dinov2/data/masking.py`, `dinov2/data/collate.py`; commit `7764ea0`), Copyright (c) Meta Platforms, Inc. and affiliates. Modified: a seeded random generator, vectorized mask filling, and the per-sample masking of `collate_data_and_cast` as a standalone function | Apache-2.0 ([licenses/LICENSE-DINOv2.txt](licenses/LICENSE-DINOv2.txt)) |

The rest of the code was written for this repository, following the algorithms and conventions of the official
[DINO](https://github.com/facebookresearch/dino) and [DINOv2](https://github.com/facebookresearch/dinov2) implementations
(see [docs/implementation.md](docs/implementation.md)); the CIFAR-10 augmentation settings follow
[mrnghia90/dino-from-scratch](https://huggingface.co/mrnghia90/dino-from-scratch).
