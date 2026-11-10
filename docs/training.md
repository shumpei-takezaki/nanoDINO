# Training and evaluation

## Setup

```bash
uv sync
echo 'WANDB_API_KEY=...' > .env   # optional; scripts load it. Without a key, wandb logs offline (`wandb sync outputs/*/wandb/offline-run-*`)
# CIFAR-10 is downloaded to ./data automatically (or symlink an existing data/cifar-10-batches-py)
```

## Configs

| config | method | registers | role |
|---|---|---:|---|
| `configs/dino.yaml` | DINO (CLS loss) | 0 | strict comparison |
| `configs/dinov2.yaml` | DINO + iBOT + KoLeo | 0 | strict comparison |
| `configs/dinov2_reg4.yaml` | DINO + iBOT + KoLeo | 4 | RAE-aligned variant, reported separately |

All three share `configs/base.yaml` (via `_base_`): the same data, `mrnghia90` augmentation, backbone, head, optimizer and schedules.
`configs/dinov2_vits.yaml` is the same DINOv2 recipe with a ViT-S/4 backbone (384-d, 6 heads; trained with TF32), used to
compare encoder sizes in [nanoRAE](https://github.com/shumpei-takezaki/nanoRAE).

## Training

Run long jobs in a detached tmux session. The scripts resume automatically from `checkpoint_last.pt`.

```bash
# smoke (2 epochs)
uv run python -m nanodino.train --config configs/dino.yaml   --epochs 2 --batch-size 32 --output-dir outputs/smoke_dino
uv run python -m nanodino.train --config configs/dinov2.yaml --epochs 2 --batch-size 32 --output-dir outputs/smoke_dinov2

# primary 1000-epoch runs (seed 0; another seed: append --seed N --output-dir outputs/<config>_seedN)
tmux new-session -d -s dino        'scripts/train_dino.sh        > outputs/dino.log 2>&1'
tmux new-session -d -s dinov2      'scripts/train_dinov2.sh      > outputs/dinov2.log 2>&1'
tmux new-session -d -s dinov2_reg4 'scripts/train_dinov2_reg4.sh > outputs/dinov2_reg4.log 2>&1'

# explicit resume
uv run python -m nanodino.train --config configs/dinov2.yaml --output-dir outputs/dinov2 \
  --resume outputs/dinov2/checkpoints/checkpoint_last.pt

# DDP (the global batch stays 256; the per-GPU batch is 256 / world_size)
uv run torchrun --nproc_per_node=2 -m nanodino.train --config configs/dino.yaml --output-dir outputs/dino
```

Measured on one A100-40GB, bs 256, fp32: about 55–75 s/epoch, so each 1000-epoch run takes about 16–20 h.
The GPU stays at ~98% utilization, and TF32 (`train.allow_tf32: true`) only saves about 6%.

Each run writes to `outputs/<config>/`:
- `config.yaml`: the resolved config, including all augmentation parameters
- `env.json`: git commit, library versions and GPU
- `train.jsonl`: step records every 50 steps and epoch averages, with each loss logged separately, plus kNN every 50 epochs. This is the canonical log.
- `wandb/`, `wandb_run_id.txt`: wandb run (project `dino-cifar`, run `<config>`). Step records go under `train/*` and epoch records under `epoch/*`, and a resumed run continues the same wandb run.
- `checkpoints/checkpoint_last.pt` and `checkpoints/checkpoint_epoch_XXXX.pt`
- `teacher_backbone.pt`
- `metrics.json`: final kNN and linear probe results

## Evaluation

```bash
uv run python -m nanodino.evaluate --checkpoint outputs/dinov2/teacher_backbone.pt --data-root ./data
uv run python -m nanodino.evaluate --table outputs/{dino,dinov2,dinov2_reg4}/metrics.json  # -> results.md / results.json
```

- **kNN:** k=20, T=0.07, run on the L2-normalized CLS token and on the mean patch token.
- **Linear probe:** trains on frozen, standardized features with SGD for 100 epochs (lr 0.1, cosine). The hyperparameters are fixed and identical for all methods.

## Qualitative analysis

```bash
uv run python -m nanodino.analyze outputs/{dino,dinov2,dinov2_reg4}/teacher_backbone.pt --out-dir outputs/analysis
```

Writes `tsne.png` (t-SNE of the L2-normalized CLS / mean-patch features on 5000 test images) and `patch_pca.png`
(DINOv2-style PCA of patch tokens on 2 test images per class). The per-position mean patch token is subtracted before
the PCA; without this, PC1 mostly separates the top of the image from the bottom. Takes about 1.5 min on one A100.
