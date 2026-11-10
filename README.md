# nanoDINO

[![DINOv2 paper](https://img.shields.io/badge/arXiv-2304.07193-b31b1b.svg)](https://arxiv.org/abs/2304.07193)
[![Weights](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-weights-ffcc4d)](https://huggingface.co/shumpei777/nanoRAE)
[![Used by nanoRAE](https://img.shields.io/badge/used%20by-nanoRAE-24292f?logo=github)](https://github.com/shumpei-takezaki/nanoRAE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-3776ab.svg?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

DINO and DINOv2 self-supervised pretraining from scratch on CIFAR-10, with a ViT-Tiny/4 (192-d, 8×8 = 64 patch tokens) and a ViT-S/4.
The DINOv2 teacher backbones are the frozen encoders of [nanoRAE](https://github.com/shumpei-takezaki/nanoRAE).

## 📰 Updates

- **2026-11-10 (v0.2):** renamed to nanoDINO; the Python package is now `nanodino` (was `cifar_dino`); reorganized README and docs;
  third-party license notices.
- **2026-11-06 (v0.1.3):** the encoders are released on Hugging Face ([shumpei777/nanoRAE](https://huggingface.co/shumpei777/nanoRAE)).
- **2026-10-30 (v0.1):** first release: DINO vs. DINOv2, patch tokens and registers, ViT-Tiny/4 vs. ViT-S/4 (experiments 01-03).

## 📊 Results

CIFAR-10 test top-1 (%), seed 0, 1000 epochs:

| Model | kNN CLS | kNN PatchMean | Linear CLS | Linear PatchMean |
|---|---:|---:|---:|---:|
| DINO, ViT-Tiny/4 | 82.04 | 78.27 | 79.86 | 77.74 |
| DINOv2, ViT-Tiny/4 | 85.49 | 86.05 | 81.25 | 86.59 |
| DINOv2-Reg4, ViT-Tiny/4 | 84.59 | 84.04 | 79.84 | 85.00 |
| DINOv2, ViT-S/4 | 86.96 | 87.04 | 76.59 | 89.54 |

The linear-probe hyperparameters are fixed and untuned, so the CLS linear numbers are provisional.
All experiments, with training curves and the patch-token / register analysis, are listed in [docs/README.md](docs/README.md).

## 🗺️ Roadmap

🚧 in progress · 📋 planned

- 📋 Larger encoder: DINOv2 ViT-B/4 (for nanoRAE)
- 📋 More seeds, and a linear-probe protocol with hyperparameters chosen on a validation split
- 📋 More datasets: CIFAR-100, Tiny ImageNet

## ⚡ Quick start

```bash
git clone https://github.com/shumpei-takezaki/nanoDINO.git && cd nanoDINO
uv sync
uvx --from huggingface_hub hf download shumpei777/nanoRAE --local-dir pretrained_models --include encoders/encoder_dinov2_vit_tiny4.pt
```

```python
import torch
from nanodino.export import load_backbone

model, meta = load_backbone("pretrained_models/encoders/encoder_dinov2_vit_tiny4.pt")  # no projection head needed
with torch.no_grad():
    out = model.forward_features(images)        # images: CIFAR-normalized [B,3,32,32]
cls, regs, patches = out["cls"], out["reg_tokens"], out["patch_tokens"]  # [B,192], [B,R,192], [B,64,192]
patch_map = patches.transpose(1, 2).reshape(-1, 192, 8, 8)
```

Released backbones: `encoders/encoder_dinov2_vit_tiny4.pt` (DINOv2 ViT-Tiny/4), `encoders/encoder_dinov2_vit_s4.pt` (DINOv2 ViT-S/4)
and `encoders/disc_dino_vit_tiny4.pt` (DINO ViT-Tiny/4).

## 📚 Documentation

- [Experiments](docs/README.md): one write-up per experiment, with figures; final metrics in [results.json](docs/results.json).
- [Training and evaluation](docs/training.md): configs, commands, training times, outputs, kNN / linear probe and the qualitative analysis.
- [Implementation notes](docs/implementation.md): where this code follows, and departs from, the official DINO / DINOv2.

## 🙏 Acknowledgements

- [DINO](https://github.com/facebookresearch/dino) ([Emerging Properties in Self-Supervised Vision Transformers](https://arxiv.org/abs/2104.14294))
  and [DINOv2](https://github.com/facebookresearch/dinov2) ([DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193)):
  the algorithms and the reference implementations (losses, centering, heads, iBOT masking, KoLeo).
- [iBOT](https://arxiv.org/abs/2111.07832) (masked patch objective) and [Vision Transformers Need Registers](https://arxiv.org/abs/2309.16588) (register tokens).
- [mrnghia90/dino-from-scratch](https://huggingface.co/mrnghia90/dino-from-scratch): the CIFAR-10 ViT-Tiny/4 setup and the `mrnghia90` augmentation recipe.
- [DOV4CL](https://github.com/xieyc99/DOV4CL) and [solo-learn](https://github.com/vturrisi/solo-learn): CIFAR DINO implementations used as references.

## 📝 Citation

If you use this repository, please cite it:

```bibtex
@misc{takezaki2026nanodino,
      title={nanoDINO: Small-Scale DINO and DINOv2 Pretraining from Scratch},
      author={Shumpei Takezaki},
      year={2026},
      howpublished={\url{https://github.com/shumpei-takezaki/nanoDINO}},
}
```

and the DINO and DINOv2 papers:

```bibtex
@inproceedings{caron2021emerging,
  title={Emerging Properties in Self-Supervised Vision Transformers},
  author={Caron, Mathilde and Touvron, Hugo and Misra, Ishan and J\'egou, Herv\'e  and Mairal, Julien and Bojanowski, Piotr and Joulin, Armand},
  booktitle={Proceedings of the International Conference on Computer Vision (ICCV)},
  year={2021}
}

@misc{oquab2023dinov2,
  title={DINOv2: Learning Robust Visual Features without Supervision},
  author={Oquab, Maxime and Darcet, Timothée and Moutakanni, Theo and Vo, Huy V. and Szafraniec, Marc and Khalidov, Vasil and Fernandez, Pierre and Haziza, Daniel and Massa, Francisco and El-Nouby, Alaaeldin and Howes, Russell and Huang, Po-Yao and Xu, Hu and Sharma, Vasu and Li, Shang-Wen and Galuba, Wojciech and Rabbat, Mike and Assran, Mido and Ballas, Nicolas and Synnaeve, Gabriel and Misra, Ishan and Jegou, Herve and Mairal, Julien and Labatut, Patrick and Joulin, Armand and Bojanowski, Piotr},
  journal={arXiv:2304.07193},
  year={2023}
}
```

## 📄 License

MIT ([LICENSE](LICENSE)). `src/nanodino/masking.py` is adapted from DINOv2 and keeps its Apache-2.0 license: see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
