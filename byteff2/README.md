---
license: apache-2.0
tags:
- chemistry
- biology
---
# ByteFF2

This repository contains the model used for the paper [Bridging Quantum Mechanics to Organic Liquid Properties via a Universal Force Field](https://arxiv.org/abs/2508.08575)。

[ByteFF-Pol](https://arxiv.org/abs/2508.08575) is a polarizable force field parameterized by a graph neural network (GNN), trained on high-level quantum mechanics (QM) data, thus eliminating the need for experimental calibration. ByteFF-Pol achieves exceptional accuracy in predicting the thermodynamic and transport properties of small-molecule liquids and electrolytes, outperforming SOTA traditional and ML force fields

# Trained Models
The `trained_models` folder contains the trained model for ByteFF-Pol and its corresponding configuration (.yaml) file.

# How to use
Code and examples are available in the [byteff2](https://github.com/ByteDance-Seed/byteff2) repository.

## Citation
If you find ByteFF-Pol is useful for your research and applications, feel free to give us a star ⭐ or cite us using:

```bibtex

@misc{zheng2025bridgingquantummechanicsorganic,
  title         = {Bridging Quantum Mechanics to Organic Liquid Properties via a Universal Force Field},
  author        = {Tianze Zheng and Xingyuan Xu and Zhi Wang and Xu Han and Zhenliang Mu and Ziqing Zhang and Sheng Gong and Kuang Yu and Wen Yan},
  year          = {2025},
  eprint        = {2508.08575},
  archivePrefix = {arXiv},
  primaryClass  = {physics.comp-ph},
  url           = {https://arxiv.org/abs/2508.08575}
}
```