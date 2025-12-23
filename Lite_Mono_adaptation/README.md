# Training Adaptation for Lite-Mono on colonoscopy synthetic datasets

This folder contains **only the modified components** used in our **ablation studies**
and **benchmark experiments** based on **Lite-Mono**.

Specifically, it includes:
- `options.py`
- `custom_dataset.py`

These files are **adapted from the original Lite-Mono repository** and were modified
to enable training and evaluation of the Lite-Mono network on the
**RealSynCol**, **C3VD**, and **SimCol3D** datasets.

All remaining components of the codebase, such as the model architecture,
training pipeline, and environment setup, follow the original Lite-Mono
implementation and are not modified in this work.

For the complete codebase and setup instructions, please refer to the original
repository:
👉 https://github.com/noahzn/Lite-Mono

---

## Content of This Folder

This folder **does not represent a standalone implementation** of Lite-Mono.

It only provides:
- a **customized options parser** (`options.py`)
- a **modified dataset loader** (`custom_dataset.py`)

These components were introduced to support:
- dataset-specific paths and flags
- custom data formats and splits
- experiments on non-KITTI datasets used in our ablation and benchmark studies

---


## Required Modification in `trainer.py`

To use the provided `custom_dataset.py`, a small change is required in the
`trainer.py` file from the original Lite-Mono repository.

### Original code (Lite-Mono)
```python
# data
datasets_dict = {"kitti": datasets.KITTIRAWDataset,
                 "kitti_odom": datasets.KITTIOdomDataset}

```
### Modified code (Lite-Mono)
```python
# data
datasets_dict = {"custom": datasets.CustomRAWDataset}
```

## Acknowledgments

This code is based on **Lite-Mono**, a lightweight framework for self-supervised
monocular depth estimation. We gratefully acknowledge the authors for making
their code publicly available.

The original implementation and full training pipeline are available at:
https://github.com/noahzn/Lite-Mono

All credit for the model architecture, training methodology, and core
implementation belongs to the original authors.