# RealSynCol

This repository contains code and configuration files used to evaluate RealSynCol, including fine-tuning, ablation studies, and benchmark
evaluations.

The repository is organized into two main folders, each corresponding to a
different experimental setup and model family. Detailed documentation and
usage instructions are provided in the README files inside each folder.

---

## Repository Structure

### 1. DAMv2 Fine-tuning

This folder contains the code used to fine-tune **Depth Anything V2** using
parameter-efficient training techniques (LoRA) for metric depth estimation.

📄 Refer to the README inside this folder for:
- dataset format and input specification
- training and validation setup
- usage examples and commands
- acknowledgments and references

---

### 2. Lite-Mono Custom Dataset and Options

This folder provides the modified components required to run **Lite-Mono**
experiments on datasets different from the original KITTI benchmark.

It includes custom dataset loaders and option definitions adapted to support
**RealSynCol**, **C3VD**, and **SimCol3D**, and was used for ablation and benchmark
studies.

📄 Refer to the README inside this folder for:
- description of the modified files
- required changes to the original Lite-Mono codebase
- supported datasets
- acknowledgments and references

---

## Notes

- This repository does **not** duplicate the full implementations of
  Depth Anything V2 or Lite-Mono.
- Environment setup, base model code, and training pipelines should be obtained
  from the respective original repositories.
- Each subfolder README provides the necessary details to reproduce the
  corresponding experiments.

---

## Acknowledgments

This work builds upon the following open-source projects:

- **Depth Anything V2**  
  https://github.com/DepthAnything/Depth-Anything-V2

- **Lite-Mono**  
  https://github.com/noahzn/Lite-Mono

We thank the authors for releasing their code and pretrained models to the
research community.