# Depth Anything V2 (DAMv2) LoRA Fine-tuning for Metric Depth Estimation

This repository provides a PyTorch training script to fine-tune **Depth Anything V2 Small (HF)** for **metric depth estimation** using **LoRA (PEFT)** and **EXR depth maps**.

## Input

- **RGB images**  
  Standard RGB images (e.g., PNG or JPG).

- **Depth labels**  
  Depth maps stored as `.exr` files with `float32` values in the range **[0, 1]**.

- **Metric depth conversion**  
  Depth values are interpreted as a normalized range and converted to millimeters as:
  ```text
  depth_mm = depth_exr * max_depth_mm
  
- Training data is provided through two CSV files, `train.csv` and `validation.csv`.
Each row in the CSV files specifies the paths to a single RGB frame and its
corresponding depth map, allowing the training pipeline to load paired RGB–depth
samples directly from disk.

## Implementation Details

The model is fine-tuned starting from **Depth Anything V2 Small** using Hugging Face
`AutoModelForDepthEstimation`. Parameter-efficient fine-tuning is performed via
**LoRA (Low-Rank Adaptation)**, which is injected into the encoder attention layers
and the MLP linear layers of the backbone.

During training, all base model parameters are frozen. Only the LoRA parameters
and the depth prediction head are updated, enabling efficient adaptation while
preserving the pretrained representations.

Training and validation are performed in a **metric depth setting**, with losses
and metrics computed directly in millimeters. Model performance is evaluated using
mean absolute error (**L1**, in mm) and root mean squared error (**RMSE**, in mm).

The training pipeline automatically saves per-epoch checkpoints, tracks the best
model according to validation metrics, and produces a fully merged final checkpoint
(`final_merged`) in which the LoRA weights are injected into the base model.
Qualitative visualizations of predictions can also be saved during validation in the
form of side-by-side panels (**RGB | GT | Pred**).


## Requirements

Recommended (example versions; adjust to your setup):

- Python 3.10
- PyTorch
- transformers
- peft
- pillow, numpy, imageio, opencv-python

Install dependencies:

```bash
pip install -r requirements.txt
```


## Training Example

The training script can be launched directly from the command line by specifying
the training and validation CSV files, along with the desired output directory.
The parameter max_depth_mm is customized on realSynCol dataset, where the maximum depth is 200 mm. 

```bash
python train.py \
  --train_csv train.csv \
  --val_csv validation.csv \
  --model_name depth-anything/Depth-Anything-V2-Small-hf \
  --output_dir ./outputs_damv2_lora \
  --batch_size 4 \
  --lr 1e-4 \
  --epochs 10 \
  --max_depth_mm 200
```