import os
import argparse
import csv

import torch
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import numpy as np
import imageio.v3 as iio
import matplotlib.pyplot as plt

from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from peft import PeftModel


# -----------------------------
# Dataset (RGB image + EXR depth in mm)
# -----------------------------
class DepthDatasetEXRMetric(Dataset):
    """
    CSV format:
        image_path,depth_path
        /path/to/img.png,/path/to/depth.exr

    depth_path: EXR file with float32 values in [0,1] representing 0-max_depth_mm mm.
    We convert them to millimeters (0-max_depth_mm).
    """

    def __init__(self, csv_path, max_depth_mm=200.0):
        self.samples = []
        self.max_depth_mm = max_depth_mm

        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_path = row["image_path"]
                depth_path = row["depth_path"]
                self.samples.append((img_path, depth_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, depth_path = self.samples[idx]

        # Load RGB image
        image = Image.open(img_path).convert("RGB")

        # Load depth EXR -> float32 (H, W) in [0,1]
        depth_np = iio.imread(depth_path).astype("float32")

        # If EXR is H x W x C (e.g. 3 channels), take the first channel
        if depth_np.ndim == 3:
            depth_np = depth_np[..., 0]

        # Convert to millimeters
        depth_mm = depth_np * self.max_depth_mm  # [0, max_depth_mm]

        depth_tensor = torch.from_numpy(depth_mm)  # (H, W)

        return {
            "image": image,
            "depth": depth_tensor,
            "image_path": img_path,
        }


# -----------------------------
# Collate function (labels in mm)
# -----------------------------
def make_collate_fn(image_processor):
    """
    Uses AutoImageProcessor for rgb preprocessing and resizes depth maps
    to match the model input resolution.
    Keeps image_path as list of strings for debugging/plots.
    """

    def collate_fn(batch):
        images = [b["image"] for b in batch]
        depths = [b["depth"] for b in batch]  # already in mm
        image_paths = [b["image_path"] for b in batch]

        inputs = image_processor(
            images=images,
            return_tensors="pt",
        )
        pixel_values = inputs["pixel_values"]  # (B, C, H, W)
        _, _, h, w = pixel_values.shape

        depth_stack = []
        for d in depths:
            d = d.unsqueeze(0).unsqueeze(0)  # (1,1,H_d,W_d)
            d_resized = torch.nn.functional.interpolate(
                d, size=(h, w), mode="bilinear", align_corners=False
            )
            depth_stack.append(d_resized)

        labels = torch.cat(depth_stack, dim=0).squeeze(1)  # (B, H, W)

        return {
            "pixel_values": pixel_values,
            "labels": labels,
            "image_paths": image_paths,
        }

    return collate_fn


# -----------------------------
# Metric computation helpers
# -----------------------------
def compute_depth_metrics_per_pixels(pred_mm, gt_mm, max_depth_mm=200.0, eps=1e-6):
    """
    Compute abs_rel, sq_rel, rmse, rmse_log, delta1, delta2, delta3
    over valid pixels, all quantities in *millimeters* where applicable.

    pred_mm, gt_mm: 1D tensors with valid depth values in millimeters.

    Returns sums (not averaged):
      abs_rel:   sum(|d - d*| / d*)               (unitless)
      sq_rel:    sum((d - d*)^2 / d*)            (in mm)
      rmse:      sum((d - d*)^2)                 (in mm^2)
      rmse_log:  sum((log d - log d*)^2)         (unitless)
      delta1/2/3: counts
      n: number of pixels
    """

    # Stay in millimeters
    pred = pred_mm
    gt = gt_mm

    # Clamp to avoid division by zero / log(0)
    gt = torch.clamp(gt, min=eps)
    pred = torch.clamp(pred, min=eps)

    n = gt.numel()
    if n == 0:
        device = gt.device
        return {
            "abs_rel": torch.tensor(0.0, device=device),
            "sq_rel": torch.tensor(0.0, device=device),
            "rmse": torch.tensor(0.0, device=device),
            "rmse_log": torch.tensor(0.0, device=device),
            "delta1": torch.tensor(0.0, device=device),
            "delta2": torch.tensor(0.0, device=device),
            "delta3": torch.tensor(0.0, device=device),
            "n": torch.tensor(0, device=device),
        }

    diff = pred - gt
    abs_rel = torch.sum(torch.abs(diff) / gt)
    sq_rel = torch.sum((diff ** 2) / gt)
    mse = torch.sum(diff ** 2)

    log_diff = torch.log(pred) - torch.log(gt)
    mse_log = torch.sum(log_diff ** 2)

    # Threshold deltas
    ratio = torch.maximum(pred / gt, gt / pred)
    delta1 = torch.sum(ratio < 1.25)
    delta2 = torch.sum(ratio < (1.25 ** 2))
    delta3 = torch.sum(ratio < (1.25 ** 3))

    return {
        "abs_rel": abs_rel,
        "sq_rel": sq_rel,
        "rmse": mse,
        "rmse_log": mse_log,
        "delta1": delta1,
        "delta2": delta2,
        "delta3": delta3,
        "n": torch.tensor(n, device=gt.device),
    }


def recover_rgb_for_plot(image_path):
    """
    Load an RGB image from disk and return as uint8 numpy array (H, W, 3).
    """
    img = Image.open(image_path).convert("RGB")
    return np.array(img)


def save_debug_plot(rgb, gt_mm, pred_mm, save_path, max_depth_mm=200.0):
    """
    Save a debug plot with 3 columns:
    - Left: RGB image
    - Middle: ground truth depth (mm, colormap)
    - Right: predicted depth (mm, colormap)
    """

    gt = gt_mm.detach().cpu().numpy()
    pred = pred_mm.detach().cpu().numpy()

    vmin, vmax = 0.0, max_depth_mm

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # RGB
    axes[0].imshow(rgb)
    axes[0].set_title("RGB")
    axes[0].axis("off")

    # GT depth
    im1 = axes[1].imshow(gt, cmap="inferno", vmin=vmin, vmax=vmax)
    axes[1].set_title("GT depth (mm)")
    axes[1].axis("off")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # Pred depth
    im2 = axes[2].imshow(pred, cmap="inferno", vmin=vmin, vmax=vmax)
    axes[2].set_title("Pred depth (mm)")
    axes[2].axis("off")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# -----------------------------
# Test / evaluation loop
# -----------------------------
@torch.no_grad()
def evaluate(
    model,
    dataloader,
    device,
    max_depth_mm=200.0,
    debug_dir=None,
    max_debug_images=3,
):
    """
    Evaluate model on test set and compute depth metrics PER IMAGE.

    For each image:
      - Abs Rel
      - Sq Rel
      - RMSE (mm)
      - RMSE log
      - δ < 1.25, δ < 1.25^2, δ < 1.25^3

    Final output:
      mean ± std over images (NOT pixel-wise averages).

    All depths are in millimeters.
    """

    model.eval()

    # Per-image metric containers
    abs_rel_list = []
    sq_rel_list = []
    rmse_list = []
    rmse_log_list = []
    delta1_list = []
    delta2_list = []
    delta3_list = []

    debug_count = 0
    if debug_dir is not None:
        os.makedirs(debug_dir, exist_ok=True)

    num_batches = len(dataloader)
    print(f"[EVAL] Number of batches: {num_batches}")

    for batch_idx, batch in enumerate(dataloader):
        print(f"[EVAL] Batch {batch_idx+1}/{num_batches}")

        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)  # (B, H, W) in mm
        image_paths = batch["image_paths"]

        outputs = model(pixel_values=pixel_values)
        pred = outputs.predicted_depth.clamp(0.0, max_depth_mm)

        B = labels.shape[0]
        for i in range(B):
            valid = (labels[i] > 0.0) & (labels[i] <= max_depth_mm)
            if not valid.any():
                continue

            pred_i = pred[i][valid]
            gt_i = labels[i][valid]

            m = compute_depth_metrics_per_pixels(
                pred_i, gt_i, max_depth_mm=max_depth_mm
            )
            n = m["n"].item()

            # Per-image metrics
            abs_rel = m["abs_rel"].item() / n
            sq_rel = m["sq_rel"].item() / n
            rmse = np.sqrt(m["rmse"].item() / n)
            rmse_log = np.sqrt(m["rmse_log"].item() / n)
            delta1 = m["delta1"].item() / n
            delta2 = m["delta2"].item() / n
            delta3 = m["delta3"].item() / n

            abs_rel_list.append(abs_rel)
            sq_rel_list.append(sq_rel)
            rmse_list.append(rmse)
            rmse_log_list.append(rmse_log)
            delta1_list.append(delta1)
            delta2_list.append(delta2)
            delta3_list.append(delta3)

            # Optional debug (plots + print)
            if debug_dir is not None and debug_count < max_debug_images:
                print(f"\n[DEBUG IMAGE {debug_count+1}] {image_paths[i]}")
                print(f"  Abs Rel:    {abs_rel:.4f}")
                print(f"  Sq Rel:     {sq_rel:.4f}  (mm)")
                print(f"  RMSE (mm):  {rmse:.4f}")
                print(f"  RMSE log:   {rmse_log:.4f}")
                print(f"  δ<1.25:     {delta1:.4f}")
                print(f"  δ<1.25^2:   {delta2:.4f}")
                print(f"  δ<1.25^3:   {delta3:.4f}")

                rgb = recover_rgb_for_plot(image_paths[i])
                save_path = os.path.join(debug_dir, f"debug_{debug_count+1}.png")
                save_debug_plot(
                    rgb=rgb,
                    gt_mm=labels[i],
                    pred_mm=pred[i],
                    save_path=save_path,
                    max_depth_mm=max_depth_mm,
                )

                print(f"  Debug plot saved to: {save_path}")
                debug_count += 1

    if len(abs_rel_list) == 0:
        print("[EVAL] No valid images found.")
        return None

    # Mean ± std over images
    metrics = {
        "abs_rel_mean": float(np.mean(abs_rel_list)),
        "abs_rel_std":  float(np.std(abs_rel_list)),

        "sq_rel_mean":  float(np.mean(sq_rel_list)),
        "sq_rel_std":   float(np.std(sq_rel_list)),

        "rmse_mean":    float(np.mean(rmse_list)),
        "rmse_std":     float(np.std(rmse_list)),

        "rmse_log_mean": float(np.mean(rmse_log_list)),
        "rmse_log_std":  float(np.std(rmse_log_list)),

        "delta1_mean":  float(np.mean(delta1_list)),
        "delta1_std":   float(np.std(delta1_list)),

        "delta2_mean":  float(np.mean(delta2_list)),
        "delta2_std":   float(np.std(delta2_list)),

        "delta3_mean":  float(np.mean(delta3_list)),
        "delta3_std":   float(np.std(delta3_list)),
    }

    return metrics



# -----------------------------
# Argument parsing & main
# -----------------------------
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--test_csv",
        type=str,
        required=True,
        help="CSV with columns: image_path, depth_path (EXR in [0,1] → 0-max_depth_mm mm)",
    )
    parser.add_argument(
        "--adapter_dir",
        type=str,
        required=True,
        help="Directory with the MERGED full model + image processor (e.g. ./experiments/.../best_merged)",
    )
    parser.add_argument(
        "--base_model_name",
        type=str,
        default="depth-anything/Depth-Anything-V2-Small-hf",
        help="Base model name from Hugging Face hub.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--max_depth_mm",
        type=float,
        default=200.0,
    )
    parser.add_argument(
        "--debug_dir",
        type=str,
        default="./test_debug_plots",
        help="Directory where debug plots (RGB/GT/Pred) will be saved.",
    )
    parser.add_argument(
        "--max_debug_images",
        type=int,
        default=3,
        help="Number of debug images (with metrics + plots) to save/print.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # 1) Load processor and merged full model
    # args.adapter_dir should point to something like: .../best_merged
    image_processor = AutoImageProcessor.from_pretrained(args.adapter_dir, use_fast=False)
    model = AutoModelForDepthEstimation.from_pretrained(args.adapter_dir)
    model.to(args.device)

    # 2) Build test dataset + dataloader
    test_dataset = DepthDatasetEXRMetric(
        args.test_csv,
        max_depth_mm=args.max_depth_mm,
    )
    print(f"[MAIN] Loaded test dataset with {len(test_dataset)} samples")

    if len(test_dataset) == 0:
        print("[MAIN] WARNING: test dataset is empty! Check your CSV paths and headers.")
        return

    collate_fn = make_collate_fn(image_processor)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=True,   # <--- shuffle to get random debug images
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # 3) Evaluate
    metrics = evaluate(
        model=model,
        dataloader=test_loader,
        device=args.device,
        max_depth_mm=args.max_depth_mm,
        debug_dir=args.debug_dir,
        max_debug_images=args.max_debug_images,
    )

    if metrics is None:
        print("No valid pixels found in the test set.")
        return

    print("\n=== TEST METRICS (per-image mean ± std) ===")
    print(f"Abs Rel:      {metrics['abs_rel_mean']:.4f} ± {metrics['abs_rel_std']:.4f}")
    print(f"Sq Rel (mm):  {metrics['sq_rel_mean']:.4f} ± {metrics['sq_rel_std']:.4f}")
    print(f"RMSE (mm):    {metrics['rmse_mean']:.4f} ± {metrics['rmse_std']:.4f}")
    print(f"RMSE log:     {metrics['rmse_log_mean']:.4f} ± {metrics['rmse_log_std']:.4f}")
    print(f"δ < 1.25:     {metrics['delta1_mean']:.4f} ± {metrics['delta1_std']:.4f}")
    print(f"δ < 1.25²:    {metrics['delta2_mean']:.4f} ± {metrics['delta2_std']:.4f}")
    print(f"δ < 1.25³:    {metrics['delta3_mean']:.4f} ± {metrics['delta3_std']:.4f}")
    print("==========================================\n")


if __name__ == "__main__":
    main()
