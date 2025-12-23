import os
import argparse
import math
import csv

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import numpy as np
import imageio.v3 as iio
import cv2

from transformers import (
    AutoImageProcessor,
    AutoModelForDepthEstimation,
    get_cosine_schedule_with_warmup,
)

from peft import LoraConfig, get_peft_model, PeftModel


# ====================================================
# Dataset (RGB image + EXR depth in mm)
# ====================================================
class DepthDatasetEXRMetric(Dataset):
    """
    CSV format:
        image_path,depth_path
        /path/to/img.png,/path/to/depth.exr

    depth_path: EXR file with float32 values in [0,1] representing 0-200 mm.
    We convert them to millimeters (0-200).
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
        depth_mm = depth_np * self.max_depth_mm  # [0, 200]

        depth_tensor = torch.from_numpy(depth_mm)  # (H, W)

        return {
            "image": image,
            "depth": depth_tensor,
        }


# ====================================================
# Collate function (labels in mm)
# ====================================================
def make_collate_fn(image_processor):
    """
    Uses AutoImageProcessor for rgb preprocessing and resizes depth maps
    to match the model input resolution.
    """

    def collate_fn(batch):
        images = [b["image"] for b in batch]
        depths = [b["depth"] for b in batch]  # already in mm

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
        }

    return collate_fn


# ====================================================
# Metric depth loss in mm (Huber + optional SI-log)
# ====================================================
class DepthMetricLoss(nn.Module):
    """
    Metric depth loss in millimeters.
    - Main term: Huber (Smooth L1) on valid pixels (0 < depth <= max_depth_mm), in mm.
    - Optional: scale-invariant log term with weight lambda_si.
    """

    def __init__(self, max_depth_mm=200.0, lambda_si=0.0, huber_delta=5.0, eps=1e-6):
        """
        huber_delta: transition point between L2 and L1 (in millimeters).
                     Smaller -> more robust to outliers, larger -> more like plain L2.
        """
        super().__init__()
        self.max_depth_mm = max_depth_mm
        self.lambda_si = lambda_si
        self.huber_delta = huber_delta
        self.eps = eps

    def forward(self, pred, target):
        # pred, target: (B, H, W) in mm
        valid = (target > 0.0) & (target <= self.max_depth_mm)

        if not valid.any():
            # No valid pixels at all (unlikely but safe-guard)
            return torch.tensor(0.0, device=pred.device)

        pred_valid = pred[valid]
        target_valid = target[valid]

        # -------------------------------
        # 1) Huber / Smooth L1 in mm
        # -------------------------------
        diff = pred_valid - target_valid
        abs_diff = torch.abs(diff)

        # Huber loss per element (in mm^2 inside)
        # 0.5 * d^2                         if |d| <= delta
        # delta * (|d| - 0.5 * delta)       otherwise
        delta = self.huber_delta
        quadratic = torch.clamp(abs_diff, max=delta)
        quadratic_loss = 0.5 * quadratic ** 2
        linear_loss = delta * (abs_diff - quadratic)

        huber = quadratic_loss + linear_loss  # (N,)
        huber = huber.mean()  # average over valid pixels

        loss = huber

        # -------------------------------
        # 2) Optional SI-log term (in meters for numerical stability)
        # -------------------------------
        if self.lambda_si > 0.0:
            pred_m = torch.clamp(pred_valid / 1000.0, min=self.eps)
            target_m = torch.clamp(target_valid / 1000.0, min=self.eps)

            log_diff = torch.log(pred_m) - torch.log(target_m)
            n = log_diff.numel()
            si = (log_diff ** 2).sum() / n - (log_diff.sum() ** 2) / (n ** 2)
            loss = loss + self.lambda_si * si

        return loss


# ====================================================
# LoRA configuration helper
# ====================================================
def add_lora_to_model(model, r=8, alpha=16, dropout=0.0):
    """
    LoRA on encoder attention + MLP for Depth Anything V2 Small.

    Target modules are matched by substring on the Linear layer name:
    - query / key / value
    - attention.output.dense
    - mlp.fc1 / mlp.fc2
    """

    target_modules = [
        "query",
        "key",
        "value",
        "dense",  # attention.output.dense
        "fc1",
        "fc2",
    ]

    lora_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=dropout,
        bias="none",
        # task_type can stay None for vision; PEFT will just wrap Linear layers
    )

    model = get_peft_model(model, lora_config)
    return model


def setup_trainable_params(model, unfreeze_depth_head=True, verbose=True):
    """
    Freeze all base parameters, then:
    - unfreeze all LoRA parameters (containing 'lora_')
    - optionally unfreeze the depth head parameters (heuristic on their names)

    Returns the number of trainable params vs total.
    """

    # 1) Freeze everything
    for _, p in model.named_parameters():
        p.requires_grad = False

    # 2) Unfreeze LoRA params
    for name, p in model.named_parameters():
        if "lora_" in name:
            p.requires_grad = True

    # 3) Unfreeze depth head (heuristics on parameter names)
    if unfreeze_depth_head:
        for name, p in model.named_parameters():
            lname = name.lower()
            # Heuristic: anything that looks like a "head" and is not part of the backbone/neck
            if (
                ("head" in lname or "depth" in lname or "pred" in lname)
                and "backbone" not in lname
                and "neck" not in lname
                and "lora_" not in lname  # LoRA already handled above
            ):
                p.requires_grad = True

    # 4) Debug: print trainable summary
    if verbose:
        total_params = 0
        trainable_params = 0
        print("\n=== TRAINABLE PARAMETERS (AFTER SETUP) ===")
        for name, p in model.named_parameters():
            n = p.numel()
            total_params += n
            if p.requires_grad:
                trainable_params += n
                print(f"{name:90s} {n:10d}")
        print(
            f"Total trainable params: {trainable_params:,} | "
            f"All params: {total_params:,} | "
            f"Trainable %: {100.0 * trainable_params / total_params:.4f}"
        )
        print("==========================================\n")

    return model


# ====================================================
# Visualization utilities
# ====================================================
def depth_to_colormap(depth_mm, max_depth_mm=200.0):
    """
    Convert a single depth map in millimeters to a RGB colormap using OpenCV.
    depth_mm: torch.Tensor (H, W)
    Returns: np.uint8 colormap (H, W, 3) in RGB order.
    """
    depth_np = depth_mm.detach().cpu().numpy()
    depth_norm = np.clip(depth_np / max_depth_mm, 0.0, 1.0)
    depth_uint8 = (depth_norm * 255).astype(np.uint8)
    depth_color = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_INFERNO)
    depth_color = cv2.cvtColor(depth_color, cv2.COLOR_BGR2RGB)
    return depth_color


def recover_rgb_from_pixel_values(tensor_3chw, image_processor):
    """
    Approximate inverse of the image_processor:
    - tensor_3chw: (3, H, W) preprocessed by image_processor
    - use processor.image_mean and image_std (if available) to denormalize
    Returns a uint8 RGB image (H, W, 3).
    """
    arr = tensor_3chw.detach().cpu().numpy()
    arr = np.transpose(arr, (1, 2, 0))  # (C,H,W) -> (H,W,C)

    mean = getattr(image_processor, "image_mean", None)
    std = getattr(image_processor, "image_std", None)

    if mean is not None and std is not None:
        mean = np.array(mean)
        std = np.array(std)
        arr = arr * std + mean

    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    return arr


@torch.no_grad()
def visualize_batch(
    rgb_batch,
    pred_batch,
    gt_batch,
    image_processor,
    save_dir,
    epoch,
    max_depth_mm=200.0,
    max_samples=4,
):
    """
    Save RGB, GT depth and predicted depth as colorized PNGs.
    Also creates a side-by-side panel: [RGB | GT | Pred].
    rgb_batch: tensor (B, 3, H, W) AFTER image_processor.
    pred_batch, gt_batch: tensors (B, H, W) in millimeters.
    """
    os.makedirs(save_dir, exist_ok=True)

    B = min(max_samples, rgb_batch.shape[0])

    for i in range(B):
        rgb = recover_rgb_from_pixel_values(rgb_batch[i], image_processor)
        pred = pred_batch[i]
        gt = gt_batch[i]

        pred_color = depth_to_colormap(pred, max_depth_mm)
        gt_color = depth_to_colormap(gt, max_depth_mm)

        panel = np.hstack([rgb, gt_color, pred_color])

        cv2.imwrite(
            os.path.join(save_dir, f"epoch{epoch:03d}_sample{i}_rgb.png"),
            cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
        )
        cv2.imwrite(
            os.path.join(save_dir, f"epoch{epoch:03d}_sample{i}_gt.png"),
            cv2.cvtColor(gt_color, cv2.COLOR_RGB2BGR),
        )
        cv2.imwrite(
            os.path.join(save_dir, f"epoch{epoch:03d}_sample{i}_pred.png"),
            cv2.cvtColor(pred_color, cv2.COLOR_RGB2BGR),
        )
        cv2.imwrite(
            os.path.join(save_dir, f"epoch{epoch:03d}_sample{i}_panel.png"),
            cv2.cvtColor(panel, cv2.COLOR_RGB2BGR),
        )


# ====================================================
# Validation loop (with metrics + optional visualization)
# ====================================================
@torch.no_grad()
def validate(
    model,
    dataloader,
    device,
    max_depth_mm=200.0,
    image_processor=None,
    output_dir=None,
    epoch=None,
    visualize=True,
    max_vis_samples=4,
):
    model.eval()
    total_l1 = 0.0
    total_rmse = 0.0
    total_pixels = 0

    did_visualize = False

    for step, batch in enumerate(dataloader):
        # Keep a CPU copy of pixel_values for visualization before .to(device)
        pixel_values_cpu = batch["pixel_values"]
        labels_cpu = batch["labels"]

        pixel_values = pixel_values_cpu.to(device, non_blocking=True)
        labels = labels_cpu.to(device, non_blocking=True)  # mm

        outputs = model(pixel_values=pixel_values)
        pred = outputs.predicted_depth  # (B, H, W)
        pred = pred.clamp(0.0, max_depth_mm)

        valid = (labels > 0.0) & (labels <= max_depth_mm)
        if not valid.any():
            continue

        pred_valid = pred[valid]
        target_valid = labels[valid]
        diff = pred_valid - target_valid

        l1 = torch.abs(diff).sum()
        rmse = torch.sqrt((diff ** 2).sum())
        n = valid.sum().item()

        total_l1 += l1.item()
        total_rmse += rmse.item()
        total_pixels += n

        # One-shot visualization on the first batch of validation for this epoch
        if (
            visualize
            and (not did_visualize)
            and image_processor is not None
            and output_dir is not None
            and epoch is not None
        ):
            vis_dir = os.path.join(output_dir, "visualizations")
            visualize_batch(
                rgb_batch=pixel_values_cpu,
                pred_batch=pred.cpu(),
                gt_batch=labels_cpu,
                image_processor=image_processor,
                save_dir=vis_dir,
                epoch=epoch,
                max_depth_mm=max_depth_mm,
                max_samples=max_vis_samples,
            )
            did_visualize = True

    if total_pixels == 0:
        # No valid pixels over the whole validation set
        return float("nan"), float("nan")

    mean_l1 = total_l1 / total_pixels         # mean absolute error (mm)
    mean_rmse = total_rmse / total_pixels     # RMSE per-pixel (mm)

    return mean_l1, mean_rmse


# ====================================================
# Training loop (with validation)
# ====================================================
def train(
    model,
    image_processor,
    train_dataset,
    output_dir,
    batch_size=4,
    lr=1e-4,
    num_epochs=10,
    warmup_ratio=0.05,
    gradient_accumulation_steps=1,
    num_workers=4,
    device="cuda",
    max_depth_mm=200.0,
    val_dataset=None,
    base_model_name=None,
):

    collate_fn = make_collate_fn(image_processor)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )

    model.to(device)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=1e-2,
    )

    num_update_steps_per_epoch = math.ceil(len(train_loader) / gradient_accumulation_steps)
    max_train_steps = num_epochs * num_update_steps_per_epoch
    num_warmup_steps = int(warmup_ratio * max_train_steps)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=max_train_steps,
    )

    # Huber-based metric loss in mm
    criterion = DepthMetricLoss(
        max_depth_mm=max_depth_mm,
        lambda_si=0.0,
        huber_delta=5.0,  # 5 mm transition between L2 and L1 regime
    )

    global_step = 0
    model.train()

    os.makedirs(output_dir, exist_ok=True)
    best_val_l1 = float("inf")
    best_ckpt_path = None

    for epoch in range(num_epochs):
        running_loss = 0.0

        for step, batch in enumerate(train_loader):
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)  # mm

            outputs = model(pixel_values=pixel_values)
            pred_depth = outputs.predicted_depth  # (B, H, W)
            pred_depth = pred_depth.clamp(0.0, max_depth_mm)

            loss = criterion(pred_depth, labels)
            loss = loss / gradient_accumulation_steps
            loss.backward()

            if (step + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            running_loss += loss.item() * gradient_accumulation_steps

            if (step + 1) % 50 == 0:
                avg_loss = running_loss / 50
                print(
                    f"[Train] Epoch {epoch+1}/{num_epochs} | "
                    f"Step {step+1}/{len(train_loader)} | "
                    f"Loss {avg_loss:.4f}"
                )
                running_loss = 0.0

        # Validation
        if val_loader is not None:
            val_l1, val_rmse = validate(
                model=model,
                dataloader=val_loader,
                device=device,
                max_depth_mm=max_depth_mm,
                image_processor=image_processor,
                output_dir=output_dir,
                epoch=epoch + 1,
                visualize=True,
                max_vis_samples=4,
            )
            print(
                f"[Val] Epoch {epoch+1}/{num_epochs} | "
                f"L1 (mm) = {val_l1:.4f} | RMSE (mm) = {val_rmse:.4f}"
            )

            # Save best checkpoint according to L1
            if val_l1 < best_val_l1:
                best_val_l1 = val_l1
                best_ckpt_path = os.path.join(output_dir, "best")
                os.makedirs(best_ckpt_path, exist_ok=True)
                model.save_pretrained(best_ckpt_path)
                image_processor.save_pretrained(best_ckpt_path)
                print(
                    f"New best checkpoint (L1={best_val_l1:.4f}) saved at {best_ckpt_path}"
                )

        # Per-epoch checkpoint (regardless of validation)
        ckpt_path = os.path.join(output_dir, f"epoch-{epoch+1}")
        os.makedirs(ckpt_path, exist_ok=True)
        model.save_pretrained(ckpt_path)
        image_processor.save_pretrained(ckpt_path)
        print(f"Epoch checkpoint saved at {ckpt_path}")

        model.train()  # back to train mode after validation


    if best_ckpt_path is not None:
        print(f"Training finished. Best checkpoint at (LoRA only): {best_ckpt_path}")
    else:
        print("Training finished. No validation set provided, use epoch-* checkpoints.")

        # ---- merge current trained model and save a full checkpoint ----
    if base_model_name is not None:
        print("[MERGE] Merging current trained model (LoRA + head) into a full checkpoint...")
        import copy

        # Deep copy to avoid destroying 'model' in case you want to use it later
        model_to_merge = copy.deepcopy(model)

        # This merges LoRA into the base model and removes adapters.
        merged_model = model_to_merge.merge_and_unload()

        merged_dir = os.path.join(output_dir, "final_merged")
        os.makedirs(merged_dir, exist_ok=True)
        merged_model.save_pretrained(merged_dir)
        image_processor.save_pretrained(merged_dir)
        print(f"[MERGE] Merged full model saved to: {merged_dir}")
    else:
        print("[MERGE] base_model_name is None, skipped merging.")


# ====================================================
# Argument parsing & main
# ====================================================
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train_csv",
        type=str,
        required=True,
        help="CSV with columns: image_path, depth_path (EXR in [0,1] → 0-200 mm)",
    )
    parser.add_argument(
        "--val_csv",
        type=str,
        default=None,
        help="Validation CSV (same format as train). If omitted, no validation.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="depth-anything/Depth-Anything-V2-Small-hf",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./lora_depthanything_v2_exr_metric",
    )
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--max_depth_mm", type=float, default=200.0)

    return parser.parse_args()


def main():
    args = parse_args()

    # 1) Load model + image processor
    image_processor = AutoImageProcessor.from_pretrained(args.model_name)
    model = AutoModelForDepthEstimation.from_pretrained(args.model_name)

    # 2) Add LoRA adapters
    model = add_lora_to_model(
        model,
        r=args.lora_r,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
    )

    # 3) Setup trainable params: LoRA + depth head
    model = setup_trainable_params(
        model,
        unfreeze_depth_head=True,
        verbose=True,
    )

    # 4) Build train and (optional) validation datasets
    train_dataset = DepthDatasetEXRMetric(
        args.train_csv,
        max_depth_mm=args.max_depth_mm,
    )

    val_dataset = None
    if args.val_csv is not None:
        val_dataset = DepthDatasetEXRMetric(
            args.val_csv,
            max_depth_mm=args.max_depth_mm,
        )

    # 5) Train + validate
    train(
        model=model,
        image_processor=image_processor,
        train_dataset=train_dataset,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        lr=args.lr,
        num_epochs=args.epochs,
        warmup_ratio=args.warmup_ratio,
        gradient_accumulation_steps=args.grad_accum,
        num_workers=args.num_workers,
        device=args.device,
        max_depth_mm=args.max_depth_mm,
        val_dataset=val_dataset,
        base_model_name=args.model_name,
    )



if __name__ == "__main__":
    main()
