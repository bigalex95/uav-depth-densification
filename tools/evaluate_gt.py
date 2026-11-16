#!/usr/bin/env python3
"""
Evaluate densified depth maps against ETH3D ground truth depth.

ETH3D ground truth depth files are raw float32 binaries (inf = invalid,
finite positive values = depth); see load_gt_depth().

This script computes per-image and aggregate metrics:
- MAE (Mean Absolute Error)
- RMSE (Root Mean Square Error)
- AbsRel (Absolute Relative Error)
- δ < 1.25 accuracy (percentage of pixels within 25% of GT)
"""

import argparse
import json
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image
from tqdm import tqdm

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def load_gt_depth(gt_path: Path, width: int = 752, height: int = 480) -> np.ndarray:
    """
    Load ETH3D ground truth depth map.

    ETH3D stores depth as raw float32 binary data:
    - inf = invalid (no depth measurement)
    - Finite positive values = metric depth

    Returns float32 array with inf replaced by 0.
    """
    raw = np.fromfile(str(gt_path), dtype=np.float32)
    total_pixels = raw.shape[0]

    # Try to figure out dimensions
    if total_pixels == width * height:
        depth = raw.reshape(height, width)
    else:
        # Try common ETH3D resolutions
        for h, w in [(480, 752), (465, 732), (512, 938)]:
            if total_pixels == h * w:
                depth = raw.reshape(h, w)
                break
        else:
            # Fallback: try sqrt-ish
            side = int(np.sqrt(total_pixels))
            depth = raw.reshape(-1)
            return depth

    return depth


def load_predicted_depth(pred_path: Path) -> np.ndarray:
    """Load predicted (densified) depth map from TIFF."""
    return tifffile.imread(pred_path).astype(np.float32)


def compute_metrics(pred: np.ndarray, gt: np.ndarray, valid_mask: np.ndarray) -> dict:
    """
    Compute depth estimation metrics on valid pixels.

    Args:
        pred: Predicted depth (H, W)
        gt: Ground truth depth (H, W)
        valid_mask: Boolean mask of valid GT pixels

    Returns:
        Dictionary with metrics
    """
    pred_valid = pred[valid_mask]
    gt_valid = gt[valid_mask]

    if len(pred_valid) == 0:
        return {"valid_pixels": 0, "error": "No valid pixels"}

    # Absolute errors
    abs_err = np.abs(pred_valid - gt_valid)

    # Metrics
    mae = float(np.mean(abs_err))
    rmse = float(np.sqrt(np.mean(abs_err ** 2)))

    # Relative errors (avoid division by zero)
    safe_gt = np.maximum(gt_valid, 1e-6)
    abs_rel = float(np.mean(abs_err / safe_gt))

    # δ < threshold accuracy
    ratio = np.maximum(pred_valid / safe_gt, safe_gt / pred_valid)
    delta_1_25 = float(np.mean(ratio < 1.25) * 100)
    delta_1_25_sq = float(np.mean(ratio < 1.25 ** 2) * 100)
    delta_1_25_cu = float(np.mean(ratio < 1.25 ** 3) * 100)

    return {
        "valid_pixels": int(np.sum(valid_mask)),
        "mae": mae,
        "rmse": rmse,
        "abs_rel": abs_rel,
        "delta_1.25": delta_1_25,
        "delta_1.25^2": delta_1_25_sq,
        "delta_1.25^3": delta_1_25_cu,
        "pred_min": float(pred_valid.min()),
        "pred_max": float(pred_valid.max()),
        "gt_min": float(gt_valid.min()),
        "gt_max": float(gt_valid.max()),
    }


def align_scale_shift(pred: np.ndarray, gt: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """
    Align predicted depth to GT using least-squares scale + shift.

    Solves: aligned = scale * pred + shift to minimize ||aligned - gt||^2
    on valid pixels.

    Returns aligned prediction.
    """
    pred_valid = pred[valid_mask].flatten()
    gt_valid = gt[valid_mask].flatten()

    if len(pred_valid) < 2:
        return pred.copy()

    # Solve [pred, 1] @ [scale, shift]^T = gt
    A = np.column_stack([pred_valid, np.ones_like(pred_valid)])
    result = np.linalg.lstsq(A, gt_valid, rcond=None)
    scale, shift = result[0]

    aligned = pred * scale + shift
    return aligned


def save_comparison_figure(
    image_name: str,
    pred: np.ndarray,
    gt: np.ndarray,
    valid_mask: np.ndarray,
    metrics: dict,
    save_path: Path,
) -> None:
    """Save side-by-side comparison figure."""
    if not HAS_MPL:
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Predicted depth
    im0 = axes[0].imshow(pred, cmap="turbo")
    axes[0].set_title(f"Predicted Depth\n[{pred.min():.1f}, {pred.max():.1f}]")
    axes[0].axis("off")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # Ground truth depth
    gt_vis = gt.copy()
    gt_vis[~valid_mask] = 0
    im1 = axes[1].imshow(gt_vis, cmap="turbo")
    axes[1].set_title(f"Ground Truth\n[{gt[valid_mask].min():.1f}, {gt[valid_mask].max():.1f}]")
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # Error heatmap
    error = np.abs(pred - gt)
    error[~valid_mask] = 0
    im2 = axes[2].imshow(error, cmap="hot")
    axes[2].set_title(
        f"Abs Error\nMAE={metrics['mae']:.3f} RMSE={metrics['rmse']:.3f}"
    )
    axes[2].axis("off")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    fig.suptitle(image_name, fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate densified depth maps against ETH3D ground truth"
    )
    parser.add_argument(
        "--densified-dir",
        type=Path,
        required=True,
        help="Directory with densified depth maps (.tif)",
    )
    parser.add_argument(
        "--gt-dir",
        type=Path,
        required=True,
        help="Directory with ground truth depth maps (.png)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("evaluation"),
        help="Output directory for comparison figures",
    )
    parser.add_argument(
        "--recon",
        type=Path,
        default=Path("sfm_output/reconstruction.json"),
        help="Reconstruction JSON (for image names)",
    )
    parser.add_argument(
        "--align",
        action="store_true",
        help="Scale-shift align predictions to GT before computing metrics",
    )
    parser.add_argument(
        "--gt-scale",
        type=float,
        default=1.0,
        help="Scale factor for GT depth (e.g., 0.001 if GT is in mm)",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="Max images to evaluate (0 = all)",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "comparisons").mkdir(exist_ok=True)

    # Load reconstruction for image names
    print(f"Loading reconstruction from {args.recon}...")
    with open(args.recon, "r") as f:
        recon = json.load(f)

    image_names = sorted([d["name"] for d in recon["images"].values()])
    print(f"Found {len(image_names)} images in reconstruction")

    if args.max_images > 0:
        image_names = image_names[:args.max_images]
        print(f"Limiting to first {args.max_images} images")

    results = []
    processed = 0
    skipped = 0

    for image_name in tqdm(image_names, desc="Evaluating"):
        # Build paths
        depth_name = str(Path(image_name).with_suffix(".tif"))
        pred_path = args.densified_dir / depth_name

        # GT path: image name maps from undistorted to raw cam dirs
        # e.g., "images_rig_cam4_undistorted/xxx.png" -> GT at "images_rig_cam4/xxx.png"
        gt_image_name = image_name.replace("_undistorted", "")
        gt_path = args.gt_dir / gt_image_name

        if not pred_path.exists():
            skipped += 1
            continue
        if not gt_path.exists():
            skipped += 1
            continue

        # Load data
        pred = load_predicted_depth(pred_path)
        gt = load_gt_depth(gt_path) * args.gt_scale

        # Resize pred to GT size if needed
        if pred.shape != gt.shape:
            from PIL import Image as PILImage
            pred_img = PILImage.fromarray(pred)
            pred_img = pred_img.resize((gt.shape[1], gt.shape[0]), PILImage.BILINEAR)
            pred = np.array(pred_img)

        # Valid mask (finite GT values are valid, inf means no depth)
        valid_mask = np.isfinite(gt) & (gt > 0)

        if valid_mask.sum() == 0:
            skipped += 1
            continue

        # Optionally align scale + shift
        if args.align:
            pred = align_scale_shift(pred, gt, valid_mask)

        # Compute metrics
        metrics = compute_metrics(pred, gt, valid_mask)
        metrics["image"] = image_name
        results.append(metrics)

        # Save comparison figure
        safe_name = image_name.replace("/", "_").rsplit(".", 1)[0] + ".png"
        save_path = args.output_dir / "comparisons" / safe_name
        save_comparison_figure(image_name, pred, gt, valid_mask, metrics, save_path)

        processed += 1

    # Aggregate statistics
    print(f"\n{'=' * 60}")
    print(f"EVALUATION SUMMARY")
    print(f"{'=' * 60}")
    print(f"Processed: {processed}, Skipped: {skipped}")

    if results:
        valid_results = [r for r in results if "mae" in r]
        if valid_results:
            avg_mae = np.mean([r["mae"] for r in valid_results])
            avg_rmse = np.mean([r["rmse"] for r in valid_results])
            avg_abs_rel = np.mean([r["abs_rel"] for r in valid_results])
            avg_d125 = np.mean([r["delta_1.25"] for r in valid_results])
            avg_d125_sq = np.mean([r["delta_1.25^2"] for r in valid_results])
            avg_d125_cu = np.mean([r["delta_1.25^3"] for r in valid_results])

            print(f"\nAggregate Metrics ({len(valid_results)} images):")
            print(f"  MAE:          {avg_mae:.4f}")
            print(f"  RMSE:         {avg_rmse:.4f}")
            print(f"  AbsRel:       {avg_abs_rel:.4f}")
            print(f"  δ < 1.25:     {avg_d125:.2f}%")
            print(f"  δ < 1.25²:    {avg_d125_sq:.2f}%")
            print(f"  δ < 1.25³:    {avg_d125_cu:.2f}%")

    # Save results JSON
    summary_path = args.output_dir / "evaluation_results.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {summary_path}")
    print(f"Comparisons saved to {args.output_dir}/comparisons/")


if __name__ == "__main__":
    main()
