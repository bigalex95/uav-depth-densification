#!/usr/bin/env python3
"""
Monocular depth estimation using Depth Anything V2.

This script processes all images under --data-dir and writes the raw
predicted (relative) depth as 32-bit float TIFFs, mirroring the input
directory layout. Depth Anything V2 predictions are relative inverse depth (affine-invariant disparity),
not metric; scale is recovered later by the C++ densifier.
"""

import argparse
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image
from tqdm import tqdm

import torch
from transformers import pipeline


def find_images(data_dir: Path) -> list[Path]:
    """Find all image files in the data directory."""
    image_extensions = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
    images = []

    for ext in image_extensions:
        images.extend(data_dir.rglob(f"*{ext}"))
        images.extend(data_dir.rglob(f"*{ext.upper()}"))

    return sorted(set(images))


def estimate_depth(image_path: Path, pipe) -> np.ndarray:
    """
    Estimate depth for a single image using Depth Anything V2.

    Args:
        image_path: Path to input image
        pipe: HuggingFace depth estimation pipeline

    Returns:
        Depth map as numpy array (H, W) with float32 values (raw predicted depth)
    """
    # Load RGB image
    image = Image.open(image_path).convert("RGB")

    # Run depth estimation
    result = pipe(image)

    # Extract RAW predicted depth tensor (not normalized 'depth' PIL image)
    # The pipeline returns:
    # - 'predicted_depth': raw tensor with inverse depth predictions
    # - 'depth': normalized PIL image for visualization (we don't want this)
    predicted_depth = result["predicted_depth"]

    # Convert to numpy and ensure float32
    depth_array = predicted_depth.cpu().numpy().squeeze()

    return depth_array.astype(np.float32)


def save_depth_tiff(depth: np.ndarray, output_path: Path) -> None:
    """Save depth map as 32-bit TIFF."""
    tifffile.imwrite(output_path, depth.astype(np.float32))


def main():
    parser = argparse.ArgumentParser(
        description="Run monocular depth estimation on dataset images"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing input images (default: data)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("depth_maps"),
        help="Directory for output depth maps (default: depth_maps)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="depth-anything/Depth-Anything-V2-Large-hf",
        help="Hugging Face model ID for depth estimation",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda/cpu)",
    )
    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Find all images
    print(f"Searching for images in: {args.data_dir}")
    images = find_images(args.data_dir)
    print(f"Found {len(images)} images")

    if not images:
        print("No images found. Check the data directory.")
        return

    # Initialize Depth Anything V2 pipeline
    device = 0 if args.device == "cuda" and torch.cuda.is_available() else -1
    if args.device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available. Using CPU.")
        args.device = "cpu"

    print(f"Initializing Depth Anything V2 model on {args.device}...")
    pipe = pipeline(task="depth-estimation", model=args.model, device=device)
    print(f"Model loaded successfully!")

    # Test on first image to verify output format
    if images:
        print(f"Testing model on first image: {images[0].name}")
        test_depth = estimate_depth(images[0], pipe)
        print(f"  Shape: {test_depth.shape}, dtype: {test_depth.dtype}")
        print(f"  Range: [{test_depth.min():.4f}, {test_depth.max():.4f}]")

    # Process each image
    print(f"\nGenerating depth maps...")
    for idx, image_path in enumerate(tqdm(images, desc="Processing")):
        # Generate output path
        relative_path = image_path.relative_to(args.data_dir)
        output_path = args.output_dir / relative_path.with_suffix(".tif")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Skip if already processed
        if output_path.exists():
            continue

        # Estimate depth with error handling
        try:
            depth = estimate_depth(image_path, pipe)

            # Save as 32-bit TIFF
            save_depth_tiff(depth, output_path)

            # Clear CUDA cache every 10 images to prevent OOM
            if args.device == "cuda" and (idx + 1) % 10 == 0:
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"\nError processing {image_path.name}: {e}")
            continue

    print(f"\nDepth estimation complete!")
    print(f"Output directory: {args.output_dir}")
    print(f"Depth maps saved as 32-bit TIFFs")


if __name__ == "__main__":
    main()
