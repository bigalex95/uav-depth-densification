#!/usr/bin/env python3
"""
Convert ETH3D ground truth depth maps (raw float32 binary) into viewable images.
This script can convert a single file or a whole directory of images.
"""
import argparse
import os
from pathlib import Path
import numpy as np

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("Warning: matplotlib not installed. Please install it to generate colormapped images.")

def load_gt_depth(gt_path: Path, width: int = 752, height: int = 480) -> np.ndarray:
    """Load ETH3D ground truth depth map from raw float32 binary."""
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
            print(f"Warning: Could not determine dimensions for {gt_path}. Total pixels: {total_pixels}")
            return None
    return depth

def convert_to_visual(input_path: Path, output_path: Path):
    """Convert a single raw float32 depth map to a visual colormap image."""
    depth = load_gt_depth(input_path)
    if depth is None:
        return False
    
    # Replace inf and invalid depth with 0 for visualization
    valid_mask = np.isfinite(depth) & (depth > 0)
    
    if valid_mask.sum() == 0:
        print(f"Warning: No valid depth values found in {input_path}")
        return False
        
    depth_vis = depth.copy()
    depth_vis[~valid_mask] = 0
    
    if HAS_MPL:
        # Save as an image using matplotlib colormap
        # Invalid pixels (0) will appear at the bottom of the colormap (dark blue in turbo)
        plt.imsave(str(output_path), depth_vis, cmap="turbo")
    else:
        # Fallback to saving normalized grayscale image using PIL
        from PIL import Image
        depth_vis_normalized = (depth_vis / depth_vis.max() * 255).astype(np.uint8)
        img = Image.fromarray(depth_vis_normalized, mode='L')
        img.save(str(output_path))
        
    return True

def main():
    parser = argparse.ArgumentParser(description="Convert raw ETH3D GT depth files to viewable images")
    parser.add_argument("-i", "--input", type=Path, required=True, 
                        help="Path to a specific .png file or a directory containing them")
    parser.add_argument("-o", "--output", type=Path, 
                        help="Path to save the output file or directory")
    
    args = parser.parse_args()
    
    if not args.input.exists():
        print(f"Error: Input path '{args.input}' does not exist.")
        return
        
    if args.input.is_file():
        # Convert a single file
        output_path = args.output
        if output_path is None:
            output_path = args.input.parent / f"{args.input.stem}_vis.png"
        elif output_path.is_dir():
            output_path = output_path / f"{args.input.stem}_vis.png"
            
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if convert_to_visual(args.input, output_path):
            print(f"Successfully converted image to: {output_path}")
        
    elif args.input.is_dir():
        # Convert all .png files in the folder
        output_dir = args.output
        if output_dir is None:
            output_dir = args.input.parent / f"{args.input.name}_vis"
            
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all .png files that are likely GT files
        files = list(args.input.glob("*.png"))
        if not files:
            print(f"No .png files found in directory '{args.input}'")
            return
            
        print(f"Found {len(files)} files to convert. Saving to '{output_dir}'...")
        success_count = 0
        for f in files:
            out_file = output_dir / f"{f.stem}_vis.png"
            if convert_to_visual(f, out_file):
                success_count += 1
                
        print(f"Successfully converted {success_count}/{len(files)} files.")

if __name__ == "__main__":
    main()
