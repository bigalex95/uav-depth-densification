#!/usr/bin/env python3
"""
Convert COLMAP TXT output to JSON format.

This script reads COLMAP's TXT output (cameras.txt, images.txt, points3D.txt)
and converts them to a structured JSON format with poses {R, t, K} and sparse points.
"""

import json
import argparse
from pathlib import Path

import numpy as np


def parse_cameras_txt(filepath: Path) -> dict:
    """Parse COLMAP cameras.txt file."""
    cameras = {}
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            camera_id = int(parts[0])
            model = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            params = [float(p) for p in parts[4:]]

            # Build intrinsic matrix K
            if model == "SIMPLE_PINHOLE":
                # params: f, cx, cy
                f, cx, cy = params[:3]
                fx = fy = f
            elif model in (
                "PINHOLE",
                "OPENCV",
                "OPENCV_FISHEYE",
                "FULL_OPENCV",
                "THIN_PRISM_FISHEYE",
            ):
                # params: fx, fy, cx, cy [, distortion...]
                fx, fy, cx, cy = params[:4]
            elif model == "SIMPLE_RADIAL":
                # params: f, cx, cy, k
                f, cx, cy = params[:3]
                fx = fy = f
            else:
                # Generic fallback: first param is focal length
                fx = fy = params[0] if params else 1.0
                cx = width / 2
                cy = height / 2

            K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]

            cameras[camera_id] = {
                "model": model,
                "width": width,
                "height": height,
                "params": params,
                "K": K,
            }
    return cameras


def parse_images_txt(filepath: Path) -> dict:
    """Parse COLMAP images.txt file."""
    images = {}
    with open(filepath, "r") as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]

    # Images.txt has pairs of lines: image info, then 2D points
    i = 0
    while i < len(lines):
        parts = lines[i].split()
        image_id = int(parts[0])
        qw, qx, qy, qz = map(float, parts[1:5])
        tx, ty, tz = map(float, parts[5:8])
        camera_id = int(parts[8])
        name = parts[9]

        # Convert quaternion to rotation matrix
        R = quat_to_rotation_matrix(qw, qx, qy, qz)
        t = [tx, ty, tz]

        images[image_id] = {
            "name": name,
            "camera_id": camera_id,
            "R": R,
            "t": t,
            "qvec": [qw, qx, qy, qz],
        }

        i += 2  # Skip the 2D points line

    return images


def parse_points3d_txt(filepath: Path) -> list:
    """Parse COLMAP points3D.txt file."""
    points = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            point_id = int(parts[0])
            x, y, z = map(float, parts[1:4])
            r, g, b = map(int, parts[4:7])
            error = float(parts[7])

            points.append(
                {
                    "id": point_id,
                    "xyz": [x, y, z],
                    "rgb": [r, g, b],
                    "error": error,
                }
            )
    return points


def quat_to_rotation_matrix(qw: float, qx: float, qy: float, qz: float) -> list:
    """Convert quaternion to 3x3 rotation matrix."""
    # Normalize quaternion
    norm = np.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    qw, qx, qy, qz = qw / norm, qx / norm, qy / norm, qz / norm

    R = np.array(
        [
            [
                1 - 2 * qy * qy - 2 * qz * qz,
                2 * qx * qy - 2 * qz * qw,
                2 * qx * qz + 2 * qy * qw,
            ],
            [
                2 * qx * qy + 2 * qz * qw,
                1 - 2 * qx * qx - 2 * qz * qz,
                2 * qy * qz - 2 * qx * qw,
            ],
            [
                2 * qx * qz - 2 * qy * qw,
                2 * qy * qz + 2 * qx * qw,
                1 - 2 * qx * qx - 2 * qy * qy,
            ],
        ]
    )
    return R.tolist()


def convert_colmap_to_json(sfm_dir: Path, output_path: Path) -> None:
    """Convert COLMAP TXT output to JSON."""
    cameras_file = sfm_dir / "cameras.txt"
    images_file = sfm_dir / "images.txt"
    points_file = sfm_dir / "points3D.txt"

    # Check files exist
    for f in [cameras_file, images_file, points_file]:
        if not f.exists():
            raise FileNotFoundError(f"Required file not found: {f}")

    print(f"Parsing cameras from: {cameras_file}")
    cameras = parse_cameras_txt(cameras_file)

    print(f"Parsing images from: {images_file}")
    images = parse_images_txt(images_file)

    print(f"Parsing 3D points from: {points_file}")
    points = parse_points3d_txt(points_file)

    # Build output structure
    output = {
        "cameras": cameras,
        "images": {
            img_id: {
                "name": img["name"],
                "camera_id": img["camera_id"],
                "R": img["R"],
                "t": img["t"],
                "K": cameras[img["camera_id"]]["K"],
            }
            for img_id, img in images.items()
        },
        "points3D": points,
        "num_cameras": len(cameras),
        "num_images": len(images),
        "num_points": len(points),
    }

    # Write JSON
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nConversion complete!")
    print(f"Output: {output_path}")
    print(f"  - Cameras: {len(cameras)}")
    print(f"  - Images: {len(images)}")
    print(f"  - 3D Points: {len(points)}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert COLMAP TXT output to JSON format"
    )
    parser.add_argument(
        "--sfm-dir",
        type=Path,
        default=Path("sfm_output"),
        help="Directory containing COLMAP TXT files (default: sfm_output)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("sfm_output/reconstruction.json"),
        help="Output JSON file path (default: sfm_output/reconstruction.json)",
    )
    args = parser.parse_args()

    convert_colmap_to_json(args.sfm_dir, args.output)


if __name__ == "__main__":
    main()
