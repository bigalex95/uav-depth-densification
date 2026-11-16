#!/usr/bin/env python3
"""
Visualization suite for SfM and depth estimation results.

Provides three visualization modes:
- pointcloud: Interactive 3D viewer with sparse points and camera frustums
- depth: Side-by-side image and depth map visualization
- overlay: Project 3D sparse points onto 2D images
"""

import argparse
import json
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image

import matplotlib

# Use Agg backend by default (non-interactive, works everywhere)
# For interactive display, users can set MPLBACKEND environment variable
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import open3d as o3d


# =============================================================================
# Data Loaders
# =============================================================================


def load_reconstruction(json_path: Path) -> dict:
    """
    Load SfM reconstruction from JSON file.

    Args:
        json_path: Path to reconstruction.json

    Returns:
        Dictionary with cameras, images, points3D, and counts
    """
    if not json_path.exists():
        raise FileNotFoundError(f"Reconstruction file not found: {json_path}")

    with open(json_path, "r") as f:
        return json.load(f)


def load_depth_map(depth_path: Path) -> np.ndarray:
    """
    Load depth map from 32-bit TIFF file.

    Args:
        depth_path: Path to .tif depth map

    Returns:
        np.ndarray of shape (H, W) with float32 depth values
    """
    if not depth_path.exists():
        raise FileNotFoundError(f"Depth map not found: {depth_path}")

    return tifffile.imread(depth_path).astype(np.float32)


def load_image(image_path: Path) -> np.ndarray:
    """
    Load RGB image.

    Args:
        image_path: Path to image file

    Returns:
        np.ndarray of shape (H, W, 3) with uint8 RGB values
    """
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    return np.array(Image.open(image_path).convert("RGB"))


# =============================================================================
# Geometry Utilities
# =============================================================================


def compute_camera_center(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Compute camera center in world coordinates.

    COLMAP convention: world point X projects as x = K @ (R @ X + t)
    Camera center C in world coords: C = -R^T @ t

    Args:
        R: 3x3 rotation matrix
        t: 3x1 translation vector

    Returns:
        3D camera center position
    """
    return -R.T @ t


def project_points_to_image(
    points_3d: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    K: np.ndarray,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Project 3D points onto 2D image plane.

    Projection: p_2d = K @ (R @ p_3d + t)

    Args:
        points_3d: (N, 3) array of 3D points
        R: 3x3 rotation matrix
        t: 3x1 translation vector
        K: 3x3 camera intrinsic matrix
        width: Image width for bounds checking
        height: Image height for bounds checking

    Returns:
        Tuple of:
        - valid_points_2d: (M, 2) array of projected (u, v) coordinates
        - valid_indices: (M,) indices into original points
    """
    # Transform to camera coords: X_cam = R @ X_world + t
    X_cam = (R @ points_3d.T).T + t  # (N, 3)

    # Filter points behind camera
    valid_z = X_cam[:, 2] > 0.1

    # Project: x = K @ X_cam
    X_proj = (K @ X_cam.T).T  # (N, 3)

    # Normalize: (u, v) = (x/z, y/z)
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = X_proj[:, :2] / X_proj[:, 2:3]

    # Filter points within image bounds
    valid_bounds = (
        (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
    )

    valid_mask = valid_z & valid_bounds

    return uv[valid_mask], np.where(valid_mask)[0]


def create_camera_frustum(
    R: np.ndarray,
    t: np.ndarray,
    K: np.ndarray,
    width: int,
    height: int,
    scale: float = 1.0,
    color: list[float] | None = None,
) -> o3d.geometry.LineSet:
    """
    Create Open3D LineSet representing camera frustum.

    Args:
        R: 3x3 rotation matrix
        t: 3x1 translation vector
        K: 3x3 intrinsic matrix
        width: Image width
        height: Image height
        scale: Frustum depth (size of visualization)
        color: RGB color for frustum lines

    Returns:
        Open3D LineSet with pyramid frustum
    """
    if color is None:
        color = [1.0, 0.0, 0.0]

    # Camera center in world coords
    C = compute_camera_center(R, t)

    # Image corners in pixel coordinates
    corners_img = np.array(
        [[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float64
    )

    # Unproject to camera coords at depth=scale
    K_inv = np.linalg.inv(K)
    corners_cam = []
    for corner in corners_img:
        p_norm = K_inv @ np.array([corner[0], corner[1], 1.0])
        p_cam = p_norm * scale
        corners_cam.append(p_cam)
    corners_cam = np.array(corners_cam)

    # Transform to world coords: X_world = R^T @ (X_cam - t)
    corners_world = (R.T @ (corners_cam.T - t.reshape(3, 1))).T

    # Build LineSet: center + 4 corners
    points = np.vstack([C, corners_world])  # (5, 3)

    # Lines: center to each corner + rectangle edges
    lines = [
        [0, 1],
        [0, 2],
        [0, 3],
        [0, 4],  # Center to corners
        [1, 2],
        [2, 3],
        [3, 4],
        [4, 1],  # Rectangle edges
    ]

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector([color] * len(lines))

    return line_set


# =============================================================================
# Visualization Functions
# =============================================================================


def visualize_pointcloud(
    recon: dict,
    show_cameras: bool = True,
    frustum_scale: float = 1.0,
    point_size: float = 1.0,
    camera_filter: str | None = None,
) -> None:
    """
    Interactive 3D visualization with Open3D.

    Args:
        recon: Loaded reconstruction dict
        show_cameras: Whether to display camera frustums
        frustum_scale: Scale factor for camera frustum size
        point_size: Size of rendered points
        camera_filter: Optional camera-folder prefix (e.g. 'nadir', 'cam') to filter cameras
    """
    print(f"Loading {recon['num_points']} 3D points...")

    # Create point cloud from sparse points
    points_3d = np.array([p["xyz"] for p in recon["points3D"]])
    colors = np.array([p["rgb"] for p in recon["points3D"]]) / 255.0

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3d)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    geometries = [pcd]

    # Add camera frustums
    if show_cameras:
        print(f"Adding {recon['num_images']} camera frustums...")

        # Get image dimensions from cameras
        cam_dims = {}
        for cam_id, cam_data in recon["cameras"].items():
            cam_dims[int(cam_id)] = (cam_data["width"], cam_data["height"])

        palette = [[0.0, 0.5, 1.0], [1.0, 0.5, 0.0], [0.2, 0.8, 0.2],
                   [0.8, 0.2, 0.8], [0.9, 0.9, 0.1], [0.1, 0.8, 0.8]]
        folder_colors = {}

        for img_id, img_data in recon["images"].items():
            cam_name = img_data["name"]

            # Apply filter if specified
            if camera_filter:
                if not cam_name.startswith(camera_filter):
                    continue

            R = np.array(img_data["R"])
            t = np.array(img_data["t"])
            K = np.array(img_data["K"])

            # Get dimensions from camera
            cam_id = img_data["camera_id"]
            width, height = cam_dims.get(cam_id, (1000, 1000))

            # Color: one colour per camera folder (first path component)
            folder = cam_name.split("/")[0] if "/" in cam_name else ""
            if folder not in folder_colors:
                folder_colors[folder] = palette[len(folder_colors) % len(palette)]
            color = folder_colors[folder]

            frustum = create_camera_frustum(
                R, t, K, width, height, scale=frustum_scale, color=color
            )
            geometries.append(frustum)

    # Launch viewer
    print("Launching Open3D viewer...")
    print("Controls: Mouse drag = rotate, Scroll = zoom, Q = quit")

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="SfM Reconstruction", width=1280, height=720)

    for geom in geometries:
        vis.add_geometry(geom)

    # Set render options
    opt = vis.get_render_option()
    opt.point_size = point_size
    opt.background_color = np.array([0.1, 0.1, 0.1])

    vis.run()
    vis.destroy_window()


def visualize_depth(
    image_name: str,
    data_dir: Path,
    depth_dir: Path,
    colormap: str = "turbo",
    save_path: Path | None = None,
) -> None:
    """
    Side-by-side image and depth map visualization.

    Args:
        image_name: Name from reconstruction (e.g., "nadir/0001.jpg")
        data_dir: Base directory for images
        depth_dir: Base directory for depth maps
        colormap: Matplotlib colormap name
        save_path: If provided, saves figure instead of showing
    """
    # Build paths
    image_path = data_dir / image_name
    depth_name = str(Path(image_name).with_suffix(".tif"))
    depth_path = depth_dir / depth_name

    # Load data
    print(f"Loading image: {image_path}")
    image = load_image(image_path)

    print(f"Loading depth: {depth_path}")
    depth = load_depth_map(depth_path)

    print(f"Image shape: {image.shape}, Depth shape: {depth.shape}")
    print(f"Depth range: [{depth.min():.4f}, {depth.max():.4f}]")

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Original image
    axes[0].imshow(image)
    axes[0].set_title(f"Original: {image_name}")
    axes[0].axis("off")

    # Depth map with colormap
    im = axes[1].imshow(depth, cmap=colormap)
    axes[1].set_title(f"Depth Map (range: {depth.min():.2f} - {depth.max():.2f})")
    axes[1].axis("off")

    # Colorbar
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")
        plt.close()
    else:
        # For interactive display, save to temp file and open with system viewer
        import tempfile
        import subprocess

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            plt.savefig(f.name, dpi=150, bbox_inches="tight")
            print(f"Opening: {f.name}")
            subprocess.run(["xdg-open", f.name], check=False)
        plt.close()


def visualize_overlay(
    image_name: str,
    recon: dict,
    data_dir: Path,
    marker_size: int = 8,
    show_reprojection_error: bool = False,
    save_path: Path | None = None,
) -> None:
    """
    Project 3D sparse points onto 2D image.

    Args:
        image_name: Image name from reconstruction
        recon: Loaded reconstruction dict
        data_dir: Base directory for images
        marker_size: Size of projected point markers
        show_reprojection_error: Color points by reprojection error
        save_path: If provided, saves figure instead of showing
    """
    # Find image in reconstruction
    img_data = None
    for img_id, data in recon["images"].items():
        if data["name"] == image_name:
            img_data = data
            break

    if img_data is None:
        available = [d["name"] for d in recon["images"].values()]
        print(f"Image '{image_name}' not found in reconstruction.")
        print(f"Available images: {sorted(available)[:10]}...")
        return

    # Load image
    image_path = data_dir / image_name
    print(f"Loading image: {image_path}")
    image = load_image(image_path)
    height, width = image.shape[:2]

    # Get camera parameters
    R = np.array(img_data["R"])
    t = np.array(img_data["t"])
    K = np.array(img_data["K"])

    # Get 3D points
    points_3d = np.array([p["xyz"] for p in recon["points3D"]])
    colors_rgb = np.array([p["rgb"] for p in recon["points3D"]]) / 255.0
    errors = np.array([p["error"] for p in recon["points3D"]])

    print(f"Projecting {len(points_3d)} 3D points...")

    # Project points
    points_2d, valid_indices = project_points_to_image(
        points_3d, R, t, K, width, height
    )

    print(f"Visible points: {len(points_2d)}")

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.imshow(image)

    if len(points_2d) > 0:
        if show_reprojection_error:
            # Color by reprojection error
            valid_errors = errors[valid_indices]
            scatter = ax.scatter(
                points_2d[:, 0],
                points_2d[:, 1],
                c=valid_errors,
                cmap="hot",
                s=marker_size,
                edgecolors="black",
                linewidths=0.3,
                alpha=0.8,
            )
            plt.colorbar(scatter, ax=ax, label="Reprojection Error")
        else:
            # High-contrast cyan markers with black edges for visibility
            ax.scatter(
                points_2d[:, 0],
                points_2d[:, 1],
                c="#00FFFF",
                edgecolors="black",
                linewidths=0.3,
                s=marker_size,
                alpha=0.8,
            )

    ax.set_title(f"{image_name} - {len(points_2d)} projected points")
    ax.axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved to {save_path}")
        plt.close()
    else:
        # For interactive display, save to temp file and open with system viewer
        import tempfile
        import subprocess

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            plt.savefig(f.name, dpi=150, bbox_inches="tight")
            print(f"Opening: {f.name}")
            subprocess.run(["xdg-open", f.name], check=False)
        plt.close()


def list_available_images(recon: dict) -> None:
    """List all images in the reconstruction."""
    print("Available images in reconstruction:")
    names = sorted([d["name"] for d in recon["images"].values()])
    for name in names:
        print(f"  {name}")


def batch_visualize_depth(
    output_dir: Path,
    recon_path: Path,
    data_dir: Path,
    depth_dir: Path,
    colormap: str = "turbo",
) -> None:
    """
    Generate depth visualizations for all images and save to output directory.
    """
    import matplotlib

    matplotlib.use("Agg")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load reconstruction to get image list
    print(f"Loading reconstruction from {recon_path}...")
    recon = load_reconstruction(recon_path)
    image_names = sorted([d["name"] for d in recon["images"].values()])
    print(f"Found {len(image_names)} images")

    print(f"\nGenerating depth visualizations to {output_dir}/...")
    for i, image_name in enumerate(image_names, 1):
        safe_name = image_name.replace("/", "_").rsplit(".", 1)[0] + ".png"
        save_path = output_dir / safe_name
        try:
            visualize_depth(
                image_name,
                data_dir,
                depth_dir,
                colormap=colormap,
                save_path=save_path,
            )
            print(f"  [{i}/{len(image_names)}] {image_name} -> {save_path.name}")
        except FileNotFoundError as e:
            print(f"  [{i}/{len(image_names)}] {image_name} - SKIPPED: {e}")

    print(
        f"\nDone! Depth visualizations saved to {output_dir}/ ({len(list(output_dir.glob('*.png')))} files)"
    )


def batch_visualize_overlay(
    output_dir: Path,
    recon_path: Path,
    data_dir: Path,
    marker_size: int = 8,
    show_reprojection_error: bool = False,
) -> None:
    """
    Generate overlay visualizations for all images and save to output directory.
    """
    import matplotlib

    matplotlib.use("Agg")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load reconstruction
    print(f"Loading reconstruction from {recon_path}...")
    recon = load_reconstruction(recon_path)
    image_names = sorted([d["name"] for d in recon["images"].values()])
    print(f"Found {len(image_names)} images, {recon['num_points']} 3D points")

    print(f"\nGenerating overlay visualizations to {output_dir}/...")
    for i, image_name in enumerate(image_names, 1):
        safe_name = image_name.replace("/", "_").rsplit(".", 1)[0] + ".png"
        save_path = output_dir / safe_name
        try:
            visualize_overlay(
                image_name,
                recon,
                data_dir,
                marker_size=marker_size,
                show_reprojection_error=show_reprojection_error,
                save_path=save_path,
            )
            print(f"  [{i}/{len(image_names)}] {image_name} -> {save_path.name}")
        except FileNotFoundError as e:
            print(f"  [{i}/{len(image_names)}] {image_name} - SKIPPED: {e}")

    print(
        f"\nDone! Overlay visualizations saved to {output_dir}/ ({len(list(output_dir.glob('*.png')))} files)"
    )


def batch_visualize_all(
    output_dir: Path,
    recon_path: Path,
    data_dir: Path,
    depth_dir: Path,
    colormap: str = "turbo",
) -> None:
    """
    Generate all visualizations and save to output directory.

    Creates:
    - output_dir/depth/*.png - Side-by-side depth visualizations
    - output_dir/overlay/*.png - Sparse points projected on images
    """
    import matplotlib

    matplotlib.use("Agg")  # Non-interactive backend for batch processing

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    depth_out = output_dir / "depth"
    overlay_out = output_dir / "overlay"
    depth_out.mkdir(exist_ok=True)
    overlay_out.mkdir(exist_ok=True)

    # Load reconstruction
    print(f"Loading reconstruction from {recon_path}...")
    recon = load_reconstruction(recon_path)
    print(f"Found {recon['num_images']} images, {recon['num_points']} 3D points")

    # Get all images from reconstruction
    image_names = sorted([d["name"] for d in recon["images"].values()])

    print(f"\nGenerating depth visualizations...")
    for i, image_name in enumerate(image_names, 1):
        safe_name = image_name.replace("/", "_").rsplit(".", 1)[0] + ".png"
        save_path = depth_out / safe_name
        try:
            visualize_depth(
                image_name,
                data_dir,
                depth_dir,
                colormap=colormap,
                save_path=save_path,
            )
            print(f"  [{i}/{len(image_names)}] {image_name} -> {save_path.name}")
        except FileNotFoundError as e:
            print(f"  [{i}/{len(image_names)}] {image_name} - SKIPPED: {e}")

    print(f"\nGenerating overlay visualizations...")
    for i, image_name in enumerate(image_names, 1):
        safe_name = image_name.replace("/", "_").replace(".JPG", ".png")
        save_path = overlay_out / safe_name
        try:
            visualize_overlay(
                image_name,
                recon,
                data_dir,
                marker_size=8,
                show_reprojection_error=False,
                save_path=save_path,
            )
            print(f"  [{i}/{len(image_names)}] {image_name} -> {save_path.name}")
        except FileNotFoundError as e:
            print(f"  [{i}/{len(image_names)}] {image_name} - SKIPPED: {e}")

    print(f"\nDone! Results saved to {output_dir}/")
    print(f"  - Depth maps: {depth_out}/ ({len(list(depth_out.glob('*.png')))} files)")
    print(
        f"  - Overlays: {overlay_out}/ ({len(list(overlay_out.glob('*.png')))} files)"
    )


# =============================================================================
# CLI Entry Point
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Visualization suite for depth densification pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""examples:
  %(prog)s pointcloud                          # Interactive 3D point cloud viewer
  %(prog)s depth --image nadir/0001.jpg            # View depth map for a single image
  %(prog)s depth --list                        # List all available images
  %(prog)s overlay --image cam03/0003.jpg        # Show 3D points projected onto image
  %(prog)s all --output visualize_out          # Batch generate all visualizations
  %(prog)s all --depth-dir densified_cpp       # Use C++ densified maps instead
""",
    )
    subparsers = parser.add_subparsers(
        dest="command", required=True, help="Visualization mode to run"
    )

    # === pointcloud subcommand ===
    pc_parser = subparsers.add_parser(
        "pointcloud",
        help="Interactive 3D point cloud viewer with camera poses (requires Open3D)",
        description="Launch an interactive 3D viewer showing the sparse SfM point cloud "
        "and camera frustums. Use mouse to rotate/zoom/pan.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""examples:
  %(prog)s                                     # Default: show all cameras + points
  %(prog)s --no-cameras                        # Hide camera frustums
  %(prog)s --camera-filter nadir               # Show only cameras whose name starts with "nadir"
  %(prog)s --frustum-scale 10 --point-size 2   # Larger frustums and points
""",
    )
    pc_parser.add_argument(
        "--recon",
        type=Path,
        default=Path("sfm_output/reconstruction.json"),
        help="Path to reconstruction JSON file (default: sfm_output/reconstruction.json)",
    )
    pc_parser.add_argument(
        "--no-cameras",
        action="store_true",
        help="Hide camera frustum wireframes, show only the point cloud",
    )
    pc_parser.add_argument(
        "--frustum-scale",
        type=float,
        default=5.0,
        help="Scale factor for camera frustum wireframes (default: 5.0)",
    )
    pc_parser.add_argument(
        "--point-size",
        type=float,
        default=1.0,
        help="Rendering size of 3D points in the viewer (default: 1.0)",
    )
    pc_parser.add_argument(
        "--camera-filter",
        help="Only show cameras whose image name starts with this prefix (e.g. 'nadir', 'cam')",
    )

    # === depth subcommand ===
    depth_parser = subparsers.add_parser(
        "depth",
        help="Side-by-side original image and depth map visualization",
        description="Display a single image alongside its corresponding depth map "
        "with a color-coded visualization. The turbo colormap is used by default "
        "(red=near, blue=far).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""examples:
  %(prog)s --image nadir/0001.jpg                          # View depth for a nadir image
  %(prog)s --image cam03/0003.jpg --depth-dir densified_cpp  # Use densified depth maps
  %(prog)s --image nadir/0002.jpg --colormap plasma         # Use plasma colormap
  %(prog)s --image nadir/0002.jpg --save output.png         # Save to file instead of showing
  %(prog)s --list                                       # List all available images
""",
    )
    depth_parser.add_argument(
        "--image",
        type=str,
        help="Image name relative to data-dir (e.g., 'nadir/0001.jpg', 'cam03/0003.jpg')",
    )
    depth_parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/images"),
        help="Base directory containing the original images (default: data/images)",
    )
    depth_parser.add_argument(
        "--depth-dir",
        type=Path,
        default=Path("depth_maps"),
        help="Directory containing depth map TIFFs. Use 'depth_maps' for monocular, "
        "'densified_python' or 'densified_cpp' for densified maps (default: depth_maps)",
    )
    depth_parser.add_argument(
        "--colormap",
        default="turbo",
        help="Matplotlib colormap name for depth visualization. Options: turbo, plasma, "
        "viridis, inferno, magma, hot, etc. (default: turbo)",
    )
    depth_parser.add_argument(
        "--save",
        type=Path,
        help="Save the figure to this file path (PNG/JPG/PDF) instead of displaying interactively",
    )
    depth_parser.add_argument(
        "--list",
        action="store_true",
        help="List all available images in data-dir and exit (useful to find image names)",
    )
    depth_parser.add_argument(
        "--recon",
        type=Path,
        default=Path("sfm_output/reconstruction.json"),
        help="Path to reconstruction JSON (required for batch mode without --image)",
    )
    depth_parser.add_argument(
        "--output",
        type=Path,
        help="Output directory for batch mode. When specified without --image, processes all images.",
    )

    # === overlay subcommand ===
    overlay_parser = subparsers.add_parser(
        "overlay",
        help="Project sparse 3D SfM points onto a 2D image as colored dots",
        description="Overlay the projected SfM 3D points onto the original image. "
        "Each point is colored by its metric depth (turbo colormap). "
        "This visualizes anchor point distribution and projection accuracy.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""examples:
  %(prog)s --image nadir/0001.jpg                          # Overlay 3D points on nadir image
  %(prog)s --image cam03/0003.jpg --marker-size 5        # Larger point markers
  %(prog)s --image nadir/0002.jpg --color-by-error         # Color by reprojection error
  %(prog)s --image nadir/0002.jpg --save overlay.png       # Save to file
  %(prog)s --list                                      # List registered images
""",
    )
    overlay_parser.add_argument(
        "--image",
        type=str,
        help="Image name from the reconstruction (e.g., 'nadir/0001.jpg', 'cam03/0003.jpg')",
    )
    overlay_parser.add_argument(
        "--recon",
        type=Path,
        default=Path("sfm_output/reconstruction.json"),
        help="Path to reconstruction JSON file (default: sfm_output/reconstruction.json)",
    )
    overlay_parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/images"),
        help="Base directory containing the original images (default: data/images)",
    )
    overlay_parser.add_argument(
        "--marker-size",
        type=int,
        default=3,
        help="Radius in pixels for each projected point marker (default: 3)",
    )
    overlay_parser.add_argument(
        "--color-by-error",
        action="store_true",
        help="Color points by reprojection error instead of depth (red=high error)",
    )
    overlay_parser.add_argument(
        "--save",
        type=Path,
        help="Save the figure to this file path (PNG/JPG/PDF) instead of displaying interactively",
    )
    overlay_parser.add_argument(
        "--list",
        action="store_true",
        help="List all registered images in the reconstruction and exit",
    )
    overlay_parser.add_argument(
        "--output",
        type=Path,
        help="Output directory for batch mode. When specified without --image, processes all images.",
    )

    # === all subcommand ===
    all_parser = subparsers.add_parser(
        "all",
        help="Batch-generate depth maps and overlay PNGs for ALL registered images",
        description="Generate depth map visualizations and 3D point overlay images for every "
        "registered image in the reconstruction. Outputs are saved as PNGs to "
        "the specified output directory with subdirectories depth/ and overlay/.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""examples:
  %(prog)s                                             # Default output to visualize/
  %(prog)s --output visualize_cpp                      # Custom output directory
  %(prog)s --depth-dir densified_cpp                   # Visualize C++ densified maps
  %(prog)s --depth-dir densified_python --colormap plasma  # Python maps with plasma colormap
  %(prog)s --data-dir data/images --recon sfm_output/reconstruction.json
""",
    )
    all_parser.add_argument(
        "--output",
        type=Path,
        default=Path("visualize"),
        help="Output directory for generated PNGs. Creates depth/ and overlay/ "
        "subdirectories inside (default: visualize/)",
    )
    all_parser.add_argument(
        "--recon",
        type=Path,
        default=Path("sfm_output/reconstruction.json"),
        help="Path to reconstruction JSON file (default: sfm_output/reconstruction.json)",
    )
    all_parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/images"),
        help="Base directory containing the original images (default: data/images)",
    )
    all_parser.add_argument(
        "--depth-dir",
        type=Path,
        default=Path("depth_maps"),
        help="Directory containing depth map TIFFs to visualize. Use 'depth_maps' for "
        "monocular or 'densified_cpp'/'densified_python' for densified maps "
        "(default: depth_maps)",
    )
    all_parser.add_argument(
        "--colormap",
        default="turbo",
        help="Matplotlib colormap for depth visualizations (default: turbo)",
    )

    args = parser.parse_args()

    # Dispatch to appropriate visualization
    if args.command == "pointcloud":
        recon = load_reconstruction(args.recon)
        visualize_pointcloud(
            recon,
            show_cameras=not args.no_cameras,
            frustum_scale=args.frustum_scale,
            point_size=args.point_size,
            camera_filter=args.camera_filter,
        )

    elif args.command == "depth":
        if args.list:
            # List images from directory
            images = sorted(args.data_dir.rglob("*.JPG"))
            print("Available images:")
            for img in images:
                rel = img.relative_to(args.data_dir)
                print(f"  {rel}")
        elif args.image:
            visualize_depth(
                args.image,
                args.data_dir,
                args.depth_dir,
                colormap=args.colormap,
                save_path=args.save,
            )
        elif args.output:
            # Batch mode: process all images
            batch_visualize_depth(
                output_dir=args.output,
                recon_path=args.recon,
                data_dir=args.data_dir,
                depth_dir=args.depth_dir,
                colormap=args.colormap,
            )
        else:
            print(
                "Error: --image or --output required (or use --list to see available images)"
            )

    elif args.command == "overlay":
        recon = load_reconstruction(args.recon)
        if args.list:
            list_available_images(recon)
        elif args.image:
            visualize_overlay(
                args.image,
                recon,
                args.data_dir,
                marker_size=args.marker_size,
                show_reprojection_error=args.color_by_error,
                save_path=args.save,
            )
        elif args.output:
            # Batch mode: process all images
            batch_visualize_overlay(
                output_dir=args.output,
                recon_path=args.recon,
                data_dir=args.data_dir,
                marker_size=args.marker_size,
                show_reprojection_error=args.color_by_error,
            )
        else:
            print(
                "Error: --image or --output required (or use --list to see available images)"
            )

    elif args.command == "all":
        batch_visualize_all(
            output_dir=args.output,
            recon_path=args.recon,
            data_dir=args.data_dir,
            depth_dir=args.depth_dir,
            colormap=args.colormap,
        )


if __name__ == "__main__":
    main()
