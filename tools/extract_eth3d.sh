#!/bin/bash
# Extract the public ETH3D "terrains" rig archives (images + ground-truth depth)
# downloaded manually from https://www.eth3d.net/datasets into ./dataset/.
# Used only for the optional ground-truth evaluation (tools/evaluate_gt.py).
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DATASET_DIR="${1:-$PROJECT_ROOT/dataset}"
DATA_DIR="$PROJECT_ROOT/data"

echo "============================================"
echo "  ETH3D Terrains Dataset Extraction"
echo "============================================"
echo ""
echo "Source:  $DATASET_DIR"
echo "Target:  $DATA_DIR"
echo ""

mkdir -p "$DATA_DIR"

# Extract images (rig cameras)
if [ -f "$DATASET_DIR/terrains_rig.7z" ]; then
    echo "[1/3] Extracting terrains_rig.7z (raw images)..."
    7z x -o"$DATA_DIR" "$DATASET_DIR/terrains_rig.7z" -aoa -bsp1
    echo "  Done."
else
    echo "[1/3] SKIP: terrains_rig.7z not found"
fi

# Extract undistorted images
if [ -f "$DATASET_DIR/terrains_rig_undistorted.7z" ]; then
    echo "[2/3] Extracting terrains_rig_undistorted.7z (undistorted images)..."
    7z x -o"$DATA_DIR" "$DATASET_DIR/terrains_rig_undistorted.7z" -aoa -bsp1
    echo "  Done."
else
    echo "[2/3] SKIP: terrains_rig_undistorted.7z not found"
fi

# Extract ground truth depth maps
if [ -f "$DATASET_DIR/terrains_rig_depth.7z" ]; then
    echo "[3/3] Extracting terrains_rig_depth.7z (ground truth depths)..."
    7z x -o"$DATA_DIR" "$DATASET_DIR/terrains_rig_depth.7z" -aoa -bsp1
    echo "  Done."
else
    echo "[3/3] SKIP: terrains_rig_depth.7z not found"
fi

echo ""
echo "============================================"
echo "  Extraction Complete!"
echo "============================================"

# Show structure
echo ""
echo "Directory structure:"
find "$DATA_DIR/terrains" -maxdepth 2 -type d 2>/dev/null | sort | head -20

# Count images
NUM_IMAGES=$(find "$DATA_DIR/terrains/images" -name "*.png" -type f 2>/dev/null | wc -l)
NUM_GT=$(find "$DATA_DIR/terrains/ground_truth_depth" -name "*.png" -type f 2>/dev/null | wc -l)
echo ""
echo "Images:            $NUM_IMAGES"
echo "Ground truth maps: $NUM_GT"
