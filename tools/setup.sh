#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "============================================"
echo "  UAV Depth Densification - Setup Script"
echo "============================================"
echo ""

# Step 1: Install system dependencies
echo "[1/3] Installing system dependencies..."
"$PROJECT_ROOT/scripts/install_deps.sh"
echo ""

# Step 2: Install Python dependencies
echo "[2/3] Installing Python dependencies..."
cd "$PROJECT_ROOT"
uv sync
echo ""

# Step 3: Build C++ components
echo "[3/3] Building C++ components..."
"$PROJECT_ROOT/scripts/build.sh"
echo ""

echo "============================================"
echo "  Setup completed successfully!"
echo "============================================"
echo ""
echo "Next steps (put your images under data/images/<camera>/*.JPG):"
echo "  1. Run SfM:      ./scripts/run_sfm.sh"
echo "  2. Convert:      uv run python scripts/convert_colmap.py"
echo "  3. Depth maps:   uv run python scripts/run_depth.py --data-dir data/images"
echo "  4. Densify:      ./build/densify_main --recon sfm_output/reconstruction.json"
echo "  5. Run tests:    cd build && ctest --output-on-failure"
