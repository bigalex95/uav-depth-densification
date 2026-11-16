#!/bin/bash
set -e

# ============================================
# Default Configuration
# ============================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

NUM_THREADS=8  # Number of CPU cores to use (adjust based on your system)
MATCHING_TYPE="exhaustive"  # Options: exhaustive, sequential, spatial, vocab_tree
DATA_DIR="$PROJECT_ROOT/data"  # Default data directory
SFM_OUTPUT="$PROJECT_ROOT/sfm_output"  # Default output directory

# ============================================
# Parse Command Line Arguments
# ============================================
show_help() {
    cat << EOF
Usage: $(basename "$0") [OPTIONS]

Structure-from-Motion pipeline using COLMAP with SIMPLE_PINHOLE camera model.

Options:
    -t, --num_threads NUM       Number of CPU threads to use (default: 8)
    -m, --matching_type TYPE    Matching algorithm to use (default: exhaustive)
                                Options: exhaustive, sequential, spatial, vocab_tree
    -d, --data_path PATH        Path to data directory containing images (default: data/)
    -o, --output_path PATH      Path to output directory for SfM results (default: sfm_output/)
    -h, --help                  Show this help message

Examples:
    $(basename "$0") --num_threads 16 --matching_type sequential
    $(basename "$0") -t 10 -m exhaustive
    $(basename "$0") -d /path/to/images -o /path/to/output
    $(basename "$0") --data_path ./custom_data --output_path ./custom_output
    $(basename "$0")  # Use default settings

Matching Types:
    exhaustive  - Match all image pairs (slow, most accurate)
    sequential  - Match sequential images (fast, good for ordered data)
    spatial     - Match based on GPS location (requires EXIF GPS data)
    vocab_tree  - Use vocabulary tree (fast for large datasets, requires vocab file)

EOF
    exit 0
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--num_threads)
            NUM_THREADS="$2"
            shift 2
            ;;
        -m|--matching_type)
            MATCHING_TYPE="$2"
            shift 2
            ;;
        -d|--data_path)
            DATA_DIR="$2"
            shift 2
            ;;
        -o|--output_path)
            SFM_OUTPUT="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            ;;
        *)
            echo "Error: Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Validate NUM_THREADS
if ! [[ "$NUM_THREADS" =~ ^[0-9]+$ ]] || [ "$NUM_THREADS" -lt 1 ]; then
    echo "Error: NUM_THREADS must be a positive integer (got: $NUM_THREADS)"
    exit 1
fi

# Validate MATCHING_TYPE
case "$MATCHING_TYPE" in
    exhaustive|sequential|spatial|vocab_tree)
        # Valid option
        ;;
    *)
        echo "Error: Invalid matching type '$MATCHING_TYPE'"
        echo "Valid options: exhaustive, sequential, spatial, vocab_tree"
        exit 1
        ;;
esac

# Convert relative paths to absolute paths
DATA_DIR=$(cd "$DATA_DIR" 2>/dev/null && pwd) || {
    echo "Error: Data directory '$DATA_DIR' does not exist"
    exit 1
}

# Create output directory if it doesn't exist and convert to absolute path
mkdir -p "$SFM_OUTPUT"
SFM_OUTPUT=$(cd "$SFM_OUTPUT" && pwd)

DATABASE_PATH="$SFM_OUTPUT/database.db"

echo "============================================"
echo "  Structure-from-Motion (COLMAP)"
echo "============================================"
echo ""
echo "Configuration:"
echo "  - CPU threads: $NUM_THREADS"
echo "  - Matching type: $MATCHING_TYPE"
echo "  - Camera model: SIMPLE_PINHOLE (fx=fy)"
echo "  - Data directory: $DATA_DIR"
echo "  - Output directory: $SFM_OUTPUT"
echo ""

# Find images directory
IMAGES_DIR="$DATA_DIR/images"
if [ ! -d "$IMAGES_DIR" ]; then
    # Try to find images in nested structure
    IMAGES_DIR=$(find "$DATA_DIR" -type d -name "images" | head -1)
    if [ -z "$IMAGES_DIR" ]; then
        # Use the data directory itself
        IMAGES_DIR="$DATA_DIR"
    fi
fi

echo "Images directory: $IMAGES_DIR"
echo ""

# Create output directories
mkdir -p "$SFM_OUTPUT/sparse"

# Step 1: Feature extraction
echo "[1/5] Extracting features (SIFT)..."
colmap feature_extractor \
    --database_path "$DATABASE_PATH" \
    --image_path "$IMAGES_DIR" \
    --ImageReader.single_camera 0 \
    --ImageReader.camera_model SIMPLE_PINHOLE \
    --SiftExtraction.use_gpu 1 \
    --SiftExtraction.num_threads "$NUM_THREADS" \
    2>/dev/null || colmap feature_extractor \
    --database_path "$DATABASE_PATH" \
    --image_path "$IMAGES_DIR" \
    --ImageReader.single_camera 0 \
    --ImageReader.camera_model SIMPLE_PINHOLE \
    --SiftExtraction.use_gpu 0 \
    --SiftExtraction.num_threads "$NUM_THREADS"

# Step 2: Update camera parameters from cameras.json
CAMERAS_JSON=""
# Try different locations for cameras.json
if [ -f "$DATA_DIR/cameras.json" ]; then
    CAMERAS_JSON="$DATA_DIR/cameras.json"
else
    # Try to find it recursively
    CAMERAS_JSON=$(find "$DATA_DIR" -name "cameras.json" -type f | head -1)
fi

if [ -n "$CAMERAS_JSON" ]; then
    echo ""
    echo "[2/5] Updating camera parameters from cameras.json..."
    echo "  Camera config: $CAMERAS_JSON"
    uv run python "$SCRIPT_DIR/update_camera_db.py" "$DATABASE_PATH" "$CAMERAS_JSON"
else
    echo ""
    echo "[2/5] Skipping camera update (cameras.json not found in $DATA_DIR)"
fi

# Step 3: Feature matching
echo ""
echo "[3/5] Matching features with $MATCHING_TYPE matching..."

case "$MATCHING_TYPE" in
    exhaustive)
        colmap exhaustive_matcher \
            --database_path "$DATABASE_PATH" \
            --SiftMatching.use_gpu 1 \
            --SiftMatching.num_threads "$NUM_THREADS" \
            2>/dev/null || colmap exhaustive_matcher \
            --database_path "$DATABASE_PATH" \
            --SiftMatching.use_gpu 0 \
            --SiftMatching.num_threads "$NUM_THREADS"
        ;;

    sequential)
        colmap sequential_matcher \
            --database_path "$DATABASE_PATH" \
            --SequentialMatching.overlap 10 \
            --SiftMatching.use_gpu 1 \
            --SiftMatching.num_threads "$NUM_THREADS" \
            2>/dev/null || colmap sequential_matcher \
            --database_path "$DATABASE_PATH" \
            --SequentialMatching.overlap 10 \
            --SiftMatching.use_gpu 0 \
            --SiftMatching.num_threads "$NUM_THREADS"
        ;;

    spatial)
        colmap spatial_matcher \
            --database_path "$DATABASE_PATH" \
            --SpatialMatching.max_num_neighbors 50 \
            --SiftMatching.use_gpu 1 \
            --SiftMatching.num_threads "$NUM_THREADS" \
            2>/dev/null || colmap spatial_matcher \
            --database_path "$DATABASE_PATH" \
            --SpatialMatching.max_num_neighbors 50 \
            --SiftMatching.use_gpu 0 \
            --SiftMatching.num_threads "$NUM_THREADS"
        ;;

    vocab_tree)
        VOCAB_TREE_PATH="$PROJECT_ROOT/vocab_tree_flickr100K_words256K.bin"
        if [ ! -f "$VOCAB_TREE_PATH" ]; then
            echo "Error: Vocabulary tree not found at $VOCAB_TREE_PATH"
            echo "Download it from: https://demuc.de/colmap/"
            exit 1
        fi
        colmap vocab_tree_matcher \
            --database_path "$DATABASE_PATH" \
            --VocabTreeMatching.vocab_tree_path "$VOCAB_TREE_PATH" \
            --SiftMatching.use_gpu 1 \
            --SiftMatching.num_threads "$NUM_THREADS" \
            2>/dev/null || colmap vocab_tree_matcher \
            --database_path "$DATABASE_PATH" \
            --VocabTreeMatching.vocab_tree_path "$VOCAB_TREE_PATH" \
            --SiftMatching.use_gpu 0 \
            --SiftMatching.num_threads "$NUM_THREADS"
        ;;

    *)
        echo "Error: Unknown matching type '$MATCHING_TYPE'"
        echo "Valid options: exhaustive, sequential, spatial, vocab_tree"
        exit 1
        ;;
esac

# Step 4: Sparse reconstruction
echo ""
echo "[4/5] Running sparse reconstruction..."
colmap mapper \
    --database_path "$DATABASE_PATH" \
    --image_path "$IMAGES_DIR" \
    --output_path "$SFM_OUTPUT/sparse" \
    --Mapper.num_threads "$NUM_THREADS" \
    --Mapper.ba_global_max_num_iterations 50 \
    --Mapper.ba_local_max_num_iterations 25

# Step 5: Export to TXT format
echo ""
echo "[5/5] Exporting to TXT format..."
SPARSE_MODEL=$(ls -d "$SFM_OUTPUT/sparse"/*/ 2>/dev/null | head -1)
if [ -n "$SPARSE_MODEL" ]; then
    colmap model_converter \
        --input_path "$SPARSE_MODEL" \
        --output_path "$SFM_OUTPUT" \
        --output_type TXT

    # Print statistics
    echo ""
    echo "============================================"
    echo "  SfM Completed Successfully!"
    echo "============================================"

    if [ -f "$SFM_OUTPUT/points3D.txt" ]; then
        NUM_POINTS=$(tail -n +4 "$SFM_OUTPUT/points3D.txt" | wc -l)
        echo "  - 3D points: $NUM_POINTS"
    fi

    if [ -f "$SFM_OUTPUT/images.txt" ]; then
        NUM_REGISTERED=$(grep -E "\.jpg$|\.JPG$|\.png$|\.PNG$" "$SFM_OUTPUT/images.txt" | wc -l)
        echo "  - Registered images: $NUM_REGISTERED"
    fi

    if [ -f "$SFM_OUTPUT/cameras.txt" ]; then
        NUM_CAMERAS=$(tail -n +4 "$SFM_OUTPUT/cameras.txt" | wc -l)
        echo "  - Cameras: $NUM_CAMERAS"
    fi

    echo ""
    echo "Output files:"
    echo "  - $SFM_OUTPUT/cameras.txt"
    echo "  - $SFM_OUTPUT/images.txt"
    echo "  - $SFM_OUTPUT/points3D.txt"
    echo ""
    echo "Next step: uv run python scripts/convert_colmap.py"
else
    echo ""
    echo "============================================"
    echo "  Warning: No sparse model found"
    echo "============================================"
    echo "Check COLMAP output for errors."
    echo ""
    echo "Common issues:"
    echo "  - Not enough feature matches"
    echo "  - Incorrect camera parameters"
    echo "  - Images too dissimilar"
    echo ""
    echo "Try adjusting:"
    echo "  - MATCHING_TYPE (try 'sequential' or 'spatial')"
    echo "  - Feature extraction parameters"
    echo "  - Camera model settings"
fi
