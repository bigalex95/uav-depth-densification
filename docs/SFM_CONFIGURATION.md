# SfM Pipeline Configuration Guide

## Quick Start

Run with command-line options:

```bash
./scripts/run_sfm.sh --num_threads 10 --matching_type sequential
```

Or use short flags:

```bash
./scripts/run_sfm.sh -t 10 -m sequential
```

With custom data and output paths:

```bash
./scripts/run_sfm.sh -d ./my_data -o ./my_output -t 10 -m sequential
```

Or use defaults (8 threads, exhaustive matching, data/, sfm_output/):

```bash
./scripts/run_sfm.sh
```

Get help:

```bash
./scripts/run_sfm.sh --help
```

## Available Options

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--num_threads` | `-t` | Number of CPU cores to use | 8 |
| `--matching_type` | `-m` | Matching algorithm (exhaustive, sequential, spatial, vocab_tree) | exhaustive |
| `--data_path` | `-d` | Path to data directory containing images | data/ |
| `--output_path` | `-o` | Path to output directory for SfM results | sfm_output/ |
| `--help` | `-h` | Show help message | - |

## Detailed Configuration

### 1. --num_threads (-t)

Controls parallel processing. Set to the number of CPU cores available on your system.

- **Recommended**: 8-16 for most systems
- **Low memory**: 4-6
- **High-end workstation**: 16-32

### 2. --matching_type (-m)

Choose the matching algorithm based on your dataset characteristics:

#### **exhaustive** (Default)

- Matches every image with every other image
- **Best for**: Small datasets (<100 images), unordered images
- **Pros**: Most accurate, finds all matches
- **Cons**: Slow (O(n²) complexity)
- **Use when**: You need maximum accuracy and have time

#### **sequential**

- Matches each image with N neighboring images in sequence
- **Best for**: Ordered images (video frames, drone flights)
- **Pros**: Fast, works well for sequential data
- **Cons**: May miss long-range matches
- **Parameters**: `--SequentialMatching.overlap 10` (matches with 10 neighbors)
- **Use when**: Images are captured in sequence

#### **spatial**

- Uses GPS/EXIF location data to match nearby images
- **Best for**: Geotagged images with GPS coordinates
- **Pros**: Very fast, scales to large datasets
- **Cons**: Requires GPS data
- **Parameters**: `--SpatialMatching.max_num_neighbors 50`
- **Use when**: Your images have GPS coordinates

#### **vocab_tree**

- Uses visual vocabulary tree for efficient matching
- **Best for**: Large datasets (>1000 images)
- **Pros**: Fastest for large datasets, good accuracy
- **Cons**: Requires vocabulary tree file (1.5GB download)
- **Setup**: Download vocab tree from https://demuc.de/colmap/
- **Use when**: You have many images and need speed

### 3. --data_path (-d)

Specify the input data directory.

- The script will automatically search for:
  - `cameras.json` (for camera parameters)
  - `images/` subdirectory
- Supports absolute or relative paths
- Default: `data/` in project root

**Example:**

```bash
./scripts/run_sfm.sh -d /path/to/my/dataset
./scripts/run_sfm.sh --data_path ./custom_data
```

### 4. --output_path (-o)

Specify the output directory for SfM results.

- Creates directory if it doesn't exist
- Outputs include:
  - `database.db` (COLMAP database)
  - `sparse/` (sparse reconstruction)
  - `cameras.txt`, `images.txt`, `points3D.txt` (exported results)
- Supports absolute or relative paths
- Default: `sfm_output/` in project root

**Example:**

```bash
./scripts/run_sfm.sh -o /path/to/output
./scripts/run_sfm.sh --output_path ./results/sfm
```

## Camera Model

The script uses **SIMPLE_PINHOLE** model:

- Parameters: `[f, cx, cy]` (focal length, principal point)
- Used when: `fx = fy` (square pixels)
- Optionally set from `data/cameras.json` (see the README for its format)

## Choosing a Matching Strategy for UAV Flights

1. **Exhaustive** (default): matches every pair. For the multi-camera survey set
   used for the reported results (90 images) it registered all 90 images, so it
   is the recommended starting point for small sets.

2. **Sequential**: much faster, but only useful if the file ordering follows the
   flight path. On the same 90-image survey set, two sequential runs registered
   only 17 and 18 of 90 images (see the README results table), because the
   images come from several cameras and are not ordered along a single track.

3. **Spatial**: only if the images carry GPS EXIF data.

   - Check with: `exiftool data/images/<camera>/*.JPG | grep GPS`

## Troubleshooting

### "No sparse model found"

```bash
# Try different matching types
./scripts/run_sfm.sh -m sequential
./scripts/run_sfm.sh -m exhaustive
```

- Check camera parameters are correct in cameras.json

### Slow processing

```bash
# Reduce threads if running out of memory
./scripts/run_sfm.sh -t 4

# Use faster matching
./scripts/run_sfm.sh -m sequential
```

### Low number of registered images

```bash
# Use exhaustive matching for maximum coverage
./scripts/run_sfm.sh -m exhaustive
```

- Check that camera parameters match your images
- Verify image quality and overlap
