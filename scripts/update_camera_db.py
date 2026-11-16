#!/usr/bin/env python3
"""
Update COLMAP database with camera parameters from cameras.json.
Uses SIMPLE_PINHOLE model since fx=fy for all cameras.
"""
import json
import sqlite3
import struct
from pathlib import Path


def params_to_blob(params):
    """Convert camera parameters to binary blob for COLMAP database."""
    return struct.pack(f"{len(params)}d", *params)


def update_database_cameras(db_path, cameras_json_path):
    """Update camera parameters in COLMAP database from cameras.json."""

    # Load camera parameters
    with open(cameras_json_path, "r") as f:
        cameras = json.load(f)

    # Connect to database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # COLMAP camera model IDs
    SIMPLE_PINHOLE = 0  # params: f, cx, cy

    print("Updating camera parameters in database...")
    print(f"Using SIMPLE_PINHOLE model (model_id={SIMPLE_PINHOLE})")
    print()

    # Get all images and their camera assignments
    cursor.execute("SELECT image_id, camera_id, name FROM images")
    images = cursor.fetchall()

    # Create mapping of camera prefix to camera info
    camera_mapping = {}
    for cam_name, cam_params in cameras.items():
        f = cam_params["focalLength"]
        cx = cam_params["principalPointX"]
        cy = cam_params["principalPointY"]
        width = int(cam_params["width"])
        height = int(cam_params["height"])

        # SIMPLE_PINHOLE: [f, cx, cy]
        params = [f, cx, cy]
        params_blob = params_to_blob(params)

        camera_mapping[cam_name] = {
            "width": width,
            "height": height,
            "params_blob": params_blob,
            "prior_focal_length": 1,
        }

    print(f"Loaded {len(camera_mapping)} camera configurations from cameras.json:")
    for cam_name, info in camera_mapping.items():
        print(f"  - {cam_name}: {info['width']}x{info['height']}")
    print()

    # Get unique cameras needed
    image_to_camera = {}
    for image_id, camera_id, image_name in images:
        # Extract camera prefix from image name (e.g., "cam00/0001.jpg" -> "cam00")
        prefix = image_name.split("/")[0] if "/" in image_name else "default"

        if prefix not in camera_mapping:
            print(
                f"Warning: No camera config found for prefix '{prefix}' in image '{image_name}'"
            )
            continue

        image_to_camera[image_id] = (prefix, image_name)

    # Track which camera_ids we need for each camera type
    camera_id_mapping = {}

    # First pass: find existing camera_ids or create new ones
    for prefix in set(p for p, _ in image_to_camera.values()):
        # Find if we already have a camera_id for this prefix
        cursor.execute(
            """
            SELECT camera_id FROM cameras
            WHERE width = ? AND height = ?
            LIMIT 1
        """,
            (camera_mapping[prefix]["width"], camera_mapping[prefix]["height"]),
        )

        result = cursor.fetchone()
        if result:
            camera_id_mapping[prefix] = result[0]
        else:
            # Create new camera entry
            cursor.execute(
                """
                INSERT INTO cameras (model, width, height, params, prior_focal_length)
                VALUES (?, ?, ?, ?, ?)
            """,
                (
                    SIMPLE_PINHOLE,
                    camera_mapping[prefix]["width"],
                    camera_mapping[prefix]["height"],
                    camera_mapping[prefix]["params_blob"],
                    camera_mapping[prefix]["prior_focal_length"],
                ),
            )
            camera_id_mapping[prefix] = cursor.lastrowid

    # Update existing cameras to SIMPLE_PINHOLE model
    for prefix, camera_id in camera_id_mapping.items():
        cursor.execute(
            """
            UPDATE cameras
            SET model = ?, params = ?, prior_focal_length = ?
            WHERE camera_id = ?
        """,
            (
                SIMPLE_PINHOLE,
                camera_mapping[prefix]["params_blob"],
                camera_mapping[prefix]["prior_focal_length"],
                camera_id,
            ),
        )

    # Update image-to-camera assignments
    for image_id, (prefix, image_name) in image_to_camera.items():
        cursor.execute(
            """
            UPDATE images
            SET camera_id = ?
            WHERE image_id = ?
        """,
            (camera_id_mapping[prefix], image_id),
        )

    # Clean up unused cameras
    cursor.execute(
        """
        DELETE FROM cameras
        WHERE camera_id NOT IN (SELECT DISTINCT camera_id FROM images)
    """
    )

    deleted_count = cursor.rowcount

    conn.commit()

    # Verify updates
    cursor.execute(
        "SELECT DISTINCT camera_id, model, width, height FROM cameras ORDER BY camera_id"
    )
    updated_cameras = cursor.fetchall()

    print(f"✓ Database updated successfully!")
    print(f"  - Active cameras: {len(updated_cameras)}")
    print(f"  - Removed unused cameras: {deleted_count}")
    print(f"  - All cameras now use SIMPLE_PINHOLE model")
    print()
    print("Active cameras in database:")
    for cam_id, model, width, height in updated_cameras:
        model_name = "SIMPLE_PINHOLE" if model == 0 else f"MODEL_{model}"
        print(f"  Camera {cam_id}: {width}x{height} ({model_name})")

    conn.close()


if __name__ == "__main__":
    import sys

    # Parse command-line arguments
    if len(sys.argv) == 3:
        # Paths provided as arguments
        db_path = Path(sys.argv[1])
        cameras_json_path = Path(sys.argv[2])
    elif len(sys.argv) == 1:
        # Use default paths (for backward compatibility)
        project_root = Path(__file__).parent.parent
        db_path = project_root / "sfm_output" / "database.db"
        cameras_json_path = project_root / "data" / "cameras.json"
    else:
        print("Usage: python update_camera_db.py [DB_PATH CAMERAS_JSON_PATH]")
        print("  DB_PATH: Path to COLMAP database file")
        print("  CAMERAS_JSON_PATH: Path to cameras.json file")
        print("")
        print("If no arguments provided, uses default paths:")
        print("  - sfm_output/database.db")
        print("  - data/cameras.json")
        exit(1)

    if not db_path.exists():
        print(f"Error: Database not found at {db_path}")
        print("Run feature extraction first to create the database.")
        exit(1)

    if not cameras_json_path.exists():
        print(f"Error: cameras.json not found at {cameras_json_path}")
        exit(1)

    update_database_cameras(db_path, cameras_json_path)
