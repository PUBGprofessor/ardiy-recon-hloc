"""Re-run only the dense TSDF fusion on an existing hloc-native map.

Useful for iterating on --voxel_length / --sdf_trunc without re-running feature
extraction, matching, and SfM. The reconstruction in <outputs>/sfm_reference and
the depth maps are reused as-is.

Example:
    python run_refuse.py --outputs f:/dev2/prjs1/data1/obj-12/hloc_map \
        --image_dir f:/dev2/prjs1/data1/obj-12 \
        --voxel_length 0.002 --sdf_trunc 0.008
"""
import argparse
import json
import sys
from pathlib import Path

import pycolmap

from run_build_map import build_dense_model_from_depth


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--outputs",
        type=Path,
        required=True,
        help="Build output dir (must contain sfm_reference).",
    )
    parser.add_argument(
        "--image_dir",
        type=Path,
        required=True,
        help="Scene root directory (image names are relative to it).",
    )
    parser.add_argument(
        "--depth_dir_name",
        type=str,
        default="depth",
        help="Depth directory name, see run_build_map.py.",
    )
    parser.add_argument(
        "--min_depth", type=float, default=0.2, help="Minimum valid depth in meters."
    )
    parser.add_argument(
        "--max_depth", type=float, default=4.0, help="Maximum valid depth in meters."
    )
    parser.add_argument(
        "--voxel_length",
        type=float,
        default=0.002,
        help="TSDF voxel size in meters (fine enough for thin parts, e.g. 0.002-0.003).",
    )
    parser.add_argument(
        "--sdf_trunc",
        type=float,
        default=0.008,
        help="TSDF truncation distance in meters (e.g. 0.008-0.012 for thin parts).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sfm_dir = args.outputs / "sfm_reference"
    if not (sfm_dir / "cameras.bin").exists() and not (sfm_dir / "cameras.txt").exists():
        print(f"[error] no reconstruction found in {sfm_dir}", file=sys.stderr)
        return 1

    reconstruction = pycolmap.Reconstruction(str(sfm_dir))
    print(
        f"[info] loaded {len(reconstruction.images)} images from {sfm_dir}; "
        f"voxel={args.voxel_length}, sdf_trunc={args.sdf_trunc}"
    )
    result = build_dense_model_from_depth(
        reconstruction=reconstruction,
        output_dir=args.outputs / "dense",
        image_dir=args.image_dir,
        depth_dir_name=args.depth_dir_name,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        voxel_length=args.voxel_length,
        sdf_trunc=args.sdf_trunc,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
