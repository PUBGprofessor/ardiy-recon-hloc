import argparse
import csv
from pathlib import Path

import numpy as np

from hloc.utils.read_write_model import qvec2rotmat, read_model


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare poses_optimized.txt against a COLMAP model and diagnose whether "
            "their difference is explained by a global Sim(3) and/or a fixed camera-frame rotation."
        )
    )
    parser.add_argument("--poses_file", type=Path, required=True)
    parser.add_argument("--model_dir", type=Path, required=True)
    parser.add_argument(
        "--image_prefixes",
        type=str,
        default="scan1,scan2",
        help="Comma-separated image path prefixes to include, e.g. scan1,scan2.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        help="Optional directory where summary.txt and per_image.csv are written.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Number of worst images to print for each metric.",
    )
    return parser.parse_args()


def parse_poses_optimized(poses_file: Path):
    lines = [line.strip() for line in poses_file.read_text(encoding="utf-8").splitlines()]
    lines = [line for line in lines if line]
    if len(lines) % 3 != 0:
        raise ValueError(
            f"Unexpected poses_optimized.txt format in {poses_file}: expected 3 lines per image."
        )

    poses = {}
    for idx in range(0, len(lines), 3):
        image_name = lines[idx]
        intrinsics = np.fromstring(lines[idx + 1], sep=" ", dtype=float)
        pose_values = np.fromstring(lines[idx + 2], sep=" ", dtype=float)
        if intrinsics.size < 6:
            raise ValueError(f"Invalid intrinsics line for {image_name} in {poses_file}.")
        if pose_values.size != 12:
            raise ValueError(f"Invalid pose line for {image_name} in {poses_file}.")
        pose_c2w = pose_values.reshape(3, 4)
        rotation_c2w = pose_c2w[:, :3]
        center = pose_c2w[:, 3]
        poses[image_name] = {
            "intrinsics": intrinsics,
            "rotation_c2w": rotation_c2w,
            "center": center,
        }
    return poses


def camera_center_from_w2c(rotation_w2c: np.ndarray, translation_w2c: np.ndarray) -> np.ndarray:
    return -(rotation_w2c.T @ translation_w2c)


def rotation_angle_deg(rotation: np.ndarray) -> float:
    cos_angle = np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.rad2deg(np.arccos(cos_angle)))


def estimate_similarity_transform(source_points: np.ndarray, target_points: np.ndarray):
    if source_points.shape != target_points.shape:
        raise ValueError("Source and target points must have the same shape.")
    if source_points.ndim != 2 or source_points.shape[1] != 3:
        raise ValueError("Similarity alignment expects Nx3 point arrays.")
    if source_points.shape[0] < 3:
        raise ValueError("At least three point correspondences are required for alignment.")

    mean_source = np.mean(source_points, axis=0)
    mean_target = np.mean(target_points, axis=0)
    centered_source = source_points - mean_source
    centered_target = target_points - mean_target

    covariance = (centered_target.T @ centered_source) / source_points.shape[0]
    u, singular_values, vh = np.linalg.svd(covariance)
    correction = np.eye(3)
    if np.linalg.det(u @ vh) < 0:
        correction[-1, -1] = -1.0

    rotation = u @ correction @ vh
    source_variance = np.mean(np.sum(centered_source**2, axis=1))
    if source_variance <= 0:
        raise ValueError("Source points are degenerate, cannot estimate scale.")

    scale = float(np.sum(singular_values * np.diag(correction)) / source_variance)
    translation = mean_target - scale * (rotation @ mean_source)
    return scale, rotation, translation


def mean_rotation(rotations: list[np.ndarray]) -> np.ndarray:
    accumulator = np.zeros((3, 3), dtype=float)
    for rotation in rotations:
        accumulator += rotation
    u, _, vh = np.linalg.svd(accumulator)
    mean_rot = u @ vh
    if np.linalg.det(mean_rot) < 0:
        u[:, -1] *= -1.0
        mean_rot = u @ vh
    return mean_rot


def summarize_metric(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "min": float(np.min(array)),
        "median": float(np.median(array)),
        "mean": float(np.mean(array)),
        "max": float(np.max(array)),
    }


def format_summary_line(name: str, values: list[float], unit: str) -> str:
    stats = summarize_metric(values)
    return (
        f"{name}: min={stats['min']:.6f}{unit} median={stats['median']:.6f}{unit} "
        f"mean={stats['mean']:.6f}{unit} max={stats['max']:.6f}{unit}"
    )


def print_worst(title: str, rows: list[dict], key: str, top_k: int) -> None:
    print(title)
    ranking = sorted(rows, key=lambda row: row[key], reverse=True)[:top_k]
    for row in ranking:
        print(
            f"  {row['image_name']}: {key}={row[key]:.6f} "
            f"center_error_m={row['center_error_m']:.6f} "
            f"rot_error_deg={row['rot_error_deg']:.6f} "
            f"residual_after_fixed_cam_deg={row['residual_after_fixed_cam_deg']:.6f}"
        )


def main():
    args = parse_args()
    prefixes = tuple(
        prefix.strip() for prefix in args.image_prefixes.split(",") if prefix.strip()
    )
    poses = parse_poses_optimized(args.poses_file)
    _, images, _ = read_model(str(args.model_dir))

    shared_rows = []
    source_centers = []
    target_centers = []
    for image in images.values():
        if prefixes and not image.name.startswith(prefixes):
            continue
        pose = poses.get(image.name)
        if pose is None:
            continue

        rotation_w2c_model = qvec2rotmat(image.qvec)
        center_model = camera_center_from_w2c(rotation_w2c_model, image.tvec)
        source_centers.append(center_model)
        target_centers.append(pose["center"])
        shared_rows.append(
            {
                "image_name": image.name,
                "prefix": image.name.split("/", 1)[0],
                "rotation_c2w_model": rotation_w2c_model.T,
                "rotation_c2w_pose": pose["rotation_c2w"],
                "center_model": center_model,
                "center_pose": pose["center"],
            }
        )

    if len(shared_rows) < 3:
        raise RuntimeError(
            "Need at least 3 shared images between poses_optimized.txt and the model."
        )

    scale, rotation_align, translation_align = estimate_similarity_transform(
        np.asarray(source_centers, dtype=float), np.asarray(target_centers, dtype=float)
    )

    camera_residuals = []
    for row in shared_rows:
        center_aligned = scale * (rotation_align @ row["center_model"]) + translation_align
        rotation_c2w_model_aligned = rotation_align @ row["rotation_c2w_model"]
        camera_residual = rotation_c2w_model_aligned.T @ row["rotation_c2w_pose"]
        row["center_error_m"] = float(np.linalg.norm(center_aligned - row["center_pose"]))
        row["rot_error_deg"] = rotation_angle_deg(camera_residual)
        row["camera_residual"] = camera_residual
        camera_residuals.append(camera_residual)

    fixed_camera_rotation = mean_rotation(camera_residuals)

    for row in shared_rows:
        corrected_residual = row["camera_residual"] @ fixed_camera_rotation.T
        row["residual_after_fixed_cam_deg"] = rotation_angle_deg(corrected_residual)

    output_lines = [
        f"poses_file: {args.poses_file}",
        f"model_dir: {args.model_dir}",
        f"image_prefixes: {','.join(prefixes) if prefixes else '<all>'}",
        f"shared_images: {len(shared_rows)}",
        f"sim3_scale: {scale:.9f}",
        "sim3_rotation:",
        np.array2string(rotation_align, precision=6, suppress_small=True),
        f"sim3_translation: {np.array2string(translation_align, precision=6, suppress_small=True)}",
        format_summary_line("center_error_m", [row["center_error_m"] for row in shared_rows], "m"),
        format_summary_line("rot_error_deg", [row["rot_error_deg"] for row in shared_rows], "deg"),
        (
            "Interpretation: rot_error_deg is the orientation mismatch after only world-frame Sim(3) "
            "alignment."
        ),
        "fixed_camera_rotation:",
        np.array2string(fixed_camera_rotation, precision=6, suppress_small=True),
        f"fixed_camera_rotation_angle_deg: {rotation_angle_deg(fixed_camera_rotation):.6f}",
        format_summary_line(
            "residual_after_fixed_cam_deg",
            [row["residual_after_fixed_cam_deg"] for row in shared_rows],
            "deg",
        ),
        (
            "Interpretation: residual_after_fixed_cam_deg measures how much mismatch remains if a "
            "single global camera-frame rotation is fitted. Low values mean the two pose sources differ "
            "mostly by a fixed camera convention."
        ),
    ]

    prefixes_present = sorted({row["prefix"] for row in shared_rows})
    for prefix in prefixes_present:
        rows = [row for row in shared_rows if row["prefix"] == prefix]
        output_lines.append(f"prefix: {prefix}")
        output_lines.append(
            format_summary_line(
                f"  {prefix}_center_error_m",
                [row["center_error_m"] for row in rows],
                "m",
            )
        )
        output_lines.append(
            format_summary_line(
                f"  {prefix}_rot_error_deg",
                [row["rot_error_deg"] for row in rows],
                "deg",
            )
        )
        output_lines.append(
            format_summary_line(
                f"  {prefix}_residual_after_fixed_cam_deg",
                [row["residual_after_fixed_cam_deg"] for row in rows],
                "deg",
            )
        )

    print("\n".join(output_lines))
    print_worst(
        f"Top {args.top_k} images by rot_error_deg:",
        shared_rows,
        "rot_error_deg",
        args.top_k,
    )
    print_worst(
        f"Top {args.top_k} images by residual_after_fixed_cam_deg:",
        shared_rows,
        "residual_after_fixed_cam_deg",
        args.top_k,
    )

    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        summary_path = args.output_dir / "summary.txt"
        summary_path.write_text("\n".join(output_lines) + "\n", encoding="utf-8")

        csv_path = args.output_dir / "per_image.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "image_name",
                    "prefix",
                    "center_error_m",
                    "rot_error_deg",
                    "residual_after_fixed_cam_deg",
                ],
            )
            writer.writeheader()
            for row in sorted(shared_rows, key=lambda row: row["image_name"]):
                writer.writerow(
                    {
                        "image_name": row["image_name"],
                        "prefix": row["prefix"],
                        "center_error_m": f"{row['center_error_m']:.9f}",
                        "rot_error_deg": f"{row['rot_error_deg']:.9f}",
                        "residual_after_fixed_cam_deg": f"{row['residual_after_fixed_cam_deg']:.9f}",
                    }
                )
        print(f"Wrote summary to {summary_path}")
        print(f"Wrote per-image CSV to {csv_path}")


if __name__ == "__main__":
    main()