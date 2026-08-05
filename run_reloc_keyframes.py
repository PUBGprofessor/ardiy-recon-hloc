import argparse
import csv
from pathlib import Path

import numpy as np

DEFAULT_ROOT_DIR = Path('.')
DEFAULT_FILE_A = Path("keyframe_trajectory.txt")
DEFAULT_FILE_B = Path("poses_optimized.txt")
DEFAULT_OUTPUT = Path("map_aligns.csv")


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description=(
			"Compare poses from keyframe_trajectory.txt (file A) and "
			"poses_optimized.txt (file B) under root_dir, then export align "
			"transforms for images that exist in both files."
		)
	)
	parser.add_argument("--root_dir", type=Path, default=DEFAULT_ROOT_DIR)
	parser.add_argument("--file_a", type=Path, default=DEFAULT_FILE_A)
	parser.add_argument("--file_b", type=Path, default=DEFAULT_FILE_B)
	parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
	return parser.parse_args()


def _resolve_from_root(root_dir: Path, relative_path: Path, arg_name: str) -> Path:
	if relative_path.is_absolute():
		raise ValueError(
			f"{arg_name} must be a path relative to --root_dir, got absolute path: {relative_path}"
		)
	return (root_dir / relative_path).resolve()


def _skew_to_vector(matrix: np.ndarray) -> np.ndarray:
	return np.array([matrix[2, 1], matrix[0, 2], matrix[1, 0]], dtype=float)


def rotation_matrix_to_rvec(rotation: np.ndarray) -> np.ndarray:
	trace = float(np.trace(rotation))
	cos_theta = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
	theta = float(np.arccos(cos_theta))

	if theta < 1e-12:
		return np.zeros(3, dtype=float)

	if np.pi - theta < 1e-6:
		r_plus = (rotation + np.eye(3, dtype=float)) / 2.0
		axis = np.array(
			[
				np.sqrt(max(r_plus[0, 0], 0.0)),
				np.sqrt(max(r_plus[1, 1], 0.0)),
				np.sqrt(max(r_plus[2, 2], 0.0)),
			],
			dtype=float,
		)
		if axis[0] > 1e-6:
			axis[1] = np.copysign(axis[1], rotation[0, 1] + rotation[1, 0])
			axis[2] = np.copysign(axis[2], rotation[0, 2] + rotation[2, 0])
		elif axis[1] > 1e-6:
			axis[2] = np.copysign(axis[2], rotation[1, 2] + rotation[2, 1])
		else:
			axis = np.array([1.0, 0.0, 0.0], dtype=float)
		axis_norm = np.linalg.norm(axis)
		if axis_norm < 1e-12:
			axis = np.array([1.0, 0.0, 0.0], dtype=float)
		else:
			axis = axis / axis_norm
		return axis * theta

	skew = (rotation - rotation.T) / (2.0 * np.sin(theta))
	axis = _skew_to_vector(skew)
	return axis * theta


def _to_pose_matrix_4x4(pose_3x4: np.ndarray) -> np.ndarray:
	pose_4x4 = np.eye(4, dtype=float)
	pose_4x4[:3, :4] = pose_3x4
	return pose_4x4


def _invert_se3(transform: np.ndarray) -> np.ndarray:
	rotation = transform[:3, :3]
	translation = transform[:3, 3]
	inverse = np.eye(4, dtype=float)
	inverse[:3, :3] = rotation.T
	inverse[:3, 3] = -(rotation.T @ translation)
	return inverse


def _read_non_empty_lines(path: Path) -> list[str]:
	lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
	return [line for line in lines if line]


def parse_keyframe_trajectory(file_a: Path) -> dict[str, dict[str, np.ndarray | str]]:
	lines = _read_non_empty_lines(file_a)
	if len(lines) % 2 != 0:
		raise ValueError(
			f"Unexpected format in {file_a}: expected 2 lines per image (name + 12 pose values)."
		)

	entries: dict[str, dict[str, np.ndarray | str]] = {}
	for idx in range(0, len(lines), 2):
		image_name_raw = lines[idx]
		pose_values = np.fromstring(lines[idx + 1], sep=" ", dtype=float)
		if pose_values.size != 12:
			raise ValueError(
				f"Invalid 3x4 pose for image {image_name_raw} in {file_a}: got {pose_values.size} values."
			)
		image_key = Path(image_name_raw).name
		entries[image_key] = {
			"name_raw": image_name_raw,
			"pose_3x4": pose_values.reshape(3, 4),
		}
	return entries


def parse_poses_optimized(file_b: Path) -> dict[str, dict[str, np.ndarray | str]]:
	lines = _read_non_empty_lines(file_b)
	if len(lines) % 3 != 0:
		raise ValueError(
			f"Unexpected format in {file_b}: expected 3 lines per image (name + intrinsics + 12 pose values)."
		)

	entries: dict[str, dict[str, np.ndarray | str]] = {}
	for idx in range(0, len(lines), 3):
		image_name_raw = lines[idx]
		intrinsics = np.fromstring(lines[idx + 1], sep=" ", dtype=float)
		pose_values = np.fromstring(lines[idx + 2], sep=" ", dtype=float)

		if intrinsics.size != 9:
			raise ValueError(
				f"Invalid intrinsics for image {image_name_raw} in {file_b}: got {intrinsics.size} values."
			)
		if pose_values.size != 12:
			raise ValueError(
				f"Invalid 3x4 pose for image {image_name_raw} in {file_b}: got {pose_values.size} values."
			)

		image_key = Path(image_name_raw).name
		entries[image_key] = {
			"name_raw": image_name_raw,
			"intrinsics_3x3": intrinsics.reshape(3, 3),
			"pose_3x4": pose_values.reshape(3, 4),
		}
	return entries


def compare_common_poses(
	poses_a: dict[str, dict[str, np.ndarray | str]],
	poses_b: dict[str, dict[str, np.ndarray | str]],
) -> list[dict[str, float | str]]:
	common_keys = sorted(set(poses_a.keys()) & set(poses_b.keys()))
	rows: list[dict[str, float | str]] = []

	for key in common_keys:
		pose_a = poses_a[key]["pose_3x4"]
		pose_b = poses_b[key]["pose_3x4"]
		if not isinstance(pose_a, np.ndarray) or not isinstance(pose_b, np.ndarray):
			continue

		transform_a = _to_pose_matrix_4x4(pose_a)
		transform_b = _to_pose_matrix_4x4(pose_b)
		align = transform_b @ _invert_se3(transform_a)
		align_rotation = align[:3, :3]
		align_translation = align[:3, 3]
		align_rvec = rotation_matrix_to_rvec(align_rotation)

		rows.append(
			{
				"image_key": key,
				"image_name_a": str(poses_a[key]["name_raw"]),
				"image_name_b": str(poses_b[key]["name_raw"]),
				"align_rx": float(align_rvec[0]),
				"align_ry": float(align_rvec[1]),
				"align_rz": float(align_rvec[2]),
				"align_tx": float(align_translation[0]),
				"align_ty": float(align_translation[1]),
				"align_tz": float(align_translation[2]),
			}
		)

	return rows


def write_output(rows: list[dict[str, float | str]], output_path: Path) -> None:
	output_path.parent.mkdir(parents=True, exist_ok=True)

	fieldnames = [
		"image_key",
		"image_name_a",
		"image_name_b",
		"align_rx",
		"align_ry",
		"align_rz",
		"align_tx",
		"align_ty",
		"align_tz",
	]

	with output_path.open("w", encoding="utf-8", newline="") as f:
		writer = csv.DictWriter(f, fieldnames=fieldnames)
		writer.writeheader()
		writer.writerows(rows)


def main() -> None:
	args = parse_args()
	root_dir = args.root_dir.resolve()
	file_a = _resolve_from_root(root_dir, args.file_a, "--file_a")
	file_b = _resolve_from_root(root_dir, args.file_b, "--file_b")
	output = _resolve_from_root(root_dir, args.output, "--output")

	poses_a = parse_keyframe_trajectory(file_a)
	poses_b = parse_poses_optimized(file_b)

	common = set(poses_a.keys()) & set(poses_b.keys())
	only_a = set(poses_a.keys()) - set(poses_b.keys())
	only_b = set(poses_b.keys()) - set(poses_a.keys())

	rows = compare_common_poses(poses_a, poses_b)
	write_output(rows, output)

	print(f"Root dir: {root_dir}")
	print(f"File A path: {file_a}")
	print(f"File B path: {file_b}")
	print(f"File A images: {len(poses_a)}")
	print(f"File B images: {len(poses_b)}")
	print(f"Common images: {len(common)}")
	print(f"Only in A: {len(only_a)}")
	print(f"Only in B: {len(only_b)}")
	print(f"Wrote pose differences to: {output}")

	if rows:
		rvec_values = np.asarray(
			[
				[row["align_rx"], row["align_ry"], row["align_rz"]]
				for row in rows
			],
			dtype=float,
		)
		tvec_values = np.asarray(
			[
				[row["align_tx"], row["align_ty"], row["align_tz"]]
				for row in rows
			],
			dtype=float,
		)
		mean_rvec = np.mean(rvec_values, axis=0)
		mean_tvec = np.mean(tvec_values, axis=0)

		print(
			"Mean align rvec (rad): "
			f"[{mean_rvec[0]:.9f}, {mean_rvec[1]:.9f}, {mean_rvec[2]:.9f}]"
		)
		print(
			"Mean align tvec: "
			f"[{mean_tvec[0]:.9f}, {mean_tvec[1]:.9f}, {mean_tvec[2]:.9f}]"
		)


if __name__ == "__main__":
	main()
