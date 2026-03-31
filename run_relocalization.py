import argparse
import json
import os
import pickle
import sys
from pathlib import Path

if os.name == "nt":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from hloc import extract_features, logger, match_features
from hloc import pairs_from_poses, pairs_from_retrieval
from hloc.utils.io import read_image
from hloc.utils.read_write_model import (
    Camera as ColmapCamera,
    Image as ColmapImage,
    qvec2rotmat,
    read_model,
    rotmat2qvec,
    write_model,
)

import os
os.chdir('F:/dev2/prjs1/data1/office2')
#sys.argv='xxx build-map --image_dir  ./  --image_list ./scan1/image_list.txt --poses_file ./_scanx/poses_optimized.txt  --outputs ./hloc_map'.split()
#sys.argv='xxx build-map --image_dir  ./  --image_list ./scan1/image_list.txt --outputs ./hloc_map --build_mode hloc-native --native_pairing exhaustive --overwrite'.split()
#sys.argv='xxx localize-query --map_dir ./hloc_map --query_image scan2/color/000001.jpg --query_image_root ./ --camera_model PINHOLE --camera_params 644.565674,643.857605,657.600647,411.930359'.split()
#sys.argv='xxx localize-list --map_dir ./hloc_map_native --query_root ./ --train_list ./scan1/image_list.txt --query_list ./scan2/image_list.txt --camera_model PINHOLE --camera_params 644.565674,643.857605,657.600647,411.930359 --poses_file ./hloc_map_native/poses_optimized.txt'.split()
#sys.argv='xxx visualize-matches --results ./hloc_map_native/localization/image_list_1/image_list_1_poses.txt  --image_root ./  --query_name scan2/color/000001.jpg'.split()
sys.argv='xxx visualize-matches --results ./hloc_map_native/localization/image_list/image_list_poses.txt  --image_root ./  --query_list scan2/image_list_1.txt'.split()

def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a train reference map and localize query images."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser(
        "build-map",
        help="Extract MegaLoc descriptors for train images and optionally triangulate a reference map.",
    )
    build_parser.add_argument(
        "--image_dir",
        type=Path,
        required=True,
        help="Scene root directory. image_list entries are interpreted relative to this path.",
    )
    build_parser.add_argument(
        "--image_list",
        type=Path,
        required=True,
        help="Text file listing train images relative to --image_dir, one per line.",
    )
    build_parser.add_argument("--outputs", type=Path, required=True)
    build_parser.add_argument(
        "--poses_file",
        type=Path,
        help="Pose file in poses_optimized.txt format. If set, build-map does not require --reference_model.",
    )
    build_parser.add_argument(
        "--reference_model",
        type=Path,
        help="Optional COLMAP model with known train poses. Kept for compatibility when --poses_file is not provided.",
    )
    build_parser.add_argument(
        "--build_mode",
        type=str,
        default="posed",
        choices=["posed", "hloc-native"],
        help="Build the reference map either from known poses ('posed') or with hloc's native incremental SfM ('hloc-native').",
    )
    build_parser.add_argument(
        "--retrieval_conf",
        type=str,
        default="megaloc",
        choices=sorted(extract_features.confs.keys()),
    )
    build_parser.add_argument(
        "--local_feature_conf",
        type=str,
        default="superpoint_inloc",
        choices=sorted(extract_features.confs.keys()),
    )
    build_parser.add_argument(
        "--matcher_conf",
        type=str,
        default="superglue",
        choices=sorted(match_features.confs.keys()),
    )
    build_parser.add_argument("--num_pose_pairs", type=int, default=20)
    build_parser.add_argument("--rotation_threshold", type=float, default=30.0)
    build_parser.add_argument(
        "--native_pairing",
        type=str,
        default="exhaustive",
        choices=["exhaustive", "retrieval"],
        help="Pair generation strategy for --build_mode hloc-native.",
    )
    build_parser.add_argument(
        "--native_num_matched",
        type=int,
        default=20,
        help="Number of retrieved neighbors when --native_pairing retrieval is used.",
    )
    build_parser.add_argument("--overwrite", action="store_true")

    loc_parser = subparsers.add_parser(
        "localize-query",
        help="Localize one query image against a train map built from known poses.",
    )
    loc_parser.add_argument("--map_dir", type=Path, required=True)
    loc_parser.add_argument("--query_image", type=Path, required=True)
    loc_parser.add_argument(
        "--query_image_root",
        type=Path,
        help="Root used to name the query in H5. Set this when basename collisions are possible.",
    )
    loc_parser.add_argument(
        "--retrieval_conf",
        type=str,
        default="megaloc",
        choices=sorted(extract_features.confs.keys()),
    )
    loc_parser.add_argument(
        "--local_feature_conf",
        type=str,
        default="superpoint_inloc",
        choices=sorted(extract_features.confs.keys()),
    )
    loc_parser.add_argument(
        "--matcher_conf",
        type=str,
        default="superglue",
        choices=sorted(match_features.confs.keys()),
    )
    loc_parser.add_argument("--num_retrieval", type=int, default=20)
    loc_parser.add_argument("--ransac_thresh", type=float, default=12.0)
    loc_parser.add_argument(
        "--camera_model",
        type=str,
        help="COLMAP camera model, e.g. PINHOLE or SIMPLE_RADIAL.",
    )
    loc_parser.add_argument(
        "--camera_params",
        type=str,
        help="Comma-separated COLMAP camera parameters matching --camera_model.",
    )
    loc_parser.add_argument(
        "--fallback_to_nearest_db",
        action="store_true",
        help="If PnP fails, write the nearest retrieved database pose instead of skipping the query.",
    )
    loc_parser.add_argument(
        "--results",
        type=Path,
        help="Output file for the estimated pose. Defaults to <map_dir>/localization/<query_stem>.txt.",
    )
    loc_parser.add_argument("--overwrite", action="store_true")

    list_parser = subparsers.add_parser(
        "localize-list",
        help="Localize a list of query images and optionally evaluate against poses_optimized.txt.",
    )
    list_parser.add_argument("--map_dir", type=Path, required=True)
    list_parser.add_argument(
        "--query_root",
        type=Path,
        required=True,
        help="Root directory that query_list entries are relative to.",
    )
    list_parser.add_argument(
        "--query_list",
        type=Path,
        required=True,
        help="Text file listing query images relative to --query_root.",
    )
    list_parser.add_argument(
        "--train_list",
        type=Path,
        help=(
            "Optional text file listing the subset of map images to use as the train database for retrieval and matching. "
            "Image names must match the map image names."
        ),
    )
    list_parser.add_argument(
        "--retrieval_conf",
        type=str,
        default="megaloc",
        choices=sorted(extract_features.confs.keys()),
    )
    list_parser.add_argument(
        "--local_feature_conf",
        type=str,
        default="superpoint_inloc",
        choices=sorted(extract_features.confs.keys()),
    )
    list_parser.add_argument(
        "--matcher_conf",
        type=str,
        default="superglue",
        choices=sorted(match_features.confs.keys()),
    )
    list_parser.add_argument("--num_retrieval", type=int, default=20)
    list_parser.add_argument("--ransac_thresh", type=float, default=12.0)
    list_parser.add_argument(
        "--camera_model",
        type=str,
        help="COLMAP camera model, e.g. PINHOLE or SIMPLE_RADIAL.",
    )
    list_parser.add_argument(
        "--camera_params",
        type=str,
        help="Comma-separated COLMAP camera parameters matching --camera_model.",
    )
    list_parser.add_argument(
        "--fallback_to_nearest_db",
        action="store_true",
        help="If PnP fails, write the nearest retrieved database pose instead of skipping the query.",
    )
    list_parser.add_argument(
        "--poses_file",
        type=Path,
        help="Optional poses_optimized.txt used to evaluate localization error.",
    )
    list_parser.add_argument(
        "--eval_model",
        type=Path,
        help=(
            "Optional COLMAP model used as evaluation ground truth. "
            "Prefer this when poses_optimized.txt uses an incompatible camera convention."
        ),
    )
    list_parser.add_argument(
        "--eval_mode",
        type=str,
        default="direct",
        choices=["auto", "direct", "align"],
        help=(
            "Evaluation mode. 'direct' compares poses in the same world frame, while "
            "'align' first aligns the SfM map to poses_optimized.txt with a Sim3 transform. "
            "Use 'align' for --build_mode hloc-native maps."
        ),
    )
    list_parser.add_argument(
        "--eval_translation_thresh",
        type=float,
        default=0.25,
        help="Translation threshold in meters for accuracy computation.",
    )
    list_parser.add_argument(
        "--eval_rotation_thresh",
        type=float,
        default=5.0,
        help="Rotation threshold in degrees for accuracy computation.",
    )
    list_parser.add_argument(
        "--results",
        type=Path,
        help="Output file for estimated poses. Defaults to <map_dir>/localization/<query_list_stem>/<query_list_stem>_poses.txt.",
    )
    list_parser.add_argument("--overwrite", action="store_true")

    vis_parser = subparsers.add_parser(
        "visualize-matches",
        help="Export 2D match visualizations for one localized query from the localization logs.",
    )
    vis_parser.add_argument(
        "--results",
        type=Path,
        required=True,
        help="Pose result file whose companion *_logs.pkl file will be used.",
    )
    vis_parser.add_argument(
        "--image_root",
        type=Path,
        required=True,
        help="Root directory containing both query and database images.",
    )
    query_input_group = vis_parser.add_mutually_exclusive_group(required=True)
    query_input_group.add_argument(
        "--query_name",
        type=str,
        help="Query image name in the logs. Basename is also accepted if unique.",
    )
    query_input_group.add_argument(
        "--query_list",
        type=Path,
        help="Optional text file listing query image names to visualize in batch.",
    )
    vis_parser.add_argument(
        "--reference_sfm",
        type=Path,
        help="Optional reference SfM model. Defaults to <results>/../../sfm_reference when available.",
    )
    vis_parser.add_argument(
        "--output_dir",
        type=Path,
        help=(
            "Directory where PNG visualizations are written. Defaults to <results_parent>/visualizations/<query_stem>/ "
            "for single-query mode and <results_parent>/visualizations/<query_list_stem>/<query_stem>/ for batch mode."
        ),
    )
    vis_parser.add_argument("--top_k_db", type=int, default=3)
    vis_parser.add_argument("--dpi", type=int, default=120)
    return parser.parse_args()


def make_query_name(query_image: Path, query_root: Path | None) -> tuple[Path, str]:
    if query_root is None:
        query_root = query_image.parent
    query_root = query_root.resolve()
    query_image = query_image.resolve()
    try:
        query_name = query_image.relative_to(query_root).as_posix()
    except ValueError as exc:
        raise ValueError(
            f"Query image {query_image} is not under query root {query_root}."
        ) from exc
    return query_root, query_name


def build_camera(query_image: Path, camera_model: str | None, camera_params: str | None):
    import pycolmap

    if camera_model or camera_params:
        if not camera_model or not camera_params:
            raise ValueError("Both --camera_model and --camera_params must be set together.")
        image = read_image(query_image)
        height, width = image.shape[:2]
        params = np.array([float(value) for value in camera_params.split(",")])
        return pycolmap.Camera(
            model=camera_model,
            width=width,
            height=height,
            params=params,
        )
    return pycolmap.infer_camera_from_image(query_image)


def write_query_list(query_list_path: Path, query_name: str, camera):
    query_list_path.parent.mkdir(parents=True, exist_ok=True)
    params = " ".join(map(str, camera.params.tolist()))
    line = f"{query_name} {camera.model} {camera.width} {camera.height} {params}\n"
    query_list_path.write_text(line, encoding="utf-8")


def write_query_list_entries(query_list_path: Path, query_entries):
    query_list_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for query_name, camera in query_entries:
        params = " ".join(map(str, camera.params.tolist()))
        lines.append(
            f"{query_name} {camera.model} {camera.width} {camera.height} {params}"
        )
    query_list_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_image_list(image_list_path: Path, image_names: list[str]) -> None:
    image_list_path.parent.mkdir(parents=True, exist_ok=True)
    image_list_path.write_text("\n".join(image_names) + "\n", encoding="utf-8")


def read_pose_line(results_path: Path) -> str:
    lines = [line.strip() for line in results_path.read_text(encoding="utf-8").splitlines()]
    lines = [line for line in lines if line]
    if not lines:
        raise RuntimeError(f"No pose was written to {results_path}.")
    return lines[0]


def read_image_names(image_list_path: Path) -> list[str]:
    image_names = []
    for line in image_list_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        image_names.append(line.split()[0])
    if not image_names:
        raise ValueError(f"No image names found in {image_list_path}.")
    return image_names


def resolve_query_name_in_logs(logs: dict, query_name: str) -> str:
    loc_logs = logs.get("loc", {})
    if query_name in loc_logs:
        return query_name

    matches = [name for name in loc_logs if Path(name).name == query_name]
    if len(matches) == 1:
        return matches[0]

    camera_matches = [
        name for name in loc_logs if "/".join(name.split("/")[-2:]) == query_name
    ]
    if len(camera_matches) == 1:
        return camera_matches[0]

    if not matches and not camera_matches:
        raise KeyError(f"Query {query_name} was not found in localization logs.")
    raise KeyError(
        f"Query name {query_name} is ambiguous in localization logs. Use the full relative path."
    )


def filter_pairs_to_image_list(pairs_path: Path, valid_names: set[str]) -> None:
    kept_pairs = []
    total_pairs = 0
    for line in pairs_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        total_pairs += 1
        name0, name1 = line.split()
        if name0 in valid_names and name1 in valid_names:
            kept_pairs.append(f"{name0} {name1}")

    if not kept_pairs:
        raise RuntimeError(
            "No valid training pairs remain after filtering to --image_list. "
            "Check that --reference_model image names match the entries in --image_list."
        )

    pairs_path.write_text("\n".join(kept_pairs) + "\n", encoding="utf-8")
    removed_pairs = total_pairs - len(kept_pairs)
    if removed_pairs > 0:
        logger.info(
            "Filtered %d/%d train pairs that reference images outside --image_list.",
            removed_pairs,
            total_pairs,
        )


def parse_poses_optimized(poses_file: Path) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
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
        translation_c2w = pose_c2w[:, 3]
        poses[image_name] = (intrinsics, rotation_c2w, translation_c2w)
    return poses


def filter_image_names_to_available_poses(
    image_names: list[str], poses_file: Path
) -> tuple[list[str], dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    poses = parse_poses_optimized(poses_file)
    filtered_names = [image_name for image_name in image_names if image_name in poses]
    skipped_names = [image_name for image_name in image_names if image_name not in poses]

    if skipped_names:
        logger.info(
            "Skipping %d train images missing from %s.",
            len(skipped_names),
            poses_file,
        )

    if not filtered_names:
        raise RuntimeError(
            f"No train images from --image_list were found in {poses_file}."
        )

    return filtered_names, poses


def create_reference_model_from_poses(
    image_dir: Path,
    image_names: list[str],
    poses_file: Path,
    output_dir: Path,
    poses: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] | None = None,
) -> Path:
    if poses is None:
        poses = parse_poses_optimized(poses_file)
    output_dir.mkdir(parents=True, exist_ok=True)

    cameras = {}
    images = {}
    points3d = {}
    camera_ids = {}

    for image_id, image_name in enumerate(image_names, start=1):
        intrinsics, rotation_c2w, translation_c2w = poses[image_name]
        image = read_image(image_dir / image_name)
        height, width = image.shape[:2]
        params = np.array(
            [intrinsics[0], intrinsics[4], intrinsics[2], intrinsics[5]], dtype=float
        )
        camera_key = (width, height, *(np.round(params, 6).tolist()))
        camera_id = camera_ids.get(camera_key)
        if camera_id is None:
            camera_id = len(camera_ids) + 1
            camera_ids[camera_key] = camera_id
            cameras[camera_id] = ColmapCamera(
                id=camera_id,
                model="PINHOLE",
                width=width,
                height=height,
                params=params,
            )

        rotation_w2c = rotation_c2w.T
        translation_w2c = -(rotation_w2c @ translation_c2w)
        images[image_id] = ColmapImage(
            id=image_id,
            qvec=rotmat2qvec(rotation_w2c),
            tvec=translation_w2c,
            camera_id=camera_id,
            name=image_name,
            xys=np.zeros((0, 2), dtype=float),
            point3D_ids=np.zeros((0,), dtype=int),
        )

    write_model(cameras, images, points3d, path=str(output_dir), ext=".bin")
    logger.info(
        "Reference pose model created from %s with %d images and %d cameras.",
        poses_file,
        len(images),
        len(cameras),
    )
    return output_dir


def parse_result_poses(results_path: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    poses = {}
    for line in results_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 8:
            raise ValueError(f"Unexpected pose line in {results_path}: {line}")
        image_name = parts[0]
        qvec = np.array([float(value) for value in parts[1:5]], dtype=float)
        tvec = np.array([float(value) for value in parts[5:8]], dtype=float)
        poses[image_name] = (qvec, tvec)
    return poses


def compute_rotation_error_deg(rotation_a: np.ndarray, rotation_b: np.ndarray) -> float:
    cos_angle = np.clip((np.trace(rotation_a @ rotation_b.T) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.rad2deg(np.arccos(cos_angle)))


def load_gt_poses_from_poses_file(
    poses_file: Path,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    gt_poses = {}
    for image_name, (_, rotation_c2w, translation_c2w) in parse_poses_optimized(
        poses_file
    ).items():
        gt_poses[image_name] = (rotation_c2w.T, translation_c2w)
    return gt_poses


def load_gt_poses_from_model(
    model_dir: Path,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    _, images, _ = read_model(str(model_dir))
    gt_poses = {}
    for image in images.values():
        rotation_w2c = qvec2rotmat(image.qvec)
        center = camera_center_from_w2c(rotation_w2c, image.tvec)
        gt_poses[image.name] = (rotation_w2c, center)
    return gt_poses


def load_evaluation_ground_truth(
    poses_file: Path | None,
    eval_model: Path | None,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], str]:
    if eval_model is not None:
        return load_gt_poses_from_model(eval_model), str(eval_model)
    if poses_file is not None:
        return load_gt_poses_from_poses_file(poses_file), str(poses_file)
    raise ValueError("Either --poses_file or --eval_model must be provided for evaluation.")


def read_build_info(map_dir: Path) -> dict | None:
    build_info_path = map_dir / "build_info.json"
    if not build_info_path.exists():
        return None
    return json.loads(build_info_path.read_text(encoding="utf-8"))


def camera_center_from_w2c(rotation_w2c: np.ndarray, translation_w2c: np.ndarray) -> np.ndarray:
    return -(rotation_w2c.T @ translation_w2c)


def estimate_similarity_transform(
    source_points: np.ndarray, target_points: np.ndarray
) -> tuple[float, np.ndarray, np.ndarray]:
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


def resolve_evaluation_mode(
    map_dir: Path,
    requested_mode: str,
    eval_model: Path | None = None,
) -> str:
    if requested_mode != "auto":
        return requested_mode

    if eval_model is not None:
        return "align"

    build_info = read_build_info(map_dir)
    if build_info is not None:
        build_mode = build_info.get("build_mode")
        if build_mode == "hloc-native":
            return "align"
        if build_mode == "posed":
            return "direct"

    if (map_dir / "reference_from_poses").exists():
        return "direct"
    return "align"


def estimate_map_to_gt_alignment(
    map_dir: Path,
    gt_poses: dict[str, tuple[np.ndarray, np.ndarray]],
    gt_source: str,
) -> tuple[float, np.ndarray, np.ndarray, int]:
    _, reference_images, _ = read_model(str(map_dir / "sfm_reference"))

    source_centers = []
    target_centers = []
    for image in reference_images.values():
        if image.name not in gt_poses:
            continue
        _, translation_c2w_gt = gt_poses[image.name]
        rotation_w2c_ref = qvec2rotmat(image.qvec)
        center_ref = camera_center_from_w2c(rotation_w2c_ref, image.tvec)
        source_centers.append(center_ref)
        target_centers.append(translation_c2w_gt)

    if len(source_centers) < 3:
        raise RuntimeError(
            "Need at least 3 train images shared by sfm_reference and "
            f"{gt_source} "
            "to evaluate in aligned mode."
        )

    source_array = np.asarray(source_centers, dtype=float)
    target_array = np.asarray(target_centers, dtype=float)
    scale, rotation, translation = estimate_similarity_transform(
        source_array, target_array
    )
    return scale, rotation, translation, len(source_centers)


def summarize_localization_logs(results_path: Path):
    logs_path = Path(f"{results_path}_logs.pkl")
    if not logs_path.exists():
        return None

    with logs_path.open("rb") as handle:
        logs = pickle.load(handle)

    total_queries = len(logs.get("loc", {}))
    successful_pnp = 0
    fallback_queries = 0
    inlier_counts = []

    for loc in logs.get("loc", {}).values():
        if loc.get("covisibility_clustering"):
            best_ret = None
            for cluster_log in loc.get("log_clusters", []):
                ret = cluster_log.get("PnP_ret")
                if ret is None:
                    continue
                if best_ret is None or ret.get("num_inliers", -1) > best_ret.get(
                    "num_inliers", -1
                ):
                    best_ret = ret
            if best_ret is not None:
                successful_pnp += 1
                inlier_counts.append(int(best_ret.get("num_inliers", 0)))
            continue

        ret = loc.get("PnP_ret")
        if ret is not None:
            successful_pnp += 1
            inlier_counts.append(int(ret.get("num_inliers", 0)))
        if loc.get("used_fallback"):
            fallback_queries += 1

    summary_lines = [
        f"total_queries: {total_queries}",
        f"successful_pnp_queries: {successful_pnp}",
        f"fallback_queries: {fallback_queries}",
    ]
    if inlier_counts:
        summary_lines.extend(
            [
                f"median_inliers: {float(np.median(inlier_counts)):.2f}",
                f"mean_inliers: {float(np.mean(inlier_counts)):.2f}",
                f"min_inliers: {int(np.min(inlier_counts))}",
                f"max_inliers: {int(np.max(inlier_counts))}",
            ]
        )

    summary_path = results_path.parent / f"{results_path.stem}_summary.txt"
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    for line in summary_lines:
        logger.info(line)
    return {
        "total_queries": total_queries,
        "successful_pnp_queries": successful_pnp,
        "fallback_queries": fallback_queries,
        "summary_path": summary_path,
    }


def visualize_matches_from_logs(
    results_path: Path,
    image_root: Path,
    query_name: str,
    reference_sfm: Path | None,
    output_dir: Path | None,
    top_k_db: int,
    dpi: int,
):
    import matplotlib.pyplot as plt
    import pycolmap

    from hloc.utils.io import read_image as read_image_file
    from hloc.utils.viz import add_text, cm_RdGn, plot_images, plot_matches, save_plot

    logs_path = Path(f"{results_path}_logs.pkl")
    if not logs_path.exists():
        raise FileNotFoundError(logs_path)

    with logs_path.open("rb") as handle:
        logs = pickle.load(handle)

    resolved_query_name = resolve_query_name_in_logs(logs, query_name)
    loc = logs["loc"][resolved_query_name]
    if loc.get("covisibility_clustering"):
        best_cluster = loc.get("best_cluster")
        loc = loc["log_clusters"][best_cluster or 0]

    if reference_sfm is None:
        guessed = results_path.parents[2] / "sfm_reference"
        if not guessed.exists():
            raise FileNotFoundError(
                "Could not infer --reference_sfm from --results. Please pass it explicitly."
            )
        reference_sfm = guessed

    reconstruction = pycolmap.Reconstruction(reference_sfm)
    q_image = read_image_file(image_root / resolved_query_name)

    pnp_ret = loc.get("PnP_ret")
    points3d_ids = loc.get("points3D_ids", [])
    keypoints_query = np.asarray(loc.get("keypoints_query", []))
    if pnp_ret is not None and "inlier_mask" in pnp_ret:
        inliers = np.asarray(pnp_ret["inlier_mask"], dtype=bool)
    else:
        inliers = np.zeros(len(points3d_ids), dtype=bool)

    kp_indices, kp_to_3d_to_db = loc["keypoint_index_to_db"]
    db_entries = []
    db_scores = []
    db_match_counts = []
    for db_idx, db_image_id in enumerate(loc["db"]):
        image = reconstruction.images[db_image_id]
        track_pairs = []
        inliers_db = []
        for match_index, (point3d_id, db_indices) in enumerate(kp_to_3d_to_db):
            if db_idx not in db_indices:
                continue
            track = reconstruction.points3D[point3d_id].track
            track = {el.image_id: el.point2D_idx for el in track.elements}
            if db_image_id not in track:
                continue
            query_kp = keypoints_query[match_index]
            db_kp = np.array(image.points2D[track[db_image_id]].xy)
            track_pairs.append((query_kp, db_kp))
            inliers_db.append(bool(inliers[match_index]))
        db_entries.append((image.name, track_pairs, np.asarray(inliers_db, dtype=bool)))
        db_match_counts.append(len(track_pairs))
        db_scores.append(int(np.count_nonzero(inliers_db)))

    if not any(db_match_counts):
        raise RuntimeError(f"No visualizable matches were found for {resolved_query_name}.")

    if np.any(inliers):
        ranking = np.argsort(-np.asarray(db_scores))
    else:
        ranking = np.argsort(-np.asarray(db_match_counts))

    query_stem = Path(resolved_query_name).stem
    output_dir = output_dir or (results_path.parent / "visualizations" / query_stem)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = []
    for rank, db_rank in enumerate(ranking[:top_k_db], start=1):
        db_name, track_pairs, inliers_db = db_entries[db_rank]
        if not track_pairs:
            continue
        kp_q = np.stack([pair[0] for pair in track_pairs], axis=0)
        kp_db = np.stack([pair[1] for pair in track_pairs], axis=0)
        color = cm_RdGn(inliers_db.astype(float)).tolist()
        db_image = read_image_file(image_root / db_name)

        plot_images([q_image, db_image], dpi=dpi)
        plot_matches(kp_q, kp_db, color=color, a=0.35, lw=1.0, ps=6)
        if np.any(inliers):
            text = f"inliers: {int(np.count_nonzero(inliers_db))}/{len(inliers_db)}"
        else:
            text = f"matches: {len(inliers_db)} (PnP failed)"
        add_text(0, text)
        add_text(0, resolved_query_name, pos=(0.01, 0.01), fs=7, lcolor=None, va="bottom")
        add_text(1, db_name, pos=(0.01, 0.01), fs=7, lcolor=None, va="bottom")

        output_path = output_dir / (
            f"{query_stem}_{rank:02d}_{Path(db_name).stem}.png"
        )
        save_plot(output_path, dpi=dpi)
        saved_paths.append(output_path)
        plt.close("all")

    if not saved_paths:
        raise RuntimeError(f"No match visualization could be exported for {resolved_query_name}.")

    logger.info("Saved %d match visualizations to %s", len(saved_paths), output_dir)
    for path in saved_paths:
        logger.info("Visualization: %s", path)
    return saved_paths


def visualize_matches_for_list(
    results_path: Path,
    image_root: Path,
    query_list_path: Path,
    reference_sfm: Path | None,
    output_dir: Path | None,
    top_k_db: int,
    dpi: int,
):
    query_names = read_image_names(query_list_path)
    batch_root = output_dir or (results_path.parent / "visualizations" / query_list_path.stem)
    batch_root.mkdir(parents=True, exist_ok=True)

    all_saved_paths = []
    failures = []
    for query_name in query_names:
        query_output_dir = batch_root / Path(query_name).stem
        try:
            saved_paths = visualize_matches_from_logs(
                results_path,
                image_root,
                query_name,
                reference_sfm,
                query_output_dir,
                top_k_db,
                dpi,
            )
            all_saved_paths.extend(saved_paths)
        except Exception as exc:
            logger.warning("Failed to visualize %s: %s", query_name, exc)
            failures.append((query_name, str(exc)))

    if not all_saved_paths:
        raise RuntimeError(
            f"No match visualizations could be exported for any query in {query_list_path}."
        )

    failure_report = batch_root / f"{query_list_path.stem}_visualization_failures.txt"
    if failures:
        lines = ["query_name error"]
        for query_name, error in failures:
            lines.append(f"{query_name} {error}")
        failure_report.write_text("\n".join(lines) + "\n", encoding="utf-8")
        logger.info("Visualization failures written to %s", failure_report)

    logger.info(
        "Saved %d visualization images for %d/%d queries under %s",
        len(all_saved_paths),
        len(query_names) - len(failures),
        len(query_names),
        batch_root,
    )
    return all_saved_paths


def evaluate_localization_results(
    map_dir: Path,
    results_path: Path,
    poses_file: Path | None,
    eval_model: Path | None,
    translation_thresh: float,
    rotation_thresh: float,
    query_names: list[str] | None = None,
    eval_mode: str = "auto",
):
    gt_poses, gt_source = load_evaluation_ground_truth(poses_file, eval_model)
    estimated_poses = parse_result_poses(results_path)
    resolved_eval_mode = resolve_evaluation_mode(map_dir, eval_mode, eval_model)
    alignment = None
    if resolved_eval_mode == "align":
        alignment = estimate_map_to_gt_alignment(map_dir, gt_poses, gt_source)
        logger.info(
            "Aligned sfm_reference to GT using %d shared train images.",
            alignment[3],
        )

    name_mapping = {}
    if query_names is not None:
        basename_to_full = {}
        camera_name_to_full = {}
        for query_name in query_names:
            basename = Path(query_name).name
            camera_name = "/".join(query_name.split("/")[-2:])
            basename_to_full.setdefault(basename, []).append(query_name)
            camera_name_to_full.setdefault(camera_name, []).append(query_name)
        for basename, full_names in basename_to_full.items():
            if len(full_names) == 1:
                name_mapping[basename] = full_names[0]
        for camera_name, full_names in camera_name_to_full.items():
            if len(full_names) == 1:
                name_mapping[camera_name] = full_names[0]

    evaluated = []

    for image_name, (qvec, tvec) in estimated_poses.items():
        gt_name = image_name
        if gt_name not in gt_poses:
            gt_name = name_mapping.get(image_name, image_name)
        if gt_name not in gt_poses:
            continue
        rotation_w2c_gt, translation_c2w_gt = gt_poses[gt_name]
        rotation_w2c_est = qvec2rotmat(qvec)
        center_est = camera_center_from_w2c(rotation_w2c_est, tvec)
        if alignment is not None:
            scale, rotation_align, translation_align, _ = alignment
            center_est = scale * (rotation_align @ center_est) + translation_align
            rotation_c2w_est = rotation_w2c_est.T
            rotation_c2w_est = rotation_align @ rotation_c2w_est
            rotation_w2c_est = rotation_c2w_est.T
        center_gt = translation_c2w_gt
        translation_error = float(np.linalg.norm(center_est - center_gt))
        rotation_error = compute_rotation_error_deg(rotation_w2c_est, rotation_w2c_gt)
        evaluated.append(
            {
                "estimated_name": image_name,
                "gt_name": gt_name,
                "translation_error_m": translation_error,
                "rotation_error_deg": rotation_error,
            }
        )

    if not evaluated:
        logger.info("No localized queries were found in %s, skipping evaluation.", gt_source)
        return None

    translation_errors = np.array(
        [item["translation_error_m"] for item in evaluated], dtype=float
    )
    rotation_errors = np.array(
        [item["rotation_error_deg"] for item in evaluated], dtype=float
    )
    accuracy = np.mean(
        np.logical_and(
            translation_errors <= translation_thresh,
            rotation_errors <= rotation_thresh,
        )
    )
    summary_lines = [
        f"gt_source: {gt_source}",
        f"eval_mode: {resolved_eval_mode}",
        f"evaluated_queries: {len(evaluated)}",
        f"median_translation_error_m: {np.median(translation_errors):.6f}",
        f"mean_translation_error_m: {np.mean(translation_errors):.6f}",
        f"median_rotation_error_deg: {np.median(rotation_errors):.6f}",
        f"mean_rotation_error_deg: {np.mean(rotation_errors):.6f}",
        (
            "accuracy"
            f" @ {translation_thresh:.3f}m, {rotation_thresh:.3f}deg: {accuracy * 100:.2f}%"
        ),
    ]
    if alignment is not None:
        scale, _, _, num_alignment_images = alignment
        summary_lines.insert(1, f"alignment_images: {num_alignment_images}")
        summary_lines.insert(2, f"alignment_scale: {scale:.6f}")
    metrics_path = results_path.parent / f"{results_path.stem}_metrics.txt"
    error_path = results_path.parent / f"{results_path.stem}_errors.txt"
    metrics_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    error_lines = [
        "estimated_name gt_name translation_error_m rotation_error_deg",
    ]
    for item in sorted(evaluated, key=lambda row: row["translation_error_m"], reverse=True):
        error_lines.append(
            f"{item['estimated_name']} {item['gt_name']} "
            f"{item['translation_error_m']:.6f} {item['rotation_error_deg']:.6f}"
        )
    error_path.write_text("\n".join(error_lines) + "\n", encoding="utf-8")
    for line in summary_lines:
        logger.info(line)
    logger.info("Per-image errors written to %s", error_path)
    return {
        "eval_mode": resolved_eval_mode,
        "evaluated_queries": len(evaluated),
        "median_translation_error_m": float(np.median(translation_errors)),
        "mean_translation_error_m": float(np.mean(translation_errors)),
        "median_rotation_error_deg": float(np.median(rotation_errors)),
        "mean_rotation_error_deg": float(np.mean(rotation_errors)),
        "accuracy": float(accuracy),
        "metrics_path": metrics_path,
        "error_path": error_path,
    }


def localize_queries(
    map_dir: Path,
    query_root: Path,
    query_names: list[str],
    loc_dir: Path,
    results_path: Path,
    retrieval_conf_name: str,
    local_feature_conf_name: str,
    matcher_conf_name: str,
    num_retrieval: int,
    ransac_thresh: float,
    camera_model: str | None,
    camera_params: str | None,
    fallback_to_nearest_db: bool,
    train_list: Path | None = None,
):
    from hloc import localize_sfm

    retrieval_conf = extract_features.confs[retrieval_conf_name]
    feature_conf = extract_features.confs[local_feature_conf_name]
    matcher_conf = match_features.confs[matcher_conf_name]

    train_descriptors = map_dir / f"{retrieval_conf['output']}.h5"
    train_features = map_dir / f"{feature_conf['output']}.h5"
    reference_sfm = map_dir / "sfm_reference"
    default_train_image_list = map_dir / "train_images.txt"
    train_image_list = train_list or default_train_image_list

    for path in [train_descriptors, train_features, reference_sfm]:
        if not path.exists():
            raise FileNotFoundError(path)
    if train_list is not None and not train_image_list.exists():
        raise FileNotFoundError(train_image_list)

    selected_train_names = None
    if train_image_list.exists():
        selected_train_names = read_image_names(train_image_list)
        if default_train_image_list.exists():
            available_train_names = set(read_image_names(default_train_image_list))
            missing_train_names = [
                image_name for image_name in selected_train_names if image_name not in available_train_names
            ]
            if missing_train_names:
                preview = ", ".join(missing_train_names[:5])
                raise ValueError(
                    f"{len(missing_train_names)} images from {train_image_list} are not part of the map train set, e.g. {preview}"
                )
        logger.info(
            "Train database list: %s | selected_images=%d",
            train_image_list,
            len(selected_train_names),
        )
        if default_train_image_list.exists() and train_image_list != default_train_image_list:
            logger.info(
                "Train subset coverage: %d/%d map train images",
                len(selected_train_names),
                len(available_train_names),
            )
    elif default_train_image_list.exists():
        logger.info("Map train image list exists at %s but was not used.", default_train_image_list)

    loc_dir.mkdir(parents=True, exist_ok=True)

    results_path.parent.mkdir(parents=True, exist_ok=True)

    query_desc = extract_features.main(
        retrieval_conf,
        query_root,
        export_dir=loc_dir,
        image_list=query_names,
        feature_path=loc_dir / f"{retrieval_conf['output']}_query.h5",
        overwrite=True,
    )

    train_tag = train_image_list.stem if train_image_list.exists() else "sfm"
    pairs_path = loc_dir / f"pairs-query-{retrieval_conf_name}{num_retrieval}-{train_tag}.txt"
    if selected_train_names is not None:
        logger.info("Using %s as the fixed retrieval database list.", train_image_list)
        pairs_from_retrieval.main(
            query_desc,
            pairs_path,
            num_retrieval,
            query_list=query_names,
            db_list=selected_train_names,
            db_descriptors=train_descriptors,
        )
    else:
        logger.warning(
            "No fixed train image list found at %s, falling back to reference_sfm for retrieval candidates.",
            train_image_list,
        )
        pairs_from_retrieval.main(
            query_desc,
            pairs_path,
            num_retrieval,
            query_list=query_names,
            db_model=reference_sfm,
            db_descriptors=train_descriptors,
        )

    query_features = extract_features.main(
        feature_conf,
        query_root,
        export_dir=loc_dir,
        image_list=query_names,
        feature_path=loc_dir / f"{feature_conf['output']}_query.h5",
        overwrite=True,
    )

    matches_path = match_features.main(
        matcher_conf,
        pairs_path,
        features=query_features,
        matches=loc_dir / f"{matcher_conf['output']}_query-{train_tag}.h5",
        features_ref=train_features,
        overwrite=True,
    )

    query_cameras = [
        (query_name, build_camera(query_root / query_name, camera_model, camera_params))
        for query_name in query_names
    ]
    query_list = loc_dir / "query_with_intrinsics.txt"
    write_query_list_entries(query_list, query_cameras)

    localize_sfm.main(
        reference_sfm,
        query_cameras,
        pairs_path,
        query_features,
        matches_path,
        results_path,
        ransac_thresh=ransac_thresh,
        covisibility_clustering=False,
        fallback_to_nearest_db=fallback_to_nearest_db,
    )
    summarize_localization_logs(results_path)
    return results_path


def build_map(args):
    outputs = args.outputs
    outputs.mkdir(parents=True, exist_ok=True)
    build_info = {
        "build_mode": args.build_mode,
        "retrieval_conf": args.retrieval_conf,
        "local_feature_conf": args.local_feature_conf,
        "matcher_conf": args.matcher_conf,
    }
    (outputs / "build_info.json").write_text(
        json.dumps(build_info, indent=2) + "\n",
        encoding="utf-8",
    )
    image_names = read_image_names(args.image_list)
    poses = None
    if args.build_mode == "posed" and args.poses_file is not None:
        image_names, poses = filter_image_names_to_available_poses(
            image_names, args.poses_file
        )
    image_name_set = set(image_names)
    write_image_list(outputs / "train_images.txt", image_names)

    retrieval_conf = extract_features.confs[args.retrieval_conf]
    retrieval_path = extract_features.main(
        retrieval_conf,
        args.image_dir,
        outputs,
        image_list=image_names,
        overwrite=args.overwrite,
    )
    logger.info("Train global descriptors saved to %s", retrieval_path)

    feature_conf = extract_features.confs[args.local_feature_conf]
    matcher_conf = match_features.confs[args.matcher_conf]
    features_path = extract_features.main(
        feature_conf,
        args.image_dir,
        outputs,
        image_list=image_names,
        overwrite=args.overwrite,
    )

    if args.build_mode == "hloc-native":
        from hloc import pairs_from_exhaustive, reconstruction

        sfm_pairs = outputs / (
            "pairs-train-exhaustive.txt"
            if args.native_pairing == "exhaustive"
            else f"pairs-train-{args.retrieval_conf}{args.native_num_matched}.txt"
        )
        if args.native_pairing == "exhaustive":
            pairs_from_exhaustive.main(sfm_pairs, image_list=image_names)
        else:
            pairs_from_retrieval.main(
                retrieval_path,
                sfm_pairs,
                args.native_num_matched,
                query_list=image_names,
                db_list=image_names,
                db_descriptors=retrieval_path,
            )

        matches_path = match_features.main(
            matcher_conf,
            sfm_pairs,
            features=features_path,
            matches=outputs / f"train_{matcher_conf['output']}.h5",
            overwrite=args.overwrite,
        )

        reconstruction.main(
            outputs / "sfm_reference",
            args.image_dir,
            sfm_pairs,
            features_path,
            matches_path,
            image_list=image_names,
        )
        logger.info("Native hloc SfM reference map ready at %s", outputs / "sfm_reference")
        return

    if args.reference_model is None and args.poses_file is None:
        logger.info("No reference model provided, stopping after global descriptor extraction.")
        return

    from hloc import triangulation as triangulation_module

    reference_sfm = outputs / "sfm_reference"
    pose_model = args.reference_model
    if args.poses_file is not None:
        pose_model = create_reference_model_from_poses(
            args.image_dir,
            image_names,
            args.poses_file,
            outputs / "reference_from_poses",
            poses,
        )

    train_pairs = outputs / f"pairs-train-poses{args.num_pose_pairs}.txt"
    pairs_from_poses.main(
        pose_model,
        train_pairs,
        args.num_pose_pairs,
        rotation_threshold=args.rotation_threshold,
    )
    filter_pairs_to_image_list(train_pairs, image_name_set)

    matches_path = match_features.main(
        matcher_conf,
        train_pairs,
        features=features_path,
        matches=outputs / f"train_{matcher_conf['output']}.h5",
        overwrite=args.overwrite,
    )

    triangulation_module.main(
        reference_sfm,
        pose_model,
        args.image_dir,
        train_pairs,
        features_path,
        matches_path,
    )

    logger.info("Reference map ready at %s", reference_sfm)


def localize_query(args):
    query_root, query_name = make_query_name(args.query_image, args.query_image_root)
    query_tag = Path(query_name).stem
    loc_dir = args.map_dir / "localization" / query_tag
    results_path = args.results or (loc_dir / f"{query_tag}_pose.txt")
    localize_queries(
        args.map_dir,
        query_root,
        [query_name],
        loc_dir,
        results_path,
        args.retrieval_conf,
        args.local_feature_conf,
        args.matcher_conf,
        args.num_retrieval,
        args.ransac_thresh,
        args.camera_model,
        args.camera_params,
        args.fallback_to_nearest_db,
    )

    pose_line = read_pose_line(results_path)
    logger.info("Estimated pose: %s", pose_line)
    print(pose_line)


def localize_list(args):
    query_names = read_image_names(args.query_list)
    query_tag = args.query_list.stem
    loc_dir = args.map_dir / "localization" / query_tag
    results_path = args.results or (loc_dir / f"{query_tag}_poses.txt")
    localize_queries(
        args.map_dir,
        args.query_root,
        query_names,
        loc_dir,
        results_path,
        args.retrieval_conf,
        args.local_feature_conf,
        args.matcher_conf,
        args.num_retrieval,
        args.ransac_thresh,
        args.camera_model,
        args.camera_params,
        args.fallback_to_nearest_db,
        args.train_list,
    )
    logger.info("Estimated poses written to %s", results_path)
    print(results_path)

    if args.eval_model is not None and args.poses_file is not None:
        logger.info("Both --eval_model and --poses_file were provided, using --eval_model.")

    if args.poses_file is not None or args.eval_model is not None:
        evaluate_localization_results(
            args.map_dir,
            results_path,
            args.poses_file,
            args.eval_model,
            args.eval_translation_thresh,
            args.eval_rotation_thresh,
            query_names,
            args.eval_mode,
        )


def visualize_matches(args):
    if args.query_list is not None:
        saved_paths = visualize_matches_for_list(
            args.results,
            args.image_root,
            args.query_list,
            args.reference_sfm,
            args.output_dir,
            args.top_k_db,
            args.dpi,
        )
    else:
        saved_paths = visualize_matches_from_logs(
            args.results,
            args.image_root,
            args.query_name,
            args.reference_sfm,
            args.output_dir,
            args.top_k_db,
            args.dpi,
        )
    for path in saved_paths:
        print(path)


def main():
    args = parse_args()
    if args.command == "build-map":
        build_map(args)
    elif args.command == "localize-query":
        localize_query(args)
    elif args.command == "localize-list":
        localize_list(args)
    elif args.command == "visualize-matches":
        visualize_matches(args)
    else:
        raise ValueError(f"Unknown command {args.command}.")


if __name__ == "__main__":
    main()