import argparse
import cv2
import json
import open3d as o3d
import shlex
import sys
from pathlib import Path

import numpy as np
import pycolmap
import torch

from hloc import extract_features, logger, match_features, pairs_from_retrieval
from hloc import reconstruction as reconstruction_module
from hloc import pairs_from_exhaustive
from hloc.triangulation import import_features, import_matches
from hloc.utils.io import read_image

# =============================================================================
# 本文件作用：使用 HLOC 构建参考地图（稀疏重建 + 特征提取），为后续重定位做准备。
#   1. 汇总多个图像列表，并为每张图像关联（可选的）相机内参；
#   2. 提取图像级全局检索描述子，供检索式配对召回相似图像；
#   3. 提取局部特征（默认 SuperPoint）；
#   4. 生成匹配图像对：全排列(exhaustive) 或 基于检索(retrieval)；
#   5. 用局部特征匹配器（默认 SuperGlue）计算匹配；
#   6. 交给 COLMAP 增量式 SfM，得到稀疏点云、相机位姿与内参（sfm_reference）；
#   7. 可选：借助深度图恢复真实尺度；用 TSDF 融合稠密点云/网格；
#   8. 可选：导出 poses_optimized.txt 供下游定位直接使用；
#   9. 写 build_info.json 记录本次建图配置与结果。
# =============================================================================

#python run_build_map.py --config build_map.txt --overwrite
#python f:/devpy/hloc/run_build_map.py `
#   --image_dir f:/dev2/prjs1/data1/office2 `
#   --image_list f:/dev2/prjs1/data1/office2/scan1/image_list.txt `
#   --camera_info f:/dev2/prjs1/data1/office2/scan1/camera_info.txt `
#   --image_list f:/dev2/prjs1/data1/office2/scan2/image_list.txt `
#   --camera_info f:/dev2/prjs1/data1/office2/scan2/camera_info.txt `
#   --outputs f:/dev2/prjs1/data1/office2/hloc_map_native `
#   --native_pairing retrieval `
#   --fix_intrinsics `
#   --recover_metric_scale `
#   --export_poses_file f:/dev2/prjs1/data1/office2/hloc_map_native/poses_optimized.txt `
#   --overwrite

def parse_args():
    "解析命令行参数。"
    argv = expand_config_args(sys.argv[1:])
    parser = argparse.ArgumentParser(
        description=(
            "Build an hloc-native reference map with support for multiple image lists, "
            "explicit per-image intrinsics, and optional fixed camera intrinsics."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        help=(
            "Optional config file. Each non-empty line before 'args=' should contain "
            "'<image_list> <camera_info>' or just '<image_list>'. The final 'args=...' "
            "line can provide extra command-line options."
        ),
    )
    parser.add_argument(
        "--image_dir",
        type=Path,
        required=True,
        help="Scene root directory. Image names in image lists are interpreted relative to this path.",
    )
    parser.add_argument(
        "--image_list",
        type=Path,
        action="append",
        help="Text file listing train images relative to --image_dir. Can be passed multiple times.",
    )
    parser.add_argument("--outputs", type=Path, required=True)
    parser.add_argument(
        "--retrieval_conf",
        type=str,
        default="megaloc",
        choices=sorted(extract_features.confs.keys()),
    )
    parser.add_argument(
        "--local_feature_conf",
        type=str,
        default="superpoint_inloc",
        choices=sorted(extract_features.confs.keys()),
    )
    parser.add_argument(
        "--matcher_conf",
        type=str,
        default="superglue",
        choices=sorted(match_features.confs.keys()),
    )
    parser.add_argument(
        "--native_pairing",
        type=str,
        default="exhaustive",
        choices=["exhaustive", "retrieval"],
        help="Pair generation strategy for native SfM.",
    )
    parser.add_argument(
        "--native_num_matched",
        type=int,
        default=20,
        help="Number of retrieved neighbors when --native_pairing retrieval is used.",
    )
    parser.add_argument(
        "--native_cross_list_min_matched",
        type=int,
        default=10,
        help=(
            "When multiple --image_list are provided with --native_pairing retrieval, "
            "reserve at least this many matches per query for images from other image lists in total. "
            "0 disables list-aware balancing."
        ),
    )
    parser.add_argument(
        "--camera_info",
        type=Path,
        action="append",
        help=(
            "camera_info.txt-like file applied to an image list. If provided once, it is used for all image lists. "
            "If provided multiple times, it must match the number of --image_list arguments."
        ),
    )
    parser.add_argument(
        "--camera_assignments",
        type=Path,
        help=(
            "Optional text file with lines '<image_name> <camera_info_path>'. "
            "This overrides --camera_info on a per-image basis."
        ),
    )
    parser.add_argument(
        "--default_camera_model",
        type=str,
        default="PINHOLE",
        help="Default COLMAP camera model used when camera_info files omit an explicit model.",
    )
    parser.add_argument(
        "--fix_intrinsics",
        action="store_true",
        help="Keep COLMAP camera intrinsics fixed during incremental mapping.",
    )
    parser.add_argument(
        "--recover_metric_scale",
        action="store_true",
        help=(
            "Estimate a global metric scale from depth maps after reconstruction and "
            "overwrite the output COLMAP .bin files with the scaled model."
        ),
    )
    parser.add_argument(
        "--dense",
        action="store_true",
        help=(
            "Fuse all registered images with valid depth into a dense point cloud and mesh "
            "after mapping, using Open3D."
        ),
    )
    parser.add_argument(
        "--depth_dir_name",
        type=str,
        default="depth",
        help=(
            "Depth directory name used by metric scale recovery. For an image like "
            "scan/color/name.jpg, the depth is expected at scan/../<depth_dir_name>/name.png "
            "relative to the image path."
        ),
    )
    parser.add_argument(
        "--min_depth",
        type=float,
        default=0.2,
        help="Minimum valid depth in meters for metric scale recovery.",
    )
    parser.add_argument(
        "--max_depth",
        type=float,
        default=4.0,
        help="Maximum valid depth in meters for metric scale recovery.",
    )
    parser.add_argument(
        "--voxel_length",
        type=float,
        default=0.01,
        help=(
            "TSDF voxel size in meters for --dense fusion. To preserve thin structures, "
            "use roughly 1/4 to 1/5 of the thinnest part you need to keep (e.g. 0.002-0.003 "
            "for ~1-2cm thin parts). Finer voxels cost more memory/time."
        ),
    )
    parser.add_argument(
        "--sdf_trunc",
        type=float,
        default=0.04,
        help=(
            "TSDF truncation distance in meters for --dense fusion. Must be smaller than "
            "half the thinnest structure's thickness, otherwise thin parts are erased "
            "(e.g. for ~1-2cm thin parts use 0.008-0.012). Recommended 4-6x voxel_length."
        ),
    )
    parser.add_argument(
        "--export_poses_file",
        type=Path,
        help=(
            "Optional output path for exporting the final reconstruction in poses_optimized.txt format. "
            "If metric scale recovery is enabled, the exported poses use the scaled model."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite extracted features and matches if they already exist.",
    )
    parser.add_argument(
        "--skip_geometric_verification",
        action="store_true",
        help="Skip COLMAP geometric verification before mapping.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show full COLMAP output during reconstruction.",
    )
    args = parser.parse_args(argv)
    if not args.image_list:
        parser.error("At least one --image_list must be provided, either directly or through --config.")
    return args


def expand_config_args(argv: list[str]) -> list[str]:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=Path)
    pre_args, remaining = pre_parser.parse_known_args(argv)
    if pre_args.config is None:
        return argv

    config_path = pre_args.config.resolve()
    config_args = parse_config_file(config_path)
    config_args = resolve_config_arg_paths(config_args, config_path.parent)
    return ["--config", str(config_path), *config_args, *remaining]


def parse_config_file(config_path: Path) -> list[str]:
    config_args = []
    for line in config_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("args="):
            config_args.extend(shlex.split(line[len("args=") :].strip()))
            continue

        parts = line.split()
        if len(parts) == 1:
            config_args.extend(["--image_list", parts[0]])
            continue
        if len(parts) == 2:
            config_args.extend(["--image_list", parts[0], "--camera_info", parts[1]])
            continue
        raise ValueError(
            f"Invalid line in config file {config_path}: {line}\n"
            "Expected '<image_list>', '<image_list> <camera_info>', or 'args=...'."
        )
    return config_args


def resolve_config_arg_paths(config_args: list[str], config_dir: Path) -> list[str]:
    path_flags = {
        "--image_dir",
        "--image_list",
        "--outputs",
        "--camera_info",
        "--camera_assignments",
        "--export_poses_file",
        "--config",
    }
    resolved = []
    idx = 0
    while idx < len(config_args):
        token = config_args[idx]
        resolved.append(token)
        if token in path_flags and idx + 1 < len(config_args):
            value = Path(config_args[idx + 1])
            if not value.is_absolute():
                value = (config_dir / value).resolve()
            resolved.append(str(value))
            idx += 2
            continue
        idx += 1
    return resolved


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


def merge_image_lists(image_list_paths: list[Path]) -> tuple[list[str], dict[Path, list[str]]]:
    merged = []
    seen = set()
    names_by_list = {}
    for image_list_path in image_list_paths:
        image_names = read_image_names(image_list_path)
        names_by_list[image_list_path] = image_names
        for image_name in image_names:
            if image_name in seen:
                continue
            seen.add(image_name)
            merged.append(image_name)
    return merged, names_by_list


def build_image_to_list_id(names_by_list: dict[Path, list[str]]) -> dict[str, int]:
    image_to_list_id = {}
    for list_id, (_, image_names) in enumerate(names_by_list.items()):
        for image_name in image_names:
            image_to_list_id.setdefault(image_name, list_id)
    return image_to_list_id


def write_image_list(image_list_path: Path, image_names: list[str]) -> None:
    image_list_path.parent.mkdir(parents=True, exist_ok=True)
    image_list_path.write_text("\n".join(image_names) + "\n", encoding="utf-8")


def write_pairs(pairs_path: Path, pairs: list[tuple[str, str]]) -> None:
    pairs_path.parent.mkdir(parents=True, exist_ok=True)
    pairs_path.write_text(
        "\n".join(f"{name0} {name1}" for name0, name1 in pairs) + "\n",
        encoding="utf-8",
    )


def generate_list_balanced_retrieval_pairs(
    retrieval_path: Path,
    pairs_path: Path,
    image_names: list[str],
    image_to_list_id: dict[str, int],
    num_matched: int,
    cross_list_min_matched: int,
) -> None:
    if cross_list_min_matched <= 0:
        raise ValueError("cross_list_min_matched must be > 0 for list-balanced retrieval pairing.")

    db_names = list(image_names)
    query_names = list(image_names)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    descriptors = pairs_from_retrieval.get_descriptors(query_names, retrieval_path)
    sim = torch.einsum("id,jd->ij", descriptors.to(device), descriptors.to(device))
    sim = sim.cpu()

    db_names_arr = np.array(db_names)
    list_ids = np.array([image_to_list_id[name] for name in db_names], dtype=np.int32)
    unique_list_ids = np.unique(list_ids)

    pairs = []
    for query_idx, query_name in enumerate(query_names):
        query_list_id = image_to_list_id[query_name]
        selected_db_indices = []
        other_list_ids = [list_id for list_id in unique_list_ids if list_id != query_list_id]
        if other_list_ids:
            total_cross_quota = min(cross_list_min_matched, num_matched)
            base_quota = total_cross_quota // len(other_list_ids)
            remainder = total_cross_quota % len(other_list_ids)
        else:
            total_cross_quota = 0
            base_quota = 0
            remainder = 0

        for offset, other_list_id in enumerate(other_list_ids):
            invalid = np.ones(len(db_names), dtype=bool)
            valid_mask = list_ids == other_list_id
            invalid[valid_mask] = False

            quota = base_quota + (1 if offset < remainder else 0)
            num_select = min(quota, int(np.count_nonzero(valid_mask)))
            if num_select <= 0:
                continue

            sub_pairs = pairs_from_retrieval.pairs_from_score_matrix(
                sim[query_idx : query_idx + 1].clone(),
                invalid[None],
                num_select=num_select,
                min_score=0,
            )
            selected_db_indices.extend(db_idx for _, db_idx in sub_pairs)

        if len(selected_db_indices) < num_matched:
            invalid = np.zeros(len(db_names), dtype=bool)
            invalid[query_idx] = True
            if selected_db_indices:
                invalid[np.array(selected_db_indices, dtype=np.int64)] = True

            num_select = min(num_matched - len(selected_db_indices), len(db_names) - int(np.sum(invalid)))
            if num_select > 0:
                sub_pairs = pairs_from_retrieval.pairs_from_score_matrix(
                    sim[query_idx : query_idx + 1].clone(),
                    invalid[None],
                    num_select=num_select,
                    min_score=0,
                )
                selected_db_indices.extend(db_idx for _, db_idx in sub_pairs)

        for db_idx in selected_db_indices[:num_matched]:
            if db_names_arr[db_idx] == query_name:
                continue
            pairs.append((query_name, db_names_arr[db_idx]))

    logger.info(
        "Found %d list-balanced retrieval pairs with cross-list minimum %d per query.",
        len(pairs),
        cross_list_min_matched,
    )
    write_pairs(pairs_path, pairs)


def parse_camera_info_file(camera_info_path: Path, default_camera_model: str) -> dict:
    raw = {}
    for line in camera_info_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        raw[key.strip()] = value.strip()

    camera_model = raw.get("camera_model") or raw.get("model") or default_camera_model
    width = raw.get("width")
    height = raw.get("height")
    if width is None or height is None:
        raise ValueError(
            f"camera_info file {camera_info_path} must contain width=... and height=...."
        )

    if "params" in raw:
        params = np.array(
            [float(value) for value in raw["params"].replace(",", " ").split()],
            dtype=float,
        )
    else:
        params = build_camera_params_from_keys(camera_model, raw, camera_info_path)

    return {
        "model": camera_model,
        "width": int(width),
        "height": int(height),
        "params": params,
        "source": str(camera_info_path),
    }


def build_camera_params_from_keys(camera_model: str, raw: dict, camera_info_path: Path) -> np.ndarray:
    camera_model = camera_model.upper()
    key_orders = {
        "SIMPLE_PINHOLE": ["f", "cx", "cy"],
        "PINHOLE": ["fx", "fy", "cx", "cy"],
        "SIMPLE_RADIAL": ["f", "cx", "cy", "k1"],
        "RADIAL": ["f", "cx", "cy", "k1", "k2"],
        "OPENCV": ["fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2"],
        "OPENCV_FISHEYE": ["fx", "fy", "cx", "cy", "k1", "k2", "k3", "k4"],
    }
    aliases = {
        "f": ["f", "fx", "focal", "focal_length"],
        "fx": ["fx", "f"],
        "fy": ["fy", "f"],
        "cx": ["cx"],
        "cy": ["cy"],
        "k1": ["k1"],
        "k2": ["k2"],
        "k3": ["k3"],
        "k4": ["k4"],
        "p1": ["p1"],
        "p2": ["p2"],
    }
    if camera_model not in key_orders:
        raise ValueError(
            f"camera_info file {camera_info_path} uses unsupported camera model {camera_model}. "
            "Add params=... explicitly for this model."
        )

    params = []
    for key in key_orders[camera_model]:
        value = None
        for alias in aliases[key]:
            if alias in raw:
                value = raw[alias]
                break
        if value is None:
            raise ValueError(
                f"camera_info file {camera_info_path} is missing parameter '{key}' for {camera_model}."
            )
        params.append(float(value))
    return np.array(params, dtype=float)


def resolve_camera_info_paths(
    camera_info_paths: list[Path] | None,
    image_list_paths: list[Path],
) -> list[Path] | None:
    if not camera_info_paths:
        return None
    if len(camera_info_paths) == 1 and len(image_list_paths) > 1:
        return camera_info_paths * len(image_list_paths)
    if len(camera_info_paths) != len(image_list_paths):
        raise ValueError(
            "--camera_info must be passed either once or the same number of times as --image_list."
        )
    return camera_info_paths


def parse_camera_assignments(assignments_path: Path) -> dict[str, Path]:
    assignments = {}
    for line in assignments_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 2:
            raise ValueError(
                f"Expected '<image_name> <camera_info_path>' in {assignments_path}, got: {line}"
            )
        image_name, camera_info = parts
        camera_info_path = Path(camera_info)
        if not camera_info_path.is_absolute():
            camera_info_path = (assignments_path.parent / camera_info_path).resolve()
        assignments[image_name] = camera_info_path
    return assignments


def build_camera_specs_by_image(
    image_list_paths: list[Path],
    names_by_list: dict[Path, list[str]],
    camera_info_paths: list[Path] | None,
    camera_assignments_path: Path | None,
    default_camera_model: str,
) -> dict[str, dict]:
    camera_specs_by_image = {}
    parsed_cache = {}

    resolved_camera_info_paths = resolve_camera_info_paths(camera_info_paths, image_list_paths)
    if resolved_camera_info_paths is not None:
        for image_list_path, camera_info_path in zip(image_list_paths, resolved_camera_info_paths):
            camera_info_path = camera_info_path.resolve()
            spec = parsed_cache.setdefault(
                camera_info_path,
                parse_camera_info_file(camera_info_path, default_camera_model),
            )
            for image_name in names_by_list[image_list_path]:
                camera_specs_by_image[image_name] = spec

    if camera_assignments_path is not None:
        assignments = parse_camera_assignments(camera_assignments_path)
        for image_name, camera_info_path in assignments.items():
            camera_info_path = camera_info_path.resolve()
            spec = parsed_cache.setdefault(
                camera_info_path,
                parse_camera_info_file(camera_info_path, default_camera_model),
            )
            camera_specs_by_image[image_name] = spec

    return camera_specs_by_image


def verify_images_exist(image_dir: Path, image_names: list[str]) -> None:
    missing = [image_name for image_name in image_names if not (image_dir / image_name).exists()]
    if missing:
        preview = ", ".join(missing[:5])
        raise FileNotFoundError(
            f"{len(missing)} images from the provided image lists do not exist under {image_dir}: {preview}"
        )


def depth_path_from_image_name(image_dir: Path, image_name: str, depth_dir_name: str) -> Path:
    image_path = Path(image_name)
    return (image_dir / image_path.parent / ".." / depth_dir_name / f"{image_path.stem}.png").resolve()


def recover_metric_scale_from_depth(
    reconstruction: pycolmap.Reconstruction,
    sfm_dir: Path,
    image_dir: Path,
    depth_dir_name: str,
    min_depth: float,
    max_depth: float,
) -> dict:
    image_results = []
    total_depth_images = 0

    for image in reconstruction.images.values():
        depth_path = depth_path_from_image_name(image_dir, image.name, depth_dir_name)
        if not depth_path.exists():
            continue

        total_depth_images += 1
        depth_raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if depth_raw is None:
            logger.warning("Failed to read depth image: %s", depth_path)
            continue
        if depth_raw.ndim != 2:
            logger.warning("Depth image is not single-channel, skipping: %s", depth_path)
            continue

        depth = depth_raw.astype(np.float32) * 0.001
        sum_scale = 0.0
        valid_count = 0

        cam_from_world = image.cam_from_world()
        for point2D in image.points2D:
            if not point2D.has_point3D():
                continue

            if point2D.point3D_id not in reconstruction.points3D:
                continue
            point3D = reconstruction.points3D[point2D.point3D_id]

            projected = image.project_point(point3D.xyz)
            if projected is None or not np.all(np.isfinite(projected)):
                continue

            cam_point = cam_from_world * point3D.xyz
            z_sfm = float(cam_point[2])
            if not np.isfinite(z_sfm) or z_sfm <= 0.0:
                continue

            xi = int(projected[0] + 0.5)
            yi = int(projected[1] + 0.5)
            if xi < 0 or yi < 0 or xi >= depth.shape[1] or yi >= depth.shape[0]:
                continue

            depth_value = float(depth[yi, xi])
            if not np.isfinite(depth_value) or depth_value <= min_depth or depth_value >= max_depth:
                continue

            sum_scale += depth_value / z_sfm
            valid_count += 1

        if valid_count < 10:
            continue

        image_scale = sum_scale / valid_count
        logger.info(
            "Depth metric scale for %s: %.6f from %d correspondences",
            image.name,
            image_scale,
            valid_count,
        )
        image_results.append(
            {
                "image_name": image.name,
                "depth_path": str(depth_path),
                "scale": image_scale,
                "num_correspondences": valid_count,
            }
        )

    if not image_results:
        raise RuntimeError(
            "Metric scale recovery was requested, but no image produced enough valid depth-backed correspondences."
        )

    global_scale = float(np.mean([result["scale"] for result in image_results]))
    result = {
        "enabled": True,
        "applied": False,
        "scale": global_scale,
        "depth_dir_name": depth_dir_name,
        "min_depth": min_depth,
        "max_depth": max_depth,
        "num_depth_images_found": total_depth_images,
        "num_images_used": len(image_results),
        "num_correspondences": int(sum(r["num_correspondences"] for r in image_results)),
        "images": image_results,
    }

    if abs(global_scale - 1.0) <= 1e-4:
        logger.info("Estimated metric scale is already near 1.0: %.6f", global_scale)
        return result

    transform = pycolmap.Sim3d()
    transform.scale = global_scale
    reconstruction.transform(transform)
    reconstruction.write_binary(sfm_dir)
    result["applied"] = True
    logger.info("Applied global metric scale %.6f and overwrote %s", global_scale, sfm_dir)
    return result


def _camera_intrinsics_from_params(camera: pycolmap.Camera) -> tuple[float, float, float, float]:
    params = np.asarray(camera.params, dtype=np.float64)
    if params.size >= 4:
        fx, fy, cx, cy = params[:4]
    elif params.size >= 3:
        f, cx, cy = params[:3]
        fx = f
        fy = f
    else:
        raise ValueError(
            f"Unsupported camera params for dense fusion: model={camera.model}, params={params.tolist()}"
        )
    return float(fx), float(fy), float(cx), float(cy)


def _camera_extrinsic_matrix(image: pycolmap.Image) -> np.ndarray:
    matrix = np.array(image.cam_from_world().matrix(), dtype=np.float64, copy=True)
    if matrix.shape == (3, 4):
        extrinsic = np.eye(4, dtype=np.float64)
        extrinsic[:3, :] = matrix
    elif matrix.shape == (4, 4):
        extrinsic = matrix
    else:
        raise ValueError(
            f"Unsupported cam_from_world matrix shape for dense fusion: {matrix.shape}"
        )
    return np.ascontiguousarray(extrinsic, dtype=np.float64)


def build_dense_model_from_depth(
    reconstruction: pycolmap.Reconstruction,
    output_dir: Path,
    image_dir: Path,
    depth_dir_name: str,
    min_depth: float,
    max_depth: float,
    voxel_length: float = 0.01,
    sdf_trunc: float = 0.04,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    if voxel_length <= 0 or sdf_trunc <= 0:
        raise ValueError("--voxel_length and --sdf_trunc must be positive.")
    if voxel_length > 0.004:
        logger.warning(
            "voxel_length=%.3fm is coarse: structures thinner than ~%.0fmm cannot be "
            "represented. Use ~0.002-0.003m to keep ~1-2cm thin parts.",
            voxel_length,
            voxel_length * 1000 * 4,
        )
    if sdf_trunc > 0.02:
        logger.warning(
            "sdf_trunc=%.3fm is large: structures thinner than ~%.0fmm will be erased "
            "by TSDF truncation (need thickness > 2*sdf_trunc). Use ~0.008-0.012m for "
            "thin parts.",
            sdf_trunc,
            sdf_trunc * 1000 * 2,
        )

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_length,
        sdf_trunc=sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )

    num_depth_images_found = 0
    num_images_used = 0
    skipped_images = []

    for image in reconstruction.images.values():
        depth_path = depth_path_from_image_name(image_dir, image.name, depth_dir_name)
        if not depth_path.exists():
            continue

        num_depth_images_found += 1
        image_path = image_dir / image.name

        color_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        if color_bgr is None:
            skipped_images.append({"image_name": image.name, "reason": "failed_to_read_color"})
            continue

        depth_raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if depth_raw is None:
            skipped_images.append({"image_name": image.name, "reason": "failed_to_read_depth"})
            continue
        if depth_raw.ndim != 2:
            skipped_images.append({"image_name": image.name, "reason": "depth_not_single_channel"})
            continue

        if color_bgr.shape[:2] != depth_raw.shape[:2]:
            skipped_images.append({
                "image_name": image.name,
                "reason": (
                    f"shape_mismatch_color_{color_bgr.shape[1]}x{color_bgr.shape[0]}_"
                    f"depth_{depth_raw.shape[1]}x{depth_raw.shape[0]}"
                ),
            })
            continue

        try:
            fx, fy, cx, cy = _camera_intrinsics_from_params(image.camera)
        except ValueError as exc:
            skipped_images.append({"image_name": image.name, "reason": str(exc)})
            continue

        depth_m = depth_raw.astype(np.float32) * 0.001
        valid = np.isfinite(depth_m) & (depth_m > min_depth) & (depth_m < max_depth)
        if not np.any(valid):
            skipped_images.append({"image_name": image.name, "reason": "no_valid_depth"})
            continue

        color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
        depth_for_tsdf = depth_raw.copy()
        depth_for_tsdf[~valid] = 0

        color_o3d = o3d.geometry.Image(np.ascontiguousarray(color_rgb))
        depth_o3d = o3d.geometry.Image(np.ascontiguousarray(depth_for_tsdf))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color=color_o3d,
            depth=depth_o3d,
            depth_scale=1000.0,
            depth_trunc=max_depth,
            convert_rgb_to_intensity=False,
        )

        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            color_rgb.shape[1],
            color_rgb.shape[0],
            fx,
            fy,
            cx,
            cy,
        )
        try:
            world_to_cam = _camera_extrinsic_matrix(image)
        except ValueError as exc:
            skipped_images.append({"image_name": image.name, "reason": str(exc)})
            continue
        volume.integrate(rgbd, intrinsic, world_to_cam)
        num_images_used += 1

    if num_images_used == 0:
        raise RuntimeError(
            "--dense was enabled, but no registered image produced valid depth for TSDF fusion."
        )

    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    pcd = volume.extract_point_cloud()

    points_path = output_dir / "dense_points.ply"
    mesh_path = output_dir / "dense_mesh.ply"
    o3d.io.write_point_cloud(str(points_path), pcd)
    o3d.io.write_triangle_mesh(str(mesh_path), mesh)

    logger.info("Dense point cloud saved to %s", points_path)
    logger.info("Dense mesh saved to %s", mesh_path)

    return {
        "enabled": True,
        "applied": True,
        "depth_dir_name": depth_dir_name,
        "min_depth": min_depth,
        "max_depth": max_depth,
        "num_depth_images_found": num_depth_images_found,
        "num_images_used": num_images_used,
        "num_images_skipped": len(skipped_images),
        "num_points_before_filter": int(len(pcd.points)),
        "num_points_after_filter": int(len(pcd.points)),
        "num_mesh_vertices": int(len(mesh.vertices)),
        "num_mesh_triangles": int(len(mesh.triangles)),
        "voxel_length": voxel_length,
        "sdf_trunc": sdf_trunc,
        "points_path": str(points_path),
        "mesh_path": str(mesh_path),
    }


def export_poses_optimized(reconstruction: pycolmap.Reconstruction, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    for image in sorted(reconstruction.images.values(), key=lambda item: item.name):
        if hasattr(image, "has_pose") and not image.has_pose:
            continue

        intrinsics = image.camera.calibration_matrix().reshape(-1)
        pose_c2w = image.cam_from_world().inverse().matrix().reshape(-1)

        lines.append(image.name)
        lines.append(" ".join(f"{value:.9g}" for value in intrinsics))
        lines.append(" ".join(f"{value:.9g}" for value in pose_c2w))

    if not lines:
        raise RuntimeError("No registered images with valid poses were available for poses_optimized export.")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Exported poses_optimized.txt to %s", output_path)


def create_db_with_explicit_cameras(
    image_dir: Path,
    image_names: list[str],
    camera_specs_by_image: dict[str, dict],
    database_path: Path,
) -> dict[str, int]:
    reconstruction_module.create_empty_db(database_path)

    camera_ids = {}
    image_ids = {}
    next_camera_id = 1
    with pycolmap.Database.open(database_path) as db:
        for image_id, image_name in enumerate(image_names, start=1):
            spec = camera_specs_by_image[image_name]
            image_path = image_dir / image_name
            image = read_image(image_path)
            height, width = image.shape[:2]
            if spec["width"] != width or spec["height"] != height:
                raise ValueError(
                    f"Camera info for {image_name} says {spec['width']}x{spec['height']} but the image is {width}x{height}."
                )

            camera_key = (
                spec["model"],
                spec["width"],
                spec["height"],
                *(np.round(spec["params"], 8).tolist()),
            )
            camera_id = camera_ids.get(camera_key)
            if camera_id is None:
                camera_id = next_camera_id
                next_camera_id += 1
                camera_ids[camera_key] = camera_id
                camera = pycolmap.Camera(
                    camera_id=camera_id,
                    model=spec["model"],
                    width=spec["width"],
                    height=spec["height"],
                    params=spec["params"],
                )
                db.write_camera(camera, use_camera_id=True)

            db_image = pycolmap.Image(
                image_id=image_id,
                name=image_name,
                camera_id=camera_id,
            )
            db.write_image(db_image, use_image_id=True)
            image_ids[image_name] = image_id
    return image_ids


def run_reconstruction_with_explicit_cameras(
    sfm_dir: Path,
    image_dir: Path,
    image_names: list[str],
    pairs_path: Path,
    features_path: Path,
    matches_path: Path,
    camera_specs_by_image: dict[str, dict],
    fix_intrinsics: bool,
    skip_geometric_verification: bool,
    verbose: bool,
):
    sfm_dir.mkdir(parents=True, exist_ok=True)
    database_path = sfm_dir / "database.db"
    logger.info("Writing COLMAP logs to %s", sfm_dir / "colmap.LOG.*")
    pycolmap.logging.set_log_destination(pycolmap.logging.INFO, sfm_dir / "colmap.LOG.")

    image_ids = create_db_with_explicit_cameras(
        image_dir=image_dir,
        image_names=image_names,
        camera_specs_by_image=camera_specs_by_image,
        database_path=database_path,
    )
    with pycolmap.Database.open(database_path) as db:
        import_features(image_ids, db, features_path)
        import_matches(
            image_ids,
            db,
            pairs_path,
            matches_path,
            min_match_score=None,
            skip_geometric_verification=skip_geometric_verification,
        )
    if not skip_geometric_verification:
        reconstruction_module.estimation_and_geometric_verification(
            database_path, pairs_path, verbose
        )

    mapper_options = {"constant_cameras": True} if fix_intrinsics else None
    reconstruction = reconstruction_module.run_reconstruction(
        sfm_dir,
        database_path,
        image_dir,
        verbose=verbose,
        options=mapper_options,
    )
    if reconstruction is not None:
        logger.info(
            "Reconstruction statistics:\n%s\n\tnum_input_images = %d",
            reconstruction.summary(),
            len(image_ids),
        )
    return reconstruction


def main():
    args = parse_args()
    if args.native_cross_list_min_matched < 0:
        raise ValueError("--native_cross_list_min_matched must be >= 0.")
    if args.native_cross_list_min_matched > args.native_num_matched:
        raise ValueError("--native_cross_list_min_matched cannot exceed --native_num_matched.")

    outputs = args.outputs
    outputs.mkdir(parents=True, exist_ok=True)

    image_names, names_by_list = merge_image_lists(args.image_list)
    image_to_list_id = build_image_to_list_id(names_by_list)
    verify_images_exist(args.image_dir, image_names)
    write_image_list(outputs / "train_images.txt", image_names)

    camera_specs_by_image = build_camera_specs_by_image(
        image_list_paths=args.image_list,
        names_by_list=names_by_list,
        camera_info_paths=args.camera_info,
        camera_assignments_path=args.camera_assignments,
        default_camera_model=args.default_camera_model,
    )

    if camera_specs_by_image and len(camera_specs_by_image) != len(image_names):
        uncovered = [image_name for image_name in image_names if image_name not in camera_specs_by_image]
        preview = ", ".join(uncovered[:5])
        raise ValueError(
            "Explicit camera intrinsics were provided but do not cover all images. "
            f"Missing {len(uncovered)} images, e.g. {preview}"
        )

    build_info = {
        "build_mode": "hloc-native",
        "config": str(args.config) if args.config else None,
        "retrieval_conf": args.retrieval_conf,
        "local_feature_conf": args.local_feature_conf,
        "matcher_conf": args.matcher_conf,
        "native_pairing": args.native_pairing,
        "native_num_matched": args.native_num_matched,
        "native_cross_list_min_matched": args.native_cross_list_min_matched,
        "fix_intrinsics": args.fix_intrinsics,
        "image_lists": [str(path) for path in args.image_list],
        "camera_info_files": [str(path) for path in args.camera_info] if args.camera_info else [],
        "camera_assignments": str(args.camera_assignments) if args.camera_assignments else None,
        "explicit_camera_coverage": len(camera_specs_by_image),
        "recover_metric_scale": args.recover_metric_scale,
        "dense": args.dense,
        "depth_dir_name": args.depth_dir_name,
        "min_depth": args.min_depth,
        "max_depth": args.max_depth,
        "voxel_length": args.voxel_length,
        "sdf_trunc": args.sdf_trunc,
        "export_poses_file": str(args.export_poses_file) if args.export_poses_file else None,
    }

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

    sfm_pairs = outputs / (
        "pairs-train-exhaustive.txt"
        if args.native_pairing == "exhaustive"
        else f"pairs-train-{args.retrieval_conf}{args.native_num_matched}.txt"
    )
    if args.native_pairing == "exhaustive":
        pairs_from_exhaustive.main(sfm_pairs, image_list=image_names)
    else:
        use_list_balanced_retrieval = (
            len(names_by_list) > 1 and args.native_cross_list_min_matched > 0
        )
        if use_list_balanced_retrieval:
            generate_list_balanced_retrieval_pairs(
                retrieval_path=retrieval_path,
                pairs_path=sfm_pairs,
                image_names=image_names,
                image_to_list_id=image_to_list_id,
                num_matched=args.native_num_matched,
                cross_list_min_matched=args.native_cross_list_min_matched,
            )
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

    reconstruction = None
    if camera_specs_by_image:
        reconstruction = run_reconstruction_with_explicit_cameras(
            sfm_dir=outputs / "sfm_reference",
            image_dir=args.image_dir,
            image_names=image_names,
            pairs_path=sfm_pairs,
            features_path=features_path,
            matches_path=matches_path,
            camera_specs_by_image=camera_specs_by_image,
            fix_intrinsics=args.fix_intrinsics,
            skip_geometric_verification=args.skip_geometric_verification,
            verbose=args.verbose,
        )
    else:
        mapper_options = {"constant_cameras": True} if args.fix_intrinsics else None
        reconstruction = reconstruction_module.main(
            outputs / "sfm_reference",
            args.image_dir,
            sfm_pairs,
            features_path,
            matches_path,
            image_list=image_names,
            verbose=args.verbose,
            skip_geometric_verification=args.skip_geometric_verification,
            mapper_options=mapper_options,
        )

    if reconstruction is None:
        raise RuntimeError("3D reconstruction did not produce a valid model.")

    metric_scale_result = {"enabled": False, "applied": False}
    if args.recover_metric_scale:
        metric_scale_result = recover_metric_scale_from_depth(
            reconstruction=reconstruction,
            sfm_dir=outputs / "sfm_reference",
            image_dir=args.image_dir,
            depth_dir_name=args.depth_dir_name,
            min_depth=args.min_depth,
            max_depth=args.max_depth,
        )

    dense_result = {"enabled": args.dense, "applied": False}
    if args.dense:
        dense_result = build_dense_model_from_depth(
            reconstruction=reconstruction,
            output_dir=outputs / "dense",
            image_dir=args.image_dir,
            depth_dir_name=args.depth_dir_name,
            min_depth=args.min_depth,
            max_depth=args.max_depth,
            voxel_length=args.voxel_length,
            sdf_trunc=args.sdf_trunc,
        )

    if args.export_poses_file:
        export_poses_optimized(reconstruction, args.export_poses_file)

    build_info["metric_scale_result"] = metric_scale_result
    build_info["dense_result"] = dense_result
    (outputs / "build_info.json").write_text(
        json.dumps(build_info, indent=2) + "\n",
        encoding="utf-8",
    )

    logger.info("Native hloc SfM reference map ready at %s", outputs / "sfm_reference")


if __name__ == "__main__":
    main()
