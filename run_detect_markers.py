import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pycolmap


@dataclass
class CornerObservation:
    image_name: str
    projection: np.ndarray
    world_to_cam: np.ndarray
    image: pycolmap.Image
    u: float
    v: float


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Detect ArUco markers in images registered in a COLMAP model and triangulate "
            "the 3D world coordinates of each marker corner."
        )
    )
    parser.add_argument(
        "--image_dir",
        type=Path,
        required=True,
        help="Scene root directory that contains images used to build the map.",
    )
    parser.add_argument(
        "--model_dir",
        type=Path,
        required=True,
        help="Path to COLMAP model directory (e.g. outputs/sfm_reference).",
    )
    parser.add_argument(
        "--aruco_dict",
        type=str,
        default="DICT_4X4_50",
        help=(
            "ArUco dictionary name (e.g. DICT_4X4_50, DICT_5X5_100) or integer enum value "
            "from cv2.aruco."
        ),
    )
    parser.add_argument(
        "--min_views",
        type=int,
        default=2,
        help="Minimum number of views required to triangulate one marker corner.",
    )
    parser.add_argument(
        "--max_reproj_error",
        type=float,
        default=4.0,
        help="Maximum reprojection error in pixels for inlier filtering.",
    )
    parser.add_argument(
        "--output_json",
        type=Path,
        help="Output JSON path. Default: <model_dir>/markers.json",
    )
    return parser.parse_args()


def get_aruco_dictionary(dictionary_name: str):
    if not hasattr(cv2, "aruco"):
        raise RuntimeError(
            "OpenCV ArUco module is unavailable. Install opencv-contrib-python to enable marker detection."
        )

    name = dictionary_name.strip()
    if name.isdigit():
        return cv2.aruco.getPredefinedDictionary(int(name))

    if not name.startswith("DICT_"):
        name = f"DICT_{name}"

    if not hasattr(cv2.aruco, name):
        options = sorted(attr for attr in dir(cv2.aruco) if attr.startswith("DICT_"))
        raise ValueError(
            f"Unknown aruco dictionary '{dictionary_name}'. Supported values include: {', '.join(options)}"
        )

    dictionary_id = getattr(cv2.aruco, name)
    return cv2.aruco.getPredefinedDictionary(dictionary_id)


def create_detector(dictionary):
    if hasattr(cv2.aruco, "ArucoDetector"):
        params = cv2.aruco.DetectorParameters()
        return cv2.aruco.ArucoDetector(dictionary, params)
    return None


def detect_markers(gray_image: np.ndarray, dictionary, detector):
    if detector is not None:
        corners, ids, _ = detector.detectMarkers(gray_image)
    else:
        params = cv2.aruco.DetectorParameters_create()
        corners, ids, _ = cv2.aruco.detectMarkers(gray_image, dictionary, parameters=params)

    if ids is None or len(ids) == 0:
        return []

    results = []
    ids = ids.reshape(-1)
    for marker_idx, marker_id in enumerate(ids):
        pts = corners[marker_idx].reshape(-1, 2)
        if pts.shape[0] != 4:
            continue
        results.append((int(marker_id), pts.astype(np.float64)))
    return results


def get_world_to_cam_matrix(image: pycolmap.Image) -> np.ndarray:
    matrix = np.asarray(image.cam_from_world().matrix(), dtype=np.float64)
    if matrix.shape == (3, 4):
        return matrix
    if matrix.shape == (4, 4):
        return matrix[:3, :]
    raise ValueError(f"Unsupported cam_from_world matrix shape: {matrix.shape}")


def get_projection_matrix(image: pycolmap.Image) -> np.ndarray:
    world_to_cam = get_world_to_cam_matrix(image)
    calibration = np.asarray(image.camera.calibration_matrix(), dtype=np.float64)
    if calibration.shape != (3, 3):
        raise ValueError(f"Unsupported camera calibration matrix shape: {calibration.shape}")
    return calibration @ world_to_cam


def triangulate_linear(observations: list[CornerObservation]) -> np.ndarray | None:
    if len(observations) < 2:
        return None

    rows = []
    for obs in observations:
        p = obs.projection
        rows.append(obs.u * p[2, :] - p[0, :])
        rows.append(obs.v * p[2, :] - p[1, :])

    a = np.asarray(rows, dtype=np.float64)
    _, _, vh = np.linalg.svd(a)
    x_h = vh[-1]
    if abs(x_h[3]) < 1e-12:
        return None

    xyz = x_h[:3] / x_h[3]
    if not np.all(np.isfinite(xyz)):
        return None

    for obs in observations:
        z = float((obs.world_to_cam @ np.array([xyz[0], xyz[1], xyz[2], 1.0], dtype=np.float64))[2])
        if z <= 0.0:
            return None
    return xyz


def reprojection_error(obs: CornerObservation, xyz: np.ndarray) -> float:
    projected = None
    if hasattr(obs.image, "project_point"):
        projected = obs.image.project_point(xyz)
    if projected is None or not np.all(np.isfinite(projected)):
        homogeneous = np.array([xyz[0], xyz[1], xyz[2], 1.0], dtype=np.float64)
        projected_h = obs.projection @ homogeneous
        if projected_h[2] <= 0:
            return float("inf")
        projected = projected_h[:2] / projected_h[2]
    diff = projected - np.array([obs.u, obs.v], dtype=np.float64)
    return float(np.linalg.norm(diff))


def robust_triangulation(
    observations: list[CornerObservation],
    min_views: int,
    max_reproj_error: float,
) -> dict | None:
    if len(observations) < min_views:
        return None

    active = observations[:]
    for _ in range(4):
        if len(active) < min_views:
            return None
        xyz = triangulate_linear(active)
        if xyz is None:
            return None

        errors = [reprojection_error(obs, xyz) for obs in active]
        inliers = [obs for obs, err in zip(active, errors) if err <= max_reproj_error]

        if len(inliers) == len(active):
            mean_err = float(np.mean(errors)) if errors else 0.0
            max_err = float(np.max(errors)) if errors else 0.0
            return {
                "xyz": xyz,
                "observations": active,
                "mean_reproj_error": mean_err,
                "max_reproj_error": max_err,
            }

        active = inliers

    if len(active) < min_views:
        return None
    xyz = triangulate_linear(active)
    if xyz is None:
        return None
    errors = [reprojection_error(obs, xyz) for obs in active]
    return {
        "xyz": xyz,
        "observations": active,
        "mean_reproj_error": float(np.mean(errors)) if errors else 0.0,
        "max_reproj_error": float(np.max(errors)) if errors else 0.0,
    }


def main():
    args = parse_args()

    if args.min_views < 2:
        raise ValueError("--min_views must be >= 2.")

    model_dir = args.model_dir.resolve()
    image_dir = args.image_dir.resolve()
    output_json = (args.output_json or (model_dir / "markers.json")).resolve()

    reconstruction = pycolmap.Reconstruction(model_dir)
    if len(reconstruction.images) == 0:
        raise RuntimeError(f"No registered images found in model: {model_dir}")

    dictionary = get_aruco_dictionary(args.aruco_dict)
    detector = create_detector(dictionary)

    observations_by_corner = {}
    images_total = 0
    images_with_markers = 0
    detections_total = 0
    skipped_images = []

    for image in reconstruction.images.values():
        image_path = image_dir / image.name
        images_total += 1
        if not image_path.exists():
            skipped_images.append({"image_name": image.name, "reason": "image_not_found"})
            continue

        gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            skipped_images.append({"image_name": image.name, "reason": "failed_to_read"})
            continue

        detections = detect_markers(gray, dictionary, detector)
        if not detections:
            continue

        images_with_markers += 1
        try:
            projection = get_projection_matrix(image)
            world_to_cam = get_world_to_cam_matrix(image)
        except ValueError as exc:
            skipped_images.append({"image_name": image.name, "reason": str(exc)})
            continue

        for marker_id, corners in detections:
            detections_total += 1
            for corner_index in range(4):
                u, v = corners[corner_index]
                key = (marker_id, corner_index)
                observations_by_corner.setdefault(key, []).append(
                    CornerObservation(
                        image_name=image.name,
                        projection=projection,
                        world_to_cam=world_to_cam,
                        image=image,
                        u=float(u),
                        v=float(v),
                    )
                )

    marker_ids = sorted({marker_id for marker_id, _ in observations_by_corner.keys()})
    markers_result = []
    corners_solved = 0

    for marker_id in marker_ids:
        corner_results = []
        for corner_index in range(4):
            key = (marker_id, corner_index)
            observations = observations_by_corner.get(key, [])
            triangulated = robust_triangulation(
                observations=observations,
                min_views=args.min_views,
                max_reproj_error=args.max_reproj_error,
            )

            corner_payload = {
                "corner_index": corner_index,
                "num_observations": len(observations),
            }

            if triangulated is None:
                corner_payload["status"] = "insufficient_or_inconsistent_observations"
            else:
                xyz = triangulated["xyz"]
                corner_payload.update(
                    {
                        "status": "ok",
                        "world_xyz": [float(xyz[0]), float(xyz[1]), float(xyz[2])],
                        "num_inlier_observations": len(triangulated["observations"]),
                        "reprojection_error_mean_px": triangulated["mean_reproj_error"],
                        "reprojection_error_max_px": triangulated["max_reproj_error"],
                    }
                )
                corners_solved += 1

            corner_results.append(corner_payload)

        markers_result.append({"marker_id": marker_id, "corners": corner_results})

    payload = {
        "image_dir": str(image_dir),
        "model_dir": str(model_dir),
        "aruco_dict": args.aruco_dict,
        "min_views": args.min_views,
        "max_reproj_error": args.max_reproj_error,
        "summary": {
            "num_registered_images": images_total,
            "num_images_with_markers": images_with_markers,
            "num_marker_detections": detections_total,
            "num_markers_detected": len(marker_ids),
            "num_corners_solved": corners_solved,
            "num_corners_total": len(marker_ids) * 4,
            "num_skipped_images": len(skipped_images),
        },
        "skipped_images": skipped_images,
        "markers": markers_result,
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"markers.json saved to {output_json}")
    print(
        f"markers={len(marker_ids)}, corners_solved={corners_solved}/{len(marker_ids) * 4}, "
        f"images_with_markers={images_with_markers}/{images_total}"
    )


if __name__ == "__main__":
    main()
