"""
run_reloc_server.py —— 基于 hloc 的单图重定位后端服务

职责：
    接收前端上传的 BGR 图像（可附带相机内参），调用 hloc 重定位管线估计相机位姿，
    向前端返回 world-to-cam 位姿矩阵（3x4）、内点数量与内点率等精简结果。

定位管线（RelocService.localize）：
    1. 全局检索：MegaLoc 提取查询图全局描述子，与库图描述子内积取 Top-K；
    2. 局部特征：SuperPoint 提取查询图关键点与描述子；
    3. 特征匹配：SuperGlue 与检索到的库图逐一匹配，建立 2D-3D 对应关系；
    4. 位姿估计：pycolmap PnP + RANSAC 求解相机位姿并统计内点。

运行模式：
    - 单地图：通过 --map_dir 等参数指定一张地图，启动时即加载；
    - 多地图：通过 --config 配置文件声明多张地图，请求按 map_name 懒加载对应服务；
    - 本地自测：指定 --test_image 时不启动网络服务，仅跑一次定位并打印结果。
"""

import argparse
import importlib
import json
import shlex
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import h5py
import numpy as np
import PIL.Image
import pycolmap
import torch
import torch.nn.functional as F

from hloc import extract_features, logger, match_features, matchers
from hloc.localize_sfm import QueryLocalizer
from hloc.utils.base_model import dynamic_load
import sys
#sys.path.append('../')
#sys.argv=r'run_reloc_server.py --root_dir F:\dev2\prjs1\data1\office3 --map_dir hloc_map --default_camera_model PINHOLE --train_list scan1/image_list.txt --device cuda:0 --retvis'.split()

#e.g. python run_reloc_server.py --map_dir "F:\dev2\prjs1\data1\office2\hloc_map" --default_camera_model PINHOLE --default_camera_params "615.0,615.0,320.0,240.0" 
# --test_image "F:\dev2\prjs1\data1\office2\scan2\color\000001.jpg" --device cuda --gpu_id 0


# 图像预处理默认配置：不转灰度、不缩放，缩放插值使用 OpenCV 的 INTER_AREA
DEFAULT_PREPROCESSING = {
    "grayscale": False,
    "resize_max": None,
    "resize_force": False,
    "interpolation": "cv2_area",
}

# 多地图配置文件中，需要按“配置文件所在目录”解析为绝对路径的参数名集合
CONFIG_PATH_FLAGS = {"--root_dir"}


def read_image_names(image_list_path: Path) -> list[str]:
    """读取图像列表文件：跳过空行与 # 注释行，取每行第一个字段作为图像名。"""
    image_names = []
    for line in image_list_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        image_names.append(line.split()[0])
    if not image_names:
        raise ValueError(f"No image names found in {image_list_path}.")
    return image_names


def add_service_args(parser: argparse.ArgumentParser, require_map_dir: bool) -> None:
    """注册重定位服务相关的命令行参数（单地图模式与 config 中的每个地图条目共用）。"""
    parser.add_argument(
        "--root_dir",
        type=Path,
        help="Optional root directory used to resolve other profile paths.",
    )
    parser.add_argument("--map_dir", type=Path, required=require_map_dir)
    parser.add_argument(
        "--train_list",
        type=Path,
        help=(
            "Optional text file listing the subset of map images to use as the train database. "
            "Image names must match the names stored in the map."
        ),
    )
    parser.add_argument(
        "--image_path",
        type=Path,
        help="Optional image root metadata for this map profile.",
    )
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
    parser.add_argument("--num_retrieval", type=int, default=20)
    parser.add_argument(
        "--num_match_db",
        type=int,
        default=8,
        help="How many retrieved DB images are actually matched. Use a smaller value than --num_retrieval to reduce latency.",
    )
    parser.add_argument(
        "--min_correspondences",
        type=int,
        default=256,
        help="Stop matching early once at least this many 2D-3D correspondences are accumulated.",
    )
    parser.add_argument(
        "--min_matched_db",
        type=int,
        default=4,
        help="Do not early-stop matching before at least this many DB images have been processed.",
    )
    parser.add_argument("--ransac_thresh", type=float, default=12.0)
    parser.add_argument(
        "--default_camera_model",
        type=str,
        help="Optional default COLMAP camera model for requests, e.g. PINHOLE.",
    )
    parser.add_argument(
        "--default_camera_params",
        type=str,
        help="Optional default camera params used when a request does not provide them.",
    )
    parser.add_argument(
        "--fallback_to_nearest_db",
        action="store_true",
        help="Return the top retrieved DB pose if PnP fails.",
    )
    parser.add_argument(
        "--retvis",
        action="store_true",
        help="Return inlier visualization image (query vs matched DB frame) in server response.",
    )
    parser.add_argument(
        "--rotation_augmentation",
        action="store_true",
        help="Run localization on 0/90/180/270 query rotations and pick the best inlier result.",
    )


def resolve_config_tokens(tokens: list[str], config_dir: Path) -> list[str]:
    """把 config 行中 --root_dir 这类路径参数的相对路径解析为绝对路径（相对配置文件目录）。"""
    resolved = []
    idx = 0
    while idx < len(tokens):
        token = tokens[idx]
        resolved.append(token)
        if token in CONFIG_PATH_FLAGS and idx + 1 < len(tokens):
            value = Path(tokens[idx + 1])
            if not value.is_absolute():
                value = (config_dir / value).resolve()
            resolved.append(str(value))
            idx += 2
            continue
        idx += 1
    return resolved


def build_service_defaults(args) -> dict:
    """提取全局命令行参数中与服务相关的字段，作为 config 各地图条目的缺省值。"""
    return {
        "root_dir": args.root_dir,
        "retrieval_conf": args.retrieval_conf,
        "local_feature_conf": args.local_feature_conf,
        "matcher_conf": args.matcher_conf,
        "num_retrieval": args.num_retrieval,
        "num_match_db": args.num_match_db,
        "min_correspondences": args.min_correspondences,
        "min_matched_db": args.min_matched_db,
        "ransac_thresh": args.ransac_thresh,
        "default_camera_model": args.default_camera_model,
        "default_camera_params": args.default_camera_params,
        "fallback_to_nearest_db": args.fallback_to_nearest_db,
        "retvis": args.retvis,
        "rotation_augmentation": args.rotation_augmentation,
        "train_list": args.train_list,
        "image_path": args.image_path,
    }


def parse_map_config_file(config_path: Path, args) -> dict[str, argparse.Namespace]:
    """解析多地图配置文件：每行 '<map_name> <参数...>'，返回 {地图名: 参数命名空间}。"""
    parser = argparse.ArgumentParser(add_help=False)
    add_service_args(parser, require_map_dir=True)
    parser.set_defaults(**build_service_defaults(args))

    profiles = {}
    for line in config_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = shlex.split(line)
        if len(parts) < 2:
            raise ValueError(
                f"Invalid config line in {config_path}: {line}. Expected '<map_name> <args...>'."
            )
        map_name = parts[0]
        if map_name in profiles:
            raise ValueError(f"Duplicate map name '{map_name}' in {config_path}.")
        map_args = parser.parse_args(resolve_config_tokens(parts[1:], config_path.parent))
        root_dir = map_args.root_dir
        if root_dir is None:
            raise ValueError(
                f"Config line for map '{map_name}' is missing --root_dir in {config_path}."
            )
        root_dir = root_dir.resolve()
        map_args.root_dir = root_dir

        # map_dir / train_list / image_path 若为相对路径，统一相对 root_dir 解析
        for attr in ["map_dir", "train_list", "image_path"]:
            value = getattr(map_args, attr)
            if value is not None and not value.is_absolute():
                setattr(map_args, attr, (root_dir / value).resolve())
        profiles[map_name] = map_args

    if not profiles:
        raise ValueError(f"No map profiles found in {config_path}.")
    return profiles


def parse_args():
    """解析服务启动参数；单地图模式必须给 --map_dir，多地图模式必须给 --config。"""
    parser = argparse.ArgumentParser(
        description="Run a low-latency single-image relocalization service with in-memory models and caches."
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Optional config file with lines '<map_name> <service args...>' for lazy-loaded multi-map serving.",
    )
    add_service_args(parser, require_map_dir=False)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Inference device: auto, cpu, cuda, cuda:0, ...",
    )
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=None,
        help="Optional CUDA device index. Works with --device auto or --device cuda.",
    )
    parser.add_argument(
        "--local_cache_device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Where to keep cached DB local features. cpu is safer, cuda is faster for small maps.",
    )
    parser.add_argument(
        "--test_image",
        type=Path,
        help="Run one local test request with this image instead of starting the server.",
    )
    parser.add_argument(
        "--test_map_name",
        type=str,
        help="Map name used with --test_image when --config is provided.",
    )
    parser.add_argument(
        "--test_output",
        type=Path,
        help="Optional path where the test response is written as JSON.",
    )
    parser.add_argument(
        "--host",
        "--ip",
        dest="host",
        type=str,
        default="localhost",
        help=(
            "Server bind IP address. Defaults to localhost, which xbase.netcall "
            "resolves to the current local IP."
        ),
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Server bind port. Defaults to 8000.",
    )
    parser.add_argument(
        "--log_server_response",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to log the minimal server response for each request.",
    )
    args = parser.parse_args()
    if args.config is None and args.map_dir is None:
        parser.error("Either --map_dir or --config must be provided.")
    return args


def resolve_service_paths_from_root(service_args) -> None:
    """将命令行中的相对路径统一解析为相对 root_dir 的绝对路径，避免依赖启动时的工作目录。"""
    root_dir = service_args.root_dir
    if root_dir is None:
        return
    root_dir = root_dir.resolve()
    service_args.root_dir = root_dir
    for attr in ["map_dir", "train_list", "image_path", "test_image", "test_output"]:
        if not hasattr(service_args, attr):
            continue
        value = getattr(service_args, attr)
        if value is not None and not value.is_absolute():
            setattr(service_args, attr, (root_dir / value).resolve())


def resolve_device(device_arg: str) -> str:
    """'auto' 时优先返回 CUDA，否则 CPU；其余取值原样返回。"""
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_arg


def resolve_device_with_gpu_id(device_arg: str, gpu_id: int | None) -> str:
    """结合 --device 与 --gpu_id 计算最终推理设备字符串，并校验取值合法性。"""
    if gpu_id is not None and gpu_id < 0:
        raise ValueError("--gpu_id must be >= 0.")

    if device_arg == "auto":
        if not torch.cuda.is_available():
            if gpu_id is not None:
                raise ValueError("--gpu_id was provided but CUDA is not available.")
            return "cpu"
        if gpu_id is None:
            return "cuda"
        if gpu_id >= torch.cuda.device_count():
            raise ValueError(
                f"--gpu_id {gpu_id} is out of range. Found {torch.cuda.device_count()} CUDA device(s)."
            )
        return f"cuda:{gpu_id}"

    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("--device cuda requested but CUDA is not available.")
        if gpu_id is None:
            return "cuda"
        if gpu_id >= torch.cuda.device_count():
            raise ValueError(
                f"--gpu_id {gpu_id} is out of range. Found {torch.cuda.device_count()} CUDA device(s)."
            )
        return f"cuda:{gpu_id}"

    if device_arg.startswith("cuda:"):
        if gpu_id is not None:
            raise ValueError("Do not pass --gpu_id together with --device cuda:<id>.")
        return device_arg

    if gpu_id is not None:
        raise ValueError("--gpu_id can only be used with --device auto or --device cuda.")

    return device_arg


def resize_image(image: np.ndarray, size: tuple[int, int], interp: str) -> np.ndarray:
    """按指定插值方式缩放图像；放大时 INTER_AREA 退化为 INTER_LINEAR 以保证质量。"""
    if interp.startswith("cv2_"):
        interpolation = getattr(cv2, "INTER_" + interp[len("cv2_") :].upper())
        h, w = image.shape[:2]
        if interpolation == cv2.INTER_AREA and (w < size[0] or h < size[1]):
            interpolation = cv2.INTER_LINEAR
        return cv2.resize(image, size, interpolation=interpolation)
    if interp.startswith("pil_"):
        interpolation = getattr(PIL.Image, interp[len("pil_") :].upper())
        pil_image = PIL.Image.fromarray(image.astype(np.uint8))
        pil_image = pil_image.resize(size, resample=interpolation)
        return np.asarray(pil_image, dtype=image.dtype)
    raise ValueError(f"Unknown interpolation {interp}.")


def preprocess_image(image_bgr: np.ndarray, prepro_conf: dict) -> tuple[np.ndarray, np.ndarray]:
    """BGR 图像预处理：可选灰度/缩放，输出归一化到 [0,1] 的 CHW float 数组及原始尺寸 (W,H)。"""
    conf = SimpleNamespace(**{**DEFAULT_PREPROCESSING, **prepro_conf})
    original_size = np.array(image_bgr.shape[:2][::-1], dtype=np.int64)

    if conf.grayscale:
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    else:
        image = image_bgr[:, :, ::-1]

    image = image.astype(np.float32)
    size = tuple(original_size.tolist())
    if conf.resize_max and (conf.resize_force or max(size) > conf.resize_max):
        scale = conf.resize_max / max(size)
        size_new = tuple(int(round(x * scale)) for x in size)
        image = resize_image(image, size_new, conf.interpolation)

    if conf.grayscale:
        image = image[None]
    else:
        image = image.transpose((2, 0, 1))
    image = image / 255.0
    return image, original_size


def tensor_from_preprocessed(image: np.ndarray, device: str) -> torch.Tensor:
    """预处理结果加 batch 维并搬到推理设备，得到模型输入张量 [1,C,H,W]。"""
    return torch.from_numpy(image).float().unsqueeze(0).to(device, non_blocking=True)


def build_camera_from_array(
    image_bgr: np.ndarray,
    camera_model: str | None,
    camera_params: str | None,
) -> pycolmap.Camera:
    """按相机模型名与逗号分隔的内参字符串构建 pycolmap 相机对象（PnP 需要）。"""
    if not camera_model or not camera_params:
        raise ValueError(
            "Camera intrinsics are required for the service. Provide them in the request or as defaults."
        )
    height, width = image_bgr.shape[:2]
    params = np.array([float(value) for value in camera_params.split(",")], dtype=float)
    return pycolmap.Camera(model=camera_model, width=width, height=height, params=params)


def serialize_cam_from_world(cam_from_world: pycolmap.Rigid3d) -> dict:
    """将 world-to-cam 刚性变换序列化为 {'qvec': [qw,qx,qy,qz], 'tvec': [x,y,z]}（COLMAP 约定）。"""
    # pycolmap 内部四元数顺序为 (x,y,z,w)，重排为 COLMAP 惯用的 (qw,qx,qy,qz)
    qvec = cam_from_world.rotation.quat[[3, 0, 1, 2]].tolist()
    tvec = cam_from_world.translation.tolist()
    return {"qvec": qvec, "tvec": tvec}


def qvec_to_rotation_matrix(qvec: list[float]) -> list[list[float]]:
    """四元数 (qw,qx,qy,qz) 转 3x3 旋转矩阵，先归一化以容忍数值误差。"""
    if len(qvec) != 4:
        raise ValueError(f"Expected qvec with 4 values, got {len(qvec)}.")

    qw, qx, qy, qz = [float(v) for v in qvec]
    norm = np.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if norm <= 0:
        raise ValueError("Invalid zero-norm quaternion.")

    qw /= norm
    qx /= norm
    qy /= norm
    qz /= norm

    r00 = 1.0 - 2.0 * (qy * qy + qz * qz)
    r01 = 2.0 * (qx * qy - qz * qw)
    r02 = 2.0 * (qx * qz + qy * qw)
    r10 = 2.0 * (qx * qy + qz * qw)
    r11 = 1.0 - 2.0 * (qx * qx + qz * qz)
    r12 = 2.0 * (qy * qz - qx * qw)
    r20 = 2.0 * (qx * qz - qy * qw)
    r21 = 2.0 * (qy * qz + qx * qw)
    r22 = 1.0 - 2.0 * (qx * qx + qy * qy)

    return [
        [r00, r01, r02],
        [r10, r11, r12],
        [r20, r21, r22],
    ]


def build_minimal_server_response(response: dict) -> dict:
    """把完整的 localize 结果裁剪为发给前端的精简响应：3x4 位姿矩阵 + 内点统计（+可视化）。"""
    pose_raw = response.get("pose")
    pose = None
    if pose_raw is not None:
        r = qvec_to_rotation_matrix(pose_raw["qvec"])
        t = [float(v) for v in pose_raw["tvec"]]
        pose = np.asarray([
            [r[0][0], r[0][1], r[0][2], t[0]],
            [r[1][0], r[1][1], r[1][2], t[1]],
            [r[2][0], r[2][1], r[2][2], t[2]],
        ], dtype=np.float32)
    else:
        # 定位失败时返回 4x4 单位阵，由前端结合 num_inliers 判断结果是否可信
        pose=np.eye(4, dtype=np.float32)

    minimal = {
        "pose": pose,
        "num_inliers": np.int32(response.get("num_inliers", 0)),
        "inlier_ratio": np.float32(response.get("inlier_ratio", 0.0)),
    }
    if response.get("retvis") is not None:
        minimal["retvis:jpg"] = response["retvis"]
    return minimal


def _ensure_batched_feature_tensor(value, device: str) -> torch.Tensor:
    """统一模型输出为带 batch 维的 float 张量，兼容 list/ndarray/未升维张量等返回形式。"""
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ValueError(f"Expected batch size 1, got {len(value)} feature entries.")
        value = value[0]
    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value)
    if value.ndim == 1:
        value = value.unsqueeze(0)
    elif value.ndim == 2:
        value = value.unsqueeze(0)
    return value.to(device, non_blocking=True).float()


class RelocService:
    """单张地图的重定位服务：常驻加载检索/局部特征/匹配模型与库图特征缓存，逐请求执行定位。"""

    def __init__(
        self,
        map_dir: Path,
        retrieval_conf_name: str,
        local_feature_conf_name: str,
        matcher_conf_name: str,
        num_retrieval: int,
        num_match_db: int,
        min_correspondences: int,
        min_matched_db: int,
        ransac_thresh: float,
        device: str,
        local_cache_device: str,
        train_list: Path | None,
        map_name: str | None,
        root_dir: Path | None,
        image_path: Path | None,
        default_camera_model: str | None,
        default_camera_params: str | None,
        fallback_to_nearest_db: bool,
        retvis: bool,
        rotation_augmentation: bool,
    ):
        self.map_dir = map_dir
        self.num_retrieval = num_retrieval
        self.num_match_db = num_match_db
        self.min_correspondences = min_correspondences
        self.min_matched_db = min_matched_db
        self.device = device
        self.local_cache_device = local_cache_device
        self.train_list = train_list
        self.map_name = map_name
        self.root_dir = root_dir
        self.image_path = image_path
        self.default_camera_model = default_camera_model
        self.default_camera_params = default_camera_params
        self.fallback_to_nearest_db = fallback_to_nearest_db
        self.retvis = retvis
        self.rotation_augmentation = rotation_augmentation

        self.retrieval_conf = extract_features.confs[retrieval_conf_name]
        self.local_feature_conf = extract_features.confs[local_feature_conf_name]
        self.matcher_conf = match_features.confs[matcher_conf_name]

        # ---- 加载 COLMAP 重建（sfm_reference），确定参与定位的库图集合 ----
        self.reconstruction = pycolmap.Reconstruction(map_dir / "sfm_reference")
        self.localizer = QueryLocalizer(
            self.reconstruction,
            {"estimation": {"ransac": {"max_error": ransac_thresh}}},
        )
        all_db_name_to_id = {
            image.name: image_id for image_id, image in self.reconstruction.images.items()
        }
        self.db_image_names = self._resolve_db_image_names(all_db_name_to_id)
        self.db_name_to_id = {
            image_name: all_db_name_to_id[image_name] for image_name in self.db_image_names
        }

        # ---- 常驻加载三个模型：全局检索 / 局部特征 / 匹配器 ----
        logger.info("Loading retrieval model on %s...", device)
        self.global_model = extract_features.load_model(self.retrieval_conf, device)
        logger.info("Loading local feature model on %s...", device)
        self.local_model = extract_features.load_model(self.local_feature_conf, device)

        logger.info("Loading matcher model on %s...", device)
        matcher_model_conf = dict(self.matcher_conf["model"])
        MatcherModel = dynamic_load(matchers, matcher_model_conf["name"])
        self.matcher = MatcherModel(matcher_model_conf).eval().to(device)

        # ---- 预载库图特征与 2D-3D 映射缓存，避免每次请求重复读盘 ----
        logger.info("Loading DB global descriptors into memory...")
        self.db_global_descs = self._load_db_global_descriptors()
        logger.info("Loading DB local features into %s memory...", local_cache_device)
        self.db_local_features = self._load_db_local_features()
        logger.info("Building 2D-3D lookup cache...")
        self.db_point3d_ids = self._build_db_point3d_cache()

        # 预热模型，消化 CUDA kernel 编译等一次性开销
        self._warmup()

    def _resolve_db_image_names(self, all_db_name_to_id: dict[str, int]) -> list[str]:
        """确定库图集合：未指定 train_list 时用全部已注册图像，否则按列表筛选并校验。"""
        map_image_names = [
            self.reconstruction.images[i].name for i in sorted(self.reconstruction.images)
        ]
        if self.train_list is None:
            logger.info(
                "Using all %d registered map images as the train database.",
                len(map_image_names),
            )
            return map_image_names

        if not self.train_list.exists():
            raise FileNotFoundError(self.train_list)

        selected_names = read_image_names(self.train_list)
        missing_names = [name for name in selected_names if name not in all_db_name_to_id]
        if missing_names:
            preview = ", ".join(missing_names[:5])
            raise ValueError(
                f"{len(missing_names)} images from {self.train_list} are not registered in sfm_reference, e.g. {preview}"
            )

        logger.info(
            "Using train database list %s with %d/%d registered map images.",
            self.train_list,
            len(selected_names),
            len(map_image_names),
        )
        return selected_names

    def _load_db_global_descriptors(self) -> torch.Tensor:
        """从 h5 读取全部库图全局描述子，堆叠并 L2 归一化后放到推理设备上。"""
        path = self.map_dir / f"{self.retrieval_conf['output']}.h5"
        descriptors = []
        with h5py.File(str(path), "r", libver="latest") as hfile:
            for image_name in self.db_image_names:
                descriptors.append(hfile[image_name]["global_descriptor"].__array__().astype(np.float32))
        descs = torch.from_numpy(np.stack(descriptors, axis=0)).to(self.device)
        return F.normalize(descs, p=2, dim=1)

    def _load_db_local_features(self) -> dict[str, dict]:
        """从 h5 读取库图局部特征（关键点/描述子/分数/尺寸），缓存到 CPU 或 GPU。"""
        path = self.map_dir / f"{self.local_feature_conf['output']}.h5"
        cache_device = self.device if self.local_cache_device == "cuda" else "cpu"
        cache = {}
        with h5py.File(str(path), "r", libver="latest") as hfile:
            for image_name in self.db_image_names:
                group = hfile[image_name]
                image_size = tuple(int(v) for v in group["image_size"].__array__().tolist())
                cache[image_name] = {
                    "keypoints": torch.from_numpy(group["keypoints"].__array__().astype(np.float32)).to(cache_device),
                    "descriptors": torch.from_numpy(group["descriptors"].__array__().astype(np.float32)).to(cache_device),
                    "scores": torch.from_numpy(group["scores"].__array__().astype(np.float32)).to(cache_device),
                    "image_size": image_size,
                }
        return cache

    def _build_db_point3d_cache(self) -> dict[str, np.ndarray]:
        """为每张库图建立“关键点下标 -> 3D 点 id”的映射（-1 表示该 2D 点无 3D 对应）。"""
        cache = {}
        for image_name in self.db_image_names:
            image = self.reconstruction.images[self.db_name_to_id[image_name]]
            point3d_ids = np.array(
                [point.point3D_id if point.has_point3D() else -1 for point in image.points2D],
                dtype=np.int64,
            )
            cache[image_name] = point3d_ids
        return cache

    def _warmup(self) -> None:
        """用一张假图各跑一遍检索与局部特征提取以预热模型（失败仅告警不中断）。"""
        # MegaLoc's SALAD head requires enough spatial tokens (n > 64), so
        # tiny warmup images can fail even though real requests are valid.
        dummy = np.zeros((384, 384, 3), dtype=np.uint8)
        try:
            self.extract_global(dummy)
            self.extract_local(dummy)
        except Exception as exc:
            logger.warning("Warmup failed and will be skipped: %s", exc)

    def _resolve_db_image_path(self, db_name: str) -> Path | None:
        """按若干候选根目录定位库图原始图片文件（可视化用），找不到返回 None。"""
        db_path = Path(db_name)
        if db_path.is_absolute() and db_path.exists():
            return db_path

        candidates = []
        if self.image_path is not None:
            candidates.append(self.image_path / db_name)
        if self.root_dir is not None:
            candidates.append(self.root_dir / db_name)
        candidates.append(self.map_dir / db_name)
        candidates.append(self.map_dir.parent / db_name)

        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    @staticmethod
    def _draw_inlier_visualization(
        query_image_bgr: np.ndarray,
        db_image_bgr: np.ndarray,
        query_points_xy: np.ndarray,
        db_points_xy: np.ndarray,
    ) -> np.ndarray:
        """左右拼接查询图与库图，并用彩色连线画出内点匹配对（最多画 200 对）。"""
        h0, w0 = query_image_bgr.shape[:2]
        h1, w1 = db_image_bgr.shape[:2]
        canvas_h = max(h0, h1)
        canvas_w = w0 + w1
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        canvas[:h0, :w0] = query_image_bgr
        canvas[:h1, w0:w0 + w1] = db_image_bgr

        max_draw = 200
        if len(query_points_xy) > max_draw:
            step = max(1, len(query_points_xy) // max_draw)
            query_points_xy = query_points_xy[::step][:max_draw]
            db_points_xy = db_points_xy[::step][:max_draw]

        rng = np.random.default_rng(1234)
        for qpt, dpt in zip(query_points_xy, db_points_xy):
            color = tuple(int(v) for v in rng.integers(50, 255, size=3))
            x0, y0 = int(round(float(qpt[0]))), int(round(float(qpt[1])))
            x1, y1 = int(round(float(dpt[0] + w0))), int(round(float(dpt[1])))
            cv2.circle(canvas, (x0, y0), 3, color, -1, lineType=cv2.LINE_AA)
            cv2.circle(canvas, (x1, y1), 3, color, -1, lineType=cv2.LINE_AA)
            cv2.line(canvas, (x0, y0), (x1, y1), color, 1, lineType=cv2.LINE_AA)
        return canvas

    def _build_retvis(
        self,
        query_image_bgr: np.ndarray,
        query_keypoints: np.ndarray,
        retrieved_names: list[str],
        mkp_indices: list[int],
        point3d_ids: list[int],
        inlier_mask: np.ndarray,
    ) -> np.ndarray | None:
        """挑选内点匹配最多的一张库图，绘制查询图-库图内点可视化；无内点则返回 None。"""
        if inlier_mask.size == 0 or not inlier_mask.any():
            return None

        mkp_indices_arr = np.asarray(mkp_indices, dtype=np.int64)
        point3d_ids_arr = np.asarray(point3d_ids, dtype=np.int64)
        inlier_qidx = mkp_indices_arr[inlier_mask]
        inlier_p3d = point3d_ids_arr[inlier_mask]

        best_db_name = None
        best_matches = None

        db_limit = len(retrieved_names) if self.num_match_db <= 0 else min(self.num_match_db, len(retrieved_names))
        for db_name in retrieved_names[:db_limit]:
            db_p3d_ids = self.db_point3d_ids[db_name]
            p3d_to_kp = {}
            for kp_idx, point3d_id in enumerate(db_p3d_ids):
                point3d_id = int(point3d_id)
                if point3d_id == -1 or point3d_id in p3d_to_kp:
                    continue
                p3d_to_kp[point3d_id] = int(kp_idx)

            pairs = []
            for qidx, p3d in zip(inlier_qidx, inlier_p3d):
                kp_idx = p3d_to_kp.get(int(p3d))
                if kp_idx is not None:
                    pairs.append((int(qidx), kp_idx))

            if not pairs:
                continue
            if best_matches is None or len(pairs) > len(best_matches):
                best_db_name = db_name
                best_matches = pairs

        if not best_matches or best_db_name is None:
            return None

        db_path = self._resolve_db_image_path(best_db_name)
        if db_path is None:
            logger.warning("retvis requested but DB image not found for %s", best_db_name)
            return None

        db_image = cv2.imread(str(db_path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        if db_image is None:
            logger.warning("retvis requested but failed to read DB image %s", db_path)
            return None

        db_kpts = self.db_local_features[best_db_name]["keypoints"].detach().cpu().numpy() + 0.5
        query_pts = np.asarray([query_keypoints[qidx] for qidx, _ in best_matches], dtype=np.float32)
        db_pts = np.asarray([db_kpts[kp_idx] for _, kp_idx in best_matches], dtype=np.float32)
        return self._draw_inlier_visualization(query_image_bgr, db_image, query_pts, db_pts)

    @staticmethod
    def _rotate_image(image_bgr: np.ndarray, rotation_deg: int) -> np.ndarray:
        """将查询图顺时针旋转 0/90/180/270 度（旋转增广用）。"""
        if rotation_deg == 0:
            return image_bgr
        if rotation_deg == 90:
            return cv2.rotate(image_bgr, cv2.ROTATE_90_CLOCKWISE)
        if rotation_deg == 180:
            return cv2.rotate(image_bgr, cv2.ROTATE_180)
        if rotation_deg == 270:
            return cv2.rotate(image_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
        raise ValueError(f"Unsupported rotation degree {rotation_deg}. Expected one of [0, 90, 180, 270].")

    @staticmethod
    def _map_keypoints_rotated_to_original(
        keypoints_xy: np.ndarray,
        rotation_deg: int,
        original_width: int,
        original_height: int,
    ) -> np.ndarray:
        """把旋转后图像上检测到的关键点坐标映射回原图坐标系。"""
        if rotation_deg == 0:
            return keypoints_xy

        mapped = np.empty_like(keypoints_xy, dtype=np.float32)
        x = keypoints_xy[:, 0]
        y = keypoints_xy[:, 1]

        if rotation_deg == 90:
            mapped[:, 0] = y
            mapped[:, 1] = (original_height - 1) - x
        elif rotation_deg == 180:
            mapped[:, 0] = (original_width - 1) - x
            mapped[:, 1] = (original_height - 1) - y
        elif rotation_deg == 270:
            mapped[:, 0] = (original_width - 1) - y
            mapped[:, 1] = x
        else:
            raise ValueError(f"Unsupported rotation degree {rotation_deg}. Expected one of [0, 90, 180, 270].")

        return mapped

    @torch.inference_mode()
    def extract_global(self, image_bgr: np.ndarray) -> torch.Tensor:
        """提取查询图的全局检索描述子（L2 归一化）。"""
        image, _ = preprocess_image(image_bgr, self.retrieval_conf["preprocessing"])
        tensor = tensor_from_preprocessed(image, self.device)
        pred = self.global_model({"image": tensor})
        desc = pred["global_descriptor"]
        return F.normalize(desc, p=2, dim=1)[0]

    @torch.inference_mode()
    def extract_local(self, image_bgr: np.ndarray) -> dict:
        """提取查询图局部特征，并把关键点坐标从缩放后尺寸换算回原图尺寸。"""
        image, original_size = preprocess_image(image_bgr, self.local_feature_conf["preprocessing"])
        tensor = tensor_from_preprocessed(image, self.device)
        pred = self.local_model({"image": tensor})
        pred["keypoints"] = _ensure_batched_feature_tensor(pred["keypoints"], self.device)
        pred["descriptors"] = _ensure_batched_feature_tensor(pred["descriptors"], self.device)
        if "scores" in pred:
            pred["scores"] = _ensure_batched_feature_tensor(pred["scores"], self.device)
        pred["image_size"] = torch.from_numpy(original_size).view(1, 2)
        pred["image_size_tuple"] = (int(original_size[0]), int(original_size[1]))
        if "keypoints" in pred:
            # +0.5 对齐像素中心后按缩放比例换算回原图坐标，保证 PnP 使用原图坐标系
            size = np.array(tensor.shape[-2:][::-1], dtype=np.float32)
            scales = torch.from_numpy((original_size.astype(np.float32) / size)).to(self.device)
            pred["keypoints"] = (pred["keypoints"] + 0.5) * scales.view(1, 1, 2) - 0.5
            if "scales" in pred:
                pred["scales"] = pred["scales"] * scales.mean()
        return pred

    @torch.inference_mode()
    def retrieve(self, query_desc: torch.Tensor, num_retrieval: int) -> tuple[list[str], list[float]]:
        """全局描述子与库图做内积（余弦相似度），取 Top-K 返回库图名与得分。"""
        scores = self.db_global_descs @ query_desc.unsqueeze(1)
        scores = scores.squeeze(1)
        top_k = min(num_retrieval, scores.numel())
        values, indices = torch.topk(scores, k=top_k, largest=True, sorted=True)
        names = [self.db_image_names[int(index)] for index in indices.cpu().tolist()]
        return names, values.cpu().tolist()

    def _build_matcher_input(self, query_local: dict, db_name: str) -> dict:
        """组装匹配器输入：查询侧为 0、库图侧为 1（image 张量仅需提供尺寸占位）。"""
        db_local = self.db_local_features[db_name]
        db_device = self.device
        keypoints1 = db_local["keypoints"].to(db_device, non_blocking=True).unsqueeze(0)
        descriptors1 = db_local["descriptors"].to(db_device, non_blocking=True).unsqueeze(0)
        scores1 = db_local["scores"].to(db_device, non_blocking=True).unsqueeze(0)
        width, height = db_local["image_size"]
        query_width, query_height = query_local["image_size_tuple"]
        return {
            "image0": torch.empty((1, 1, query_height, query_width), device=db_device),
            "keypoints0": query_local["keypoints"],
            "scores0": query_local.get("scores", torch.ones((1, query_local["keypoints"].shape[1]), device=db_device)),
            "descriptors0": query_local["descriptors"],
            "image1": torch.empty((1, 1, height, width), device=db_device),
            "keypoints1": keypoints1,
            "scores1": scores1,
            "descriptors1": descriptors1,
        }

    @torch.inference_mode()
    def build_correspondences(
        self,
        query_local: dict,
        retrieved_names: list[str],
        num_match_db: int,
        min_correspondences: int,
        min_matched_db: int,
    ) -> tuple[list[int], list[int], dict]:
        """逐库图匹配并汇总 2D-3D 对应；满足提前停止条件后跳出，返回关键点下标与 3D 点 id。"""
        kp_idx_to_3d = {}
        per_db = []
        total_valid_matches = 0
        db_limit = len(retrieved_names) if num_match_db <= 0 else min(num_match_db, len(retrieved_names))
        correspondences_so_far = 0
        for db_index, db_name in enumerate(retrieved_names[:db_limit], start=1):
            matcher_input = self._build_matcher_input(query_local, db_name)
            pred = self.matcher(matcher_input)
            # matches0[i] = j 表示查询关键点 i 匹配到库图关键点 j；-1 表示无匹配
            matches0 = pred["matches0"][0].detach().cpu().numpy()
            db_point3d_ids = self.db_point3d_ids[db_name]
            valid_query_indices = np.where(matches0 > -1)[0]
            matched_db_indices = matches0[valid_query_indices].astype(np.int64)
            matched_point3d_ids = db_point3d_ids[matched_db_indices]
            valid_mask = matched_point3d_ids != -1
            num_valid_for_db = int(np.count_nonzero(valid_mask))
            total_valid_matches += num_valid_for_db
            per_db.append({
                "db_name": db_name,
                "num_query_matches": int(valid_query_indices.size),
                "num_2d3d_matches": num_valid_for_db,
            })
            for query_idx, point3d_id in zip(valid_query_indices[valid_mask], matched_point3d_ids[valid_mask]):
                query_idx = int(query_idx)
                point3d_id = int(point3d_id)
                if query_idx not in kp_idx_to_3d:
                    kp_idx_to_3d[query_idx] = []
                if point3d_id not in kp_idx_to_3d[query_idx]:
                    kp_idx_to_3d[query_idx].append(point3d_id)

            correspondences_so_far = sum(len(point_ids) for point_ids in kp_idx_to_3d.values())
            # 已积累足够多的 2D-3D 对应且处理了足够多的库图时提前停止，降低匹配时延
            if (
                min_correspondences > 0
                and db_index >= min_matched_db
                and correspondences_so_far >= min_correspondences
            ):
                break

        mkp_indices = [idx for idx in kp_idx_to_3d for _ in kp_idx_to_3d[idx]]
        point3d_ids = [point3d_id for idx in kp_idx_to_3d for point3d_id in kp_idx_to_3d[idx]]
        stats = {
            "num_db_retrieved": len(retrieved_names),
            "num_db_matched": len(per_db),
            "db_limit": db_limit,
            "retrieved_db": per_db,
            "num_query_keypoints_with_3d": len(kp_idx_to_3d),
            "num_correspondences": len(point3d_ids),
            "num_valid_matches_total": total_valid_matches,
        }
        return mkp_indices, point3d_ids, stats

    def localize(self, image_bgr: np.ndarray, request: dict | None = None) -> dict:
        """单图重定位主流程：全局检索 -> 局部特征 -> 匹配建 2D-3D 对应 -> PnP 估计位姿。"""
        request = request or {}
        # 请求级参数缺省时回落到服务启动时的默认值
        num_retrieval = int(request.get("num_retrieval", self.num_retrieval))
        num_match_db = int(request.get("num_match_db", self.num_match_db))
        min_correspondences = int(request.get("min_correspondences", self.min_correspondences))
        min_matched_db = int(request.get("min_matched_db", self.min_matched_db))
        fallback_to_nearest_db = bool(request.get("fallback_to_nearest_db", self.fallback_to_nearest_db))
        retvis = bool(request.get("retvis", self.retvis))
        rotation_augmentation = bool(request.get("rotation_augmentation", self.rotation_augmentation))
        camera_model = request.get("camera_model", self.default_camera_model)
        camera_params = request.get("camera_params", self.default_camera_params)
 
        #print image size and camera_params
        logger.info("Received localization request with image of shape %s and camera_params %s", image_bgr.shape, camera_params)

        timings = {}
        time_start = time.perf_counter()
        query_camera = build_camera_from_array(image_bgr, camera_model, camera_params)
        timings["camera_s"] = time.perf_counter() - time_start

        # 为了应对手机拍摄或设备翻转导致的图像重定向问题，开启 rotation_augmentation 后会对 4 个方向分别尝试定位。
        # 旋转增广：开启时对 0/90/180/270 四个朝向分别定位，按 (内点数, 内点率, 对应数) 字典序选最优
        rotation_candidates = [0, 90, 180, 270] if rotation_augmentation else [0]
        original_height, original_width = image_bgr.shape[:2]
        best_candidate = None
        global_time_total = 0.0
        retrieve_time_total = 0.0
        local_time_total = 0.0
        match_time_total = 0.0
        pnp_time_total = 0.0

        for rotation_deg in rotation_candidates:
            image_for_candidate = self._rotate_image(image_bgr, rotation_deg)

            # 1) 全局检索：提取查询图全局描述子
            time_global = time.perf_counter()
            query_desc = self.extract_global(image_for_candidate)
            global_time_total += time.perf_counter() - time_global

            #    与库图描述子做相似度检索，取 Top-K
            time_retrieve = time.perf_counter()
            retrieved_names, retrieval_scores = self.retrieve(query_desc, num_retrieval)
            retrieve_time_total += time.perf_counter() - time_retrieve

            # 2) 提取查询图局部特征（关键点 + 描述子）
            time_local = time.perf_counter()
            query_local = self.extract_local(image_for_candidate)
            local_time_total += time.perf_counter() - time_local

            # 3) 与检索到的库图逐一匹配，建立 2D-3D 对应关系
            time_match = time.perf_counter()
            mkp_indices, point3d_ids, match_stats = self.build_correspondences(
                query_local,
                retrieved_names,
                num_match_db,
                min_correspondences,
                min_matched_db,
            )
            match_time_total += time.perf_counter() - time_match

            query_keypoints_rot = query_local["keypoints"][0].detach().cpu().numpy() + 0.5
            # 旋转增广时把关键点坐标映射回原图，保证 PnP 使用原图坐标系
            query_keypoints = self._map_keypoints_rotated_to_original(
                query_keypoints_rot,
                rotation_deg,
                original_width,
                original_height,
            )

            # 4) PnP + RANSAC 估计相机位姿（无 2D-3D 对应时跳过）
            time_pnp = time.perf_counter()
            result = None
            if point3d_ids:
                result = self.localizer.localize(query_keypoints, mkp_indices, point3d_ids, query_camera)
            pnp_time_total += time.perf_counter() - time_pnp

            num_inliers = int(result.get("num_inliers", 0)) if result is not None else 0
            num_corr = int(match_stats.get("num_correspondences", 0))
            inlier_ratio = float(num_inliers / num_corr) if num_corr > 0 else 0.0
            score_tuple = (num_inliers, inlier_ratio, num_corr)

            candidate = {
                "rotation_deg": rotation_deg,
                "result": result,
                "retrieved_names": retrieved_names,
                "retrieval_scores": retrieval_scores,
                "match_stats": match_stats,
                "mkp_indices": mkp_indices,
                "point3d_ids": point3d_ids,
                "query_keypoints": query_keypoints,
                "score": score_tuple,
            }

            if best_candidate is None or candidate["score"] > best_candidate["score"]:
                best_candidate = candidate

        result = best_candidate["result"]
        retrieved_names = best_candidate["retrieved_names"]
        retrieval_scores = best_candidate["retrieval_scores"]
        match_stats = best_candidate["match_stats"]
        mkp_indices = best_candidate["mkp_indices"]
        point3d_ids = best_candidate["point3d_ids"]
        query_keypoints = best_candidate["query_keypoints"]
        best_rotation_deg = int(best_candidate["rotation_deg"])

        # PnP 失败且允许回退时，用检索得分最高的库图位姿兜底
        used_fallback = False
        if result is None and fallback_to_nearest_db and retrieved_names:
            used_fallback = True
            nearest = self.reconstruction.images[self.db_name_to_id[retrieved_names[0]]]
            cam_from_world = nearest.cam_from_world()
        elif result is not None:
            cam_from_world = result["cam_from_world"]
        else:
            cam_from_world = None

        timings["global_s"] = float(global_time_total)
        timings["retrieve_s"] = float(retrieve_time_total)
        timings["local_s"] = float(local_time_total)
        timings["match_s"] = float(match_time_total)
        timings["pnp_s"] = float(pnp_time_total)
        timings["total_s"] = time.perf_counter() - time_start

        # 组装完整响应：状态/位姿、检索与匹配统计、各阶段耗时、本次定位配置
        response = {
            "status": "ok" if cam_from_world is not None else "failed",
            "pose": serialize_cam_from_world(cam_from_world) if cam_from_world is not None else None,
            "used_fallback": used_fallback,
            "retrieval": [
                {"db_name": name, "score": float(score)}
                for name, score in zip(retrieved_names, retrieval_scores)
            ],
            "match_stats": match_stats,
            "timings": {key: float(value) for key, value in timings.items()},
            "config": {
                "map_name": self.map_name,
                "root_dir": str(self.root_dir) if self.root_dir is not None else None,
                "map_dir": str(self.map_dir),
                "image_path": str(self.image_path) if self.image_path is not None else None,
                "num_retrieval": num_retrieval,
                "num_match_db": num_match_db,
                "min_correspondences": min_correspondences,
                "min_matched_db": min_matched_db,
                "rotation_augmentation": rotation_augmentation,
                "best_rotation_deg": best_rotation_deg,
                "train_list": str(self.train_list) if self.train_list is not None else None,
                "num_train_db_images": len(self.db_image_names),
            },
        }
        if result is not None:
            # 统计内点指标（含按去重关键点计的内点率）；retvis 开启时附带内点可视化
            num_inliers = int(result.get("num_inliers", 0))
            num_correspondences = int(match_stats.get("num_correspondences", 0))
            num_query_keypoints_with_3d = int(match_stats.get("num_query_keypoints_with_3d", 0))
            inlier_mask = np.asarray(result.get("inlier_mask", []), dtype=bool)
            num_unique_inlier_keypoints = 0
            if inlier_mask.size:
                mkp_indices_array = np.asarray(mkp_indices, dtype=np.int64)
                num_unique_inlier_keypoints = int(
                    np.unique(mkp_indices_array[inlier_mask]).size
                )
            response["num_inliers"] = num_inliers
            response["num_correspondences"] = num_correspondences
            response["inlier_ratio"] = (
                float(num_inliers / num_correspondences)
                if num_correspondences > 0
                else 0.0
            )
            response["num_unique_inlier_keypoints"] = num_unique_inlier_keypoints
            response["inlier_ratio_unique_kp"] = (
                float(num_unique_inlier_keypoints / num_query_keypoints_with_3d)
                if num_query_keypoints_with_3d > 0
                else 0.0
            )
            if retvis:
                response["retvis"] = self._build_retvis(
                    query_image_bgr=image_bgr,
                    query_keypoints=query_keypoints,
                    retrieved_names=retrieved_names,
                    mkp_indices=mkp_indices,
                    point3d_ids=point3d_ids,
                    inlier_mask=inlier_mask,
                )
        return response


def run_test_request(service: RelocService, image_path: Path, output_path: Path | None) -> None:
    """本地自测：读图跑一次 localize，打印结果 JSON，可选写入文件（不启动网络服务）。"""
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if image is None:
        raise ValueError(f"Cannot read image {image_path}.")
    response = service.localize(image, {})
    text = json.dumps(response, indent=2)
    print(text)
    if output_path is not None:
        output_path.write_text(text + "\n", encoding="utf-8")


def build_service(
    service_args,
    device: str,
    local_cache_device: str,
    map_name: str | None,
) -> RelocService:
    """按参数命名空间实例化一个 RelocService（实例化时会加载模型与库图缓存）。"""
    return RelocService(
        map_dir=service_args.map_dir,
        retrieval_conf_name=service_args.retrieval_conf,
        local_feature_conf_name=service_args.local_feature_conf,
        matcher_conf_name=service_args.matcher_conf,
        num_retrieval=service_args.num_retrieval,
        num_match_db=service_args.num_match_db,
        min_correspondences=service_args.min_correspondences,
        min_matched_db=service_args.min_matched_db,
        ransac_thresh=service_args.ransac_thresh,
        device=device,
        local_cache_device=local_cache_device,
        train_list=service_args.train_list,
        map_name=map_name,
        root_dir=service_args.root_dir,
        image_path=service_args.image_path,
        default_camera_model=service_args.default_camera_model,
        default_camera_params=service_args.default_camera_params,
        fallback_to_nearest_db=service_args.fallback_to_nearest_db,
        retvis=service_args.retvis,
        rotation_augmentation=service_args.rotation_augmentation,
    )


class RelocServiceManager:
    """多地图管理器：按 map_name 懒加载并缓存各地图的 RelocService 实例。"""

    def __init__(self, profiles: dict[str, argparse.Namespace], device: str, local_cache_device: str):
        self.profiles = profiles
        self.device = device
        self.local_cache_device = local_cache_device
        self.services: dict[str, RelocService] = {}

    def get_service(self, map_name: str) -> RelocService:
        """按地图名取服务实例；首次访问时现场加载（模型加载耗时较长）。"""
        if map_name not in self.profiles:
            available = ", ".join(sorted(self.profiles))
            raise KeyError(f"Unknown map_name '{map_name}'. Available maps: {available}")
        if map_name not in self.services:
            logger.info("Loading reloc service for map '%s'...", map_name)
            self.services[map_name] = build_service(
                self.profiles[map_name],
                self.device,
                self.local_cache_device,
                map_name,
            )
        return self.services[map_name]


def make_server_handler(
    service: RelocService | None = None,
    manager: RelocServiceManager | None = None,
    log_server_response: bool = True,
):
    """构造网络请求处理函数：校验输入图像、按 map_name 分发（多地图时）、执行定位并返回精简响应。"""

    def server_handler(objs):
        # 请求必须包含 'image' 字段：HxWx3 的 BGR ndarray
        if "image" not in objs:
            raise ValueError("Expected request dict with an 'image' field containing a BGR ndarray.")
        image = objs["image"]
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Expected 'image' to have shape HxWx3 in BGR order.")
        active_service = service
        # 多地图模式（--config）下按请求中的 map_name 分发到对应地图的服务
        if manager is not None:
            map_name = objs.get("map_name", objs.get("map"))
            if not map_name:
                raise ValueError("Expected request dict to include 'map_name' when the server is started with --config.")
            active_service = manager.get_service(str(map_name))
        response = active_service.localize(image, objs)
        minimal_response = build_minimal_server_response(response)
        if log_server_response:
            logger.info("response: num_inliers=%d, inlier_ratio=%f", minimal_response["num_inliers"], minimal_response["inlier_ratio"])
            logger.info("pose=%s", minimal_response["pose"])
        return minimal_response

    return server_handler


def main():
    """入口：解析参数 -> 构建服务（单地图/多地图） -> 本地自测或启动 netcall 网络服务。"""
    args = parse_args()
    device = resolve_device_with_gpu_id(args.device, args.gpu_id)
    local_cache_device = args.local_cache_device
    if local_cache_device == "cuda" and not device.startswith("cuda"):
        raise ValueError("--local_cache_device cuda requires a CUDA inference device.")

    service = None
    manager = None
    if args.config is not None:
        # 多地图模式：先解析配置文件，各地图服务在首次请求时懒加载
        profiles = parse_map_config_file(args.config.resolve(), args)
        logger.info("Loaded %d map profiles from %s", len(profiles), args.config)
        manager = RelocServiceManager(profiles, device, local_cache_device)
        if args.test_image is not None:
            test_map_name = args.test_map_name
            if test_map_name is None:
                if len(profiles) == 1:
                    test_map_name = next(iter(profiles))
                else:
                    raise ValueError("--test_map_name is required with --test_image when --config defines multiple maps.")
            run_test_request(manager.get_service(test_map_name), args.test_image, args.test_output)
            return
    else:
        # 单地图模式：启动时即加载唯一的地图服务
        resolve_service_paths_from_root(args)
        service = build_service(args, device, local_cache_device, None)
        if args.test_image is not None:
            run_test_request(service, args.test_image, args.test_output)
            return

    # 网络服务依赖 xbase.netcall；未安装时提示改用 --test_image 本地自测
    netcall = importlib.import_module("xbase.netcall") if importlib.util.find_spec("xbase.netcall") else None
    if netcall is None:
        raise ImportError(
            "xbase.netcall.runServer is not available. Either install xbase or use --test_image for local testing."
        )

    logger.info("Reloc service is ready.")
    logger.info("Starting reloc server on %s:%d", args.host, args.port)
    netcall.runServer(
        make_server_handler(service, manager, log_server_response=args.log_server_response),
        port=args.port,
        ip=args.host,
    )


if __name__ == "__main__":
    main()
