import json
from pathlib import Path

import cv2
import numpy as np
import pycolmap

SCENE = Path(r"F:\dev2\prjs1\data1\obj-12")
DEPTH = SCENE / "scan1" / "depth"
info = json.load(open(SCENE / "hloc_map" / "build_info.json"))
global_scale = info["metric_scale_result"]["scale"]
per_image = {r["image_name"]: r["scale"] for r in info["metric_scale_result"]["images"]}
recon = pycolmap.Reconstruction(str(SCENE / "hloc_map" / "sfm_reference"))

for name in ["scan1/color/000055.jpg", "scan1/color/000061.jpg",
             "scan1/color/000141.jpg", "scan1/color/000100.jpg"]:
    im = recon.images[[k for k, v in recon.images.items() if v.name == name][0]]
    stem = Path(name).stem
    d = cv2.imread(str(DEPTH / f"{stem}.png"), cv2.IMREAD_UNCHANGED)
    dm = d.astype(np.float32) * 0.001
    w2c = im.cam_from_world()
    p = np.asarray(im.camera.params, dtype=np.float64)
    fx, fy, cx, cy = p[0], p[1], p[2], p[3]
    ratios = []
    for p2d in im.points2D:
        if not p2d.has_point3D():
            continue
        p3d = recon.points3D[p2d.point3D_id]
        cam = w2c * p3d.xyz
        z = float(cam[2])
        if z <= 0:
            continue
        x = int(round(fx * cam[0] / z + cx))
        y = int(round(fy * cam[1] / z + cy))
        if not (0 <= x < d.shape[1] and 0 <= y < d.shape[0]):
            continue
        dv = float(dm[y, x])
        if dv <= 0.2 or dv >= 2.0:
            continue
        ratios.append(dv / z)
    r = np.array(ratios)
    if name in per_image:
        expected = per_image[name] / global_scale
        msg = f"build_info scale={per_image[name]:.4f} (expected depth/z ~{expected:.3f})"
    else:
        msg = "not in build_info per-image list"
    print(f"{name}: {msg}; measured median depth/z={np.median(r):.3f} (n={len(r)})")
