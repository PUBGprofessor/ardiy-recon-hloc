"""Diagnose why thin regions are empty in obj-12 dense reconstruction.

Checks:
  A. Pose + metric-scale consistency: project SfM 3D points into depth maps,
     compare projected camera-z against the depth value at the same pixel.
  B. Spatial depth noise (single-frame, local smoothness on static area).
  C. Effective resolution: how many voxels across a given thickness at the
     current voxel_length, and the truncation-vs-thickness rule.
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import pycolmap

SCENE = Path(r"F:\dev2\prjs1\data1\obj-12")
SFM = SCENE / "hloc_map" / "sfm_reference"
IMG_DIR = SCENE / "scan1"
DEPTH_DIR = SCENE / "scan1" / "depth"
MAX_DEPTH = 2.0


def main() -> None:
    recon = pycolmap.Reconstruction(str(SFM))
    print(f"[info] reconstruction: {len(recon.images)} images, "
          f"{len(recon.points3D)} 3D points")

    # ---------- A. pose + scale consistency ----------
    # For a set of images, project registered 3D points into the depth map and
    # compare the metric camera-z (after the Sim3 scale was applied) with the
    # depth measurement (mm->m).  Pixels on depth discontinuities are excluded
    # (they would produce spurious ratios even with perfect poses).
    sample = sorted(recon.images.values(), key=lambda im: im.name)[::5]
    all_medians = []
    bad_imgs = []
    for image in sample:
        depth_path = DEPTH_DIR / f"{Path(image.name).stem}.png"
        if not depth_path.exists():
            continue
        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if depth is None:
            continue
        depth_m = depth.astype(np.float32) * 0.001
        cam_from_world = image.cam_from_world()
        ratios = []
        for p2d in image.points2D:
            if not p2d.has_point3D():
                continue
            p3d = recon.points3D[p2d.point3D_id]
            cam = cam_from_world * p3d.xyz
            z = float(cam[2])
            if z <= 0:
                continue
            x, y = int(round(float(cam[0]))), int(round(float(cam[1])))
            if not (2 <= x < depth.shape[1] - 2 and 2 <= y < depth.shape[0] - 2):
                continue
            patch = depth_m[y - 1:y + 2, x - 1:x + 2]
            dv = float(depth_m[y, x])
            if dv <= 0.2 or dv >= MAX_DEPTH:
                continue
            # skip depth edges: require the 3x3 neighbourhood to be within 8 mm
            if np.any(np.abs(patch - dv) > 0.008):
                continue
            ratios.append(dv / z)
        if len(ratios) > 20:
            r = np.array(ratios)
            med = float(np.median(r))
            robust = float(np.percentile(np.abs(r - med) / med, 90) * 100)
            all_medians.append(med)
            tag = "OK " if abs(med - 1.0) < 0.1 else "BAD"
            if abs(med - 1.0) >= 0.1:
                bad_imgs.append((image.name, med))
            print(f"[A] [{tag}] {image.name}: median depth/z = {med:.4f} "
                  f"(robust dev {robust:.1f}%, n={len(r)})")
    if all_medians:
        arr = np.array(all_medians)
        print(f"[A] overall median depth/z across sampled images = {np.median(arr):.4f} "
              f"(consistent metric scale should be ~1.0); "
              f"{len(bad_imgs)}/{len(all_medians)} images deviate >10%")
        for name, m in bad_imgs[:6]:
            print(f"    bad: {name} -> {m:.3f}")

    # ---------- B. spatial depth noise on a frame ----------
    frame = sorted((DEPTH_DIR).glob("*.png"))[len(sorted(DEPTH_DIR.glob("*.png"))) // 2]
    d = cv2.imread(str(frame), cv2.IMREAD_UNCHANGED).astype(np.float32)
    dm = d * 0.001
    valid = (dm > 0.3) & (dm < MAX_DEPTH)
    # smoothness = |d - median_filter(d)| in valid flat areas
    med = cv2.medianBlur(d, 5).astype(np.float32)
    resid = np.abs(d - med)
    v = resid[valid]
    print(f"[B] single-frame spatial depth residual (mm) on {frame.name}: "
          f"median {np.median(v):.2f}  p90 {np.percentile(v, 90):.2f}  "
          f"p99 {np.percentile(v, 99):.2f}")

    # ---------- C. TSDF resolution analysis ----------
    print("\n[C] TSDF parameter analysis (voxel_length=0.01, sdf_trunc=0.04):")
    for thick_mm in (5, 10, 15, 20, 30, 50, 80):
        t = thick_mm * 0.001
        voxels_across = t / 0.01
        rule = "OK" if t > 2 * 0.04 else "ERASED (t < 2*sdf_trunc)"
        print(f"    thickness {thick_mm:3d} mm -> {voxels_across:.1f} voxels across, "
              f"truncation rule: {rule}")


if __name__ == "__main__":
    sys.exit(main())
