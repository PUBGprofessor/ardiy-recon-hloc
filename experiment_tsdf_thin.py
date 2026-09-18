"""Decisive experiment: does sdf_trunc/voxel_length erase a thin structure?

Fuses real obj-12 scan frames that see the yellow floor tape from both sides
into a ScalableTSDFVolume, using (a) the current params (voxel 1cm / trunc 4cm)
and (b) finer params.  Counts how many surface points land inside the tape's 3D
bounding box -> shows whether the thin tape survives fusion.
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import pycolmap

SCENE = Path(r"F:\dev2\prjs1\data1\obj-12")
SFM = SCENE / "hloc_map" / "sfm_reference"
COLOR = SCENE / "scan1" / "color"
DEPTH = SCENE / "scan1" / "depth"
MAX_DEPTH = 2.0
MIN_DEPTH = 0.2


def world_from_cam(image) -> np.ndarray:
    return np.asarray(image.cam_from_world().inverse().matrix(), dtype=np.float64)


def intrinsic_from_camera(cam) -> tuple:
    p = np.asarray(cam.params, dtype=np.float64)
    return float(p[0]), float(p[1]), float(p[2]), float(p[3])


def tape_pixels(rgb: np.ndarray) -> np.ndarray:
    """Yellow-tape mask: bright yellow in HSV, plausible for the floor tape."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, (20, 90, 120), (40, 255, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return mask > 0


def tape_3d_bbox(recon) -> tuple[np.ndarray, np.ndarray]:
    """Back-project tape pixels in several frames to find its 3D extent."""
    pts = []
    for image in sorted(recon.images.values(), key=lambda im: im.name)[::10]:
        stem = Path(image.name).stem
        rgb = cv2.cvtColor(cv2.imread(str(COLOR / f"{stem}.jpg"),
                                      cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        depth = cv2.imread(str(DEPTH / f"{stem}.png"), cv2.IMREAD_UNCHANGED)
        if rgb is None or depth is None:
            continue
        dm = depth.astype(np.float32) * 0.001
        mask = tape_pixels(rgb) & (dm > MIN_DEPTH) & (dm < MAX_DEPTH)
        ys, xs = np.nonzero(mask)
        if len(ys) < 200:
            continue
        fx, fy, cx, cy = intrinsic_from_camera(image.camera)
        c2w = np.asarray(image.cam_from_world().inverse().matrix(), dtype=np.float64)
        z = dm[ys, xs]
        u = (xs - cx) / fx
        v = (ys - cy) / fy
        cam = np.stack([u * z, v * z, z], axis=1)
        world = (c2w[:3, :3] @ cam.T).T + c2w[:3, 3]
        pts.append(world)
    if not pts:
        raise RuntimeError("no tape found")
    allp = np.concatenate(pts)
    # keep points near the floor (median height) only -> drop yellow objects
    # that are not on the floor
    med_y = float(np.median(allp[:, 1]))
    on_floor = allp[np.abs(allp[:, 1] - med_y) < 0.12]
    if len(on_floor) < 500:
        raise RuntimeError("not enough floor-tape points")
    # cluster on (x, z) with a 20cm grid; keep the densest cluster
    gx = np.floor((on_floor[:, 0] - on_floor[:, 0].min()) / 0.2).astype(int)
    gz = np.floor((on_floor[:, 2] - on_floor[:, 2].min()) / 0.2).astype(int)
    cells, counts = np.unique(gx * 10000 + gz, return_counts=True)
    best = cells[int(np.argmax(counts))]
    bx, bz = divmod(int(best), 10000)
    sel = on_floor[(gx == bx) & (gz == bz)]
    # add neighbouring cells in the cluster
    for dx, dz in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        m = (gx == bx + dx) & (gz == bz + dz)
        sel = np.concatenate([sel, on_floor[m]]) if np.any(m) else sel
    lo = np.min(sel, axis=0)
    hi = np.max(sel, axis=0)
    return lo, hi


def fuse(recon, lo, hi, voxel_length, sdf_trunc):
    """Fuse sampled frames, keeping only pixels whose 3D point is near the tape box."""
    vol = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_length, sdf_trunc=sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
    margin = 0.05
    lo_m, hi_m = lo - margin, hi + margin
    n = 0
    for image in sorted(recon.images.values(), key=lambda im: im.name)[::4]:
        stem = Path(image.name).stem
        rgb = cv2.cvtColor(cv2.imread(str(COLOR / f"{stem}.jpg"),
                                      cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        depth = cv2.imread(str(DEPTH / f"{stem}.png"), cv2.IMREAD_UNCHANGED)
        if rgb is None or depth is None:
            continue
        dm = depth.astype(np.float32) * 0.001
        fx, fy, cx, cy = intrinsic_from_camera(image.camera)
        m3x4 = np.asarray(image.cam_from_world().matrix(), dtype=np.float64)
        w2c = np.eye(4, dtype=np.float64)
        w2c[:3, :] = m3x4
        c2w = np.asarray(image.cam_from_world().inverse().matrix(), dtype=np.float64)

        ys, xs = np.nonzero((dm > MIN_DEPTH) & (dm < MAX_DEPTH))
        if len(ys) == 0:
            continue
        z = dm[ys, xs]
        u = (xs - cx) / fx
        v = (ys - cy) / fy
        cam = np.stack([u * z, v * z, z], axis=1)
        world = (c2w[:3, :3] @ cam.T).T + c2w[:3, 3]
        keep = np.all((world >= lo_m) & (world <= hi_m), axis=1)
        if not np.any(keep):
            continue
        # zero out everything outside the crop so only the tape area is fused
        depth_crop = np.zeros_like(depth)
        depth_crop[ys[keep], xs[keep]] = depth[ys[keep], xs[keep]]

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(np.ascontiguousarray(rgb)),
            o3d.geometry.Image(np.ascontiguousarray(depth_crop)),
            depth_scale=1000.0, depth_trunc=MAX_DEPTH,
            convert_rgb_to_intensity=False)
        intr = o3d.camera.PinholeCameraIntrinsic(
            rgb.shape[1], rgb.shape[0], fx, fy, cx, cy)
        vol.integrate(rgbd, intr, w2c)
        n += 1
    pcd = vol.extract_point_cloud()
    p = np.asarray(pcd.points)
    in_region = np.all((p >= lo - 0.01) & (p <= hi + 0.01), axis=1)
    reg = p[in_region]
    if len(reg) == 0:
        return n, len(p), 0, 0.0, 0
    # floor height = modal y of the region
    ybins = np.floor((reg[:, 1] - reg[:, 1].min()) / 0.005).astype(int)
    floor_y = reg[:, 1].min() + (np.bincount(ybins).argmax() + 0.5) * 0.005
    # tape slab: 5mm .. 35mm above the floor
    slab = reg[(reg[:, 1] > floor_y + 0.005) & (reg[:, 1] < floor_y + 0.035)]
    return n, len(p), int(len(slab)), float(floor_y), int(len(reg))


def main() -> None:
    recon = pycolmap.Reconstruction(str(SFM))
    lo, hi = tape_3d_bbox(recon)
    print(f"[tape] 3D bbox lo={np.round(lo,3)} hi={np.round(hi,3)} "
          f"size={np.round(hi-lo,3)} m")
    print("       floor tape region, tape is ~1-3cm thick on the floor.")
    for vox, trunc, label in [
        (0.010, 0.040, "current : voxel=1cm  trunc=4cm"),
        (0.003, 0.012, "finer   : voxel=3mm  trunc=1.2cm"),
        (0.001, 0.004, "finest  : voxel=1mm  trunc=4mm"),
    ]:
        frames, total, slab_pts, floor_y, region_pts = fuse(recon, lo, hi, vox, trunc)
        print(f"[{label}] fused {frames} frames -> total {total} pts; "
              f"region {region_pts} pts; tape-slab pts = {slab_pts} "
              f"(floor_y={floor_y:.3f})")


if __name__ == "__main__":
    sys.exit(main())
