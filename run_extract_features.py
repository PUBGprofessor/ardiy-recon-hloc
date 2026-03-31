
# In[ ]:
import os
#os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pathlib import Path

from hloc import (
    extract_features,
    match_features,
    reconstruction,
    visualization,
    pairs_from_retrieval,
)

import os
from PIL import Image

def scale_rotate_and_center_crop(
    inpath,
    outpath,
    scale=1.0,
    rotate_deg=0.0,
    ratio=0.5,
    interp=Image.BILINEAR,
    exts=(".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"),
):
    """
    对 inpath 下的所有图片：
      1. scale
      2. rotate (around center)
      3. center crop by ratio
      4. save to outpath with same filename

    参数：
        inpath      : 输入图片目录
        outpath     : 输出图片目录
        scale       : 缩放比例 (>0)
        rotate_deg  : 旋转角度（度，逆时针，正数）
        ratio       : 中心裁剪比例 (0 < ratio <= 1)
        interp      : 插值方式
        exts        : 支持的图片后缀
    """
    assert scale > 0
    assert 0 < ratio <= 1.0

    os.makedirs(outpath, exist_ok=True)

    for fname in os.listdir(inpath):
        if not fname.lower().endswith(exts):
            continue

        in_file = os.path.join(inpath, fname)
        out_file = os.path.join(outpath, fname)

        try:
            img = Image.open(in_file).convert("RGB")
        except Exception as e:
            print(f"[WARN] Failed to load {in_file}: {e}")
            continue

        # ---------- 1. scale ----------
        w, h = img.size
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        img = img.resize((new_w, new_h), interp)

        # ---------- 2. rotate ----------
        # expand=True 保证旋转后不裁掉角
        img = img.rotate(
            rotate_deg,
            resample=interp,
            expand=True
        )

        # ---------- 3. center crop ----------
        W, H = img.size
        crop_w = int(round(W * ratio))
        crop_h = int(round(H * ratio))

        left = (W - crop_w) // 2
        top = (H - crop_h) // 2
        right = left + crop_w
        bottom = top + crop_h

        img = img.crop((left, top, right, bottom))

        # ---------- 4. save ----------
        img.save(out_file)

    print(f"Done. Processed images saved to: {outpath}")



def run_for_imset(ddir, imset):
    images = Path(ddir+imset)

    retrieval_conf = extract_features.confs["megaloc"]
    #retrieval_conf = extract_features.confs["netvlad"]
    #retrieval_conf = extract_features.confs["openibl"]

    retrieval_path = extract_features.main(retrieval_conf, images, ddir)
    #add prefix of imset to the file name
    final_retrieval_path = Path(str(retrieval_path).replace('.h5', '-'+imset+'.h5'))
    #remove final_retrieval_path if it exists
    if final_retrieval_path.exists():
        os.remove(str(final_retrieval_path))
    #rename the file
    os.rename(str(retrieval_path), str(final_retrieval_path))

if __name__ == "__main__":

    ddir=r'F:/data3r/reloc/south-building/'

    #scale_rotate_and_center_crop(ddir+'query/',ddir+'query1',scale=0.5, rotate_deg=0, ratio=0.5)
    #scale_rotate_and_center_crop(ddir+'db/',ddir+'db1',scale=0.5, rotate_deg=0, ratio=0.5)

    #run_for_imset(ddir, 'db1')    
    run_for_imset(ddir, 'query1')




    



