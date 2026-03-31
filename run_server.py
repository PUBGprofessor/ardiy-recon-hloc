
import os

from pathlib import Path

from hloc import (
    extract_features,
)

import torch
import os
import cv2
import numpy as np
from PIL import Image
import PIL.Image

import sys
sys.path.append('../')
from xbase.netcall import encodeObjs,decodeObjs,runServer
from types import SimpleNamespace

def resize_image(image, size, interp):
    if interp.startswith("cv2_"):
        interp = getattr(cv2, "INTER_" + interp[len("cv2_") :].upper())
        h, w = image.shape[:2]
        if interp == cv2.INTER_AREA and (w < size[0] or h < size[1]):
            interp = cv2.INTER_LINEAR
        resized = cv2.resize(image, size, interpolation=interp)
    elif interp.startswith("pil_"):
        interp = getattr(PIL.Image, interp[len("pil_") :].upper())
        resized = PIL.Image.fromarray(image.astype(np.uint8))
        resized = resized.resize(size, resample=interp)
        resized = np.asarray(resized, dtype=image.dtype)
    else:
        raise ValueError(f"Unknown interpolation {interp}.")
    return resized


def prepro_image(image, prepro_conf, size):
    conf=prepro_conf
    if conf.resize_max and (
            conf.resize_force or max(size) > conf.resize_max
        ):
            scale = conf.resize_max / max(size)
            size_new = tuple(int(round(x * scale)) for x in size)
            image = resize_image(image, size_new, conf.interpolation)

    if conf.grayscale:
        image = image[None]
    else:
        image = image.transpose((2, 0, 1))  # HxWxC to CxHxW
    image = image / 255.0
    return image

@torch.no_grad()
def run_server():
    conf = extract_features.confs["megaloc"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model=extract_features.load_model(conf, device)

    default_conf = {
        "globs": ["*.jpg", "*.png", "*.jpeg", "*.JPG", "*.PNG"],
        "grayscale": False,
        "resize_max": None,
        "resize_force": False,
        "interpolation": "cv2_area",  # pil_linear is more accurate but slower
    }
    prepro_conf = SimpleNamespace(**{**default_conf, **conf["preprocessing"]})

    def serverHandler(objs):
        image=objs['image']
        image = image[:, :, ::-1]  # BGR to RGB
        image = image.astype(np.float32)
        size = image.shape[:2][::-1]
        image = prepro_image(image, prepro_conf, size)
        image = torch.from_numpy(image).to(device).unsqueeze(0)
        #add the batch dimension

        with torch.no_grad():
            pred = model({'image':image})
            desc=pred['global_descriptor'].cpu().numpy()
            retObj={"desc":desc}
            return retObj
        
    runServer(serverHandler)
    
    if False:
        image=cv2.imread(r'F:\data3r\reloc\south-building\query1\P1180143.JPG')
        serverHandler({'image':image})

if __name__ == "__main__":
    run_server()



