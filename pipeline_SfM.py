#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import os

#get_ipython().run_line_magic('load_ext', 'autoreload')
#get_ipython().run_line_magic('autoreload', '2')

from pathlib import Path

from hloc import (
    extract_features,
    match_features,
    reconstruction,
    visualization,
    pairs_from_retrieval,
)

if __name__ == "__main__":

    # ## Setup
    # In this notebook, we will run SfM reconstruction from scratch on a set of images. We choose the [South-Building dataset](https://openaccess.thecvf.com/content_cvpr_2013/html/Hane_Joint_3D_Scene_2013_CVPR_paper.html) - we will download it later. First, we define some paths.

    # In[ ]:


    images = Path("datasets/South-Building/images/")

    outputs = Path("outputs/sfm2/")
    sfm_pairs = outputs / "pairs-megaloc.txt"
    sfm_dir = outputs / "sfm_superpoint+superglue"

    retrieval_conf = extract_features.confs["netvlad"]
    feature_conf = extract_features.confs["superpoint_aachen"]
    matcher_conf = match_features.confs["superglue"]


    # ## Download the dataset
    # The dataset is simply a set of images. The intrinsic parameters will be extracted from the EXIF data and refined with SfM.

    # In[ ]:


    if not images.exists():
        os.system('wget https://cvg-data.inf.ethz.ch/local-feature-evaluation-schoenberger2017/South-Building.zip -P datasets/')
        os.system('unzip -q datasets/South-Building.zip -d datasets/')


    # ## Find image pairs via image retrieval
    # We extract global descriptors with NetVLAD and find for each image the most similar ones. For smaller dataset we can instead use exhaustive matching via `hloc/pairs_from_exhaustive.py`, which would find $\frac{n(n-1)}{2}$ images pairs.

    # In[ ]:


    retrieval_path = extract_features.main(retrieval_conf, images, outputs)
    feature_path = extract_features.main(feature_conf, images, outputs)

    
    pairs_from_retrieval.main(retrieval_path, sfm_pairs, num_matched=5)


    # ## Extract and match local features

    # In[ ]:


    match_path = match_features.main(
        matcher_conf, sfm_pairs, feature_conf["output"], outputs
    )


    # ## 3D reconstruction
    # Run COLMAP on the features and matches.

    # In[ ]:


    model = reconstruction.main(sfm_dir, images, sfm_pairs, feature_path, match_path)


    # ## Visualization
    # We visualize some of the registered images, and color their keypoint by visibility, track length, or triangulated depth.

    # In[ ]:


    visualization.visualize_sfm_2d(model, images, color_by="visibility", n=5)


    # In[ ]:


    visualization.visualize_sfm_2d(model, images, color_by="track_length", n=5)


    # In[ ]:


    visualization.visualize_sfm_2d(model, images, color_by="depth", n=5)

