#!/usr/bin/env python
# coding: utf-8

# In[1]:

import os

#get_ipython().run_line_magic('load_ext', 'autoreload')
#get_ipython().run_line_magic('autoreload', '2')

from pathlib import Path
from pprint import pformat

from hloc import (
    extract_features,
    match_features,
    pairs_from_covisibility,
    pairs_from_retrieval,
)
from hloc import colmap_from_nvm, triangulation, localize_sfm, visualization

if __name__ == "__main__":

    # # Pipeline for outdoor day-night visual localization

    # ## Setup
    # Here we declare the paths to the dataset, the reconstruction and localization outputs, and we choose the feature extractor and the matcher. You only need to download the [Aachen Day-Night dataset](https://www.visuallocalization.net/datasets/) and put it in `datasets/aachen/`, or change the path.

    # In[5]:


    dataset = Path("./datasets/Aachen-Day-Night/")  # change this if your dataset is somewhere else
    images = dataset / "images/images_upright"

    outputs = Path("outputs/aachen/")  # where everything will be saved
    sfm_pairs = outputs / "pairs-db-covis20.txt"  # top 20 most covisible in SIFT model
    loc_pairs = outputs / "pairs-query-netvlad20.txt"  # top 20 retrieved by NetVLAD
    reference_sfm = outputs / "sfm_superpoint+superglue"  # the SfM model we will build
    results = outputs / "Aachen_hloc_superpoint+superglue_netvlad20.txt"  # the result file

    # list the standard configurations available
    print(f"Configs for feature extractors:\n{pformat(extract_features.confs)}")
    print(f"Configs for feature matchers:\n{pformat(match_features.confs)}")


    # In[3]:


    # pick one of the configurations for image retrieval, local feature extraction, and matching
    # you can also simply write your own here!
    retrieval_conf = extract_features.confs["netvlad"]
    feature_conf = extract_features.confs["superpoint_aachen"]
    matcher_conf = match_features.confs["superglue"]


    # ## Extract local features for database and query images

    # In[ ]:


    features = extract_features.main(feature_conf, images, outputs)


    # The function returns the path of the file in which all the extracted features are stored.

    # ## Generate pairs for the SfM reconstruction
    # Instead of matching all database images exhaustively, we exploit the existing SIFT model to find which image pairs are the most covisible. We first convert the SIFT model from the NVM to the COLMAP format, and then do a covisiblity search, selecting the top 20 most covisibile neighbors for each image.

    # In[ ]:


    colmap_from_nvm.main(
        dataset / "3D-models/aachen_cvpr2018_db.nvm",
        dataset / "3D-models/database_intrinsics.txt",
        dataset / "aachen.db",
        outputs / "sfm_sift",
    )

    pairs_from_covisibility.main(outputs / "sfm_sift", sfm_pairs, num_matched=20)


    # ## Match the database images

    # In[ ]:


    sfm_matches = match_features.main(
        matcher_conf, sfm_pairs, feature_conf["output"], outputs
    )


    # The function returns the path of the file in which all the computed matches are stored.

    # ## Triangulate a new SfM model from the given poses
    # We triangulate the sparse 3D pointcloud given the matches and the reference poses stored in the SIFT COLMAP model.

    # In[ ]:


    reconstruction = triangulation.main(
        reference_sfm, outputs / "sfm_sift", images, sfm_pairs, features, sfm_matches
    )


    # ## Find image pairs via image retrieval
    # We extract global descriptors with NetVLAD and find for each image the $k$ most similar ones. A larger $k$ improves the robustness of the localization for difficult queries but makes the matching more expensive. Using $k{=}10{-}20$ is generally a good tradeoff but $k{=}50$ gives the best results for the Aachen Day-Night dataset.

    # In[ ]:


    global_descriptors = extract_features.main(retrieval_conf, images, outputs)
    pairs_from_retrieval.main(
        global_descriptors, loc_pairs, num_matched=20, db_prefix="db", query_prefix="query"
    )


    # ## Match the query images

    # In[ ]:


    loc_matches = match_features.main(
        matcher_conf, loc_pairs, feature_conf["output"], outputs
    )


    # ## Localize!
    # Perform hierarchical localization using the precomputed retrieval and matches. The file `Aachen_hloc_superpoint+superglue_netvlad50.txt` will contain the estimated query poses. Have a look at `Aachen_hloc_superpoint+superglue_netvlad50.txt_logs.pkl` to analyze some statistics and find failure cases.

    # In[ ]:


    localize_sfm.main(
        reconstruction,
        dataset / "queries/*_time_queries_with_intrinsics.txt",
        loc_pairs,
        features,
        loc_matches,
        results,
        covisibility_clustering=False,
    )  # not required with SuperPoint+SuperGlue


    # ## Visualizing the SfM model
    # We visualize some of the database images with their detected keypoints.

    # Color the keypoints by track length: red keypoints are observed many times, blue keypoints few.

    # In[22]:


    visualization.visualize_sfm_2d(reconstruction, images, n=1, color_by="track_length")


    # Color the keypoints by visibility: blue if sucessfully triangulated, red if never matched.

    # In[ ]:


    visualization.visualize_sfm_2d(reconstruction, images, n=1, color_by="visibility")


    # Color the keypoints by triangulated depth: red keypoints are far away, blue keypoints are closer.

    # In[20]:


    visualization.visualize_sfm_2d(reconstruction, images, n=1, color_by="depth")


    # ## Visualizing the localization
    # We parse the localization logs and for each query image plot matches and inliers with a few database images.

    # In[10]:


    visualization.visualize_loc(
        results, images, reconstruction, n=1, top_k_db=1, prefix="query/night", seed=2
    )

