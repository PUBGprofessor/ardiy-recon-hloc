import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

root_dir=r'F:/data3r/reloc/south-building/'
#set current directory to root_dir
os.chdir(root_dir)

featype='megaloc'
imset='query'

# ========== 配置区域 ==========
QUERY_H5_PATH = f"global-feats-{featype}-{imset}.h5"
DB_H5_PATH = f"global-feats-{featype}-db1.h5"
TOP_K = 5  # 检索前 K 个
# 如果 h5 里的 image_paths 是相对路径，你可以指定一个根目录
QUERY_IMG_ROOT = f"./{imset}/"   # e.g. "/path/to/query/images"
DB_IMG_ROOT = "./db1/"      # e.g. "/path/to/db/images"
# =============================


def load_features_from_h5_v1(h5_path, feat_key="features", path_key="image_paths"):
    """
    从 h5 文件中读取特征和图像路径。
    假设:
      - features: (N, D)
      - image_paths: (N,)
    """
    with h5py.File(h5_path, "r") as f:
        feats = np.asarray(f[feat_key])  # (N, D)
        paths = np.asarray(f[path_key])  # (N,)
        
        # 可能是 bytes，需要转成 str
        if isinstance(paths[0], bytes):
            paths = np.array([p.decode("utf-8") for p in paths])
    return feats, paths


def load_features_from_h5(h5_path,
                          desc_key="global_descriptor",
                          to_float32=True,
                          sort_keys=True):
    """
    读取类似：
        /P1180141.JPG/global_descriptor
        /P1180141.JPG/image_size
        ...
    这种结构的 h5 文件。

    返回：
        feats: (N, D) float32 or 原 dtype
        paths: (N,) 图像文件名（字符串）
    """
    feats_list = []
    paths = []

    with h5py.File(h5_path, "r") as f:
        # f.keys() 是所有图像名，比如 "P1180141.JPG"
        img_keys = list(f.keys())
        if sort_keys:
            img_keys = sorted(img_keys)

        for name in img_keys:
            grp = f[name]
            if desc_key not in grp:
                # 没有 global_descriptor 的 group 直接跳过
                continue

            desc = grp[desc_key][()]  # 读成 numpy 数组

            # 有些实现会存成 (1, D)，变成 (D,)
            if desc.ndim > 1:
                desc = desc.reshape(-1)

            if to_float32:
                desc = desc.astype(np.float32)

            feats_list.append(desc)
            paths.append(name)  # 这里就是图像文件名

    feats = np.stack(feats_list, axis=0)  # (N, D)
    paths = np.array(paths)

    return feats, paths


def l2_normalize(feats, eps=1e-12):
    """对每个特征向量做 L2 归一化，便于用点积当作余弦相似度。"""
    norms = np.linalg.norm(feats, axis=1, keepdims=True) + eps
    return feats / norms


def retrieve_topk(query_feats, db_feats, k=5):
    """
    用余弦相似度做检索。
    query_feats: (Nq, D), 已经 L2 归一化
    db_feats: (Nd, D), 已经 L2 归一化
    返回:
        indices: (Nq, k) 每个 query 对应的 topk 检索结果索引
        scores: (Nq, k) 对应的相似度
    """
    # 余弦相似度 = 归一化后点积
    # 相当于对每个 query 计算 sim = db_feats @ q
    sims = query_feats @ db_feats.T  # (Nq, Nd)
    
    # 对每个 query，取相似度最大的 k 个索引
    topk_idx = np.argsort(-sims, axis=1)[:, :k]  # 负号表示从大到小
    topk_scores = np.take_along_axis(sims, topk_idx, axis=1)
    return topk_idx, topk_scores


def load_image(img_path):
    """安全地读一张图像，失败时返回一张灰背景图。"""
    try:
        img = Image.open(img_path).convert("RGB")
    except Exception as e:
        print(f"[WARN] Failed to load image: {img_path}, error: {e}")
        # 生成一张灰色占位图
        img = Image.new("RGB", (256, 256), (128, 128, 128))
    return img


def visualize_retrieval(
    query_idx,
    query_paths,
    db_paths,
    topk_indices,
    topk_scores,
    query_root=".",
    db_root=".",
    k=5,
):
    """
    可视化一个 query 及其 top-K 检索结果。
    """
    # 防御性处理
    k = min(k, topk_indices.shape[1])

    # 创建画布：1 行 (1 + k) 列
    plt.figure(figsize=(3 * (k + 1), 4))

    # --- 显示 query 图像 ---
    q_path = os.path.join(query_root, query_paths[query_idx])
    q_img = load_image(q_path)
    ax = plt.subplot(1, k + 1, 1)
    ax.imshow(q_img)
    ax.set_title("Query", fontsize=10)
    ax.axis("off")

    # --- 显示 top-K 检索结果 ---
    for rank in range(k):
        db_idx = topk_indices[query_idx, rank]
        score = topk_scores[query_idx, rank]
        db_path = os.path.join(db_root, db_paths[db_idx])
        db_img = load_image(db_path)

        ax = plt.subplot(1, k + 1, rank + 2)
        ax.imshow(db_img)
        ax.set_title(f"Rank {rank+1}\n{score:.3f}", fontsize=9)
        ax.axis("off")

    plt.tight_layout()
    plt.show(block=True)
    #wait close
    #plt.waitforbuttonpress()
    #plt.close()


def main():
    # 1. 读取特征
    print("Loading features...")
    query_feats, query_paths = load_features_from_h5(QUERY_H5_PATH)
    db_feats, db_paths = load_features_from_h5(DB_H5_PATH)

    print(f"Query features: {query_feats.shape}, images: {len(query_paths)}")
    print(f"DB features:    {db_feats.shape}, images: {len(db_paths)}")

    # 2. L2 归一化
    query_feats = l2_normalize(query_feats)
    db_feats = l2_normalize(db_feats)

    # 3. 前 K 检索
    print("Retrieving top-K neighbors...")
    topk_indices, topk_scores = retrieve_topk(query_feats, db_feats, k=TOP_K)

    # 4. 随机查看几个 query 的检索结果
    #    你也可以指定固定的 query 索引
    num_to_show = 500
    num_to_show = min(num_to_show, len(query_paths))

    for qi in range(num_to_show):
        print(f"\n=== Query {qi} : {query_paths[qi]} ===")
        for rank in range(TOP_K):
            db_idx = topk_indices[qi, rank]
            print(
                f"  Rank {rank+1}: {db_paths[db_idx]}  "
                f"score={topk_scores[qi, rank]:.4f}"
            )

        visualize_retrieval(
            query_idx=qi,
            query_paths=query_paths,
            db_paths=db_paths,
            topk_indices=topk_indices,
            topk_scores=topk_scores,
            query_root=QUERY_IMG_ROOT,
            db_root=DB_IMG_ROOT,
            k=TOP_K,
        )


if __name__ == "__main__":
    main()
