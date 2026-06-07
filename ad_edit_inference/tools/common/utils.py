import torch
import numpy as np

def threshold_cos_partition(emb: torch.Tensor, thr: float = 0.85):
    """
    emb: (b, h)  一组按顺序排列的向量
    thr: 阈值，比如 0.8
    返回一个分割点列表 cuts，比如 [k1, k2,...]，
    表示切分后得到的区间是 [0:k1], [k1:k2], [k2:...]。
    """
    # 1) 归一化
    # emb_norm = F.normalize(emb, p=2, dim=1)      # (b, h)
    # 2) 相邻余弦相似度
    sims = torch.sum(emb[:-1] * emb[1:], dim=1)  # (b-1,)
    # 3) 找到所有 < thr 的索引 i
    #    这样 i 表示 emb[i] ↔ emb[i+1] 相似度低于 thr，
    #    分割点就在 i+1 处
    print(sims)
    low_idxs = torch.nonzero(sims < thr, as_tuple=False).squeeze(1)
    cuts = (low_idxs + 1).tolist()
    return cuts


def tensor_to_str_matrix(t: torch.Tensor, modality="video"):
    """
    输入：二维整数 tensor t，shape=(H, W)
    输出：同维度的 Python 嵌套列表，每个元素形如 "v_{列号}_{值}"
    """
    # 1. 一次性搬到 CPU 并转 NumPy
    arr = t.cpu().numpy()                     # shape (H, W)

    # 2. 构造列号矩阵
    H, W = arr.shape
    cols = np.arange(W, dtype=int).reshape(1, W)  # shape (1, W)
    cols = np.repeat(cols, H, axis=0)             # shape (H, W)

    # 3. 转成字符串矩阵
    col_str = cols.astype(str)                    # e.g. [["0","1","2"], …]
    val_str = arr.astype(str)                     # 每个位置的值转为字符串

    # 4. 批量拼接："v_" + col_str + "_" + val_str
    tmp = np.char.add(col_str, np.char.add("_", val_str))
    if modality == "video":
        full = np.char.add("<v_", tmp)                 # shape (H, W) 的 str ndarray
    else:
        full = np.char.add("<a_", tmp)
    full = np.char.add(full, ">")
    matrix = full.tolist()
    str_list = ["".join(row) for row in matrix]
    return str_list