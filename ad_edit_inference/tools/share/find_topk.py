import torch
import torch.nn.functional as F


def find_top_k_results(emb1, emb2, k=5):
    with torch.no_grad():
        emb1_norm = F.normalize(emb1, p=2, dim=1)   # (nq, d)
        emb2_norm = F.normalize(emb2, p=2, dim=1)   # (nc, d)
        # 计算相似度矩阵 (nq, nc)
        sim_matrix = emb1_norm @ emb2_norm.t()
        # 对每个查询，取 top k
        topk_vals, topk_idx = torch.topk(
            sim_matrix, k, dim=1, largest=True, sorted=True)
        return topk_idx, topk_vals
