#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Filter ads by per-chunk video_id consistency across multiple JSONL chunks.

约定:
- 每个 chunk 内的 ad_id 是局部唯一的。
- 不同 chunk 之间 ad_id 数值相同 ≠ 同一个广告，必须分开处理。
- video_id = frame_id 去掉最后 4 位字符 (若长度<=4，则使用原值)。

对每个 (chunk_id, ad_id)：
  1) 统计该 ad 的所有 clip 的 video_id 分布
  2) 找出出现次数最多的 video_id = max_count
  3) 一致性比例 = max_count / total_clips_for_that_ad
  4) 若 一致性比例 >= r 且 total_clips >= min_clips，则该 ad (在该 chunk 中) 通过

输出:
- 一个汇总 JSONL 文件，包含所有通过的 (chunk_id, ad_id) 的原始行。
- 每行增加字段 "chunk_id": <int> (放在最前面)，用于区分不同 chunk 的同号 ad_id。
- 末尾打印全局统计信息。
"""

import os
import json
import argparse
from collections import defaultdict
from typing import Dict, Tuple
from tqdm import tqdm


# ---------------------------
# 工具函数
# ---------------------------

# def iter_chunk_files(base_dir: str, num_chunks: int,
#                      prefix: str = "ct_chunk_",
#                      suffix: str = ".jsonl"):
#     """按约定命名返回存在的 chunk 文件路径 (chunk_id, path)。"""
#     for i in range(num_chunks):
#         path = os.path.join(base_dir, f"{prefix}{i}{suffix}")
#         if os.path.exists(path):
#             yield i, path
#         # 不存在就跳过，不报错

def iter_chunk_files(base_dir: str, num_chunks: int,
                     prefix: str = "ct_chunk_",
                     suffix: str = ".jsonl"):
    """只处理 ct_chunk_99.jsonl。"""
    i = 99
    path = os.path.join(base_dir, f"{prefix}{i}{suffix}")
    if os.path.exists(path):
        yield i, path
    else:
        print(f"[WARN] Chunk file not found: {path}")



def extract_video_id(frame_id: str) -> str:
    """
    根据规则从 frame_id 提取 video_id:
    - 若长度 > 4, 取除最后 4 位外的前缀
    - 否则直接返回原字符串
    """
    if not isinstance(frame_id, str):
        frame_id = str(frame_id)
    if len(frame_id) > 4:
        return frame_id[:-4]
    return frame_id


def select_valid_ads_for_chunk(chunk_ad_video_counts: Dict[int, Dict[str, int]],
                               chunk_ad_total_clips: Dict[int, int],
                               r: float,
                               min_clips: int) -> Dict[int, float]:
    """
    对单个 chunk 内:
      输入:
        - chunk_ad_video_counts: {ad_id: {video_id: count}}
        - chunk_ad_total_clips: {ad_id: total_clips}
      返回:
        - valid_ads: {ad_id: consistency_ratio}
    """
    valid_ads: Dict[int, float] = {}
    for ad_id, vid_counts in chunk_ad_video_counts.items():
        total = chunk_ad_total_clips.get(ad_id, 0)
        if total < min_clips or total <= 0:
            continue
        max_count = max(vid_counts.values()) if vid_counts else 0
        ratio = max_count / float(total)
        if ratio >= r:
            valid_ads[ad_id] = ratio
    return valid_ads


# ---------------------------
# 主流程
# ---------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Filter multi-chunk clip_table JSONL by per-chunk ad video_id consistency."
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default="/data/phd/miltonzhou/sft/data_preprocess/filter_data/data_chunk_1102",
        help="包含 ct_chunk_*.jsonl 的目录"
    )
    parser.add_argument(
        "--num_chunks",
        type=int,
        default=30,
        help="chunk 数量 (假定文件名为 ct_chunk_0.jsonl ... ct_chunk_{num_chunks-1}.jsonl)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=1,
        help="video_id 一致性阈值 r: max(video_id_count)/total_clips >= r 则通过"
    )
    parser.add_argument(
        "--min_clips",
        type=int,
        default=1,
        help="该 chunk 内每个 ad 至少需要多少个 clip 才参与筛选"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/data/phd/miltonzhou/sft/data_preprocess/filter_data/data_chunk_1102/filtered_by_video_r100_per_chunk_eval.jsonl",
        help="输出 JSONL 路径 (默认: base_dir/filtered_by_video_rXX_per_chunk.jsonl)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    base_dir = args.base_dir
    num_chunks = args.num_chunks
    r = args.threshold
    min_clips = args.min_clips

    if args.output:
        output_path = args.output
    else:
        r_tag = int(r * 100)
        output_path = os.path.join(
            base_dir, f"filtered_by_video_r{r_tag}_per_chunk.jsonl"
        )

    print(f"Base dir      : {base_dir}")
    print(f"Num chunks    : {num_chunks}")
    print(f"Threshold r   : {r}")
    print(f"Min clips/ad  : {min_clips}")
    print(f"Output        : {output_path}")

    # 全局统计
    global_total_clips = 0
    global_total_ads = 0
    global_valid_ads = 0
    global_kept_clips = 0
    global_ratio_sum = 0.0  # 用于计算 valid ads 的平均一致性

    chunk_files = list(iter_chunk_files(base_dir, num_chunks))
    if not chunk_files:
        print("[WARN] No chunk files found. Please check base_dir/num_chunks.")
        return

    # 打开输出文件，一次写入所有 chunk 的通过样本
    with open(output_path, "w", encoding="utf-8") as out_f:
        # 逐个 chunk 处理，避免跨 chunk 混 ad_id
        for chunk_id, path in tqdm(chunk_files, desc="Processing chunks", unit="chunk"):
            # ---- 第一遍：统计该 chunk 内的 (ad_id -> video_id 分布) ----
            chunk_ad_video_counts: Dict[int, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
            chunk_ad_total_clips: Dict[int, int] = defaultdict(int)

            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    s = line.strip()
                    if not s:
                        continue
                    try:
                        obj = json.loads(s)
                    except json.JSONDecodeError:
                        continue

                    ad_id = obj.get("ad_id", None)
                    frame_id = obj.get("frame_id", "")

                    if ad_id is None:
                        continue
                    try:
                        ad_id = int(ad_id)
                    except (TypeError, ValueError):
                        continue

                    video_id = extract_video_id(frame_id)
                    chunk_ad_total_clips[ad_id] += 1
                    chunk_ad_video_counts[ad_id][video_id] += 1

            # 统计该 chunk 的基本信息
            chunk_total_ads = len(chunk_ad_total_clips)
            chunk_total_clips = sum(chunk_ad_total_clips.values())
            global_total_ads += chunk_total_ads
            global_total_clips += chunk_total_clips

            if chunk_total_ads == 0:
                continue

            # ---- 决定该 chunk 内哪些 ad 通过 ----
            valid_ads = select_valid_ads_for_chunk(
                chunk_ad_video_counts,
                chunk_ad_total_clips,
                r=r,
                min_clips=min_clips,
            )

            chunk_valid_ads = len(valid_ads)
            global_valid_ads += chunk_valid_ads
            # 记录一致性比例总和用于计算平均
            global_ratio_sum += sum(valid_ads.values())

            # ---- 第二遍：写出该 chunk 内通过的行，附加 chunk_id 字段 ----
            if chunk_valid_ads > 0:
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        s = line.strip()
                        if not s:
                            continue
                        try:
                            obj = json.loads(s)
                        except json.JSONDecodeError:
                            continue

                        ad_id = obj.get("ad_id", None)
                        if ad_id is None:
                            continue
                        try:
                            ad_id = int(ad_id)
                        except (TypeError, ValueError):
                            continue

                        if ad_id in valid_ads:
                            # 把 chunk_id 放到最前面，其余字段原样附加
                            new_obj = {"chunk_id": chunk_id}
                            new_obj.update(obj)
                            out_f.write(json.dumps(new_obj, ensure_ascii=False) + "\n")
                            global_kept_clips += 1

    # ---- 全局 Summary ----
    avg_consistency_valid = (
        (global_ratio_sum / global_valid_ads)
        if global_valid_ads > 0 else 0.0
    )

    print("\n=== Summary ===")
    print(f"Total clips (all chunks)      : {global_total_clips}")
    print(f"Total ads   (all chunks)      : {global_total_ads}")
    print(f"Valid ads   (all chunks)      : {global_valid_ads}")
    print(f"Kept clips total              : {global_kept_clips}")
    print(f"Avg consistency (valid ads)   : {avg_consistency_valid:.4f}")
    print(f"Saved filtered ads to         : {output_path}")

if __name__ == "__main__":
    main()


### python filter_by_video_consistency_chunks.py --base_dir /data/phd/miltonzhou/sft/data_preprocess/filter_data/data_chunk_1102 --num_chunks 32 --threshold 1

### EVAL：/data/phd/miltonzhou/sft/data_preprocess/filter_data/data_chunk_1102/ct_chunk_99.jsonl