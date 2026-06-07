#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
统计 SFT 数据集分布：
1) clips_per_video        —— (chunk_id, ad_id) 下唯一 clip_id 数
2) seconds_per_clip       —— 单条记录 v_tok 中 <|frame_start|> 个数
3) lines_per_video        —— (chunk_id, ad_id) 下非空 text 条数
4) lines_per_clip         —— 单条记录是否有台词：非空=1，空=0
5) seconds_per_video      —— (chunk_id, ad_id) 下各 clip 秒数求和
6) clip_text_len          —— 单条记录 text 的字符数（len(text.strip())）
7) ad_text_len            —— (chunk_id, ad_id) 下 text 字符数求和
8) unique_products        —— 非空 product 的唯一种类数

结果保存到 data_stats.csv，格式：metric,key,count
"""

import os
import json
import argparse
from collections import Counter, defaultdict
from typing import Dict, Set, Tuple, Iterable, Any

FRAME_TOKEN = "<|frame_start|>"

def iter_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue

def collect_records(input_path: str):
    if os.path.isfile(input_path):
        yield from iter_jsonl(input_path)
    else:
        for name in sorted(os.listdir(input_path)):
            if name.lower().endswith(".jsonl"):
                full = os.path.join(input_path, name)
                if os.path.isfile(full):
                    yield from iter_jsonl(full)

def safe_text_len(txt: Any) -> int:
    if isinstance(txt, str):
        return len(txt.strip())
    return 0

def is_nonempty_text(txt: Any) -> bool:
    return isinstance(txt, str) and txt.strip() != ""

def count_seconds(v_tok: Any) -> int:
    if isinstance(v_tok, str):
        return v_tok.count(FRAME_TOKEN)
    return 0

def main():
    parser = argparse.ArgumentParser(description="SFT 数据集多项分布统计")
    parser.add_argument("--input", default="/data/phd/miltonzhou/sft/data_preprocess/filter_data/data_chunk_1102/Final_CT_train_1109.jsonl", help="输入的 jsonl 文件或包含 jsonl 的目录")
    parser.add_argument("--output", default="data_stats.csv", help="输出 CSV 文件路径 (默认: data_stats.csv)")
    args = parser.parse_args()

    # —— 累计容器 —— #
    video_to_clips: Dict[Tuple[Any, Any], Set[Any]] = defaultdict(set)   # 统计唯一 clip_id
    seconds_per_clip_counter: Counter = Counter()                        # clip 秒数分布
    product_set: Set[str] = set()                                        # product 去重
    clip_text_len_counter: Counter = Counter()                           # clip 级字符数分布
    ad_text_len_sum: Dict[Tuple[Any, Any], int] = defaultdict(int)       # ad 级字符数和
    lines_per_video_sum: Dict[Tuple[Any, Any], int] = defaultdict(int)   # ad 级台词数（非空 text 条数）
    seconds_per_video_sum: Dict[Tuple[Any, Any], int] = defaultdict(int) # ad 级秒数和
    lines_per_clip_counter: Counter = Counter()                          # clip 级台词数分布（非空=1，空=0）

    total_records = 0

    for rec in collect_records(args.input):
        total_records += 1

        chunk_id = rec.get("chunk_id")
        ad_id    = rec.get("ad_id")
        clip_id  = rec.get("clip_id")
        video_key = (chunk_id, ad_id)

        # —— 1) 唯一 clip 数 —— #
        if chunk_id is not None and ad_id is not None and clip_id is not None:
            video_to_clips[video_key].add(clip_id)

        # —— 2) clip 秒数 —— #
        sec = count_seconds(rec.get("v_tok"))
        seconds_per_clip_counter[sec] += 1

        # —— 5) ad 级秒数和 —— #
        if chunk_id is not None and ad_id is not None:
            seconds_per_video_sum[video_key] += sec

        # —— product 去重 —— #
        product = rec.get("product")
        if isinstance(product, str) and product.strip():
            product_set.add(product.strip())

        # —— 6) clip 级 text 字符数 —— #
        tlen = safe_text_len(rec.get("text"))
        clip_text_len_counter[tlen] += 1

        # —— 7) ad 级 text 字符数和 —— #
        if chunk_id is not None and ad_id is not None:
            ad_text_len_sum[video_key] += tlen

        # —— 3) ad 级台词数（非空 text 条数）—— #
        if chunk_id is not None and ad_id is not None and is_nonempty_text(rec.get("text")):
            lines_per_video_sum[video_key] += 1

        # —— 4) clip 级台词数（此处将单条记录的 text 是否非空视为 0/1）—— #
        lines_per_clip_counter[1 if is_nonempty_text(rec.get("text")) else 0] += 1

    # —— 汇总分布 —— #
    # 1) clips_per_video
    clips_per_video_counter: Counter = Counter()
    for clip_ids in video_to_clips.values():
        clips_per_video_counter[len(clip_ids)] += 1

    # 5) seconds_per_video
    seconds_per_video_counter: Counter = Counter()
    for total_sec in seconds_per_video_sum.values():
        seconds_per_video_counter[total_sec] += 1

    # 3) lines_per_video
    lines_per_video_counter: Counter = Counter()
    for total_lines in lines_per_video_sum.values():
        lines_per_video_counter[total_lines] += 1

    # 7) ad_text_len
    ad_text_len_counter: Counter = Counter()
    for total_len in ad_text_len_sum.values():
        ad_text_len_counter[total_len] += 1

    # 8) unique_products
    unique_products_count = len(product_set)

    # —— 写 CSV —— #
    with open(args.output, "w", encoding="utf-8") as f:
        f.write("metric,key,count\n")

        # clips_per_video
        for k in sorted(clips_per_video_counter.keys()):
            f.write(f"clips_per_video,{k},{clips_per_video_counter[k]}\n")
        f.write("\n")

        # seconds_per_clip
        for s in sorted(seconds_per_clip_counter.keys()):
            f.write(f"seconds_per_clip,{s},{seconds_per_clip_counter[s]}\n")
        f.write("\n")

        # # lines_per_video
        # for L in sorted(lines_per_video_counter.keys()):
        #     f.write(f"lines_per_video,{L},{lines_per_video_counter[L]}\n")
        # f.write("\n")

        # # lines_per_clip (0/1 统计)
        # for c in sorted(lines_per_clip_counter.keys()):
        #     f.write(f"lines_per_clip,{c},{lines_per_clip_counter[c]}\n")
        # f.write("\n")

        # seconds_per_video（新增）
        for S in sorted(seconds_per_video_counter.keys()):
            f.write(f"seconds_per_video,{S},{seconds_per_video_counter[S]}\n")
        f.write("\n")

        # clip_text_len
        for Lc in sorted(clip_text_len_counter.keys()):
            f.write(f"clip_text_len,{Lc},{clip_text_len_counter[Lc]}\n")
        f.write("\n")

        # ad_text_len
        for La in sorted(ad_text_len_counter.keys()):
            f.write(f"ad_text_len,{La},{ad_text_len_counter[La]}\n")
        f.write("\n")

        # unique_products
        f.write(f"unique_products,__all__,{unique_products_count}\n")

    print(f"Total records read: {total_records}")
    print(f"Total videos (unique (chunk_id, ad_id)): {len(video_to_clips)}")
    print(f"Unique non-empty product types: {unique_products_count}")
    print(f"Saved stats to: {args.output}")

if __name__ == "__main__":
    main()
