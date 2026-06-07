#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import argparse
from collections import defaultdict


def get_ad_key(obj):
    """广告唯一标识：(chunk_id, ad_id)，没有 chunk_id 时退化为 ad_id。"""
    ad_id = obj.get("ad_id")
    if "chunk_id" in obj:
        return (obj.get("chunk_id"), ad_id)
    return ad_id


def has_empty_caption(obj):
    """
    判断该条记录是否视为 caption 为空：
    - caption 字段缺失
    - caption 为 None
    - caption 为 "" 或 全是空白
    """
    if "caption" not in obj:
        return True
    cap = obj.get("caption")
    if cap is None:
        return True
    if isinstance(cap, str) and cap.strip() == "":
        return True
    return False


def main():
    parser = argparse.ArgumentParser(
        description=(
            "过滤 JSONL：只保留所有 clip 的 caption 均非空的广告。\n"
            "以 (chunk_id, ad_id) 作为广告单位，只要其中一个 clip caption 为空，该广告全丢弃。"
        )
    )
    parser.add_argument(
        "--input",
        type=str,
        default="/data/phd/miltonzhou/sft/data_preprocess/filter_data/BLCT_TEST_1109.jsonl",
        help="输入 JSONL 文件路径（原始数据，不会被修改）"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/data/phd/miltonzhou/sft/data_preprocess/filter_data/BLCT_TEST_no_empty_caption_1109.jsonl",
        help="输出筛选后 JSONL 文件路径"
    )
    args = parser.parse_args()

    input_path = args.input
    output_path = args.output

    # -------- 第一遍：统计每个广告是否有空 caption --------
    ad_has_empty = defaultdict(bool)
    all_ads = set()

    with open(input_path, "r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                # 坏行直接忽略
                continue

            key = get_ad_key(obj)
            all_ads.add(key)

            if has_empty_caption(obj):
                ad_has_empty[key] = True

    # 没有空 caption 的广告集合
    valid_ads = {key for key in all_ads if not ad_has_empty[key]}

    # -------- 第二遍：写入保留广告的数据到新文件 --------
    kept_ads = set()
    kept_lines = 0

    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:

        for line in fin:
            raw = line.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                continue

            key = get_ad_key(obj)
            if key in valid_ads:
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                kept_lines += 1
                kept_ads.add(key)

    # -------- 打印统计信息 --------
    print(f"原始广告数: {len(all_ads)}")
    print(f"筛选后剩余广告数: {len(kept_ads)}")
    print(f"筛选后剩余行数(clip 数): {kept_lines}")


if __name__ == "__main__":
    main()
