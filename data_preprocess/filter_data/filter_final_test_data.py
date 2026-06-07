#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import argparse
from collections import defaultdict


def is_valid_record(obj):
    # 必须包含这三个字段
    if not all(k in obj for k in ("brand", "product", "features")):
        return False

    brand = obj.get("brand")
    product = obj.get("product")
    features = obj.get("features")

    # brand: 非 None 且非空字符串
    if brand is None or (isinstance(brand, str) and brand.strip() == ""):
        return False

    # product: 非 None 且非空字符串
    if product is None or (isinstance(product, str) and product.strip() == ""):
        return False

    # features: 必须是非空列表
    if not isinstance(features, list) or len(features) == 0:
        return False

    return True


def make_ad_key(obj):
    """
    定义广告唯一ID：
    - 若存在 chunk_id，则使用 (chunk_id, ad_id)
    - 否则仅使用 ad_id
    """
    ad_id = obj.get("ad_id")
    if "chunk_id" in obj:
        return (obj.get("chunk_id"), ad_id)
    return ad_id


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Filter JSONL: "
            "1) 保留 brand/product/features 合法记录；"
            "2) 仅保留 clip 数在 [3,60] 的广告；"
            "3) 按广告和 clip_id 排序输出。"
        )
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="输入 JSONL 文件路径"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="输出筛选后 JSONL 文件路径"
    )
    args = parser.parse_args()

    total_lines = 0

    # 第一次遍历：统计每个广告的唯一 clip_id 数量（仅对 is_valid_record 的记录）
    ad_clip_ids = defaultdict(set)

    with open(args.input, "r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            total_lines += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                # 坏行直接跳过
                continue

            if not is_valid_record(obj):
                continue

            key = make_ad_key(obj)
            clip_id = obj.get("clip_id")

            # clip_id 必须存在才计数
            if clip_id is not None:
                ad_clip_ids[key].add(clip_id)

    # 根据 clip 数过滤广告：只保留 3-60 个 clip 的广告
    valid_ads = {
        key for key, clips in ad_clip_ids.items()
        if 3 <= len(clips) <= 60
    }

    # 第二次遍历：收集属于 valid_ads 的记录，后面统一排序输出
    ad_records = defaultdict(list)

    with open(args.input, "r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            if not is_valid_record(obj):
                continue

            key = make_ad_key(obj)
            if key not in valid_ads:
                continue

            ad_records[key].append(obj)

    # 排序并写出：
    # - key 顺序：按 (chunk_id, ad_id) 或 ad_id 升序
    # - 同一 key 内按 clip_id 升序
    def sort_key_for_ad(k):
        # k 可能是 tuple((chunk_id, ad_id)) 或 单个 ad_id
        if isinstance(k, tuple):
            return (k[0], k[1])
        return (0, k)

    kept_ads = 0
    kept_clips = 0

    with open(args.output, "w", encoding="utf-8") as fout:
        for key in sorted(ad_records.keys(), key=sort_key_for_ad):
            # 同一广告内按 clip_id 排序
            records = sorted(
                ad_records[key],
                key=lambda r: r.get("clip_id", 0)
            )
            if not records:
                continue

            kept_ads += 1
            kept_clips += len(records)

            for obj in records:
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"总行数(原始读取): {total_lines}")
    print(f"满足 brand/product/features 且 clip 数在[3,60]的广告数: {kept_ads}")
    print(f"最终保留的 clip 数量: {kept_clips}")


if __name__ == "__main__":
    main()


### python filter_final_test_data.py --input /data/phd/miltonzhou/sft/data_preprocess/filter_data/data_chunk_1102/Final_CT_eval_1109.jsonl --output /data/phd/miltonzhou/sft/data_preprocess/filter_data/data_chunk_1102/TEST_DATA_1109.jsonl
