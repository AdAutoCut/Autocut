#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Filter CT_train_1109.jsonl by presence of valid 'product' field.

规则:
- 输入文件中每行是一个 clip 样本, 至少包含:
    chunk_id, ad_id, clip_id, text, product, ...
- 若该行:
    - 缺少 'product' 字段, 或
    - product 为 null / 空串 / 仅空白 / 字面 "null"(大小写任意)
  => 丢弃
- 否则保留到新文件 final_CT_train_1109.jsonl

同时:
- 在保留的行中, 按 (chunk_id, ad_id) 去重计数,
  打印剩余广告(组)数量。
"""

import argparse
import json
import os
from typing import Any, Dict, Set, Tuple
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Filter JSONL rows by non-empty 'product' and count distinct (chunk_id, ad_id)."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="/data/phd/miltonzhou/sft/data_preprocess/filter_data/data_chunk_1102/CT_eval_1109.jsonl",
        help="输入 JSONL 文件路径"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/data/phd/miltonzhou/sft/data_preprocess/filter_data/data_chunk_1102/Final_CT_eval_1109.jsonl",
        help="输出 JSONL 文件路径 (仅保留有有效 product 的行)"
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        default=True,
        help="是否显示进度条"
    )
    return parser.parse_args()


def has_valid_product(obj: Dict[str, Any]) -> bool:
    """判断一条样本的 product 字段是否有效."""
    if "product" not in obj:
        return False
    val = obj["product"]
    # None 直接无效
    if val is None:
        return False
    # 如果是列表/字典, 视为有内容(按需可扩展), 这里主要针对字符串
    if isinstance(val, str):
        s = val.strip()
        if s == "" or s.lower() == "null":
            return False
        return True
    # 非字符串(例如数字等)，只要存在就当有效
    return True


def main():
    args = parse_args()

    in_path = args.input
    out_path = args.output

    if not os.path.exists(in_path):
        print(f"[ERROR] Input file not found: {in_path}")
        return

    total_rows = 0
    kept_rows = 0
    ad_keys: Set[Tuple[int, int]] = set()

    # 先简单统计一下行数用于进度条
    num_lines = sum(1 for _ in open(in_path, "r", encoding="utf-8"))

    with open(in_path, "r", encoding="utf-8") as fin, \
         open(out_path, "w", encoding="utf-8") as fout:

        iterator = fin
        if args.progress:
            iterator = tqdm(fin, total=num_lines, desc="Filtering", unit="line")

        for line in iterator:
            total_rows += 1
            s = line.strip()
            if not s:
                continue

            try:
                obj = json.loads(s)
            except json.JSONDecodeError:
                continue

            if not has_valid_product(obj):
                continue

            # 保留该行
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            kept_rows += 1

            # 统计 (chunk_id, ad_id)
            chunk_id = obj.get("chunk_id", 0)
            ad_id = obj.get("ad_id", None)
            try:
                chunk_id = int(chunk_id)
            except (TypeError, ValueError):
                chunk_id = 0
            try:
                ad_id = int(ad_id)
            except (TypeError, ValueError):
                # ad_id 异常就不计入广告数, 但行仍然保留
                ad_id = None
            if ad_id is not None:
                ad_keys.add((chunk_id, ad_id))

    print("\n=== Summary ===")
    print(f"Total rows in input         : {total_rows}")
    print(f"Rows with valid product kept: {kept_rows}")
    print(f"Distinct (chunk_id, ad_id)  : {len(ad_keys)}")
    print(f"Saved filtered data to      : {out_path}")


if __name__ == "__main__":
    main()
