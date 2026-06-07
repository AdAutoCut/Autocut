#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from collections import Counter
from typing import Dict

MANIFEST_PATH = "/data/phd/miltonzhou/sft/data_preprocess/T2atc_manifest_vidsort.json"        # 输入：你的 manifest_sort json 文件（list 格式）
PRED_INPUT_PATH = "/data/phd/qinsizhong/llm_factory_test/baselines/results/F2___audrank_atc_embsft.jsonl"          # 输入：原始的 pred_aud.jsonl
PRED_OUTPUT_PATH = "/data/phd/qinsizhong/llm_factory_test/baselines/results/F2___audrank_atc_embsft_wgt.jsonl"  # 输出：带 photo_id 的新 jsonl


def frame_to_photo_id(frame_id) -> int:
    """
    根据你的定义：photo_id = frame_id 去掉后四位
    兼容 int 或 str 输入
    """
    return int(str(frame_id)) // 10000


def build_sample_to_photo(manifest_path: str) -> Dict[int, int]:
    """
    从 manifest_sort.json 里构造：
        sample_id -> gt_photo_id（出现次数最多的 photo_id）
    """
    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)  # manifest 是一个 list

    sample_to_photo: Dict[int, int] = {}

    for item in data:
        # 有些任务可能没有 sample_id，保险起见做个防御
        if "sample_id" not in item:
            continue
        sample_id = item["sample_id"]

        clips = item.get("clips", [])
        counter = Counter()

        for clip in clips:
            orig = clip.get("orig", {})
            frame_id = orig.get("frame_id", None)
            if frame_id is None:
                continue
            photo_id = frame_to_photo_id(frame_id)
            counter[photo_id] += 1

        if not counter:
            # 这个 sample 没有有效 frame_id，跳过
            continue

        # 频次最高的 photo_id 作为该 sample 的 gt
        gt_photo_id, _ = counter.most_common(1)[0]
        sample_to_photo[sample_id] = gt_photo_id

    return sample_to_photo


def add_photo_id_to_pred(pred_in: str, pred_out: str, sample_to_photo: Dict[int, int]) -> None:
    """
    读取 pred_aud.jsonl，为每一行根据 sample_id 填充 photo_id 字段，写到新文件
    """
    missing_count = 0
    total = 0

    with open(pred_in, "r", encoding="utf-8") as fin, \
         open(pred_out, "w", encoding="utf-8") as fout:

        for line in fin:
            line = line.strip()
            if not line:
                continue

            obj = json.loads(line)
            sample_id = obj.get("sample_id", None)
            total += 1

            if sample_id in sample_to_photo:
                obj["photo_id"] = sample_to_photo[sample_id]
            else:
                # 如果在 manifest 中找不到 sample_id，可以选择：
                # 1) 直接跳过这一行（不写出）
                # 2) 写出但用 -1 或 None 占位
                # 这里选择写出并标记为 -1，你可以按需修改
                obj["photo_id"] = -1
                missing_count += 1

            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"总共处理样本数: {total}")
    print(f"其中找不到 gt photo_id 的样本数: {missing_count}")


def main():
    # 1) 从 manifest_sort.json 统计每个 sample 的 gt photo_id
    sample_to_photo = build_sample_to_photo(MANIFEST_PATH)
    print(f"从 manifest 中得到 gt photo_id 的 sample 数量: {len(sample_to_photo)}")

    # 2) 为 pred_aud.jsonl 每一行补充 photo_id
    add_photo_id_to_pred(PRED_INPUT_PATH, PRED_OUTPUT_PATH, sample_to_photo)


if __name__ == "__main__":
    main()
