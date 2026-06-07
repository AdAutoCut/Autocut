#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import csv
import random
from typing import Dict, Optional

# -------- 路径配置 --------
ROOT_DIR   = "/data/phd/qinsizhong/llm_factory_test"
ATC_DIR    = os.path.join(ROOT_DIR, "Final_user_study_atc_2")
GPT_DIR    = os.path.join(ROOT_DIR, "Final_user_study_gpt4o_2")
OUTPUT_CSV = os.path.join(ROOT_DIR, "user_study_table.csv")

# URL 前缀（从 ROOT_DIR 开始接）
BASE_URL = ""

# 为了随机顺序可复现（想每次都不同可以注释掉这一行）
random.seed(42)


sid_pattern = re.compile(r"sid(\d+)")


def parse_sid(filename: str) -> Optional[int]:
    """
    从文件名中解析 sid，例如：
    'sid0__p164625038962__a158625916887_atc.mp4' -> 0
    'sid0__a162854846790.mp4' -> 0
    """
    m = sid_pattern.search(filename)
    if not m:
        return None
    return int(m.group(1))


def collect_videos(dir_path: str) -> Dict[int, str]:
    """
    收集目录下所有 mp4 文件：
    返回 sid -> 相对 ROOT_DIR 的路径（用于拼 URL）
    """
    mapping: Dict[int, str] = {}

    for name in os.listdir(dir_path):
        if not name.lower().endswith(".mp4"):
            continue
        sid = parse_sid(name)
        if sid is None:
            print(f"[warning] 无法从文件名解析 sid: {name}")
            continue
        abs_path = os.path.join(dir_path, name)
        rel_path = os.path.relpath(abs_path, ROOT_DIR)  # 例如 'Final_user_study_atc_2/sid0__...mp4'
        if sid in mapping:
            print(f"[warning] sid={sid} 在 {dir_path} 中有多个文件，保留第一个：")
            print("  已有:", mapping[sid])
            print("  忽略:", rel_path)
            continue
        mapping[sid] = rel_path

    return mapping


def main():
    # 收集 atc 和 gpt 的视频
    atc_videos = collect_videos(ATC_DIR)
    gpt_videos = collect_videos(GPT_DIR)

    # 取两个模型共有的 sid
    common_sids = sorted(set(atc_videos.keys()) & set(gpt_videos.keys()))
    print(f"共有样本数（两个模型都有视频的 sid）：{len(common_sids)}")

    # CSV 列名
    headers = [
        "sample_id",
        "video_A",
        "video_B",
        "画面流畅度和视频逻辑(0-5)",
        "画面-台词内容一致(0-5)",
        "广告吸引程度(0-5)",
        "音乐视频匹配性(0-5)",
        "视频画面一致性(0-5)",
        "a_model",
        "b_model",
    ]

    with open(OUTPUT_CSV, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)

        sample_id = 1
        for sid in common_sids:
            atc_rel = atc_videos[sid]
            gpt_rel = gpt_videos[sid]

            atc_url = f"{BASE_URL}/{atc_rel}"
            gpt_url = f"{BASE_URL}/{gpt_rel}"

            # 随机决定谁是 A / B
            if random.random() < 0.5:
                video_A = gpt_url
                video_B = atc_url
                a_model = "gpt"
                b_model = "atc"
            else:
                video_A = atc_url
                video_B = gpt_url
                a_model = "atc"
                b_model = "gpt"

            row = [
                sample_id,
                video_A,
                video_B,
                "",  # 画面流畅度和视频逻辑(0-5)
                "",  # 画面-台词内容一致(0-5)
                "",  # 广告吸引程度(0-5)
                "",  # 音乐视频匹配性(0-5)
                "",  # 视频画面一致性(0-5)
                a_model,
                b_model,
            ]
            writer.writerow(row)
            sample_id += 1

    print(f"已写入 CSV: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
