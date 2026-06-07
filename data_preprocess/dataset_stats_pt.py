#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
统计“单行即整支广告”的 JSONL 数据（文本内含特定标记）：
输入示例（每行一个 JSON，且 text 内为整支广告串）：
{"text": "<|ad_start|> ... <|clip_start|> ... <|text_start|>台词...<|text_end|> ... <|frame_start|> ... <|clip_end|> ... <|ad_end|>"}

统计项（均汇总为全局分布并写入 CSV: metric,key,count）：
1) clips_per_video      —— 每个广告(=video)中 clip 的数量
2) seconds_per_clip     —— 每个 clip 的秒数（该 clip 内 <|frame_start|> 次数）
3) seconds_per_video    —— 每个广告的总秒数（该广告内 <|frame_start|> 次数）
4) clip_text_len        —— 每个 clip 的台词字符总数（该 clip 内所有 <|text_start|>…<|text_end|> 的字符长度和）
5) ad_text_len          —— 每个广告的台词字符总数（其所有 clip 的 clip_text_len 之和）

使用方法：
python stats_from_compact_text.py --input /path/to/data.jsonl --output data_stats.csv
或：
python stats_from_compact_text.py --input /path/to/dir_with_jsonl --output data_stats.csv
"""

import os
import json
import re
import argparse
from collections import Counter
from typing import Iterable, Dict, Any, List, Tuple

# 标记
AD_START   = r"<\|ad_start\|>"
AD_END     = r"<\|ad_end\|>"
CLIP_START = r"<\|clip_start\|>"
CLIP_END   = r"<\|clip_end\|>"
TEXT_START = r"<\|text_start\|>"
TEXT_END   = r"<\|text_end\|>"
FRAME_START = r"<\|frame_start\|>"

# 预编译正则（DOTALL 以跨行匹配；非贪婪）
RE_AD   = re.compile(f"{AD_START}(.*?){AD_END}", flags=re.DOTALL)
RE_CLIP = re.compile(f"{CLIP_START}(.*?){CLIP_END}", flags=re.DOTALL)
RE_TEXT = re.compile(f"{TEXT_START}(.*?){TEXT_END}", flags=re.DOTALL)

def iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                yield json.loads(s)
            except json.JSONDecodeError:
                # 跳过坏行
                continue

def collect_records(input_path: str) -> Iterable[Dict[str, Any]]:
    if os.path.isfile(input_path):
        yield from iter_jsonl(input_path)
    else:
        for name in sorted(os.listdir(input_path)):
            if name.lower().endswith(".jsonl"):
                full = os.path.join(input_path, name)
                if os.path.isfile(full):
                    yield from iter_jsonl(full)

def parse_ads(raw_text: str) -> List[str]:
    """从一条记录的 text 中解析所有 <|ad_start|>…<|ad_end|> 片段。
       常见情况是一条记录只有 1 段广告；但这里支持多段。"""
    if not isinstance(raw_text, str):
        return []
    return [m.group(1) for m in RE_AD.finditer(raw_text)]

def parse_clips(ad_block: str) -> List[str]:
    """从单个广告片段中解析所有 <|clip_start|>…<|clip_end|> 片段。"""
    return [m.group(1) for m in RE_CLIP.finditer(ad_block)]

def count_frames(s: str) -> int:
    """统计给定片段内 <|frame_start|> 的出现次数。"""
    return len(re.findall(FRAME_START, s))

def clip_text_len(clip_block: str) -> int:
    """统计单个 clip 的台词字符数：所有 <|text_start|>…<|text_end|> 的字符长度和。"""
    total = 0
    for m in RE_TEXT.finditer(clip_block):
        total += len(m.group(1))
    return total

def main():
    parser = argparse.ArgumentParser(description="统计“单行即整支广告”文本格式的多项分布")
    parser.add_argument("--input", default="/data/phd/qinsizhong/llm_factory_test/data/pt_831_train.jsonl", help="输入 jsonl 文件或包含 jsonl 的目录")
    parser.add_argument("--output", default="data_stats_pt.csv", help="输出 CSV 文件路径")
    args = parser.parse_args()

    # 全局分布累积器
    clips_per_video_counter   = Counter()
    seconds_per_clip_counter  = Counter()
    seconds_per_video_counter = Counter()
    clip_text_len_counter     = Counter()
    ad_text_len_counter       = Counter()

    total_rows = 0
    total_ads  = 0
    total_clips= 0

    for rec in collect_records(args.input):
        total_rows += 1
        text = rec.get("text")
        ad_blocks = parse_ads(text) if isinstance(text, str) else []

        for ad_block in ad_blocks:
            total_ads += 1

            # 该广告的 clip 列表
            clips = parse_clips(ad_block)
            n_clips = len(clips)
            clips_per_video_counter[n_clips] += 1

            # 该广告的总秒数
            ad_seconds = count_frames(ad_block)
            seconds_per_video_counter[ad_seconds] += 1

            # 该广告的台词总字符数
            ad_text_total_len = 0

            # 遍历 clip
            for clip_block in clips:
                total_clips += 1

                # clip 秒数
                sec = count_frames(clip_block)
                seconds_per_clip_counter[sec] += 1

                # clip 台词字符数
                c_len = clip_text_len(clip_block)
                clip_text_len_counter[c_len] += 1

                ad_text_total_len += c_len

            ad_text_len_counter[ad_text_total_len] += 1

    # 写 CSV（metric,key,count）
    with open(args.output, "w", encoding="utf-8") as f:
        f.write("metric,key,count\n")

        for k in sorted(clips_per_video_counter.keys()):
            f.write(f"clips_per_video,{k},{clips_per_video_counter[k]}\n")
        f.write("\n")

        for s in sorted(seconds_per_clip_counter.keys()):
            f.write(f"seconds_per_clip,{s},{seconds_per_clip_counter[s]}\n")
        f.write("\n")

        for s in sorted(seconds_per_video_counter.keys()):
            f.write(f"seconds_per_video,{s},{seconds_per_video_counter[s]}\n")
        f.write("\n")

        for L in sorted(clip_text_len_counter.keys()):
            f.write(f"clip_text_len,{L},{clip_text_len_counter[L]}\n")
        f.write("\n")

        for L in sorted(ad_text_len_counter.keys()):
            f.write(f"ad_text_len,{L},{ad_text_len_counter[L]}\n")

    print(f"Total JSON rows read: {total_rows}")
    print(f"Total ads parsed: {total_ads}")
    print(f"Total clips parsed: {total_clips}")
    print(f"Saved stats to: {args.output}")

if __name__ == "__main__":
    main()
