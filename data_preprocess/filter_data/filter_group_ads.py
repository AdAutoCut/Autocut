#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import json
import os
import re
from collections import defaultdict
from typing import Dict, Iterable, Optional, List, Any

PATTERN_SCORE = re.compile(r"匹配度[:：]\s*([0-9]+(?:\.[0-9]+)?)")

BASE_DIR = "/data/phd/miltonzhou/sft/data_preprocess/filter_data/cp_results"
INPUT_NAME = "infer_results_p1.jsonl"
OUTPUT_NAME = "match_ids_p1.jsonl"

def parse_args():
    ap = argparse.ArgumentParser(description="按匹配比例筛选优质 global_id")
    ap.add_argument("--clip-threshold", type=float, default=4.0,
                    help="匹配片段阈值，默认 5 分（>= 5 记为匹配）")
    ap.add_argument("--keep-ratio", type=float, default=0.8,
                    help="整视频通过比例阈值，默认 0.8（80%）")
    ap.add_argument("--chunks", type=int, nargs='+', required=True,
                    help="多个数据分片编号，例如 5 6 7 对应 cp_chunk5 cp_chunk6 cp_chunk7")
    return ap.parse_args()

def extract_score(answer: Optional[str]) -> Optional[float]:
    if not answer or not isinstance(answer, str):
        return None
    m = PATTERN_SCORE.search(answer)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None

def iter_jsonl(path: str) -> Iterable[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[WARN] JSON err@{ln}: {e}; {line[:60]}...")
                continue
            if not isinstance(obj, dict):
                print(f"[WARN] Non-dict JSON at line {ln}, skip: {str(obj)[:60]}...")
                continue
            yield obj

def _to_int_or_none(x: Any) -> Optional[int]:
    try:
        return int(x)
    except Exception:
        return None

def extract_by_index(input_path: str, output_path: str, index_list: list[int]) -> None:
    """
    从 input_path 的 jsonl 文件中抽取指定行号（从 1 开始计数）的数据，保存到 output_path。
    :param input_path: 输入 jsonl 文件路径
    :param output_path: 输出 jsonl 文件路径
    :param index_list: 要抽取的行号列表（从 1 开始）
    """
    index_set = set(index_list)
    results = []

    with open(input_path, "r", encoding="utf-8") as fin:
        for i, line in enumerate(fin, start=1):
            if i in index_set:
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    print(f"[WARN] Line {i} error")
                    continue
                results.append(obj)

    with open(output_path, "w", encoding="utf-8") as fout:
        for obj in results:
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"[DONE] Extracted {len(results)} entries from {input_path} to {output_path}")

def process_chunk(chunk_id: int, args, global_id_list: List[str]):
    chunk_dir = os.path.join(BASE_DIR, f"cp_chunk{chunk_id}")
    in_path = os.path.join(chunk_dir, INPUT_NAME)
    out_path = os.path.join(chunk_dir, OUTPUT_NAME)

    if not os.path.exists(in_path):
        raise FileNotFoundError(f"file not found: {in_path}")

    stats: Dict[str, List[int]] = defaultdict(lambda: [0, 0])
    contents: Dict[str, List[dict]] = defaultdict(list)

    count = 0
    for obj in iter_jsonl(in_path):
        gid = obj.get("global_id")
        if gid is None:
            continue
        gid = str(gid)

        clip_id = _to_int_or_none(obj.get("clip_id"))
        text = obj.get("raw_text")
        frame_id = obj.get("frame_id")
        score = extract_score(obj.get("model_output"))

        stats[gid][0] += 1  # total++
        if score is not None and score >= args.clip_threshold:
            stats[gid][1] += 1  # matched++

        contents[gid].append({
            "clip_id": clip_id if clip_id is not None else obj.get("clip_id"),
            "text": text,
            "frame_id": frame_id,
            "score": score
        })

        count += 1
        # if count % 10000 == 0:
        #     print(f"[INFO] Processed {count} clips ...")

    print(f"[INFO] Finished reading {count} clips in total for chunk {chunk_id}")

    kept_items = []
    for gid, (total, matched) in stats.items():
        if total <= 0:
            continue
        ratio = matched / total
        if ratio >= args.keep_ratio and total >= 4:
            items = contents.get(gid, [])

            def sort_key(x):
                ci = _to_int_or_none(x.get("clip_id"))
                return (0, ci) if ci is not None else (1, float("inf"))

            items_sorted = sorted(items, key=sort_key)

            kept_items.append({
                "global_id": int(gid) if gid.isdigit() else gid,
                "match_ratio": f"{matched}/{total}",
                "content": items_sorted
            })

            # Add global_id to the list
            global_id_list.append(gid)

    with open(out_path, "w", encoding="utf-8") as fout:
        for item in kept_items:
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"[DONE] Processing {in_path}")
    print(f"[DONE] Saved in {out_path}")
    print(f"[DONE] Filtered Ratio: {len(kept_items)} / {len(stats)}")

def main():
    args = parse_args()
    global_id_list = []

    for chunk_id in args.chunks:
        process_chunk(chunk_id, args, global_id_list)

    # 将 global_id 转换为整数，加 1 后再转回字符串
    updated_global_id_list = [int(gid) + 1 for gid in global_id_list]

    # Print all the global_ids from all chunks
    print(f"[DONE] All filtered global_ids: {updated_global_id_list}")
    print(f"[DONE] TOTAL: {len(updated_global_id_list)}")

    # Extract the corresponding entries from the prefiltered data
    extract_by_index(
        input_path="prefiltered_data_0815.jsonl",
        output_path="filtered_1019_t3r7.jsonl",
        index_list=updated_global_id_list
    )



if __name__ == "__main__":
    main()
