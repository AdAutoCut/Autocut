#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse

def iter_jsonl(paths):
    """依次读取多个 jsonl 文件，产出每条合法 JSON 对象。空行/坏行跳过并告警。"""
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            for ln, line in enumerate(f, 1):
                s = line.strip()
                if not s:
                    continue
                try:
                    yield json.loads(s)
                except Exception as e:
                    print(f"[warn] skip bad line: file={p}, line={ln}, err={e}")

def write_chunk(objs, out_dir, model_name, chunk_idx):
    """将一批对象写成一个 jsonl 文件（MR_{model_name}_{chunk}.jsonl）。"""
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"MR_{model_name}_{chunk_idx}.jsonl")
    with open(out_path, "w", encoding="utf-8") as fout:
        for obj in objs:
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    print(f"[save] {out_path} ({len(objs)} lines)")

def main():
    ap = argparse.ArgumentParser(description="Split model_result.jsonl into 15-per-chunk jsonl files.")
    ap.add_argument("--inputs", type=str, nargs="+", default=["/data/phd/qinsizhong/llm_factory_test/baselines/results/F2___vidsort_gpt4o.jsonl"],
                    help="一个或多个待拆分的 model_result.jsonl 路径")
    ap.add_argument("--model_name", type=str, choices=["atc", "gpt4o"], default="gpt4o",
                    help="用于输出命名的模型名：atc 或 gpt4o")
    # ap.add_argument("--inputs", type=str, nargs="+", default=["/data/phd/qinsizhong/llm_factory_test/baselines/results/F2___vidsort_atc_embsft.jsonl"],
    #                 help="一个或多个待拆分的 model_result.jsonl 路径")
    # ap.add_argument("--model_name", type=str, choices=["atc", "gpt4o"], default="atc",
    #                 help="用于输出命名的模型名：atc 或 gpt4o")
    ap.add_argument("--out_dir", type=str, default="model_pred_chunks2",
                    help="输出目录，默认 /model_pred_chunks")
    ap.add_argument("--chunk_size", type=int, default=20,
                    help="每个 chunk 的样本数，默认 15")
    ap.add_argument("--start_chunk", type=int, default=0,
                    help="起始 chunk 编号（默认 0），方便接着追加")
    args = ap.parse_args()

    buf = []
    total = 0
    chunk_idx = args.start_chunk

    for obj in iter_jsonl(args.inputs):
        buf.append(obj)
        total += 1
        if len(buf) == args.chunk_size:
            write_chunk(buf, args.out_dir, args.model_name, chunk_idx)
            chunk_idx += 1
            buf = []

    # 收尾（不足一个 chunk 的剩余部分也要写）
    if buf:
        write_chunk(buf, args.out_dir, args.model_name, chunk_idx)
        chunk_idx += 1

    print(f"[done] read={total}, chunks={chunk_idx - args.start_chunk}, out_dir={os.path.abspath(args.out_dir)}")

if __name__ == "__main__":
    main()
