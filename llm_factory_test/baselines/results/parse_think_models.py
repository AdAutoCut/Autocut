#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import re
from typing import Optional

# Match all standalone two-digit IDs in order (e.g., 01, 12, 09)
TWO_DIGIT_ID = re.compile(r'(?<!\d)(\d{2})(?!\d)')

def extract_ids_after_think(text: str) -> str:
    """
    Extract the comma-separated sequence of two-digit IDs from TEXT,
    preferring the substring after the last </think>. If </think> is not found,
    search the whole TEXT. Returns "" if none found.
    """
    if not isinstance(text, str):
        return ""
    pos = text.rfind("</think>")
    tail = text[pos + len("</think>"):] if pos != -1 else text
    ids = TWO_DIGIT_ID.findall(tail)
    if ids:
        return ",".join(ids)
    # Fallback: search the whole text
    ids = TWO_DIGIT_ID.findall(text)
    return ",".join(ids) if ids else ""

def process_file(src_path: str, dst_path: str) -> None:
    total, updated, missing = 0, 0, 0
    with open(src_path, "r", encoding="utf-8") as fin, \
         open(dst_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                # skip invalid json line
                continue

            raw = obj.get("model_generate", "")
            seq = extract_ids_after_think(raw)

            # write back: replace model_generate with the cleaned sequence
            obj["model_generate"] = seq
            if seq:
                updated += 1
            else:
                missing += 1

            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"[OK] Done. total={total}, updated={updated}, missing={missing}")
    print(f"[OK] Saved to: {dst_path}")

def main():
    ap = argparse.ArgumentParser(description="Extract ID sequence after </think> and rewrite JSONL.")
    ap.add_argument("--src", default="/data/phd/qinsizhong/llm_factory_test/baselines/results/selectnsort_Qwen3_32B_think.jsonl", help="Input JSONL path")
    ap.add_argument("--dst", default="/data/phd/qinsizhong/llm_factory_test/baselines/results/selectnsort_Qwen3_32B_think_parsed.jsonl", help="Output JSONL path")
    args = ap.parse_args()
    process_file(args.src, args.dst)

if __name__ == "__main__":
    main()
