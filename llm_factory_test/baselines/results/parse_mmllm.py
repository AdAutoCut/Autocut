# import json, re

# def normalize_ids(s: str) -> str:
#     # 抓取所有连续数字（自动忽略 '错误' 之类的非数字）
#     nums = re.findall(r'\d+', s or "")
#     return ",".join(f"{int(x):02d}" for x in nums)

# in_path  = "/data/phd/qinsizhong/llm_factory_test/baselines/results/selectnsort_Qwen25VL_7B.jsonl"
# out_path = "/data/phd/qinsizhong/llm_factory_test/baselines/results/selectnsort_Qwen25VL_7B_p.jsonl"

# with open(in_path, "r", encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
#     for line in fin:
#         if not line.strip():
#             continue
#         obj = json.loads(line)
#         if "model_generate" in obj and obj["model_generate"] is not None:
#             obj["model_generate"] = normalize_ids(obj["model_generate"])
#         fout.write(json.dumps(obj, ensure_ascii=False) + "\n")


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Normalize `model_generate` in a JSONL file.

- Input example:
  "model_generate": "'Picture 3,Picture 6,Picture 7,Picture 4,Picture 8,Picture 11,Picture 12,Picture 9'"

- Output example:
  "model_generate": "03,06,07,04,08,11,12,09"

All other fields are preserved as-is.

USAGE:
python parse_mmllm.py input.jsonl output.jsonl
"""

import argparse
import json
import re
from typing import Any, Dict

def normalize_model_generate(value: str) -> str:
    """
    Convert strings like:
      "'Picture 3,Picture 6,Picture 7,Picture 4,Picture 8,Picture 11,Picture 12,Picture 9'"
    to:
      "03,06,07,04,08,11,12,09"

    Rules:
    - Extract all integers in order.
    - Zero-pad to 2 digits.
    - Join with commas (no spaces).
    - If no integers are found, return the original string unchanged.
    """
    if not isinstance(value, str):
        return value

    s = value.strip()

    # If the whole string is wrapped in a single pair of quotes, strip them
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        s = s[1:-1].strip()

    # Extract integers in order
    nums = re.findall(r'\d+', s)
    if not nums:
        return value  # leave unchanged if nothing to convert

    padded = [f"{int(n):02d}" for n in nums]
    return ",".join(padded)

def process_line(obj: Dict[str, Any]) -> Dict[str, Any]:
    if "model_generate" in obj:
        obj["model_generate"] = normalize_model_generate(obj["model_generate"])
    return obj

def main():
    parser = argparse.ArgumentParser(description="Normalize `model_generate` field in a JSONL file.")
    parser.add_argument("input", help="Path to input JSONL")
    parser.add_argument("output", help="Path to output JSONL")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as fin, \
         open(args.output, "w", encoding="utf-8") as fout:
        for line_no, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Line {line_no}: invalid JSON ({e})") from e

            new_obj = process_line(obj)
            fout.write(json.dumps(new_obj, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
