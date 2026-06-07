#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import re
import random
import argparse
from typing import List, Dict, Any, Optional

ID_PATTERN = re.compile(r"\[ID=(\d+)\]")
DIGITS_PATTERN = re.compile(r"\d+")

# -------- I/O --------

def read_input(path: str) -> List[Dict[str, Any]]:
    """
    Read either JSON (list of dicts) or JSONL. Auto-detects by first non-space char.
    """
    with open(path, "r", encoding="utf-8") as f:
        head = f.read(1)
        if not head:
            return []
        f.seek(0)
        if head.lstrip() == "[" or head == "[":
            # JSON array
            obj = json.load(f)
            if not isinstance(obj, list):
                raise ValueError("Input JSON must be a list of objects.")
            return obj
        else:
            # JSONL fallback
            rows: List[Dict[str, Any]] = []
            for lineno, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise ValueError(f"{path}: line {lineno}: invalid JSON ({e})") from e
            return rows

def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for obj in rows:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

# -------- parsing helpers --------

def last_turn_value(convs: Any, role: str) -> str:
    """Return the last conversation.value for a given role ('human' or 'gpt')."""
    val = ""
    if isinstance(convs, list):
        for turn in convs:
            if isinstance(turn, dict) and turn.get("from") == role:
                v = turn.get("value", "")
                if isinstance(v, str):
                    val = v
    return val

def extract_pool_ids(human_text: str) -> List[str]:
    """Extract pool IDs as zero-padded strings by scanning [ID=..] markers."""
    if not isinstance(human_text, str):
        return []
    ids = ID_PATTERN.findall(human_text)
    seen = set()
    out: List[str] = []
    for s in ids:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out

def parse_ids_from_text(s: str) -> List[str]:
    """Extract all digit tokens from a freeform string, preserving zero padding."""
    if not isinstance(s, str):
        return []
    return DIGITS_PATTERN.findall(s)

def build_random_selection(pool_ids: List[str], n: int, rng: random.Random) -> List[str]:
    """Sample n unique IDs from pool_ids (without replacement)."""
    if not pool_ids or n <= 0:
        return []
    n = min(n, len(pool_ids))
    return rng.sample(pool_ids, k=n)

# -------- core --------

def process_records(rows: List[Dict[str, Any]], seed: Optional[int] = None) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    out_rows: List[Dict[str, Any]] = []

    for rec in rows:
        convs = rec.get("conversations", [])
        human_val = last_turn_value(convs, "human")
        gpt_val   = last_turn_value(convs, "gpt")

        pool_ids = extract_pool_ids(human_val)   # e.g., ["01","02","13",...]
        pred_ids = parse_ids_from_text(gpt_val)  # e.g., ["02","12","06",...]
        n = len(pred_ids)

        rand_ids = build_random_selection(pool_ids, n, rng)
        rec["model_generate"] = ",".join(rand_ids)
        out_rows.append(rec)

    return out_rows

def main():
    parser = argparse.ArgumentParser(description="Generate random selection baseline and append to model_generate.")
    parser.add_argument("--input", required=True, help="Path to input JSON (list) or JSONL.")
    parser.add_argument("--output", required=True, help="Path to output JSONL.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    args = parser.parse_args()

    rows = read_input(args.input)
    out_rows = process_records(rows, seed=args.seed)
    write_jsonl(args.output, out_rows)

    print(f"Done. Read {len(rows)} records from {args.input}. Wrote JSONL to {args.output}.")

if __name__ == "__main__":
    main()