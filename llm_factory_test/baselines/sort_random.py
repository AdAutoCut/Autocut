#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Random Baseline for sorting task (only save the final shuffle per record)
"""

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Iterable

CSV_BLOCK = re.compile(r"\d{1,3}(?:\s*,\s*\d{1,3})+")
DIGITS_RE = re.compile(r"\d+")


def read_any(path: str) -> List[Dict[str, Any]]:
    """Read .json (list or single dict) or .jsonl into a list of dicts."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    text = p.read_text(encoding="utf-8")

    if p.suffix.lower() == ".jsonl":
        out: List[Dict[str, Any]] = []
        for i, line in enumerate(text.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"JSONL line {i} invalid: {e}") from e
        return out

    # .json
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON invalid: {e}") from e

    if isinstance(obj, list):
        return obj
    elif isinstance(obj, dict):
        return [obj]
    else:
        raise ValueError("JSON must be a list or a dict.")


def write_jsonl(records: Iterable[Dict[str, Any]], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def extract_label_from_conversations(convs: Any) -> str:
    """Take the last turn with from=='gpt' as label string."""
    label = ""
    if isinstance(convs, list):
        for turn in convs:
            if isinstance(turn, dict) and turn.get("from") == "gpt":
                label = turn.get("value", "") or ""
    return label


def parse_seq_str(s: Any) -> List[str]:
    """
    Prefer extracting a numeric CSV block like '03,05,01,...'.
    Fallback: collect all \d+ in order.
    """
    if not isinstance(s, str):
        return []
    m = CSV_BLOCK.search(s)
    if m:
        csv = m.group(0)
        return [tok.strip() for tok in csv.split(",") if tok.strip() != ""]
    tokens = DIGITS_RE.findall(s)
    return tokens


def build_final_shuffle_outputs(
    records: List[Dict[str, Any]],
    n_shuffles: int,
    seed: int,
) -> Iterable[Dict[str, Any]]:
    """
    For each record, perform n_shuffles in-place with a stable RNG,
    and only output the final shuffled order as 'model_generate'.
    """
    for idx, rec in enumerate(records):
        label_str = extract_label_from_conversations(rec.get("conversations", []))
        label_seq = parse_seq_str(label_str)

        # Stable per-record RNG
        rng = random.Random(seed ^ (idx * 100003))

        seq = label_seq.copy()
        # Shuffle n times; only final order is kept
        for _ in range(max(n_shuffles, 1)):
            rng.shuffle(seq)

        out_rec = dict(rec)  # shallow copy
        out_rec["model_generate"] = ",".join(seq) if seq else ""
        yield out_rec


def main():
    ap = argparse.ArgumentParser(
        description="Generate random baseline (final shuffle only) and write JSONL."
    )
    ap.add_argument("--input", "-i",
                    default="/data/phd/qinsizhong/llm_factory_test/baselines/data/sft_901_eval.json",
                    help="Path to input .json or .jsonl")
    ap.add_argument("--output", "-o",
                    default="/data/phd/qinsizhong/llm_factory_test/baselines/results/sort_random.jsonl",
                    help="Path to output .jsonl")
    ap.add_argument("--n", type=int, default=10,
                    help="Number of consecutive shuffles per record (default: 10)")
    ap.add_argument("--seed", type=int, default=42,
                    help="Base RNG seed (default: 42)")
    args = ap.parse_args()

    try:
        records = read_any(args.input)
    except Exception as e:
        print(f"Failed to read input: {e}", file=sys.stderr)
        sys.exit(1)

    outputs = build_final_shuffle_outputs(records, n_shuffles=args.n, seed=args.seed)
    try:
        write_jsonl(outputs, args.output)
    except Exception as e:
        print(f"Failed to write output: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Done. Read {len(records)} record(s). Wrote 1 shuffled line per record to {args.output} "
          f"(each record shuffled {args.n} time(s), final order saved).")


if __name__ == "__main__":
    main()
