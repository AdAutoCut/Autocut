#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate vid_sort predictions against ground-truth order using order-sensitive metrics.

- Input: JSONL where each line is an object containing:
    - conversations: list of dicts; use the last item with {"from": "gpt"}["value"] as ground-truth order (comma-separated ids)
    - model_generate: string of comma-separated ids as prediction
- If the two sequences are not the same length, SKIP that sample (as requested).

Metrics per sample (computed on the common element set, preserving your preference for pairwise order):
  * pair_acc     : pairwise order accuracy = correct_pairs / C_max
  * kendall_tau  : Kendall's tau = 1 - 2 * inversions / C_max
  * inversions   : number of pairwise order inversions (lower is better)
  * spearman_rho : Spearman's rho on the common set
  * footrule_F   : Spearman footrule distance (L1 rank diff)
  * footrule_sim : 1 - F / F_max  (higher is better)
  * lis_len      : LIS length of mapped prediction (equals LCS length), reflecting kept relative order
  * lis_sim      : lis_len / n_common

Output: a JSON report with per-sample metrics and dataset-level aggregates.

Usage:
    python evaluate_vid_sort_metrics.py \
        --input data.jsonl \
        --output report.json \
        --id_field sample_id

Notes:
- Sequence strings like "10,1,3,7,9,4,11,5,8,6,2" are parsed robustly (commas/Chinese commas/spaces allowed).
- If an element repeats in prediction, only the first occurrence is kept.
- If two sequences have equal length but different element sets, metrics are computed on the intersection; coverage is recorded.
"""

from __future__ import annotations
import argparse
import json
from typing import Any, Dict, List, Tuple
import math
import bisect

# ---------------------- parsing helpers ----------------------

def parse_id_seq(s: str) -> List[str]:
    # Normalize separators (English/Chinese commas, whitespace)
    s = s.replace("，", ",").strip()
    # Split on comma, fallback to whitespace if no commas
    parts = [p.strip() for p in s.split(",") if p.strip()] if "," in s else s.split()
    return parts


def last_gpt_value(conversations: List[Dict[str, Any]]) -> str | None:
    if not isinstance(conversations, list):
        return None
    for item in reversed(conversations):
        if isinstance(item, dict) and item.get("from") == "gpt":
            val = item.get("value")
            if isinstance(val, str):
                return val
    return None

# ---------------------- metric core ----------------------

def align_to_truth(truth: List[Any], pred: List[Any]) -> List[int]:
    rank = {x: i + 1 for i, x in enumerate(truth)}
    seen = set()
    mapped: List[int] = []
    for x in pred:
        if x in rank and x not in seen:
            mapped.append(rank[x])
            seen.add(x)
    return mapped


def kendall_pairwise(truth: List[Any], pred: List[Any]) -> Dict[str, float]:
    mapped = align_to_truth(truth, pred)
    n = len(mapped)
    cmax = n * (n - 1) // 2
    if n <= 1 or cmax == 0:
        return {"inversions": 0.0, "pair_acc": 1.0, "tau": 1.0}
    D = 0
    for i in range(n):
        for j in range(i + 1, n):
            if mapped[i] > mapped[j]:
                D += 1
    C = cmax - D
    return {
        "inversions": float(D),
        "pair_acc": C / cmax,
        "tau": (C - D) / cmax,
    }


def spearman_rho(truth: List[Any], pred: List[Any]) -> float:
    # compute on the intersection
    rank_t = {x: i + 1 for i, x in enumerate(truth)}
    seen = set(); kept = []
    for x in pred:
        if x in rank_t and x not in seen:
            kept.append(x); seen.add(x)
    n = len(kept)
    if n < 2:
        return 1.0
    # Re-rank within the common subset
    sub_truth = sorted(kept, key=lambda x: rank_t[x])
    rank_sub_truth = {x: i + 1 for i, x in enumerate(sub_truth)}
    rank_sub_pred  = {x: i + 1 for i, x in enumerate(kept)}
    ssd = 0
    for x in sub_truth:
        d = rank_sub_truth[x] - rank_sub_pred[x]
        ssd += d * d
    return 1.0 - 6.0 * ssd / (n * (n * n - 1))


def footrule(truth: List[Any], pred: List[Any]) -> Tuple[float, float]:
    rank_t = {x: i + 1 for i, x in enumerate(truth)}
    seen = set(); kept = []
    for x in pred:
        if x in rank_t and x not in seen:
            kept.append(x); seen.add(x)
    n = len(kept)
    if n < 2:
        return 0.0, 1.0
    sub_truth = sorted(kept, key=lambda x: rank_t[x])
    rank_sub_truth = {x: i + 1 for i, x in enumerate(sub_truth)}
    rank_sub_pred  = {x: i + 1 for i, x in enumerate(kept)}
    F = sum(abs(rank_sub_truth[x] - rank_sub_pred[x]) for x in sub_truth)
    Fmax = (n * n) / 2 if n % 2 == 0 else (n * n - 1) / 2
    return float(F), float(1.0 - F / Fmax)


def lis_similarity(truth: List[Any], pred: List[Any]) -> Tuple[int, float]:
    mapped = align_to_truth(truth, pred)
    n = len(mapped)
    if n == 0:
        return 0, 0.0
    d: List[int] = []
    for x in mapped:
        i = bisect.bisect_left(d, x)
        if i == len(d):
            d.append(x)
        else:
            d[i] = x
    L = len(d)
    return L, L / n

# ---------------------- evaluation loop ----------------------

def eval_sample(truth_seq: List[str], pred_seq: List[str]) -> Dict[str, float]:
    # When lengths differ, the caller should skip; here we still compute coverage info
    common = [x for x in truth_seq if x in set(pred_seq)]
    coverage = len(common) / len(truth_seq) if truth_seq else 0.0
    kd = kendall_pairwise(truth_seq, pred_seq)
    rho = spearman_rho(truth_seq, pred_seq)
    F, Fsim = footrule(truth_seq, pred_seq)
    L, Lsim = lis_similarity(truth_seq, pred_seq)
    out = {
        "pair_acc": kd["pair_acc"],
        "kendall_tau": kd["tau"],
        "inversions": kd["inversions"],
        "spearman_rho": rho,
        "footrule_F": F,
        "footrule_sim": Fsim,
        "lis_len": float(L),
        "lis_sim": Lsim,
        "coverage": coverage,
        "n": float(len(common)),
    }
    return out


def aggregate(rows: List[Dict[str, Any]], keys: List[str]) -> Dict[str, float]:
    agg: Dict[str, float] = {}
    if not rows:
        return {k: float('nan') for k in keys}
    for k in keys:
        vals = [r[k] for r in rows if isinstance(r.get(k), (int, float))]
        agg[f"mean_{k}"] = sum(vals) / len(vals) if vals else float('nan')
    return agg


def main():
    ap = argparse.ArgumentParser(description="Evaluate vid_sort order metrics on JSONL")
    ap.add_argument("--input", default="/data/phd/qinsizhong/llm_factory_test/baselines/results/F___vidsort_atc.jsonl", help="input")
    ap.add_argument("--output", default="sort_metrics_report_atc1.json", help="Output report.json path")
    ap.add_argument("--id_field", default="sample_id", help="Primary key field name (default sample_id)")
    ap.add_argument("--encoding", default="utf-8")
    args = ap.parse_args()

    per_sample: List[Dict[str, Any]] = []
    skipped_len: List[Any] = []
    total, used = 0, 0

    with open(args.input, 'r', encoding=args.encoding) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            obj = json.loads(line)
            sid = obj.get(args.id_field)
            gt_str = last_gpt_value(obj.get("conversations", []))
            pred_str = obj.get("model_generate")
            if not isinstance(gt_str, str) or not isinstance(pred_str, str):
                skipped_len.append(sid)
                continue
            gt = parse_id_seq(gt_str)
            pd = parse_id_seq(pred_str)
            # Skip when lengths differ (per requirement)
            if len(gt) != len(pd):
                skipped_len.append(sid)
                continue
            used += 1
            metrics = eval_sample(gt, pd)
            per_sample.append({
                args.id_field: sid,
                "n_total": len(gt),
                **metrics,
            })

    keys = [
        "pair_acc", "kendall_tau", "inversions",
        "spearman_rho", "footrule_F", "footrule_sim",
        "lis_len", "lis_sim", "coverage", "n",
    ]
    summary = aggregate(per_sample, keys)

    report = {
        "meta": {
            "input": args.input,
            "total_rows": total,
            "used_rows": used,
            "skipped_len_mismatch": skipped_len,
        },
        "summary": summary,
        "samples": per_sample,
    }

    with open(args.output, 'w', encoding='utf-8') as wf:
        json.dump(report, wf, ensure_ascii=False, indent=2)

    print("Done. Used:", used, "/", total, "lines. Report:", args.output)


if __name__ == "__main__":
    main()
