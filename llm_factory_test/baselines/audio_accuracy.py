#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import argparse
import statistics

def main():
    parser = argparse.ArgumentParser(description="Compute average rank of GT within model_generate (Top-50 list).")
    parser.add_argument("--input", default="/data/phd/qinsizhong/llm_factory_test/baselines/results/F2___audrank_atc_embsft_wgt.jsonl", help="Path to JSONL file")
    args = parser.parse_args()

    ranks = []        # 1..50 if found, else 51
    mrr_vals = []     # 1/rank if found (rank<=50), else 0
    total = 0         # samples with non-empty gt_aud_id
    hit50 = 0

    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            gt_list = obj.get("photo_id")
            if not gt_list:
                continue  # skip samples without GT
            total += 1

            gt = str(gt_list)
            cands = obj.get("predict_aud_id", [])
            cands = [str(x) for x in cands] if isinstance(cands, list) else []

            # find 1-based rank; if not found in top-50, treat as 51
            try:
                rank = cands.index(gt) + 1
            except ValueError:
                rank = 51

            ranks.append(rank)
            if rank <= 50:
                hit50 += 1
                mrr_vals.append(1.0 / rank)
            else:
                mrr_vals.append(0.0)

    if not ranks:
        print("No valid samples with non-empty gt_aud_id found.")
        return

    mean_rank = sum(ranks) / len(ranks)
    median_rank = statistics.median(ranks)
    hit_at_50 = hit50 / len(ranks) * 100.0
    mrr = sum(mrr_vals) / len(mrr_vals)

    print(f"Samples counted: {len(ranks)}")
    print(f"Mean Rank (1=best, 51=miss): {mean_rank:.3f}")
    print(f"Median Rank: {median_rank:.1f}")
    print(f"Hit@50: {hit_at_50:.2f}% ({hit50}/{len(ranks)})")
    print(f"MRR (1/rank, miss=0): {mrr:.4f}")

if __name__ == "__main__":
    main()
