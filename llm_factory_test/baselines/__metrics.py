#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import re
from typing import List, Dict, Any, Sequence, Hashable
import os

# ----------------------------
# 基础工具
# ----------------------------

def read_jsonl(jsonl_path: str) -> List[Dict[str, Any]]:
    data: List[Dict[str, Any]] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Line {lineno}: invalid JSON ({e})") from e
            data.append(obj)
    return data


def calc_edit_distance(a: Sequence[Hashable], b: Sequence[Hashable]) -> int:
    n, m = len(a), len(b)
    if n == 0: return m
    if m == 0: return n
    dp = [[0]*(m+1) for _ in range(n+1)]
    for i in range(1, n+1): dp[i][0] = i
    for j in range(1, m+1): dp[0][j] = j
    for i in range(1, n+1):
        ai = a[i-1]
        for j in range(1, m+1):
            bj = b[j-1]
            cost = 0 if ai == bj else 1
            dp[i][j] = min(
                dp[i-1][j] + 1,        # 删除
                dp[i][j-1] + 1,        # 插入
                dp[i-1][j-1] + cost    # 替换/匹配
            )
    return dp[n][m]


def scsa_counts(pred_seq, ans_seq):
    """
    Single-clip selection accuracy counts:
    - correct = |set(pred) ∩ set(answer)|    (deduplicated)
    - total   = len(answer)
    Returns (correct, total, acc or None if total==0)
    """
    total = len(ans_seq)
    if total == 0:
        return 0, 0, None
    correct = len(set(pred_seq) & set(ans_seq))
    return correct, total, correct / total


# 解析 ID 串：更稳健，允许混入非数字字符，只提取数字 token 按顺序
_ID_RE = re.compile(r"\d+")

def parse_seq_str(s: Any) -> List[str]:
    if not isinstance(s, str):
        return []
    # 先常规逗号切，再兜底用正则把纯数字提出来，合并去空
    parts = [p.strip() for p in s.replace("\n", ",").split(",") if p.strip() != ""]
    tokens: List[str] = []
    for p in parts:
        digits = _ID_RE.findall(p)
        if digits:
            tokens.extend(digits)
    if not tokens:
        tokens = _ID_RE.findall(s)
    return tokens



# ----------------------------
# 与你的数据结构匹配的提取函数
# ----------------------------

def extract_label_from_conversations(convs: Any) -> str:
    """
    取 conversations 中最后一个 from=='gpt' 的 value 作为 label 字符串。
    """
    label = ""
    if isinstance(convs, list):
        for turn in convs:
            if isinstance(turn, dict) and turn.get("from") == "gpt":
                label = turn.get("value", "") or ""
    return label


def extract_prediction(rec: Dict[str, Any]) -> str:
    """
    预测串来自顶层 'model_generate'。若不存在则返回空串。
    """
    val = rec.get("model_generate", "")
    return val if isinstance(val, str) else ""


# ----------------------------
# 总逻辑
# ----------------------------

def compute_metrics(
    data: List[Dict[str, Any]],
    verbose: bool = False,
    record_errors_path: str = None,
) -> Dict[str, Any]:
    """
    - Edit distance (ED): 仅在 label_len == pred_len 的样本上评估（与原逻辑一致）
    - SCSA: 对所有样本都计算（即使长度不一致也统计）
    """
    # --- ED 相关 ---
    d_model: List[int] = []
    exact_match_cnt = 0
    generation_error_results: List[Dict[str, Any]] = []
    kept_count = 0

    # --- SCSA 相关（两个口径）---
    scsa_macro_list: List[float] = []  # 逐样本准确率（忽略 total==0 的样本）
    scsa_total_correct = 0             # 全局正确总数（micro 分子）
    scsa_total_answer  = 0             # 全局答案总数（micro 分母）
    scsa_zero_den_samples = 0          # 答案为空的样本计数
    
    perfect_selection_cnt = 0          # 统计“答案里所有ID都被选到”的样本

    # --- 新增：统计空的 model_generate 数量 ---
    empty_pred_cnt = 0

    for idx, rec in enumerate(data):
        label_str = extract_label_from_conversations(rec.get("conversations", []))
        pred_str  = extract_prediction(rec)

        # 统计空预测（字段缺失或只包含空白）
        if pred_str.strip() == "":
            empty_pred_cnt += 1

        label_seq = parse_seq_str(label_str)
        pred_seq  = parse_seq_str(pred_str)

        # --------- SCSA: 对所有样本都计算 ---------
        sc_correct, sc_total, sc_acc = scsa_counts(pred_seq, label_seq)
        scsa_total_correct += sc_correct
        scsa_total_answer  += sc_total
        if sc_acc is None:
            scsa_zero_den_samples += 1
        else:
            scsa_macro_list.append(sc_acc)
            # 满分则计数（答案非空时成立）
            if sc_correct == sc_total and sc_total > 0:
                perfect_selection_cnt += 1

        # --------- ED: 仅对长度一致样本 ---------
        if len(label_seq) != len(pred_seq):
            generation_error_results.append({
                "index": idx,
                "label_str": label_str,
                "pred_str": pred_str,
                "label_seq": label_seq,
                "pred_seq": pred_seq,
                "label_len": len(label_seq),
                "pred_len": len(pred_seq),
                "reason": "length_mismatch"
            })
            if verbose:
                print(f"[{idx:06d}] SKIP length_mismatch | "
                      f"label_len={len(label_seq)} pred_len={len(pred_seq)}")
            continue

        kept_count += 1

        dist_model = calc_edit_distance(label_seq, pred_seq)
        d_model.append(dist_model)
        if dist_model == 0:
            exact_match_cnt += 1

        if verbose:
            print(
                f"[{idx:06d}] ED={dist_model:.3f} | "
                f"label={label_seq} | pred={pred_seq}"
            )

    # 可选：导出长度不一致样本
    if record_errors_path and generation_error_results:
        with open(record_errors_path, "w", encoding="utf-8") as f:
            for item in generation_error_results:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # --- 汇总（ED）---
    counted = len(d_model)
    avg_m = sum(d_model)/counted if counted else float("nan")
    em_rate = (exact_match_cnt / counted) if counted else float("nan")

    # --- 汇总（SCSA）---
    scsa_micro = (scsa_total_correct / scsa_total_answer) if scsa_total_answer > 0 else float("nan")
    scsa_macro = (sum(scsa_macro_list)/len(scsa_macro_list)) if scsa_macro_list else float("nan")

    return {
        # ED
        "num_records_total": len(data),
        "num_records_kept": kept_count,
        "num_records_skipped": len(generation_error_results),
        "distances_model": d_model,
        "avg_model": avg_m,
        "exact_match_cnt": exact_match_cnt,
        "em_rate": em_rate,

        # SCSA
        "scsa_total_correct": scsa_total_correct,
        "scsa_total_answer": scsa_total_answer,
        "scsa_micro_avg": scsa_micro,   # ∑correct / ∑answer
        "scsa_macro_avg": scsa_macro,   # mean over samples
        "scsa_samples_den0": len(scsa_macro_list),
        "scsa_samples_den0_skipped": scsa_zero_den_samples,

        # NEW: 全选中统计
        "perfect_selection_cnt": perfect_selection_cnt,
        "perfect_selection_rate": (
            perfect_selection_cnt / (len(data) - scsa_zero_den_samples)
            if (len(data) - scsa_zero_den_samples) > 0 else float("nan")
        ),

        # NEW: 空预测统计
        "num_empty_model_generate": empty_pred_cnt,
    }



# ----------------------------
# 用法示例
# ----------------------------
if __name__ == "__main__":

    # JSONL = "file.jsonl"

    data = read_jsonl(JSONL)

    # 打印输入文件名（含路径 & 基名）
    print("-" * 80)
    print(f"[ Input file ] {JSONL}")
    print(f"[ Loaded {len(data)} records ]")

    report = compute_metrics(
        data,
        verbose=False,
        record_errors_path=None,
    )

    print(f"Total records: {report['num_records_total']}")
    print(f"Kept for eval: {report['num_records_kept']}")
    print(f"Skipped (length mismatch): {report['num_records_skipped']}")
    print(f"Average ED: {report['avg_model']:.4f}")
    print(f"Exact matches (ED=0): {report['exact_match_cnt']}  |  EM rate: {report['em_rate']:.2%}")

    # 新增：打印空预测数量
    print(f"Empty model_generate: {report['num_empty_model_generate']}")

    print("\nSingle-Clip Selection Accuracy (SCSA):")
    print(f"Macro-avg SCSA: {report['scsa_macro_avg']:.4f}  "
          f"(samples with nonzero denom: {report['scsa_samples_den0']}, zero-denom skipped: {report['scsa_samples_den0_skipped']})")

    print(f"Perfect selections (all required IDs hit, order ignored): "
          f"{report['perfect_selection_cnt']} / {report['scsa_samples_den0']} "
          f"= {report['perfect_selection_rate']:.2%}")

    print("-" * 80)
