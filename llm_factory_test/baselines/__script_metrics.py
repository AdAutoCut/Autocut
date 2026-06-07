#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
计算广告台词长度匹配指标：WCD 和 WCD_clip（支持行数不一致时用0补齐）

更新点：
- 行数不一致的样本不再跳过；
- WCD_clip 在 max(len(target), len(model)) 的行数上逐行计算，
  缺失的一侧按长度 0 处理，然后求平均。
"""

import json
import argparse
import re


def remove_spaces(s: str) -> str:
    """移除字符串中的所有空白字符（包括半角/全角空格、制表符等）。"""
    return re.sub(r'\s+', '', s)


def extract_target_script(sample):
    """优先从 sample['value']，否则从 conversations 中最后一条 gpt 提取。"""
    if isinstance(sample.get("value"), str):
        return sample["value"].strip()

    conv = sample.get("conversations")
    if isinstance(conv, list):
        gpt_values = [
            m.get("value", "")
            for m in conv
            if m.get("from") == "gpt" and isinstance(m.get("value"), str)
        ]
        if gpt_values:
            return gpt_values[-1].strip()

    return None


def extract_model_script(sample):
    """从 sample['model_generate'] 提取模型生成台词。"""
    text = sample.get("model_generate")
    if isinstance(text, str):
        return text.strip()
    return None


def split_lines_target(text):
    """
    目标台词按行拆分：
    - 去掉首尾空格
    - 跳过空行
    - 移除每行中的所有空白字符
    """
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        line = remove_spaces(line)
        if line:
            lines.append(line)
    return lines


def clean_model_line_prefix(line):
    """
    清洗模型台词前缀中的序号和多余符号，只作用于每一行开头。
    示例:
        "1. xxx" "2) xxx" "3）xxx" "4、xxx" "5: xxx" "(1) xxx" "（2）xxx"
        "• xxx" "- xxx" "· xxx" "第一句：xxx" "第1句: xxx"
    """
    s = line.strip()
    # 去除常见数字序号形式: 1. 1) 1） 1、 1: 1：等
    s = re.sub(r'^\s*\d+\s*[\.\)\）、:：-]\s*', '', s)
    # 去除括号数字: (1) （1）
    s = re.sub(r'^\s*[\(\（]\d+[\)\）]\s*', '', s)
    # 去除项目符号: • - ·
    s = re.sub(r'^\s*[•\-·]\s*', '', s)
    # 去除“第X句/条/行/段：”等
    s = re.sub(r'^第[一二三四五六七八九十百零0-9]+[句条行段][:：]\s*', '', s)
    return s.strip()


def split_lines_model(text):
    """
    模型台词按行拆分并清洗：
    - 去掉首尾空格和空行
    - 去掉行首编号/项目符号等前缀
    - 移除每行中的所有空白字符
    """
    lines = []
    for raw in text.splitlines():
        raw = raw.strip()
        if not raw:
            continue
        cleaned = clean_model_line_prefix(raw)
        cleaned = remove_spaces(cleaned)
        if cleaned:
            lines.append(cleaned)
    return lines


def compute_wcd_metrics(jsonl_path):
    total_samples = 0
    matched_linecount_samples = 0
    mismatched_linecount_samples = 0

    wcd_list = []
    wcd_clip_list = []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            total_samples += 1

            try:
                sample = json.loads(line)
            except json.JSONDecodeError:
                continue

            target_script = extract_target_script(sample)
            model_script = extract_model_script(sample)

            # 若两侧完全缺失，跳过
            if not target_script and not model_script:
                continue

            target_lines = split_lines_target(target_script or "")
            model_lines = split_lines_model(model_script or "")

            n_t = len(target_lines)
            n_m = len(model_lines)

            if n_t == n_m and n_t > 0:
                matched_linecount_samples += 1
            else:
                # 行数不一致或一侧为0，都纳入“不一致计数”
                mismatched_linecount_samples += 1

            # 若两侧皆 0 行，则无法定义逐句评估，跳过
            L = max(n_t, n_m)
            if L == 0:
                continue

            # ---- WCD（整段）----
            target_total_len = sum(len(s) for s in target_lines)
            model_total_len = sum(len(s) for s in model_lines)
            wcd = abs(model_total_len - target_total_len)

            # ---- WCD_clip（按最大行数补0取平均）----
            diffs = []
            for i in range(L):
                lt = len(target_lines[i]) if i < n_t else 0
                lm = len(model_lines[i]) if i < n_m else 0
                diffs.append(abs(lm - lt))
            wcd_clip = sum(diffs) / L

            wcd_list.append(wcd)
            wcd_clip_list.append(wcd_clip)

    print(f"数据: {jsonl_path}")
    print(f"总样本数: {total_samples}")
    print(f"行数一致样本数: {matched_linecount_samples}")
    print(f"行数不一致/一侧为空样本数: {mismatched_linecount_samples}")

    if not wcd_list:
        print("无可计算样本，无法计算 WCD 和 WCD_clip。")
        return

    avg_wcd = sum(wcd_list) / len(wcd_list)
    avg_wcd_clip = sum(wcd_clip_list) / len(wcd_clip_list)

    print(f"平均 WCD: {avg_wcd:.4f}")
    print(f"平均 WCD_clip: {avg_wcd_clip:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="计算广告台词 WCD 和 WCD_clip 指标（行数不一致时用0补齐）")
    parser.add_argument(
        "--input",
        type=str,
        default="/data/phd/qinsizhong/llm_factory_test/baselines/results/F___vid2text_gemini3f.jsonl",
        help="输入 JSONL 文件路径"
    )
    args = parser.parse_args()
    compute_wcd_metrics(args.input)
