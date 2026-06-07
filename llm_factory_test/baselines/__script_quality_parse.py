#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Parse SQ results JSONL and compute average "总分"

使用场景：
- 输入：SQ 评估后的 JSONL 文件，每行是一个样本，通常包含：
    {
      "product": "...",
      "brand": "...",
      "features": [...],
      "script": "...",
      "model_output": "...\n\"总分\": 85\n..."
      或
      "parsed_scores": {
          "基础质量": 25,
          "语言与语气": 12,
          "触达受众": 10,
          "创意叙事": 30,
          "总分": 77,
          "理由": "..."
      }
    }

要求：
- 对每一行样本：
    1) 优先从 parsed_scores["总分"] 读取；
    2) 若没有 parsed_scores，则从 model_output 文本中解析出 "总分"；
- 若该条样本无法可靠获取“总分”，则跳过；
- 最终输出：
    - 总样本数
    - 成功解析到“总分”的样本数
    - 平均总分（仅在成功样本上取平均）
"""

import json
import argparse
import re
from typing import Dict, Any, Iterable, List, Optional


def read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception as e:
                print(f"[WARN] 第 {ln} 行 JSON 解析失败: {e}")


def try_parse_json_like(text: str) -> Optional[Dict[str, Any]]:
    """
    尝试将模型输出转为 JSON：
    - 去掉 markdown 代码块包装
    - 去掉开头的 'json' 标记
    - 用 json.loads 解析
    """
    if not text:
        return None

    t = text

    # 去掉代码块 ```...```
    if "```" in t:
        parts = t.split("```")
        # 经验：中间块更可能是 JSON
        if len(parts) >= 3:
            t = parts[-2]

    t = t.replace("\r", "").replace("\t", "").strip()

    # 去掉可能的 'json' 前缀
    if t.lower().startswith("json"):
        t = t[4:].strip()

    # 简单兜底：如果不是以 { 开头，解析成功率会很低
    t = t.strip()
    if not t.startswith("{"):
        return None

    try:
        return json.loads(t)
    except Exception:
        return None


def extract_total_score_from_record(rec: Dict[str, Any]) -> Optional[float]:
    """
    从一条结果记录中抽取“总分”：
    优先级：
    1) parsed_scores["总分"]
    2) model_output 里的 JSON 结构中的 "总分"
    3) model_output 文本中用正则匹配 `"总分": xx` / `总分：xx`
    若都失败则返回 None
    """

    # 1) parsed_scores
    parsed = rec.get("parsed_scores")
    if isinstance(parsed, dict):
        score = parsed.get("总分")
        if isinstance(score, (int, float)):
            return float(score)

    # 2) 尝试把 model_output 当 JSON-like 解析
    model_output = rec.get("model_output")
    if isinstance(model_output, str) and model_output.strip():
        as_json = try_parse_json_like(model_output)
        if isinstance(as_json, dict):
            score = as_json.get("总分")
            if isinstance(score, (int, float)):
                return float(score)

        # 3) 直接在纯文本中用正则找 "总分"
        # 匹配形式：
        # 总分: 85
        # "总分": 85
        # 总分：85.5
        m = re.search(
            r'总分["」】]?\s*[:：]\s*([0-9]+(?:\.[0-9]+)?)',  # 较宽松一点
            model_output
        )
        if not m:
            # 再试试更普通的："总分": 85
            m = re.search(
                r'"总分"\s*:\s*([0-9]+(?:\.[0-9]+)?)',
                model_output
            )
        if m:
            try:
                return float(m.group(1))
            except Exception:
                return None

    return None


def compute_average_score(jsonl_path: str):
    total_records = 0
    valid_records = 0
    scores: List[float] = []

    for rec in read_jsonl(jsonl_path):
        total_records += 1
        score = extract_total_score_from_record(rec)
        if score is not None:
            valid_records += 1
            scores.append(score)

    print(f"评估数据: {jsonl_path}")
    print(f"总样本数: {total_records}")
    print(f"成功解析到“总分”的样本数: {valid_records}")

    if not scores:
        print("未能从任何样本中解析出有效的“总分”。")
        return

    avg_score = sum(scores) / len(scores)
    print(f"平均总分: {avg_score:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute average SQ '总分' from JSONL results")
    parser.add_argument(
        "--input",
        type=str,
        default="/data/phd/qinsizhong/llm_factory_test/baselines/results/SQ__gemini3f.jsonl",
        help="输入 SQ 评估结果 JSONL 文件路径（包含 model_output 或 parsed_scores）"
    )
    args = parser.parse_args()

    compute_average_score(args.input)
