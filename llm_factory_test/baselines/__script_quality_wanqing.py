#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script Quality Evaluation from ShareGPT-style input
WanQing OpenAI-compatible version
"""

import os
import re
import json
import argparse
from typing import Dict, Any, Iterable, List, Optional
from tqdm import tqdm
from openai import OpenAI


# ================== 评估 Prompt 模板（完全不变） ==================
EVAL_PROMPT = """
你是一名专业的广告台词评估员。你将收到：
1) 产品信息；
2) 参考脚本（人工台词，逐行对应视频片段）；
3) 待评估脚本（模型生成，逐行）；
4) 已预计算的“逐行字数统计与差值表”和“平均字数差 avg_diff”。

请你按“三个大类、总分100”的标准打分，并严格遵守下述定义与配分：
----------------------------------------
类别 1：基础质量 —— 共 30 分
----------------------------------------
- 准确性（Accuracy）[10 分]：脚本内容必须与提供的商品与品牌信息一致；不得出现错误品牌、虚假功能或事实错误；拼写语法正确。
- 可理解性（Understandability）[10 分]：语言连贯、逻辑清晰；即使不看画面也能理解。
- 内容安全（Content Safety）[10 分]：不包含冒犯或危险表述；不涉政治、宗教、性别、种族、暴力、毒品等敏感内容。

----------------------------------------
类别 2：表达与沟通 ——共 40 分
----------------------------------------
- 语言与语气自然（Language & Tone）[10 分]：口语化、亲切自然、不生硬，贴近受众真实说话方式。
- 吸引力（Engagement）[10 分]：能吸引注意力，愿意听下去；可通过提问、悬念、幽默等方式增强兴趣。
- 卖点呈现清晰（Selling Points Clarity）[20 分]：准确且有说服力地传达产品核心卖点或优势；表达自然、有感染力；若能融入创意/情感并形成记忆点，给高分。

----------------------------------------
类别 3：长度与节奏 ——共 30 分
----------------------------------------
- 字数匹配度（Length Matching）[20 分]
  分数 = max(0, 20 - avg_diff)。
- 断句与节奏（Rhythm & Pausing）[10 分]

----------------------------------------
输出格式（必须为 JSON）：
{
  "基础质量": <0-30>,
  "表达与沟通": <0-40>,
  "长度与节奏": <0-30>,
  "总分": <0-100>,
  "理由": "用 2–4 句简要说明主要优点与不足。"
}

参考信息：
产品信息：[[PRODUCT_INFO]]
参考脚本：
[[REF_SCRIPT]]
待评估脚本：
[[MODEL_SCRIPT]]
平均字数差 avg_diff：[[AVG_DIFF]]
"""


# ================== IO ==================
def read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception as e:
                print(f"[WARN] 第{ln}行 JSON 解析失败: {e}")


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ================== ShareGPT 解析 ==================
_PRODUCT_RE = re.compile(r"【商品】：(.+?)\s*(?:\n|$)")
_BRAND_RE   = re.compile(r"【品牌】：(.+?)\s*(?:\n|$)")
_FEAT_RE    = re.compile(r"【卖点】：(\[.*?\])\s*(?:\n|$)")


def extract_product_info_from_human(human_value: str) -> Dict[str, Any]:
    product = "无"
    brand   = "无"
    features: List[str] = []

    m = _PRODUCT_RE.search(human_value)
    if m:
        product = m.group(1).strip()

    m = _BRAND_RE.search(human_value)
    if m:
        brand = m.group(1).strip()

    m = _FEAT_RE.search(human_value)
    if m:
        raw = m.group(1)
        try:
            features = json.loads(raw)
        except Exception:
            body = raw.strip("[]")
            features = [x.strip() for x in re.split(r"[，,]", body) if x.strip()]

    return {"product": product, "brand": brand, "features": features}


def extract_ref_script(rec: Dict[str, Any]) -> str:
    convs = rec.get("conversations") or []
    gpt_values = [
        m.get("value", "")
        for m in convs
        if isinstance(m, dict) and m.get("from") == "gpt"
    ]
    return gpt_values[-1].strip() if gpt_values else ""


def extract_model_script(rec: Dict[str, Any]) -> str:
    return (rec.get("model_generate") or "").strip()


def extract_human_value(rec: Dict[str, Any]) -> str:
    convs = rec.get("conversations") or []
    if convs and isinstance(convs[0], dict):
        return (convs[0].get("value") or "").strip()
    return ""


def split_lines_keep(text: str) -> List[str]:
    return [ln.strip() for ln in text.splitlines() if ln.strip()]


# ================== 长度统计（完全不变） ==================
_WS_RE = re.compile(r'\s+', flags=re.UNICODE)

def remove_all_spaces(s: str) -> str:
    return _WS_RE.sub('', s)


def clean_model_line_prefix(line: str) -> str:
    s = line.strip()
    s = re.sub(r'^\s*\d+[\.\)\）、:：-]\s*', '', s)
    s = re.sub(r'^\s*[•\-·]\s*', '', s)
    s = re.sub(r'^第[一二三四五六七八九十0-9]+[句条行段][:：]\s*', '', s)
    return s.strip()


def stats_lengths(ref_lines: List[str], model_lines_raw: List[str]) -> Dict[str, Any]:
    model_clean = [remove_all_spaces(clean_model_line_prefix(x)) for x in model_lines_raw]
    ref_clean   = [remove_all_spaces(x) for x in ref_lines]

    max_len = max(len(ref_clean), len(model_clean))
    diffs, ref_lens, model_lens = [], [], []

    for i in range(max_len):
        r = ref_clean[i] if i < len(ref_clean) else ''
        m = model_clean[i] if i < len(model_clean) else ''
        ref_lens.append(len(r))
        model_lens.append(len(m))
        diffs.append(abs(len(m) - len(r)))

    avg_diff = sum(diffs) / max(1, len(diffs))
    model_lines_for_read = [clean_model_line_prefix(x) for x in model_lines_raw]

    return {
        "ref_lengths": ref_lens,
        "model_lengths": model_lens,
        "diffs": diffs,
        "avg_diff": avg_diff,
        "model_lines_for_read": model_lines_for_read
    }


# ================== 模型调用（唯一实质改动） ==================
def call_model(
    client: OpenAI,
    ep_id: str,
    product_info_text: str,
    ref_lines: List[str],
    model_lines_for_read: List[str],
    avg_diff: float,
    temperature: float = 0.0,
    max_tokens: int = 700,
) -> str:

    prompt = (
        EVAL_PROMPT
        .replace("[[PRODUCT_INFO]]", product_info_text)
        .replace("[[REF_SCRIPT]]", "\n".join(ref_lines))
        .replace("[[MODEL_SCRIPT]]", "\n".join(model_lines_for_read))
        .replace("[[AVG_DIFF]]", f"{avg_diff:.4f}")
    )

    messages = [
        {"role": "system", "content": "你是专业广告脚本质量评估助手。"},
        {"role": "user", "content": prompt},
    ]

    resp = client.chat.completions.create(
        model=ep_id,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return (resp.choices[0].message.content or "").strip()


def try_parse_json_like(text: str) -> Dict[str, Any]:
    try:
        if "```" in text:
            text = text.split("```")[-2]
        return json.loads(text)
    except Exception:
        return {}


# ================== 主流程 ==================
def main():
    ap = argparse.ArgumentParser(description="Script Quality Evaluation (WanQing)")
    ap.add_argument("--input", default="/data/phd/qinsizhong/llm_factory_test/baselines/results/F___vid2text_gemini3f.jsonl")
    ap.add_argument("--output", default="/data/phd/qinsizhong/llm_factory_test/baselines/results/SQ__gemini3f.jsonl")
    ap.add_argument("--parse_json", action="store_true")

    ap.add_argument("--endpoint", default=os.getenv("WQ_BASE_URL",""))
    ap.add_argument("--ep_id", default=os.getenv("WQ_EP_ID",""))
    ap.add_argument("--api_key", default=os.getenv("WQ_API_KEY",""))
    args = ap.parse_args()

    assert args.endpoint and args.ep_id and args.api_key, "missing wanqing API settings"

    client = OpenAI(
        base_url=args.endpoint,
        api_key=args.api_key
    )

    out_rows = []

    for rec in tqdm(list(read_jsonl(args.input)), desc="Evaluating SQ"):
        ref_script = extract_ref_script(rec)
        model_script = extract_model_script(rec)
        if not ref_script or not model_script:
            continue

        ref_lines = split_lines_keep(ref_script)
        model_lines_raw = split_lines_keep(model_script)

        human_value = extract_human_value(rec)
        info = extract_product_info_from_human(human_value)
        product_info_text = f"商品：{info['product']}；品牌：{info['brand']}；卖点：{info['features']}"

        st = stats_lengths(ref_lines, model_lines_raw)

        raw_output = call_model(
            client=client,
            ep_id=args.ep_id,
            product_info_text=product_info_text,
            ref_lines=ref_lines,
            model_lines_for_read=st["model_lines_for_read"],
            avg_diff=st["avg_diff"],
        )

        row = {
            "product": info["product"],
            "brand": info["brand"],
            "features": info["features"],
            "ref_script": "\n".join(ref_lines),
            "script": "\n".join(st["model_lines_for_read"]),
            "stats": {
                "ref_lengths": st["ref_lengths"],
                "model_lengths": st["model_lengths"],
                "diffs": st["diffs"],
                "avg_diff": st["avg_diff"],
            },
            "model_output": raw_output,
        }

        if args.parse_json:
            row["parsed_scores"] = try_parse_json_like(raw_output)

        out_rows.append(row)

    write_jsonl(args.output, out_rows)
    print(f"[OK] 完成 Script Quality 评估，结果数={len(out_rows)}，输出={args.output}")


if __name__ == "__main__":
    main()
