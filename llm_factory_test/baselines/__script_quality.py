#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script Quality Evaluation from ShareGPT-style input (Azure OpenAI)
------------------------------------------------------------------
- 输入: ShareGPT 格式 JSONL（每行包含：
    - conversations: [
          {from:"human", value:"...包含【商品】【品牌】【卖点】..."},
          {from:"gpt",   value:"参考/目标广告台词"},
          ...
      ]
    - model_generate: 模型生成的台词（待评估脚本）
  ）

- 逻辑（新版）:
    1) 提取参考广告台词 ref_script（最后一条 from=="gpt" 的 value）
    2) 提取待评估脚本 model_script（model_generate）
    3) 不强制行数一致；统计逐行字数差（用于“长度与节奏/字数匹配度”20分）：
       - 仅用于长度统计：去掉行内所有空白字符；
       - 模型行清洗行首编号/项目符号后再统计；
       - 若模型多出行：差值= len(model_i) - 0；
         若模型缺少行：差值= 0 - len(ref_i)；
       - avg_diff = 平均 |差值|；
    4) 从 human.value 中解析: 商品(product)、品牌(brand)、卖点(features)
    5) 调用 Azure OpenAI 做 Script Quality 评估（新 3 大类：30/40/30）

- 输出: JSONL（每行对应一个“已评估”的样本，包含：
    - product, brand, features
    - ref_script（参考台词，逐行）
    - script（模型台词，行首编号已清理）
    - stats: {ref_lengths, model_lengths, diffs, avg_diff}
    - model_output: SQ 打分原始结果
    - 可选 parsed_scores: 尝试解析出的结构化分数
"""

import os
import re
import json
import argparse
from typing import Dict, Any, Iterable, List, Optional
from openai import AzureOpenAI
from tqdm import tqdm


# ================== 评估 Prompt 模板（使用占位符，避免花括号转义问题） ==================
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
  规则：每句台词长度应与参考台词接近。使用已提供的 avg_diff（基于逐行 |model_len - ref_len| 的平均值，已去行内空白并处理缺/多行）计分：
  分数 = max(0, 20 - avg_diff)。
- 断句与节奏（Rhythm & Pausing）[10 分]
  规则：断句自然、语义完整；避免一句过长或过碎。

----------------------------------------
输出格式（必须为 JSON）：
{{
  "基础质量": <0-30>,
  "表达与沟通": <0-40>,
  "长度与节奏": <0-30>,
  "总分": <0-100>,
  "理由": "用 2–4 句简要说明主要优点与不足。"
}}

注意：
- 总分 = 三个大类分数之和；
- “字数匹配度（20分）”务必基于 avg_diff 使用：20 - avg_diff，低于 0 则取 0；
- 其他分项依据上面的定义独立评估，不要把长度因素重复计入非长度类目。

----------------------------------------
参考信息（供评分使用）：
产品信息：[[PRODUCT_INFO]]

参考脚本（人工台词，逐行）：
[[REF_SCRIPT]]

待评估脚本（模型生成，逐行，已清理行首编号/项目符号）：
[[MODEL_SCRIPT]]

平均字数差 avg_diff：[[AVG_DIFF]]
"""


# ================== 基础 IO ==================
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


# ================== ShareGPT 解析（商品信息） ==================
_PRODUCT_RE = re.compile(r"【商品】：(?P<prod>.+?)\s*(?:\n|$)")
_BRAND_RE   = re.compile(r"【品牌】：(?P<brand>.+?)\s*(?:\n|$)")
_FEAT_RE    = re.compile(r"【卖点】：(?P<feat>\[.*?\])\s*(?:\n|$)")


def _clean_str(x: Optional[str]) -> str:
    if x is None:
        return "无"
    s = str(x).strip()
    return s if s else "无"


def extract_product_info_from_human(human_value: str) -> Dict[str, Any]:
    """从 human.value 文本中解析: 商品 / 品牌 / 卖点（list[str]）"""
    product = "无"
    brand   = "无"
    features: List[str] = []

    # 商品
    m = _PRODUCT_RE.search(human_value)
    if m:
        product = _clean_str(m.group("prod"))

    # 品牌
    m = _BRAND_RE.search(human_value)
    if m:
        brand = _clean_str(m.group("brand"))

    # 卖点
    m = _FEAT_RE.search(human_value)
    if m:
        raw = m.group("feat").strip()
        feats = None
        try:
            feats = json.loads(raw)
        except Exception:
            cleaned = raw.replace("'", '"')
            try:
                feats = json.loads(cleaned)
            except Exception:
                body = raw.strip("[] \t\r\n")
                feats = [s.strip().strip('"').strip("'") for s in re.split(r"[，,]", body) if s.strip()]
        if isinstance(feats, list):
            features = [s for s in (str(x).strip() for x in feats) if s]

    return {"product": product, "brand": brand, "features": features}


# ================== 提取参考台词 & 模型台词 ==================
def extract_ref_script(rec: Dict[str, Any]) -> str:
    """从 conversations 中提取参考广告台词：取最后一个 from == "gpt" 的 value。"""
    convs = rec.get("conversations") or []
    if not isinstance(convs, list):
        return ""
    gpt_values = [
        m.get("value", "")
        for m in convs
        if isinstance(m, dict) and m.get("from") == "gpt" and isinstance(m.get("value"), str)
    ]
    if not gpt_values:
        return ""
    return gpt_values[-1].strip()


def extract_model_script(rec: Dict[str, Any]) -> str:
    """从 model_generate 提取待评估脚本。"""
    text = rec.get("model_generate")
    if isinstance(text, str):
        return text.strip()
    return ""


def extract_human_value(rec: Dict[str, Any]) -> str:
    """提取 human 提示（用于解析商品/品牌/卖点）。默认取 conversations[0].value。"""
    convs = rec.get("conversations") or []
    if isinstance(convs, list) and len(convs) > 0 and isinstance(convs[0], dict):
        return (convs[0].get("value") or "").strip()
    return ""


def split_lines_keep(text: str) -> List[str]:
    """通用行拆分（保留行内空格；用于传给 GPT 的可读脚本）。"""
    return [ln.strip() for ln in text.splitlines() if ln.strip()]


# ================== 文本清洗与长度统计 ==================
_WS_RE = re.compile(r'\s+', flags=re.UNICODE)

def remove_all_spaces(s: str) -> str:
    """移除所有空白字符（半角/全角空格、制表符等）。"""
    return _WS_RE.sub('', s)


def clean_model_line_prefix(line: str) -> str:
    """
    清洗模型台词前缀中的序号和多余符号，只作用于每一行开头。
    示例（都会被去掉前缀部分）:
        "1. xxx", "2) xxx", "3）xxx", "4、xxx", "5: xxx", "(1) xxx", "（2）xxx",
        "• xxx", "- xxx", "· xxx", "第1句：xxx", "第一句: xxx"
    """
    s = line.strip()
    s = re.sub(r'^\s*\d+\s*[\.\)\）、:：-]\s*', '', s)         # 1. 1) 1） 1、 1: 1： 等
    s = re.sub(r'^\s*[\(\（]\d+[\)\）]\s*', '', s)             # (1) （1）
    s = re.sub(r'^\s*[•\-·]\s*', '', s)                       # 项目符号
    s = re.sub(r'^第[一二三四五六七八九十百零0-9]+[句条行段][:：]\s*', '', s)  # 第X句：
    return s.strip()


def stats_lengths(ref_lines: List[str], model_lines_raw: List[str]) -> Dict[str, Any]:
    """
    计算长度统计（仅用于长度匹配评分，不影响传给 GPT 的可读脚本）：
    - 参考行：去除行内空白后计 len
    - 模型行：先清行首编号/项目符号，再去除行内空白后计 len
    - 对齐方式：按下标对齐；多出的模型行与缺失的参考行按 0 处理
    """
    # 清洗用于统计的模型文本（仅长度统计用）
    model_lines_clean_for_len = [remove_all_spaces(clean_model_line_prefix(x)) for x in model_lines_raw]
    ref_lines_for_len         = [remove_all_spaces(x) for x in ref_lines]

    max_len = max(len(ref_lines_for_len), len(model_lines_clean_for_len))
    ref_lens, model_lens, diffs = [], [], []

    for i in range(max_len):
        r = ref_lines_for_len[i]   if i < len(ref_lines_for_len)   else ''
        m = model_lines_clean_for_len[i] if i < len(model_lines_clean_for_len) else ''
        rl = len(r)
        ml = len(m)
        ref_lens.append(rl)
        model_lens.append(ml)
        diffs.append(abs(ml - rl))

    avg_diff = float(sum(diffs)) / max(1, len(diffs))
    # 也生成给 GPT 阅读的“可读版本”（不去空格，只清行首编号）
    model_lines_for_read = [clean_model_line_prefix(x).strip() for x in model_lines_raw]

    return {
        "ref_lengths": ref_lens,
        "model_lengths": model_lens,
        "diffs": diffs,
        "avg_diff": avg_diff,
        "model_lines_for_read": model_lines_for_read
    }


# ================== 模型调用 & 解析 ==================
def call_model(client: AzureOpenAI,
               deployment: str,
               product_info_text: str,
               ref_lines: List[str],
               model_lines_for_read: List[str],
               ref_lens: List[int],
               model_lens: List[int],
               diffs: List[int],
               avg_diff: float,
               temperature: float = 0.0,
               max_tokens: int = 700) -> str:
    """
    调用 Azure OpenAI，返回原始字符串输出（使用占位符替换，避免 format 花括号冲突）
    """
    prompt = (
        EVAL_PROMPT
        .replace("[[PRODUCT_INFO]]", product_info_text)
        .replace("[[REF_SCRIPT]]", "\n".join(ref_lines))
        .replace("[[MODEL_SCRIPT]]", "\n".join(model_lines_for_read))
        # .replace("[[REF_LENGTHS]]", json.dumps(ref_lens, ensure_ascii=False))
        # .replace("[[MODEL_LENGTHS]]", json.dumps(model_lens, ensure_ascii=False))
        # .replace("[[LENGTH_DIFFS]]", json.dumps(diffs, ensure_ascii=False))
        .replace("[[AVG_DIFF]]", f"{avg_diff:.4f}")
    )

    messages = [
        {"role": "system", "content": "你是专业广告脚本质量评估助手。"},
        {"role": "user", "content": prompt},
    ]

    resp = client.chat.completions.create(
        model=deployment,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return (resp.choices[0].message.content or "").strip()


def try_parse_json_like(text: str) -> Dict[str, Any]:
    """尝试把模型输出转成 JSON 对象（可选）"""
    if not text:
        return {}
    try:
        t = text
        if "```" in t:
            parts = t.split("```")
            if len(parts) >= 3:
                t = parts[-2]
        t = t.replace("\r", "").replace("\t", "").strip()
        if t.lower().startswith("json"):
            t = t[4:].strip()
        return json.loads(t)
    except Exception:
        return {}


# ================== 主流程 ==================
def main():
    ap = argparse.ArgumentParser(description="Script Quality Evaluation from ShareGPT-style JSONL")
    ap.add_argument("--input",  default="/data/phd/qinsizhong/llm_factory_test/baselines/results/F___vid2text_llava.jsonl", help="输入 ShareGPT JSONL 文件")
    ap.add_argument("--output", default="/data/phd/qinsizhong/llm_factory_test/baselines/results/SQ__llava.jsonl", help="输出评估 JSONL 文件")
    ap.add_argument("--endpoint", type=str, default=os.getenv("ENDPOINT_URL", ""))
    ap.add_argument("--deployment", type=str, default=os.getenv("DEPLOYMENT_NAME", ""))
    ap.add_argument("--api_key", type=str, default=os.getenv("AZURE_OPENAI_API_KEY", ""))
    ap.add_argument("--parse_json", action="store_true", help="尝试解析模型输出为 JSON 对象")
    args = ap.parse_args()

    if not args.endpoint or not args.deployment or not args.api_key:
        raise ValueError("请通过参数或环境变量提供有效的 endpoint / deployment / api_key。")

    client = AzureOpenAI(
        azure_endpoint=args.endpoint,
        api_key=args.api_key,
        api_version="2025-01-01-preview",
    )

    total_samples = 0
    evaluated_samples = 0
    skipped_samples = 0

    out_rows: List[Dict[str, Any]] = []

    for rec in tqdm(list(read_jsonl(args.input)), desc="Evaluating (length-aware)"):
        total_samples += 1

        # 1) 提取参考台词 & 模型台词
        ref_script = extract_ref_script(rec)
        model_script = extract_model_script(rec)

        # 必要字段缺失则跳过
        if not isinstance(ref_script, str) or not isinstance(model_script, str):
            skipped_samples += 1
            continue

        ref_lines = split_lines_keep(ref_script)
        model_lines_raw = split_lines_keep(model_script)

        # 两者至少要有一条非空句子才有意义
        if len(ref_lines) == 0 and len(model_lines_raw) == 0:
            skipped_samples += 1
            continue

        # 2) 解析商品信息（来自 human 提示）
        human_value = extract_human_value(rec)
        info = extract_product_info_from_human(human_value)
        product = info["product"]
        brand   = info["brand"]
        feats   = info["features"]
        product_info_text = f"商品：{product}；品牌：{brand}；卖点：{feats}"

        # 3) 统计长度（用于“长度与节奏/字数匹配度”）
        st = stats_lengths(ref_lines, model_lines_raw)
        ref_lens   = st["ref_lengths"]
        model_lens = st["model_lengths"]
        diffs      = st["diffs"]
        avg_diff   = st["avg_diff"]
        model_lines_for_read = st["model_lines_for_read"]

        # 4) 调用模型评估
        try:
            raw_output = call_model(
                client=client,
                deployment=args.deployment,
                product_info_text=product_info_text,
                ref_lines=ref_lines,
                model_lines_for_read=model_lines_for_read,
                ref_lens=ref_lens,
                model_lens=model_lens,
                diffs=diffs,
                avg_diff=avg_diff,
            )
        except Exception as e:
            print(f"[ERROR] API 调用失败，跳过该样本: {e}")
            skipped_samples += 1
            continue

        # 5) 写入结果
        out_rec = {
            "product": product,
            "brand": brand,
            "features": feats,
            "ref_script": "\n".join(ref_lines),
            "script": "\n".join(model_lines_for_read),
            "stats": {
                "ref_lengths": ref_lens,
                "model_lengths": model_lens,
                "diffs": diffs,
                "avg_diff": avg_diff,
            },
            "model_output": raw_output,
        }
        if args.parse_json:
            out_rec["parsed_scores"] = try_parse_json_like(raw_output)

        out_rows.append(out_rec)
        evaluated_samples += 1

    write_jsonl(args.output, out_rows)

    print(f"[统计] 总样本数: {total_samples}")
    print(f"[统计] 已评估样本数: {evaluated_samples}")
    print(f"[统计] 被跳过样本数: {skipped_samples}")
    print(f"[OK] 已完成评估，有效结果 {len(out_rows)} 条；保存到: {args.output}")


if __name__ == "__main__":
    main()
