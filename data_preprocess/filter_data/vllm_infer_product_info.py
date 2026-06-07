#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vLLM LLMEngine offline inference — per (chunk_id, ad_id) extract brand, product & features

Input :
  JSONL，字段至少包含:
    - chunk_id  (区分不同 chunk 来源)
    - ad_id
    - clip_id
    - text
Output:
  JSONL; 每行 = 原始行 + 新字段:
    - brand: str or null
    - product: str or null
    - features: List[str] or null

同一 (chunk_id, ad_id) 下所有 clip 共享同一套结果。
"""

import argparse
import json
import os
import re
from typing import Dict, List, Tuple, Any, Optional
import torch
import pandas as pd
from tqdm import tqdm

from transformers import AutoTokenizer
from vllm import EngineArgs, LLMEngine, RequestOutput, SamplingParams
from vllm.utils import FlexibleArgumentParser


# ---------------------------
# Helpers
# ---------------------------

def build_sampling_params(args) -> SamplingParams:
    return SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_new_tokens,
        n=args.n,
    )

def group_texts_by_chunk_ad(
    all_rows: List[Dict[str, Any]]
) -> Tuple[
    List[Tuple[int, int]],
    Dict[Tuple[int, int], List[int]],
    Dict[Tuple[int, int], List[Tuple[int, str]]]
]:
    """
    分组粒度: (chunk_id, ad_id)

    返回:
      keys_order: [(chunk_id, ad_id), ...] 按首次出现顺序
      key2indices: (chunk_id, ad_id) -> 该组在 all_rows 中的行索引列表
      key2clips:   (chunk_id, ad_id) -> [(clip_id, text), ...]
    """
    key2indices: Dict[Tuple[int, int], List[int]] = {}
    key2clips: Dict[Tuple[int, int], List[Tuple[int, str]]] = {}
    keys_order: List[Tuple[int, int]] = []

    for idx, row in enumerate(all_rows):
        # chunk_id 必须参与 key；若不存在则默认 0
        chunk_id = row.get("chunk_id", 0)
        try:
            chunk_id = int(chunk_id)
        except (TypeError, ValueError):
            chunk_id = 0

        ad_raw = row.get("ad_id", -1)
        try:
            ad_id = int(ad_raw)
        except (TypeError, ValueError):
            continue

        key = (chunk_id, ad_id)

        if key not in key2indices:
            key2indices[key] = []
            key2clips[key] = []
            keys_order.append(key)

        key2indices[key].append(idx)

        clip_raw = row.get("clip_id", 0)
        try:
            clip_id = int(clip_raw)
        except (TypeError, ValueError):
            clip_id = 0

        text = row.get("text", "")
        if not isinstance(text, str):
            text = str(text) if text is not None else ""

        key2clips[key].append((clip_id, text))

    return keys_order, key2indices, key2clips


def build_user_prompt_for_group(
    key: Tuple[int, int],
    clip_texts: List[Tuple[int, str]]
) -> str:
    """
    针对单个 (chunk_id, ad_id) 构造 prompt。
    要求输出三行:
      brand: <品牌或null>
      product: <商品或null>
      features: <用逗号或顿号分隔的多个简短卖点；若没有写null>
    """
    chunk_id, ad_id = key
    clip_texts = sorted(clip_texts, key=lambda x: x[0])
    lines: List[str] = []
    for cid, t in clip_texts:
        t = (t or "").strip()
        if t:
            lines.append(f"({cid}) {t}")
    joined = "\n".join(lines) if lines else ""

    prompt = (
        "任务：请你仅根据下面同一个广告的所有片段台词，判断该广告的品牌（brand）、商品（product），并总结核心卖点（features）。\n"
        "要求：只输出三行，格式必须严格如下（不是 JSON，不要多余解释）：\n"
        "brand: <品牌或null>\n"
        "product: <商品或null>\n"
        "features: <用逗号或顿号分隔的多个简短卖点；若没有写null>\n"
        "注意：卖点为短语，每条不超过12个字；仅依据台词信息。\n"
        f"\n广告定位信息：chunk_id={chunk_id}, ad_id={ad_id}\n"
        f"广告台词：\n{joined}\n"
    )
    return prompt


def prepare_prompts_per_group(
    keys_order: List[Tuple[int, int]],
    key2clips: Dict[Tuple[int, int], List[Tuple[int, str]]],
    tokenizer,
    debug: bool = False
) -> List[Tuple[str, str, Tuple[int, int]]]:
    """
    为每个 (chunk_id, ad_id) 生成:
      (request_id, prompt_text, key)
    request_id 用 "chunk_{c}_ad_{a}"，避免冲突。
    """
    items: List[Tuple[str, str, Tuple[int, int]]] = []
    for key in keys_order:
        user_text = build_user_prompt_for_group(key, key2clips[key])
        messages = [{"role": "user", "content": user_text}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        if debug:
            print(f"\n=== Prompt for key={key} ===\n{text}\n")

        chunk_id, ad_id = key
        rid = f"chunk_{chunk_id}_ad_{ad_id}"
        items.append((rid, text, key))
    return items


_BRAND_PAT = re.compile(r'^(?:\s*(?:brand|品牌)\s*:\s*)(.*)$', re.IGNORECASE)
_PROD_PAT  = re.compile(r'^(?:\s*(?:product|商品)\s*:\s*)(.*)$', re.IGNORECASE)
_FEAT_PAT  = re.compile(r'^(?:\s*(?:features|卖点)\s*:\s*)(.*)$', re.IGNORECASE)

def _norm_field(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    v = s.strip()
    if v == "" or v.lower() == "null":
        return None
    return v

def _split_features(s: str) -> List[str]:
    # 分隔符：中英文逗号、顿号、分号、竖线、斜杠
    parts = re.split(r'[,\uFF0C\u3001;\uFF1B\|/]+', s)
    items: List[str] = []
    seen = set()
    for p in parts:
        t = p.strip().strip("。!！?？；;，,、")
        if not t:
            continue
        if len(t) > 12:
            t = t[:12]
        if t not in seen:
            seen.add(t)
            items.append(t)
        if len(items) >= 5:
            break
    return items

def extract_brand_product_features_from_text(
    gen_text: str
) -> Tuple[Optional[str], Optional[str], Optional[List[str]]]:
    if not gen_text:
        return None, None, None

    brand: Optional[str] = None
    product: Optional[str] = None
    features: Optional[List[str]] = None

    for raw_line in gen_text.splitlines()[:15]:
        line = raw_line.strip()
        if not line:
            continue

        mb = _BRAND_PAT.match(line)
        if mb and brand is None:
            brand = _norm_field(mb.group(1))
            continue

        mp = _PROD_PAT.match(line)
        if mp and product is None:
            product = _norm_field(mp.group(1))
            continue

        mf = _FEAT_PAT.match(line)
        if mf and features is None:
            fs = _norm_field(mf.group(1))
            if fs is None:
                features = None
            else:
                feats = _split_features(fs)
                features = feats if feats else None
            continue

    return brand, product, features


# ---------------------------
# vLLM 循环：按 (chunk_id, ad_id) 组处理
# ---------------------------

def process_with_engine(
    engine: LLMEngine,
    pending_prompts: List[Tuple[str, str, Tuple[int, int]]],
    all_rows: List[Dict[str, Any]],
    key2indices: Dict[Tuple[int, int], List[int]],
    sampling_params: SamplingParams,
    out_fp
):
    """
    对每个 (chunk_id, ad_id) 发一次请求，
    将解析出的 brand/product/features 写回该组下所有 clip 行。
    """
    rid_to_key: Dict[str, Tuple[int, int]] = {rid: key for (rid, _, key) in pending_prompts}
    queue = pending_prompts.copy()

    total_groups = len(pending_prompts)
    pbar = tqdm(total=total_groups, desc="vLLM groups finished", unit="group")

    try:
        while queue or engine.has_unfinished_requests():
            if queue:
                rid, text, key = queue.pop(0)
                engine.add_request(rid, text, sampling_params)

            request_outputs: List[RequestOutput] = engine.step()

            for ro in request_outputs:
                if not ro.finished:
                    continue

                rid = ro.request_id
                if rid not in rid_to_key:
                    continue
                key = rid_to_key.pop(rid)

                # 拿生成结果
                if ro.outputs and len(ro.outputs) > 0:
                    gen_text = ro.outputs[0].text or ""
                    prompt_str = getattr(ro, "prompt", None)
                    if prompt_str and gen_text.startswith(prompt_str):
                        gen_text = gen_text[len(prompt_str):]
                    gen_text = gen_text.strip()
                else:
                    gen_text = ""

                brand, product, features = extract_brand_product_features_from_text(gen_text)

                brand_out = brand if brand is not None else None
                product_out = product if product is not None else None
                features_out = features if features is not None else None

                # 写回该组所有行
                for idx in key2indices[key]:
                    row = dict(all_rows[idx])  # copy 一份
                    row["brand"] = brand_out
                    row["product"] = product_out
                    row["features"] = features_out
                    out_fp.write(json.dumps(row, ensure_ascii=False) + "\n")
                out_fp.flush()

                pbar.update(1)

    finally:
        pbar.close()


# ---------------------------
# CLI & main
# ---------------------------

def parse_args():
    parser = FlexibleArgumentParser(
        description="vLLM brand/product/features extraction per (chunk_id, ad_id)"
    )
    parser = EngineArgs.add_cli_args(parser)

    parser.add_argument(
        "--model_path",
        type=str,
        default="/data/phd/hf_models/Qwen3-32B"
    )
    parser.add_argument(
        "--src_path",
        type=str,
        default="/data/phd/miltonzhou/sft/data_preprocess/filter_data/data_chunk_1102/filtered_by_video_r100_per_chunk_eval.jsonl",
        help="输入 JSONL（包含 chunk_id, ad_id, clip_id, text ...）"
    )
    parser.add_argument(
        "--result_path",
        type=str,
        default="/data/phd/miltonzhou/sft/data_preprocess/filter_data/data_chunk_1102/CT_eval_1109.jsonl",
        help="输出 JSONL（原行 + brand/product/features，按组写入）"
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--debug_prompt", action="store_true", default=False)
    return parser.parse_args()


def main():
    args = parse_args()
    args.model = args.model_path

    # 构造 vLLM 引擎参数
    engine_args = EngineArgs.from_cli_args(args)
    engine_args.gpu_memory_utilization = 0.90
    engine_args.enable_expert_parallel = True if "235B" in args.model else False
    # engine_args.tensor_parallel_size = torch.cuda.device_count()
    engine_args.tensor_parallel_size = 4
    engine_args.max_model_len = 4096

    engine = LLMEngine.from_engine_args(engine_args)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # 读取 JSONL
    all_rows = pd.read_json(args.src_path, lines=True).to_dict("records")
    # all_rows = pd.read_json(args.src_path, lines=True, nrows=100).to_dict("records")

    # 按 (chunk_id, ad_id) 分组
    keys_order, key2indices, key2clips = group_texts_by_chunk_ad(all_rows)

    sampling_params = build_sampling_params(args)

    # 准备 prompts
    pending_prompts = prepare_prompts_per_group(
        keys_order, key2clips, tokenizer, debug=args.debug_prompt
    )

    print(f"Starting vLLM inference: total_groups={len(keys_order)}, saving to {args.result_path}")
    # 直接写新文件；如不想追加历史结果，推荐用 "w"
    with open(args.result_path, "w", encoding="utf-8") as out_fp:
        try:
            process_with_engine(
                engine,
                pending_prompts,
                all_rows,
                key2indices,
                sampling_params,
                out_fp,
            )
        finally:
            try:
                engine.shutdown()
            except Exception:
                pass

if __name__ == "__main__":
    main()
