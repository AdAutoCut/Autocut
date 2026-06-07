#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gemini 多模态离线推理脚本（支持 select / sort 任务）

【最小改动原则】
- 推理逻辑 100% 不变
- 仅将 AzureOpenAI → OpenAI（Gemini OpenAI-compatible API）
- 输出结构完全一致
"""

import argparse
import json
import os
import re
import base64
from typing import Dict, List, Tuple

import pandas as pd
from tqdm import tqdm
from openai import OpenAI


# ---------------------------
# Helpers
# ---------------------------

def _read_records_auto(path: str) -> List[Dict]:
    """Read JSON or JSONL automatically into list[dict]."""
    try:
        return pd.read_json(path, lines=False).to_dict("records")
    except ValueError:
        return pd.read_json(path, lines=True).to_dict("records")


def _extract_human_value(rec: Dict) -> str:
    """从 conversations 取第一条 human/user 的 value。"""
    conv = rec.get("conversations", [])
    if not conv:
        return ""
    for m in conv:
        if m.get("from") in ("human", "user"):
            return m.get("value", "") or ""
    return conv[0].get("value", "") or ""


def _extract_frames_and_rewrite(text: str) -> Tuple[List[str], str]:
    """
    从文本中提取 [frame_id]，并将其替换为 "Picture k"。
    """
    frame_ids: List[str] = []

    def repl(match: re.Match) -> str:
        fid = match.group(1)
        frame_ids.append(fid)
        return f"Picture {len(frame_ids)}"

    new_text = re.sub(r"\[(\d+)\]", repl, text)
    return frame_ids, new_text


def _encode_image_to_data_url(img_path: str) -> str:
    """将本地图片编码为 data:image/jpeg;base64,..."""
    if not os.path.exists(img_path):
        print(f"[warn] image not found: {img_path}")
        return ""
    try:
        with open(img_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
        return f"data:image/jpeg;base64,{b64}"
    except Exception as e:
        print(f"[warn] fail to encode image: {img_path} ({e})")
        return ""


def build_client(api_key: str) -> OpenAI:
    """初始化 Gemini OpenAI-compatible 客户端。"""
    return OpenAI(
        api_key=api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    )


def build_output_record(src: Dict, answer: str) -> Dict:
    """构造输出记录：meta + 原始 conversations + 模型生成结果。"""
    return {
        "sample_id": src.get("sample_id", None),
        "task": src.get("task", None),
        "ad_key": src.get("ad_key", None),
        "conversations": src.get("conversations", []),
        "system": src.get("system", ""),
        "tools": src.get("tools", ""),
        "model_generate": answer,
    }


# ---------------------------
# Core inference
# ---------------------------

def infer_records_with_gemini(
    client: OpenAI,
    model_name: str,
    records: List[Dict],
    frames_dir: str,
    result_path: str,
    max_tokens: int = 256,
    temperature: float = 0.0,
):
    """
    对所有 records 逐条调用 Gemini 多模态推理，并将结果写入 result_path(JSONL)。
    metadata 透传，不影响 prompt。
    """
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    total = len(records)
    written = 0

    with open(result_path, "a", encoding="utf-8") as out_fp:
        for idx, rec in enumerate(tqdm(records, desc="Gemini inference", unit="sample")):
            raw_text = _extract_human_value(rec)
            if not raw_text:
                continue

            frame_ids, new_text = _extract_frames_and_rewrite(raw_text)
            if not frame_ids:
                continue

            image_urls = []
            for fid in frame_ids:
                img_path = os.path.join(frames_dir, f"{fid}.jpg")
                data_url = _encode_image_to_data_url(img_path)
                if data_url:
                    image_urls.append(data_url)

            if not image_urls:
                continue

            user_content = [{"type": "text", "text": new_text}]
            for url in image_urls:
                user_content.append(
                    {"type": "image_url", "image_url": {"url": url}}
                )

            try:
                completion = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": user_content}],
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
            except Exception as e:
                print(f"[error] idx={idx}: API 调用失败: {e}")
                continue

            if not completion.choices:
                continue

            answer = (completion.choices[0].message.content or "").strip()
            out_fp.write(
                json.dumps(build_output_record(rec, answer), ensure_ascii=False) + "\n"
            )
            out_fp.flush()
            written += 1

    print(f"[info] 推理完成，共写入 {written}/{total} 条样本到 {result_path}")


# ---------------------------
# CLI
# ---------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Gemini 多模态推理（select/sort 任务，OpenAI-compatible）"
    )
    parser.add_argument("--api_key", type=str, default=os.getenv("GEMINI_API_KEY",""))
    parser.add_argument("--model", type=str, default="gemini-3-flash-preview")
    parser.add_argument("--prompts_path", type=str, default="/data/phd/qinsizhong/miltonzhou/sft/data_preprocess/T_baseline_mm_vidselect.json")
    parser.add_argument("--frames_dir", type=str, default="/data/phd/qinsizhong/miltonzhou/sft/data_preprocess/filter_data/downloaded_frames")
    parser.add_argument("--result_path", type=str, default="/data/phd/qinsizhong/llm_factory_test/baselines/results/F___vidselect_gemini3f.jsonl")
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.api_key:
        raise ValueError("missing GEMINI_API_KEY")

    client = build_client(args.api_key)
    records = _read_records_auto(args.prompts_path)

    print(f"[info] 加载输入样本 {len(records)} 条。")

    infer_records_with_gemini(
        client=client,
        model_name=args.model,
        records=records,
        frames_dir=args.frames_dir,
        result_path=args.result_path,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )


if __name__ == "__main__":
    main()
