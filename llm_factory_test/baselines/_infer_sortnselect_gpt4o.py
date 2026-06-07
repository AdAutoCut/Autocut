#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Azure GPT-4o 多模态离线推理脚本（支持 select / sort 任务）

改动要点：
- 保留推理逻辑；
- 不将 meta 信息（sample_id, task, ad_key）放入 prompt；
- 输出时透传 meta 信息（放在 JSON 前部）；
- 输出结构保持统一：
    {
      "sample_id": ...,
      "task": "...",
      "ad_key": {...},
      "conversations": [...],
      "system": "",
      "tools": "",
      "model_generate": "..."
    }
"""

import argparse
import json
import os
import re
import base64
from typing import Dict, List, Tuple

import pandas as pd
from tqdm import tqdm
from openai import AzureOpenAI


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
    例如 "1)[1626601758120002]" -> "1)Picture 1"
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


def build_client(endpoint: str, api_key: str, api_version: str) -> AzureOpenAI:
    """初始化 AzureOpenAI 客户端。"""
    return AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version=api_version,
    )


def build_output_record(src: Dict, answer: str) -> Dict:
    """
    构造输出记录：meta + 原始 conversations + 模型生成结果。
    """
    out = {}
    out["sample_id"] = src.get("sample_id", None)
    out["task"] = src.get("task", None)
    out["ad_key"] = src.get("ad_key", None)
    out["conversations"] = src.get("conversations", [])
    out["system"] = src.get("system", "")
    out["tools"] = src.get("tools", "")
    out["model_generate"] = answer
    return out


# ---------------------------
# Core inference
# ---------------------------

def infer_records_with_gpt4o(
    client: AzureOpenAI,
    deployment: str,
    records: List[Dict],
    frames_dir: str,
    result_path: str,
    max_tokens: int = 256,
    temperature: float = 0.0,
):
    """
    对所有 records 逐条调用 GPT-4o，多模态推理，并将结果写入 result_path(JSONL)。
    metadata（sample_id/task/ad_key）透传，不影响 prompt。
    """
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    total = len(records)
    written = 0

    with open(result_path, "a", encoding="utf-8") as out_fp:
        for idx, rec in enumerate(tqdm(records, desc="GPT-4o inference", unit="sample")):
            raw_text = _extract_human_value(rec)
            if not raw_text:
                print(f"[warn] idx={idx}: 没有 human 文本，跳过。")
                continue

            frame_ids, new_text = _extract_frames_and_rewrite(raw_text)
            if not frame_ids:
                print(f"[warn] idx={idx}: 未解析到任何 [frame_id]，跳过。")
                continue

            # 加载图片（与 Picture k 对应顺序）
            image_urls = []
            for fid in frame_ids:
                img_path = os.path.join(frames_dir, f"{fid}.jpg")
                data_url = _encode_image_to_data_url(img_path)
                if data_url:
                    image_urls.append(data_url)
                else:
                    print(f"[warn] idx={idx}: skip missing image {fid}")

            if not image_urls:
                print(f"[warn] idx={idx}: 所有关联图片加载失败，跳过该样本。")
                continue

            # 构造 user 消息（仅文本+图片）
            user_content = [{"type": "text", "text": new_text}]
            for url in image_urls:
                user_content.append({"type": "image_url", "image_url": {"url": url}})

            user_message = {"role": "user", "content": user_content}

            # 调用 GPT-4o
            try:
                completion = client.chat.completions.create(
                    model=deployment,
                    messages=[user_message],
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
            except Exception as e:
                print(f"[error] idx={idx}: API 调用失败: {e}")
                continue

            if not completion.choices:
                print(f"[warn] idx={idx}: 无返回 choices，跳过。")
                continue

            answer = (completion.choices[0].message.content or "").strip()
            out_obj = build_output_record(rec, answer)

            out_fp.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
            out_fp.flush()
            written += 1

    print(f"[info] 推理完成，共写入 {written}/{total} 条样本结果到 {result_path}。")


# ---------------------------
# CLI
# ---------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Azure GPT-4o 多模态推理（select/sort 任务）——透传 metadata，不进入 prompt。"
    )
    parser.add_argument("--endpoint", type=str, default=os.getenv("ENDPOINT_URL", ""))
    parser.add_argument("--deployment", type=str, default=os.getenv("DEPLOYMENT_NAME", ""))
    parser.add_argument("--api_key", type=str, default=os.getenv("AZURE_OPENAI_API_KEY", ""))
    parser.add_argument("--api_version", type=str, default="2025-01-01-preview")
    parser.add_argument("--prompts_path", type=str, default="/data/phd/miltonzhou/sft/data_preprocess/T2_gpt_vidsort.json",
                        help="输入 JSON/JSONL，每条记录含 conversations，human 文本中含 1)[frame_id] ...")
    parser.add_argument("--frames_dir", type=str, default="/data/phd/miltonzhou/sft/data_preprocess/filter_data/downloaded_frames", ##!!!!!
                        help="帧图目录，文件名为 {frame_id}.jpg。")
    parser.add_argument("--result_path", type=str, default="/data/phd/qinsizhong/llm_factory_test/baselines/results/F2___vidsort_gpt4o.jsonl",
                        help="输出 JSONL 路径，包含 model_generate。")
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.endpoint or not args.deployment or not args.api_key:
        raise ValueError("请提供 --endpoint, --deployment, --api_key 或通过环境变量配置。")

    client = build_client(args.endpoint, args.api_key, args.api_version)

    records = _read_records_auto(args.prompts_path)
    print(f"[info] 加载输入样本 {len(records)} 条。")

    infer_records_with_gpt4o(
        client=client,
        deployment=args.deployment,
        records=records,
        frames_dir=args.frames_dir,
        result_path=args.result_path,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )


if __name__ == "__main__":
    main()
