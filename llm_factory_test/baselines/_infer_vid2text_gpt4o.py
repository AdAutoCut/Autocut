#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Azure GPT-4o 多模态离线推理脚本（vid2text 台词生成，无 system prompt）

改动要点：
- 透传并写回 metadata: sample_id / task / ad_key（置于输出对象最前，方便与 manifest 对齐检索）；
- metadata 不注入 prompt，不影响模型推理；
- 输出为精简对象：{ sample_id, task, ad_key, conversations, system, tools, model_generate }。
"""

import argparse
import json
import os
import re
import base64
from typing import Dict, List

import pandas as pd
from tqdm import tqdm
from openai import AzureOpenAI


# ---------------------------
# Helpers
# ---------------------------

def read_records_auto(path: str) -> List[Dict]:
    """读取 JSON 或 JSONL 为 list[dict]。"""
    try:
        return pd.read_json(path, lines=False).to_dict("records")
    except ValueError:
        return pd.read_json(path, lines=True).to_dict("records")


def extract_human_value(rec: Dict) -> str:
    """取第一条 human/user 的文本；若没有则退回第 0 条。"""
    conv = rec.get("conversations", [])
    if not conv:
        return ""
    for m in conv:
        if m.get("from") in ("human", "user"):
            return m.get("value", "") or ""
    return conv[0].get("value", "") or ""


def extract_frame_ids(text: str) -> List[str]:
    """提取所有 [frame_id]（中括号内数字）。"""
    return re.findall(r"\[(\d+)\]", text)


def remove_frame_ids_from_text(text: str) -> str:
    """删除 [frame_id]，保留其余内容与编号结构。"""
    return re.sub(r"\[(\d+)\]", "", text)


def encode_image_to_data_url(img_path: str) -> str:
    """本地图片 -> data:image/jpeg;base64,...（失败返回空串）。"""
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
    """初始化 Azure OpenAI 客户端。"""
    return AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version=api_version,
    )


def build_output_record(src: Dict, answer: str) -> Dict:
    """
    生成“精简但可溯源”的输出对象（meta 放在最前）：
    {
      "sample_id": ...,
      "task": "...",
      "ad_key": {"chunk_id": ..., "ad_id": ...},
      "conversations": [...],
      "system": "",
      "tools": "",
      "model_generate": "..."
    }
    """
    out: Dict = {}
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

def infer_vid2text_with_gpt4o(
    client: AzureOpenAI,
    deployment: str,
    records: List[Dict],
    frames_dir: str,
    result_path: str,
    max_tokens: int = 512,
    temperature: float = 0.7,
):
    """逐条推理 vid2text 任务，metadata 透传写回，不进入 prompt。"""
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    written = 0

    with open(result_path, "a", encoding="utf-8") as out_fp:
        for idx, rec in enumerate(tqdm(records, desc="GPT-4o vid2text", unit="sample")):
            raw_text = extract_human_value(rec)
            if not raw_text:
                print(f"[warn] idx={idx}: 无 human 文本，跳过。")
                continue

            frame_ids = extract_frame_ids(raw_text)
            if not frame_ids:
                print(f"[warn] idx={idx}: 无 frame_id，跳过。")
                continue

            # 保留原 prompt，仅删掉 [frame_id]
            prompt_text = remove_frame_ids_from_text(raw_text)

            # 加载帧图片
            image_urls = []
            for fid in frame_ids:
                img_path = os.path.join(frames_dir, f"{fid}.jpg")
                data_url = encode_image_to_data_url(img_path)
                if data_url:
                    image_urls.append(data_url)
                else:
                    print(f"[warn] idx={idx}: 图片缺失 {fid}")

            if not image_urls:
                print(f"[warn] idx={idx}: 无有效图片，跳过。")
                continue

            # 构造 user 消息（无 system；meta 不进 prompt）
            user_content = [{"type": "text", "text": prompt_text}]
            # 图片顺序与原 [frame_id] 出现顺序一致
            for url in image_urls:
                user_content.append({"type": "image_url", "image_url": {"url": url}})

            user_message = {"role": "user", "content": user_content}

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
                print(f"[warn] idx={idx}: 无返回结果，跳过。")
                continue

            answer = (completion.choices[0].message.content or "").strip()
            out_obj = build_output_record(rec, answer)

            out_fp.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
            out_fp.flush()
            written += 1

    print(f"[info] 推理完成：共写入 {written}/{len(records)} 条样本。")


# ---------------------------
# CLI
# ---------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Azure GPT-4o 多模态推理：vid2text（无 system prompt，最小改动，metadata 透传回写）。"
    )
    parser.add_argument("--endpoint", type=str, default=os.getenv("ENDPOINT_URL", ""))
    parser.add_argument("--deployment", type=str, default=os.getenv("DEPLOYMENT_NAME", ""))
    parser.add_argument("--api_key", type=str, default=os.getenv("AZURE_OPENAI_API_KEY", ""))
    parser.add_argument("--api_version", type=str, default="2025-01-01-preview")
    parser.add_argument("--prompts_path", type=str, default="/data/phd/miltonzhou/sft/data_preprocess/T_baseline_mm_vid2text.json")
    parser.add_argument("--frames_dir", type=str, default="/data/phd/miltonzhou/sft/data_preprocess/filter_data/downloaded_frames")
    parser.add_argument("--result_path", type=str, default="/data/phd/qinsizhong/llm_factory_test/baselines/results/F___vid2text_gpt4o.jsonl")
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.endpoint or not args.deployment or not args.api_key:
        raise ValueError("必须提供 --endpoint, --deployment, --api_key。")

    client = build_client(args.endpoint, args.api_key, args.api_version)
    records = read_records_auto(args.prompts_path)
    print(f"[info] 加载输入样本 {len(records)} 条。")

    infer_vid2text_with_gpt4o(
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
