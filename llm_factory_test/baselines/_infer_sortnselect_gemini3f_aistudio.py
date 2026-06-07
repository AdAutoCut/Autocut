#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gemini 多模态离线推理脚本（支持 select / sort 任务）
+ 支持在调用前打印 prompt（debug）
+ 支持只推理前 n 条样本（用于测试）
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


def _print_prompt_debug(
    idx: int,
    model_name: str,
    frame_ids: List[str],
    loaded_frame_ids: List[str],
    new_text: str,
    image_urls: List[str],
    max_text_chars: int = 2000
) -> None:
    """
    打印调用前的 prompt 信息（不打印 base64 全量，避免刷屏）。
    """
    print("\n" + "=" * 88)
    print(f"[DEBUG] idx={idx} model={model_name}")
    print(f"[DEBUG] frame_ids(parsed)  count={len(frame_ids)}: {frame_ids}")
    print(f"[DEBUG] frame_ids(loaded)  count={len(loaded_frame_ids)}: {loaded_frame_ids}")
    print(f"[DEBUG] images_to_send     count={len(image_urls)}")
    print("[DEBUG] user_text (rewritten):")
    if len(new_text) > max_text_chars:
        print(new_text[:max_text_chars] + "\n... (truncated) ...")
    else:
        print(new_text)

    # 只打印每张图片的“短前缀”，用于确认顺序和格式
    print("[DEBUG] image_url prefixes (order matters):")
    for i, url in enumerate(image_urls[:10], 1):
        # url 形如 data:image/jpeg;base64,xxxx
        prefix = url[:60] + "..." if len(url) > 60 else url
        print(f"  - Picture {i}: {prefix}")
    if len(image_urls) > 10:
        print(f"  ... ({len(image_urls) - 10} more images not shown) ...")
    print("=" * 88 + "\n")


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
    print_prompt: bool = False,
    n: int = 0,
):
    """
    对所有 records 逐条调用 Gemini 多模态推理，并将结果写入 result_path(JSONL)。
    - print_prompt: 调用前打印 prompt（debug）
    - n: 只推理前 n 条（n<=0 表示全部）
    """
    os.makedirs(os.path.dirname(result_path), exist_ok=True)

    if n and n > 0:
        records = records[:n]

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
            loaded_frame_ids = []
            for fid in frame_ids:
                img_path = os.path.join(frames_dir, f"{fid}.jpg")
                data_url = _encode_image_to_data_url(img_path)
                if data_url:
                    image_urls.append(data_url)
                    loaded_frame_ids.append(fid)

            if not image_urls:
                continue

            # ===== 打印 prompt（在调用前）=====
            if print_prompt:
                _print_prompt_debug(
                    idx=idx,
                    model_name=model_name,
                    frame_ids=frame_ids,
                    loaded_frame_ids=loaded_frame_ids,
                    new_text=new_text,
                    image_urls=image_urls,
                )

            user_content = [{"type": "text", "text": new_text}]
            for url in image_urls:
                user_content.append({"type": "image_url", "image_url": {"url": url}})

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
            out_fp.write(json.dumps(build_output_record(rec, answer), ensure_ascii=False) + "\n")
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
    parser.add_argument("--api_key", type=str, default=os.getenv("GEMINI_API_KEY"))
    parser.add_argument("--model", type=str, default="gemini-3-flash-preview")

    parser.add_argument("--prompts_path", type=str,
                        default="/data/phd/qinsizhong/miltonzhou/sft/data_preprocess/T_baseline_mm_vidselect.json")
    parser.add_argument("--frames_dir", type=str,
                        default="/data/phd/qinsizhong/miltonzhou/sft/data_preprocess/filter_data/downloaded_frames")
    parser.add_argument("--result_path", type=str,
                        default="/data/phd/qinsizhong/llm_factory_test/baselines/results/F___vidselect_gemini3f.jsonl")

    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)

    parser.add_argument("--n", type=int, default=0,
                        help="只推理输入中的前 n 条样本；n<=0 表示全部。")

    parser.add_argument("--print_prompt", action="store_true",
                        help="在每次调用 Gemini 前打印 prompt（文本+图片顺序信息）。")

    return parser.parse_args()


def main():
    args = parse_args()
    if not args.api_key:
        raise ValueError("missing GEMINI_API_KEY")

    client = build_client(args.api_key)
    records = _read_records_auto(args.prompts_path)

    print(f"[info] 加载输入样本 {len(records)} 条。")
    if args.n and args.n > 0:
        print(f"[info] 测试模式：只推理前 {args.n} 条。")

    infer_records_with_gemini(
        client=client,
        model_name=args.model,
        records=records,
        frames_dir=args.frames_dir,
        result_path=args.result_path,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        print_prompt=args.print_prompt,
        n=args.n,
    )


if __name__ == "__main__":
    main()
