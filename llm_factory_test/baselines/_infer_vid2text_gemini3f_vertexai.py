#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Gemini (Vertex AI) 多模态离线推理脚本（vid2text 台词生成）

等价替换 GPT-4o 版本的逻辑：
- 从 human 文本中解析 [frame_id]
- 删除 [frame_id] 后作为 prompt 文本
- 按 frame_id 出现顺序加载图片
- 单次请求：文本 + 多张图片
- metadata 透传写回（不进 prompt）
- 输出 JSONL：model_generate 写 Gemini 结果

新增功能：
- --n：只推理前 n 条（n<=0 表示全部）
- --print_prompt：调用前打印 prompt 组成（文本 + 图片顺序检查）

"""

import argparse
import json
import os
import re
from typing import Dict, List

import pandas as pd
from tqdm import tqdm
from google import genai
from google.genai import types


# ---------------------------
# Helpers（与 GPT-4o 版本保持一致）
# ---------------------------

def read_records_auto(path: str) -> List[Dict]:
    """读取 JSON 或 JSONL 为 list[dict]"""
    try:
        return pd.read_json(path, lines=False).to_dict("records")
    except ValueError:
        return pd.read_json(path, lines=True).to_dict("records")


def extract_human_value(rec: Dict) -> str:
    """取第一条 human/user 文本；若没有则退回第 0 条"""
    conv = rec.get("conversations", [])
    if not conv:
        return ""
    for m in conv:
        if m.get("from") in ("human", "user"):
            return m.get("value", "") or ""
    return conv[0].get("value", "") or ""


def extract_frame_ids(text: str) -> List[str]:
    """提取所有 [frame_id]（顺序即出现顺序）"""
    return re.findall(r"\[(\d+)\]", text)


def remove_frame_ids_from_text(text: str) -> str:
    """删除 [frame_id]，其余文本保持不变"""
    return re.sub(r"\[(\d+)\]", "", text)


def load_image_bytes(img_path: str) -> bytes:
    """读取本地图片为 bytes（失败返回 b''）"""
    if not os.path.exists(img_path):
        print(f"[warn] image not found: {img_path}")
        return b""
    with open(img_path, "rb") as f:
        return f.read()


def build_output_record(src: Dict, answer: str) -> Dict:
    """输出对象结构：与 GPT-4o 版保持一致"""
    out: Dict = {}
    out["sample_id"] = src.get("sample_id", None)
    out["task"] = src.get("task", None)
    out["ad_key"] = src.get("ad_key", None)
    out["conversations"] = src.get("conversations", [])
    out["system"] = src.get("system", "")
    out["tools"] = src.get("tools", "")
    out["model_generate"] = answer
    return out


def print_prompt_debug(
    idx: int,
    model: str,
    frame_ids: List[str],
    loaded_ids: List[str],
    prompt_text: str,
    max_chars: int = 2000,
) -> None:
    """打印调用前的 prompt 信息（不打印图片 bytes）"""
    print("\n" + "=" * 90)
    print(f"[DEBUG] idx={idx} model={model}")
    print(f"[DEBUG] frame_ids(parsed) count={len(frame_ids)}: {frame_ids}")
    print(f"[DEBUG] frame_ids(loaded) count={len(loaded_ids)}: {loaded_ids}")
    print(f"[DEBUG] images_to_send count={len(loaded_ids)}")
    print("[DEBUG] prompt_text (frame_ids removed):")
    if len(prompt_text) > max_chars:
        print(prompt_text[:max_chars] + "\n... (truncated) ...")
    else:
        print(prompt_text)
    print("=" * 90 + "\n")


# ---------------------------
# Core inference（Vertex AI Gemini）
# ---------------------------

def infer_vid2text_with_gemini_vertex(
    client: genai.Client,
    model: str,
    records: List[Dict],
    frames_dir: str,
    result_path: str,
    n: int = 0,
    print_prompt: bool = False,
):
    """逐条推理 vid2text（逻辑等价 GPT-4o）"""
    if n and n > 0:
        records = records[:n]

    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    written = 0
    total = len(records)

    with open(result_path, "a", encoding="utf-8") as out_fp:
        for idx, rec in enumerate(tqdm(records, desc="Gemini Vertex vid2text", unit="sample")):
            raw_text = extract_human_value(rec)
            if not raw_text:
                print(f"[warn] idx={idx}: 无 human 文本，跳过")
                continue

            frame_ids = extract_frame_ids(raw_text)
            if not frame_ids:
                print(f"[warn] idx={idx}: 无 frame_id，跳过")
                continue

            prompt_text = remove_frame_ids_from_text(raw_text)

            image_parts = []
            loaded_ids = []
            for fid in frame_ids:
                img_path = os.path.join(frames_dir, f"{fid}.jpg")
                img_bytes = load_image_bytes(img_path)
                if img_bytes:
                    image_parts.append(
                        types.Part.from_bytes(
                            data=img_bytes,
                            mime_type="image/jpeg"
                        )
                    )
                    loaded_ids.append(fid)
                else:
                    print(f"[warn] idx={idx}: 图片缺失 {fid}")

            if not image_parts:
                print(f"[warn] idx={idx}: 无有效图片，跳过")
                continue

            # ===== debug 打印（在调用前）=====
            if print_prompt:
                print_prompt_debug(
                    idx=idx,
                    model=model,
                    frame_ids=frame_ids,
                    loaded_ids=loaded_ids,
                    prompt_text=prompt_text,
                )

            # === 关键：等价于 GPT-4o 的 user_content ===
            contents = [prompt_text] + image_parts

            try:
                response = client.models.generate_content(
                    model=model,
                    contents=contents,
                )
            except Exception as e:
                print(f"[error] idx={idx}: Gemini 调用失败: {e}")
                continue

            answer = (response.text or "").strip()
            out_obj = build_output_record(rec, answer)

            out_fp.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
            out_fp.flush()
            written += 1

    print(f"[info] 推理完成：写入 {written}/{total} 条样本 → {result_path}")


# ---------------------------
# CLI
# ---------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Vertex AI Gemini 多模态推理：vid2text（逻辑等价 GPT-4o 版本）"
    )
    parser.add_argument("--project_id", default="")
    parser.add_argument("--location", default="global")
    parser.add_argument("--model", default="gemini-3-flash-preview")

    parser.add_argument("--prompts_path",
                        default="/data/phd/qinsizhong/miltonzhou/sft/data_preprocess/T_baseline_mm_vid2text.json")
    parser.add_argument("--frames_dir",
                        default="/data/phd/qinsizhong/miltonzhou/sft/data_preprocess/filter_data/downloaded_frames")
    parser.add_argument("--result_path",
                        default="/data/phd/qinsizhong/llm_factory_test/baselines/results/F___vid2text_gemini3f.jsonl")

    parser.add_argument("--n", type=int, default=0,
                        help="只推理前 n 条；n<=0 表示全部")
    parser.add_argument("--print_prompt", action="store_true",
                        help="在每次调用前打印 prompt（文本 + 图片顺序检查）")

    return parser.parse_args()


def main():
    args = parse_args()

    client = genai.Client(
        vertexai=True,
        project=args.project_id,
        location=args.location,
    )

    records = read_records_auto(args.prompts_path)
    print(f"[info] 加载输入样本 {len(records)} 条")
    if args.n and args.n > 0:
        print(f"[info] 测试模式：只推理前 {args.n} 条")
    if args.print_prompt:
        print("[info] debug 模式：每次调用前打印 prompt 组成")

    infer_vid2text_with_gemini_vertex(
        client=client,
        model=args.model,
        records=records,
        frames_dir=args.frames_dir,
        result_path=args.result_path,
        n=args.n,
        print_prompt=args.print_prompt,
    )


if __name__ == "__main__":
    main()