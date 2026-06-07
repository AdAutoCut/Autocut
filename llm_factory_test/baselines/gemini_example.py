#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Gemini (Vertex AI) 多模态离线推理脚本（select / sort）
- 保持原 prompt 组装逻辑不变
- 使用 Vertex AI 官方 Gemini 调用方式
- 支持：
  * 调用前打印 prompt（debug）
  * 只推理前 n 条样本
"""

import argparse
import json
import os
import re
from typing import Dict, List, Tuple

import pandas as pd
from tqdm import tqdm
from google import genai
from google.genai import types


# =========================
# Helpers
# =========================

def _read_records_auto(path: str) -> List[Dict]:
    """自动读取 JSON / JSONL"""
    try:
        return pd.read_json(path, lines=False).to_dict("records")
    except ValueError:
        return pd.read_json(path, lines=True).to_dict("records")


def _extract_human_value(rec: Dict) -> str:
    """从 conversations 中取第一条 human/user 文本"""
    conv = rec.get("conversations", [])
    if not conv:
        return ""
    for m in conv:
        if m.get("from") in ("human", "user"):
            return m.get("value", "") or ""
    return conv[0].get("value", "") or ""


def _extract_frames_and_rewrite(text: str) -> Tuple[List[str], str]:
    """
    从文本中提取 [frame_id]，并替换为 Picture k
    """
    frame_ids: List[str] = []

    def repl(match: re.Match) -> str:
        fid = match.group(1)
        frame_ids.append(fid)
        return f"Picture {len(frame_ids)}"

    new_text = re.sub(r"\[(\d+)\]", repl, text)
    return frame_ids, new_text


def _load_image_bytes(img_path: str) -> bytes:
    """读取本地图片为 bytes"""
    if not os.path.exists(img_path):
        print(f"[warn] image not found: {img_path}")
        return b""
    with open(img_path, "rb") as f:
        return f.read()


def build_output_record(src: Dict, answer: str) -> Dict:
    """构造输出记录（结构保持不变）"""
    return {
        "sample_id": src.get("sample_id", None),
        "task": src.get("task", None),
        "ad_key": src.get("ad_key", None),
        "conversations": src.get("conversations", []),
        "system": src.get("system", ""),
        "tools": src.get("tools", ""),
        "model_generate": answer,
    }


def print_prompt_debug(
    idx: int,
    model: str,
    frame_ids: List[str],
    loaded_ids: List[str],
    text: str,
    max_chars: int = 2000
):
    print("\n" + "=" * 90)
    print(f"[DEBUG] idx={idx} model={model}")
    print(f"[DEBUG] frame_ids(parsed): {frame_ids}")
    print(f"[DEBUG] frame_ids(loaded): {loaded_ids}")
    print("[DEBUG] rewritten text:")
    if len(text) > max_chars:
        print(text[:max_chars] + "\n... (truncated)")
    else:
        print(text)
    print("=" * 90 + "\n")


# =========================
# Core inference
# =========================

def infer_with_gemini_vertex(
    client: genai.Client,
    model: str,
    records: List[Dict],
    frames_dir: str,
    output_path: str,
    max_tokens: int,
    temperature: float,
    print_prompt: bool,
    n: int,
):
    if n > 0:
        records = records[:n]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    written = 0
    total = len(records)

    with open(output_path, "a", encoding="utf-8") as out_f:
        for idx, rec in enumerate(tqdm(records, desc="Gemini Vertex inference")):
            raw_text = _extract_human_value(rec)
            if not raw_text:
                continue

            frame_ids, rewritten_text = _extract_frames_and_rewrite(raw_text)
            if not frame_ids:
                continue

            image_parts = []
            loaded_ids = []

            for fid in frame_ids:
                img_path = os.path.join(frames_dir, f"{fid}.jpg")
                img_bytes = _load_image_bytes(img_path)
                if img_bytes:
                    image_parts.append(
                        types.Part.from_bytes(
                            data=img_bytes,
                            mime_type="image/jpeg"
                        )
                    )
                    loaded_ids.append(fid)

            if not image_parts:
                continue

            if print_prompt:
                print_prompt_debug(
                    idx=idx,
                    model=model,
                    frame_ids=frame_ids,
                    loaded_ids=loaded_ids,
                    text=rewritten_text,
                )

            contents = [rewritten_text] + image_parts

            try:
                response = client.models.generate_content(
                    model=model,
                    contents=contents,
                    # generation_config={
                    #     "max_output_tokens": max_tokens,
                    #     "temperature": temperature,
                    # }
                )
            except Exception as e:
                print(f"[error] idx={idx}: {e}")
                continue

            answer = (response.text or "").strip()
            out_f.write(
                json.dumps(build_output_record(rec, answer), ensure_ascii=False) + "\n"
            )
            out_f.flush()
            written += 1

    print(f"[info] 推理完成：{written}/{total} → {output_path}")


# =========================
# CLI
# =========================

def parse_args():
    ap = argparse.ArgumentParser("Gemini Vertex AI multimodal inference")
    ap.add_argument("--project_id", default="")
    ap.add_argument("--location", default="global")
    ap.add_argument("--model", default="gemini-3-flash-preview")

    ap.add_argument("--prompts_path", default="/data/phd/qinsizhong/miltonzhou/sft/data_preprocess/T2_gemini3f_vidsort.json")
    ap.add_argument("--frames_dir", default="/data/phd/qinsizhong/miltonzhou/sft/data_preprocess/filter_data/downloaded_frames")
    ap.add_argument("--output_path", default="/data/phd/qinsizhong/llm_factory_test/baselines/results/F2___vidsort_gemini3f.jsonl")

    ap.add_argument("--max_tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.0)

    ap.add_argument("--n", type=int, default=0,
                    help="只推理前 n 条，n<=0 表示全部")
    ap.add_argument("--print_prompt", action="store_true")

    return ap.parse_args()


def main():
    args = parse_args()

    client = genai.Client(
        vertexai=True,
        project=args.project_id,
        location=args.location,
    )

    records = _read_records_auto(args.prompts_path)
    print(f"[info] loaded samples: {len(records)}")
    if args.n > 0:
        print(f"[info] test mode: first {args.n} samples")

    infer_with_gemini_vertex(
        client=client,
        model=args.model,
        records=records,
        frames_dir=args.frames_dir,
        output_path=args.output_path,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        print_prompt=args.print_prompt,
        n=args.n,
    )


if __name__ == "__main__":
    main()
