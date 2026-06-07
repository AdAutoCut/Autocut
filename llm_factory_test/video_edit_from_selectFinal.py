#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Process vid_sort inference results (no label) -> stitch videos.

输入:
  --sort_result     : sort 推理结果 JSONL（每行一条，无 label）
  --sort_manifest   : 对应的 sort-manifest（JSON 数组；由 A 步生成）
  --model_name      : 模型名标识（如 atc / gpt4o），仅用于输出目录组织
  --chunk           : chunk 编号（整数或字符串），仅用于输出目录组织
  --out_dir         : 基础输出目录（默认 user_study_vids）

行为:
  - 按 sample_id 对齐 sort-manifest
  - 真实台词来自 sort human 的“【广告台词如下】”块
  - 视频顺序来自 model_generate 的编号（对应 sort-manifest 的 local_id）
  - 文本按真实台词、视频按模型顺序，一一配对；长度不等取 min 并打印告警
  - 输出两版视频：*_model_frame.mp4 与 *_model_clip.mp4
  - 若同名文件已存在，则跳过该条（避免覆盖）
"""

import os
import json
import re
import argparse
from typing import Dict, List, Tuple

# 外部工具（你项目里已有）
from tools.postprocess import generate_video
from tools.parse import parse_multimodal
from tools.video import init_video_model
from tools.audio import init_audio_model

# ------------------------
# 小工具
# ------------------------
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def parse_indices(s: str) -> List[int]:
    s = (s or "").replace("，", ",")
    out: List[int] = []
    for p in s.split(","):
        p = p.strip()
        if not p:
            continue
        try:
            out.append(int(p))
        except:
            pass
    return out

def load_sort_manifest_by_sid(path: str) -> Dict[int, dict]:
    with open(path, "r", encoding="utf-8") as f:
        root = json.loads(f.read().lstrip("\ufeff"))
    if not isinstance(root, list):
        raise TypeError("sort_manifest 根应为 JSON 数组。")

    sid_index: Dict[int, dict] = {}
    for obj in root:
        if not isinstance(obj, dict):
            continue
        sid = obj.get("sample_id")
        if sid is None:
            continue

        by_local: Dict[int, dict] = {}
        for c in obj.get("clips", []) or []:
            if isinstance(c, dict) and isinstance(c.get("local_id"), int):
                by_local[c["local_id"]] = c

        sid_index[sid] = {
            "clips_by_local": by_local,
            "meta": obj.get("meta", {}) or {},
            "ad_key": obj.get("ad_key", {}) or {}
        }
    return sid_index

def extract_script_lines_from_human(human_value: str) -> List[str]:
    """
    从 human 的“【广告台词如下】”块抽取台词行。
    兼容 atc / gpt4o 的 human；后续“【视频片段…】/【片段的代表帧…】/【现在请输出…】”等都会被截断。
    """
    if not isinstance(human_value, str):
        return []
    start = human_value.find("【广告台词如下】")
    if start == -1:
        return []

    rem = human_value[start + len("【广告台词如下】") :]
    cut_patterns = [
        "【视频片段", "【片段的代表帧", "【片段", "【视频素材", "【视频片段素材", "【现在请输出"
    ]
    cut = len(rem)
    for pat in cut_patterns:
        p = rem.find(pat)
        if p != -1:
            cut = min(cut, p)
    block = rem[:cut].strip("\n")

    lines: List[str] = []
    for line in block.splitlines():
        t = line.strip()
        if t:
            lines.append(t.rstrip("，,"))
    return lines

def dedup_and_filter(ids: List[int], clips_by_local: Dict[int, dict]) -> List[int]:
    seen = set()
    out: List[int] = []
    for x in ids:
        if x in seen:
            continue
        if x not in clips_by_local:
            continue
        seen.add(x)
        out.append(x)
    return out

def ensure_vtok_wrapped(vtok: str) -> str:
    vtok = vtok or ""
    if ("<|video_start|>" in vtok) and ("<|video_end|>" in vtok):
        return vtok
    return f"<|video_start|>{vtok}<|video_end|>"

def make_clip_segment(text: str, vtok: str) -> str:
    vtok_block = ensure_vtok_wrapped(vtok or "")
    return f"<|clip_start|><|text_start|>{text or ''}<|text_end|>{vtok_block}<|clip_end|>"

def build_ad_text_mixed_by_lists(texts: List[str], vtoks: List[str]) -> Tuple[str, dict]:
    n_text, n_vtok = len(texts), len(vtoks)
    n = min(n_text, n_vtok)
    truncated = (n_text != n_vtok)

    segments = [make_clip_segment(texts[i], vtoks[i]) for i in range(n)]
    ad_string = f"<|ad_start|>{''.join(segments)}<|ad_end|>"

    stats = {
        "len_texts": n_text,
        "len_vtoks": n_vtok,
        "paired": n,
        "truncated": truncated
    }
    return ad_string, stats

# ------------------------
# 主流程
# ------------------------
def main():
    ap = argparse.ArgumentParser(description="后处理 vid_sort 结果（无 label）→ 拼接视频")
    ap.add_argument("--pred_dir", type=str, default="/data/phd/qinsizhong/llm_factory_test/model_pred_chunks2",
                    help="预测文件所在目录，内部命名为 MR_{model_name}_{chunk}.jsonl")
    # 可选：保留 --sort_result 但默认 None，用于手动指定单文件时的兜底
    ap.add_argument("--sort_result", type=str, default=None,
                    help="(可选) 直接指定某个结果文件；若提供则优先生效")

    ap.add_argument("--sort_manifest", type=str, required=True,
                    help="对应的 sort-manifest（JSON 数组）")
    ap.add_argument("--model_name", type=str, default="atc",
                    help="模型名标识（用于输出目录组织，如 atc / gpt4o）")
    ap.add_argument("--chunk", type=str, default="0",
                    help="chunk 编号（用于输出目录组织）")
    ap.add_argument("--out_dir", type=str, default="user_study_vids2",
                    help="基础输出目录")
    ap.add_argument("--max_samples", type=int, default=None,
                    help="最多处理多少条（调试用）")
    args = ap.parse_args()

    # 计算预测文件路径
    if args.sort_result:
        pred_path = args.sort_result
    else:
        fname = f"MR_{args.model_name}_{args.chunk}.jsonl"
        pred_path = os.path.join(args.pred_dir, fname)
    if not os.path.exists(pred_path):
        raise FileNotFoundError(f"[error] 预测文件不存在：{pred_path}")
    print(f"[info] 使用预测文件：{pred_path}")

    video_rqvae_model, video_index, frame_index, clip_index = init_video_model(load_faiss=True, model_name="video_8_256_0729")
    init_audio_model(load_faiss=True, model_name="audio_8_256_0729")
    
    # 输出目录组织
    out_base = os.path.join(args.out_dir, args.model_name, f"chunk_{args.chunk}")
    ensure_dir(out_base)

    sid_index = load_sort_manifest_by_sid(args.sort_manifest)
    print(f"[info] sort-manifest 样本数：{len(sid_index)}")
    print(f"[info] 输出目录：{out_base}")

    total, ok, skip, exist = 0, 0, 0, 0

    with open(pred_path, "r", encoding="utf-8") as fin:
        for line in fin:
            if args.max_samples is not None and ok >= args.max_samples:
                break
            s = line.strip()
            if not s:
                continue
            total += 1

            try:
                mr = json.loads(s)
            except Exception as e:
                print(f"[warn] JSON 解析失败，跳过一行：{e}")
                skip += 1
                continue

            if mr.get("task") != "vid_sort":
                skip += 1
                continue

            sample_id = mr.get("sample_id")
            if sample_id is None or sample_id not in sid_index:
                print(f"[warn] 跳过：sample_id 缺失或不在 sort-manifest 中（sample_id={sample_id}）")
                skip += 1
                continue

            payload = sid_index[sample_id]
            clips_by_local = payload["clips_by_local"]

            # 真实台词（从 human 抽）
            conversations = mr.get("conversations") or []
            human_value = conversations[0].get("value", "") if (conversations and conversations[0].get("from")=="human") else ""
            script_lines = extract_script_lines_from_human(human_value)
            if not script_lines:
                print(f"[warn] sample_id={sample_id}: 未解析到台词行，跳过。")
                skip += 1
                continue

            # 模型顺序（local_id）
            pred_ids = dedup_and_filter(parse_indices(mr.get("model_generate", "")), clips_by_local)
            if not pred_ids:
                print(f"[warn] sample_id={sample_id}: model_generate 无有效编号，跳过。")
                skip += 1
                continue

            # 取 v_tok（按模型顺序）
            vtoks: List[str] = []
            for lid in pred_ids:
                clip = clips_by_local.get(lid)
                v = (clip.get("orig") or {}).get("v_tok") if clip else None
                if v:
                    vtoks.append(v)
            if not vtoks:
                print(f"[warn] sample_id={sample_id}: 未取到任何 v_tok，跳过。")
                skip += 1
                continue

            # 文本按真实台词顺序、视频按模型顺序
            ad_str, stats = build_ad_text_mixed_by_lists(script_lines, vtoks)
            if stats.get("truncated"):
                print(f"[note] sample_id={sample_id}: 台词({stats['len_texts']})与视频({stats['len_vtoks']})长度不等，配对数={stats['paired']}。")

            # 解析多模态 → 拼视频
            ans, _ = parse_multimodal(ad_str)
            clips_model = ans['clips']

            # 输出文件
            out_prefix = os.path.join(out_base, f"{args.model_name}_c{args.chunk}_sid{sample_id}")
            out_frame = f"{out_prefix}_model_frame.mp4"
            # out_clip  = f"{out_prefix}_model_clip.mp4"

            # 避免覆盖
            if os.path.exists(out_frame):
                print(f"[skip] 已存在输出文件（sid={sample_id}），跳过：{os.path.basename(out_prefix)}_*")
                exist += 1
                continue

            # 生成
            generate_video(clips_model, extract_method="frame", out_path=out_frame)
            # generate_video(clips_model, extract_method="clip",  out_path=out_clip)

            ok += 1
            print(f"[ok] sid={sample_id} → {os.path.basename(out_frame)}")

    print(f"\n[done] 读取 {total} 行；成功 {ok}；跳过 {skip}；已有文件跳过 {exist}。")
    print(f"[save] 输出目录：{out_base}")

if __name__ == "__main__":
    main()

