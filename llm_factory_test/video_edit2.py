#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from tools.postprocess import generate_video
from tools.parse import parse_multimodal
from tools.video import init_video_model
from tools.audio import init_audio_model

import json
import argparse
import os
from typing import Dict, List, Tuple, Any

# ------------------------
# 基础解析
# ------------------------
def parse_indices(s: str) -> List[int]:
    s = (s or "").replace("，", ",")
    out = []
    for p in s.split(","):
        p = p.strip()
        if not p:
            continue
        try:
            out.append(int(p))
        except:
            pass
    return out

def load_manifest_by_sample_id(manifest_path: str) -> Dict[int, dict]:
    with open(manifest_path, "r", encoding="utf-8") as f:
        root = json.loads(f.read().lstrip("\ufeff"))
    if not isinstance(root, list):
        raise TypeError("manifest 根应为 JSON 数组（list）。")

    sid_index: Dict[int, dict] = {}
    for obj in root:
        if not isinstance(obj, dict):
            continue
        sid = obj.get("sample_id")
        if sid is None:
            continue

        # clips_by_local
        by_local: Dict[int, dict] = {}
        clips = obj.get("clips", [])
        if isinstance(clips, list):
            for c in clips:
                if not isinstance(c, dict):
                    continue
                lid = c.get("local_id")
                if isinstance(lid, int):
                    by_local[lid] = c

        # correct_order
        label = (obj.get("label") or {}).get("vid_sort") or {}
        co_raw = label.get("correct_order") or []
        correct_order: List[int] = []
        for x in co_raw:
            try:
                correct_order.append(int(x))
            except:
                pass

        if by_local:
            sid_index[sid] = {
                "clips_by_local": by_local,
                "correct_order": correct_order,
                "meta": obj.get("meta", {})
            }
    return sid_index

# ------------------------
# 片段/广告拼接（可复用）
# ------------------------
def make_clip_segment(text: str, vtok: str) -> str:
    text = text or ""
    vtok = vtok or ""
    if "<|video_start|>" in vtok and "<|video_end|>" in vtok:
        video_block = vtok
    else:
        video_block = f"<|video_start|>{vtok}<|video_end|>"
    return f"<|clip_start|><|text_start|>{text}<|text_end|>{video_block}<|clip_end|>"

def build_text_list(clips_by_local: Dict[int, dict], text_ids: List[int]) -> Tuple[List[str], List[int]]:
    texts, missing = [], []
    for lid in text_ids:
        clip = clips_by_local.get(lid)
        if clip is None:
            missing.append(lid)
            continue
        texts.append((clip.get("orig") or {}).get("text") or "")
    return texts, missing

def build_vtok_list(clips_by_local: Dict[int, dict], vtok_ids: List[int]) -> Tuple[List[str], List[int]]:
    vtoks, missing = [], []
    for lid in vtok_ids:
        clip = clips_by_local.get(lid)
        if clip is None:
            missing.append(lid)
            continue
        vtoks.append((clip.get("orig") or {}).get("v_tok") or "")
    return vtoks, missing

def build_ad_text_mixed(clips_by_local: Dict[int, dict], text_ids: List[int], vtok_ids: List[int]) -> Tuple[str, dict]:
    """
    文本按 text_ids 的顺序；视频按 vtok_ids 的顺序。
    最终把第 i 个 text 与第 i 个 vtok 配对；若长度不同，按较短的长度对齐并告警。
    返回 (ad_string, stats)
    """
    texts, miss_text = build_text_list(clips_by_local, text_ids)
    vtoks, miss_vtok = build_vtok_list(clips_by_local, vtok_ids)

    n_text, n_vtok = len(texts), len(vtoks)
    n = min(n_text, n_vtok)
    truncated = (n_text != n_vtok)

    segments = [make_clip_segment(texts[i], vtoks[i]) for i in range(n)]
    ad_string = f"<|ad_start|>{''.join(segments)}<|ad_end|>"

    stats = {
        "missing_text_ids": miss_text,
        "missing_vtok_ids": miss_vtok,
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", type=str, default="./model_pred_chunks",
                    help="预测文件所在目录，内部按 MR_{model_name}_{chunk}.jsonl 命名")
    ap.add_argument("--model_name", type=str, choices=["atc", "gpt4o"], required=True,
                    help="选择要处理的模型结果：atc 或 gpt4o")
    ap.add_argument("--chunk", type=int, required=True,
                    help="要处理的 chunk 号（整数）")
    ap.add_argument("--manifest", type=str,
                    default="/data/phd/miltonzhou/sft/data_preprocess/T_manifest_vidsort.json",
                    help="manifest JSON（数组）路径")
    ap.add_argument("--out_dir", type=str, default="./user_study_vids",
                    help="导出视频保存目录，将自动创建")
    args = ap.parse_args()

    # 解析输入/输出路径
    in_path = os.path.join(args.pred_dir, f"MR_{args.model_name}_{args.chunk}.jsonl")
    if not os.path.exists(in_path):
        raise FileNotFoundError(f"找不到结果文件：{in_path}")
    os.makedirs(args.out_dir, exist_ok=True)

    # 载入 manifest 索引（按 sample_id）
    sid_index = load_manifest_by_sample_id(args.manifest)

    # 初始化模型（与你原来一致）
    video_rqvae_model, video_index, frame_index, clip_index = init_video_model(
        load_faiss=True, model_name="video_8_256_0729"
    )
    init_audio_model(load_faiss=True, model_name="audio_8_256_0729")

    print(f"[info] manifest（按 sample_id）条目：{len(sid_index)}")
    print(f"[info] 正在处理：model={args.model_name} | chunk={args.chunk} | file={in_path}")
    print(f"[info] 输出目录：{args.out_dir}")

    total, ok, skip = 0, 0, 0
    with open(in_path, "r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            total += 1
            mr = json.loads(line)

            sample_id = mr.get("sample_id")
            if sample_id is None or sample_id not in sid_index:
                skip += 1
                print(f"[warn] 跳过：sample_id 缺失或 manifest 中不存在（sample_id={sample_id}）")
                continue

            payload = sid_index[sample_id]
            clips_by_local = payload["clips_by_local"]
            correct_order = payload.get("correct_order", [])
            pred_ids = parse_indices(mr.get("model_generate", ""))

            # 组装文本（label 顺序）与画面（该模型预测顺序）
            ad_model, st_model = build_ad_text_mixed(
                clips_by_local, text_ids=correct_order, vtok_ids=pred_ids
            )

            # 解析多模态串成结构
            ans_model, _ = parse_multimodal(ad_model)
            clips_model = ans_model['clips']

            # 输出命名：包含模型、chunk、sample，全部写入 user_study_vids/
            base = f"{args.model_name}_sid{sample_id}"
            out_frame = os.path.join(args.out_dir, f"{base}_frame.mp4")
            # out_clip  = os.path.join(args.out_dir, f"{base}_clip.mp4")

            # 生成视频
            generate_video(clips_model, extract_method="frame", out_path=out_frame)
            # generate_video(clips_model, extract_method="clip",  out_path=out_clip)

            ok += 1

    print(f"\n[done] 文件 {in_path} | 总计 {total} 条；成功 {ok} 条；跳过 {skip} 条。"
          f"\n[save] 全部视频已保存到：{os.path.abspath(args.out_dir)}")

if __name__ == "__main__":
    main()
