#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Post-process vid_sort results WITHOUT label:
- 文本（台词）从 sort 的 prompt (conversations[0].value) 的【广告台词如下】解析到的行，作为真实顺序；
- 视频（v_tok）按 model_generate 的编号顺序，从 sort-manifest（local_id=1..K）查到对应 v_tok；
- 将第 i 个台词与第 i 个 v_tok 配对（长度不等取较短并告警），拼 <|ad_start|>... 串；
- 调用 parse_multimodal -> generate_video 生成视频。

Inputs:
  --sort_result_jsonl   : 没有 label 的 vid_sort 推理结果 JSONL
  --sort_manifest_json  : 与之对应的 sort-manifest（JSON 数组；clips.local_id=1..K）
  --out_dir             : 输出视频目录（默认 ./user_study_vids）
  --prefix              : 输出文件名前缀（默认 "sortrun"）

Output:
  - {out_dir}/{prefix}_{sample_id}_model_frame.mp4
  - {out_dir}/{prefix}_{sample_id}_model_clip.mp4
  - 同时在控制台打印 ad_string 的统计信息（不落盘）
"""

import os
import re
import json
import argparse
from typing import Dict, List, Tuple, Any

from tools.postprocess import generate_video
from tools.parse import parse_multimodal
from tools.video import init_video_model
from tools.audio import init_audio_model

# ------------------------
# 通用解析
# ------------------------
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
    """
    读取 sort-manifest（JSON 数组），按 sample_id 建索引：
      sid -> {
        "clips_by_local": { local_id -> clip(dict with orig) },
        "meta": dict,
        "ad_key": dict
      }
    """
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
        if by_local:
            sid_index[sid] = {
                "clips_by_local": by_local,
                "meta": obj.get("meta", {}) or {},
                "ad_key": obj.get("ad_key", {}) or {}
            }
    return sid_index

# ------------------------
# 从 sort 的 human 文本解析脚本行
# ------------------------
def extract_script_lines_from_human(human_value: str) -> List[str]:
    """
    取 '【广告台词如下】' 之后到“素材/片段块”之前的行，去空行与中文逗号尾。
    """
    if not isinstance(human_value, str):
        return []
    start = human_value.find("【广告台词如下】")
    if start == -1:
        return []
    rem = human_value[start + len("【广告台词如下】") :]
    cut_patterns = ["【视频片段", "【片段", "【视频素材", "【视频片段素材", "【现在请输出"]
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

# ------------------------
# vtok/clip 拼装
# ------------------------
def ensure_vtok_wrapped(vtok: str) -> str:
    vtok = vtok or ""
    if ("<|video_start|>" in vtok) and ("<|video_end|>" in vtok):
        return vtok
    return f"<|video_start|>{vtok}<|video_end|>"

def make_clip_segment(text: str, vtok_wrapped: str) -> str:
    text = text or ""
    return f"<|clip_start|><|text_start|>{text}<|text_end|>{vtok_wrapped}<|clip_end|>"

def build_ad_from_textlines_and_vtok_ids(clips_by_local: Dict[int, dict],
                                         text_lines: List[str],
                                         vtok_ids: List[int]) -> Tuple[str, dict]:
    """
    文本按 text_lines 的顺序；视频按 vtok_ids 的顺序。
    第 i 个文本与第 i 个 vtok 配对；长度不同取较短并记录 stats。
    """
    vtoks: List[str] = []
    miss_vtok_ids: List[int] = []
    for lid in vtok_ids:
        c = clips_by_local.get(lid)
        if not c:
            miss_vtok_ids.append(lid)
            continue
        v = (c.get("orig") or {}).get("v_tok") or ""
        vtoks.append(ensure_vtok_wrapped(v))

    n_text, n_vtok = len(text_lines), len(vtoks)
    n = min(n_text, n_vtok)
    truncated = (n_text != n_vtok)
    segments = [make_clip_segment(text_lines[i], vtoks[i]) for i in range(n)]
    ad_string = f"<|ad_start|>{''.join(segments)}<|ad_end|>"

    stats = {
        "len_texts": n_text,
        "len_vtoks": n_vtok,
        "paired": n,
        "truncated": truncated,
        "missing_vtok_ids": miss_vtok_ids
    }
    return ad_string, stats

# ------------------------
# 主流程
# ------------------------
def main():
    ap = argparse.ArgumentParser(description="Post-process vid_sort results (no label): text from prompt, vtok by model_generate.")
    ap.add_argument("--sort_result_jsonl", type=str, default="/data/phd/qinsizhong/llm_factory_test/baselines/results/F2___vidsort_atc_embsft.jsonl", help="无 label 的 vid_sort 推理结果 JSONL")
    ap.add_argument("--sort_manifest_json", type=str, default="/data/phd/miltonzhou/sft/data_preprocess/T2atc_manifest_vidsort.json", help="对应的 sort-manifest（JSON 数组；local_id=1..K）")
    ap.add_argument("--out_dir", type=str, default="./C_user_study_vids_atc", help="输出视频目录")
    ap.add_argument("--prefix", type=str, default="C_atc", help="输出文件名前缀")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    sid_index = load_sort_manifest_by_sid(args.sort_manifest_json)
    print(f"[info] sort-manifest 索引条目：{len(sid_index)}")

    # 预初始化（如果 generate_video 里用到了 embedding/FAISS）
    video_rqvae_model, video_index, frame_index, clip_index = init_video_model(
        load_faiss=True, model_name="video_8_256_0729"
    )
    init_audio_model(load_faiss=True, model_name="audio_8_256_0729")

    total, ok, skip = 0, 0, 0
    with open(args.sort_result_jsonl, "r", encoding="utf-8") as fin:
        for line in fin:
            s = line.strip()
            if not s:
                continue
            total += 1
            mr = json.loads(s)

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

            # 1) 从 human 解析“真实顺序”的台词行
            conversations = mr.get("conversations") or []
            human_value = conversations[0].get("value", "") if (conversations and conversations[0].get("from")=="human") else ""
            script_lines = extract_script_lines_from_human(human_value)
            if not script_lines:
                print(f"[warn] sample_id={sample_id}: 未从 prompt 解析到台词行，将用空串占位。")
                # 至少给出与 model_generate 同长度的空文本，避免全丢
                script_lines = [""] * len(parse_indices(mr.get("model_generate", "")))

            # 2) 解析模型预测的编号序列（vtok 顺序）
            pred_ids_raw = parse_indices(mr.get("model_generate", ""))
            # 去重与越界过滤（保持顺序）
            seen, pred_ids = set(), []
            for x in pred_ids_raw:
                if x in seen:
                    continue
                if x not in clips_by_local:
                    continue
                seen.add(x)
                pred_ids.append(x)

            if not pred_ids:
                print(f"[warn] 跳过：model_generate 无有效编号（sample_id={sample_id}）")
                skip += 1
                continue

            # 3) 组装 ad 串（文本=脚本行顺序；视频=预测顺序）
            ad_string, stats = build_ad_from_textlines_and_vtok_ids(
                clips_by_local=clips_by_local,
                text_lines=script_lines,
                vtok_ids=pred_ids
            )

            if stats.get("truncated"):
                print(f"[note] sample_id={sample_id}: 文本({stats['len_texts']}) 与 vtok({stats['len_vtoks']}) 长度不等，已按 {stats['paired']} 配对。")
            if stats.get("missing_vtok_ids"):
                print(f"[note] sample_id={sample_id}: 缺失 vtok 的 local_id: {stats['missing_vtok_ids']}")

            # 4) 解析并导出视频
            print(ad_string)
            ans, _ = parse_multimodal(ad_string)
            clips = ans["clips"]
            out_frame = os.path.join(args.out_dir, f"{args.prefix}_{sample_id}_frame.mp4")
            # out_clip  = os.path.join(args.out_dir, f"{args.prefix}_{sample_id}_clip.mp4")
            generate_video(clips, extract_method="frame", out_path=out_frame)
            # generate_video(clips, extract_method="clip",  out_path=out_clip)

            print(f"[ok] sample_id={sample_id} ")
            ok += 1

    print(f"\n[done] 共读入 {total} 行；成功处理 {ok} 条；跳过 {skip} 条。")

if __name__ == "__main__":
    main()
