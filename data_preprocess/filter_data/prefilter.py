#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import re
import argparse
from pathlib import Path
from typing import List, Optional

# -------------------------
# Precompiled regex (DOTALL)
# -------------------------
RE_AD     = re.compile(r"<\|ad_start\|>(.*?)<\|ad_end\|>", re.DOTALL)
RE_AUDIO  = re.compile(r"<\|audio_start\|>(.*?)<\|audio_end\|>", re.DOTALL)
RE_CLIP   = re.compile(r"<\|clip_start\|>(.*?)<\|clip_end\|>", re.DOTALL)
RE_TEXT   = re.compile(r"<\|text_start\|>(.*?)<\|text_end\|>", re.DOTALL)
RE_VIDEO  = re.compile(r"<\|video_start\|>(.*?)<\|video_end\|>", re.DOTALL)
RE_FRAME  = re.compile(r"<\|frame_start\|>(.*?)<\|frame_end\|>", re.DOTALL)

def count_literal(s: str, sub: str) -> int:
    return s.count(sub)

def has_audio_tokens(ad_text: str) -> bool:
    m = RE_AUDIO.search(ad_text)
    if not m:
        return False
    return "<a_" in m.group(1)

def clip_has_valid_text(clip_body: str) -> bool:
    """text 块必须存在，且非空白"""
    tm = RE_TEXT.search(clip_body)
    if not tm:
        return False
    return tm.group(1).strip() != ""

def _frame_group_has_8_slots(frame_body: str) -> bool:
    """可选：frame 组内是否至少包含 v_0..v_7"""
    for k in range(8):
        if f"<v_{k}_" not in frame_body:
            return False
    return True

def clip_has_valid_video(clip_body: str, strict_v_slots: bool = False) -> bool:
    """
    video 有效：必须有 video 块；至少 1 个 frame 组；且包含 <v_ token。
    若 strict_v_slots=True，则每个 frame 组都需包含 v_0..v_7。
    """
    vm = RE_VIDEO.search(clip_body)
    if not vm:
        return False
    video_body = vm.group(1)

    if "<v_" not in video_body:
        return False

    has_frame_group = False
    for fm in RE_FRAME.finditer(video_body):
        has_frame_group = True
        if strict_v_slots and (not _frame_group_has_8_slots(fm.group(1))):
            return False
    return has_frame_group

def extract_audio_block(ad_text: str) -> Optional[str]:
    m = RE_AUDIO.search(ad_text)
    return m.group(0) if m else None

def rebuild_ad_text(valid_clip_chunks: List[str], audio_block: str) -> str:
    """重建广告：有效 clip 拼接 + 原 audio 放末尾 + ad 边界"""
    return "<|ad_start|>" + "".join(valid_clip_chunks) + (audio_block or "") + "<|ad_end|>"

def filter_jsonl_with_semantic_rules(
    input_path: str,
    output_path: str,
    min_clip: int = 1,
    max_clip: int = 40,
    min_frame_groups: int = 1,
    max_frame_groups: int = 200,
    strict_v_slots: bool = False,
    progress_every: int = 10000
):
    """
    流程：
      A) 广告级数量阈值预筛：clip_count、frame_group_count 在给定区间
      B) 广告级：必须有 audio 且含 <a_ token
      C) clip 级清洗：若某 clip 的 text 无效 或 video 无效 => 删除该 clip
         清洗后 clip 数=0 => 丢弃整条
    """

    n_total = 0
    n_kept = 0

    # 丢弃原因统计
    drop_json_invalid = 0
    drop_qty_clip = 0
    drop_qty_frame = 0
    drop_no_audio = 0
    drop_zero_clips_after_clean = 0

    # clip 级移除统计
    removed_clips_text_invalid = 0
    removed_clips_video_invalid = 0

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            n_total += 1
            s = line.strip()
            if not s:
                continue

            try:
                obj = json.loads(s)
            except json.JSONDecodeError:
                drop_json_invalid += 1
                if (n_total % progress_every) == 0:
                    print(f"[Progress] {n_total} processed | kept={n_kept}")
                continue

            ad_text = obj.get("text", "")
            if not isinstance(ad_text, str):
                drop_json_invalid += 1
                if (n_total % progress_every) == 0:
                    print(f"[Progress] {n_total} processed | kept={n_kept}")
                continue

            # ---------- A) 数量阈值预筛（广告级，基于原始文本） ----------
            clip_count = count_literal(ad_text, "<|clip_start|>")
            frame_group_count = count_literal(ad_text, "<|frame_start|>")

            if not (min_clip <= clip_count <= max_clip):
                drop_qty_clip += 1
                if (n_total % progress_every) == 0:
                    print(f"[Progress] {n_total} processed | kept={n_kept}")
                continue

            if not (min_frame_groups <= frame_group_count <= max_frame_groups):
                drop_qty_frame += 1
                if (n_total % progress_every) == 0:
                    print(f"[Progress] {n_total} processed | kept={n_kept}")
                continue

            # ---------- B) 广告级：必须有有效 audio ----------
            if not has_audio_tokens(ad_text):
                drop_no_audio += 1
                if (n_total % progress_every) == 0:
                    print(f"[Progress] {n_total} processed | kept={n_kept}")
                continue

            # ---------- C) clip 级清洗：text 与 video 各自校验 ----------
            valid_clips: List[str] = []
            removed_text = 0
            removed_video = 0

            for cm in RE_CLIP.finditer(ad_text):
                clip_chunk = cm.group(0)   # 含 <|clip_start|>...<|clip_end|>
                clip_body  = cm.group(1)

                text_ok  = clip_has_valid_text(clip_body)
                video_ok = clip_has_valid_video(clip_body, strict_v_slots=strict_v_slots)

                if text_ok and video_ok:
                    valid_clips.append(clip_chunk)
                else:
                    if not text_ok:
                        removed_text += 1
                    if not video_ok:
                        removed_video += 1

            removed_clips_text_invalid += removed_text
            removed_clips_video_invalid += removed_video

            if len(valid_clips) == 0:
                drop_zero_clips_after_clean += 1
                if (n_total % progress_every) == 0:
                    print(f"[Progress] {n_total} processed | kept={n_kept}")
                continue

            # 重建 text：保留有效 clip，音频置于末尾（与示例一致）
            audio_block = extract_audio_block(ad_text) or ""
            obj["text"] = rebuild_ad_text(valid_clips, audio_block)

            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            n_kept += 1

            if (n_total % progress_every) == 0:
                print(f"[Progress] {n_total} processed | kept={n_kept} | "
                      f"drop_json_invalid={drop_json_invalid} | drop_qty_clip={drop_qty_clip} | "
                      f"drop_qty_frame={drop_qty_frame} | drop_no_audio={drop_no_audio} | "
                      f"drop_zero_clips={drop_zero_clips_after_clean} | "
                      f"removed_clips_text_invalid={removed_clips_text_invalid} | "
                      f"removed_clips_video_invalid={removed_clips_video_invalid}")

    # ------------- Summary -------------
    print("\n=== Summary ===")
    print(f"Total processed: {n_total}")
    print(f"Kept: {n_kept}")
    print(f"Dropped (invalid JSON/text type): {drop_json_invalid}")
    print(f"Dropped (clip count out of range): {drop_qty_clip}")
    print(f"Dropped (frame_group count out of range): {drop_qty_frame}")
    print(f"Dropped (no/empty audio tokens): {drop_no_audio}")
    print(f"Dropped (zero clips after cleaning): {drop_zero_clips_after_clean}")
    print(f"Removed clips — text invalid: {removed_clips_text_invalid}")
    print(f"Removed clips — video invalid: {removed_clips_video_invalid}")
    print(f"Saved to: {output_path}")

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="JSONL prefilter + semantic cleaning for ad/clip/frame data")
    # 全部默认值（不 required）
    p.add_argument("--input", default="/data/phd/qinsizhong/llm_factory_test/data/0729_train.jsonl")
    p.add_argument("--output", default="prefiltered_data_1102_train.jsonl")
    p.add_argument("--min_clip", type=int, default=3)
    p.add_argument("--max_clip", type=int, default=30)
    p.add_argument("--min_frame_groups", type=int, default=5, help="按 frame_start 计数（帧组数）")
    p.add_argument("--max_frame_groups", type=int, default=60)
    p.add_argument("--strict_v_slots", action="store_true", default=False,
                   help="若 True：每个 frame 组都必须包含 v_0..v_7 槽位")
    p.add_argument("--progress_every", type=int, default=10000)
    return p

if __name__ == "__main__":
    args = build_argparser().parse_args()

    filter_jsonl_with_semantic_rules(
        input_path=args.input,
        output_path=args.output,
        min_clip=args.min_clip,
        max_clip=args.max_clip,
        min_frame_groups=args.min_frame_groups,
        max_frame_groups=args.max_frame_groups,
        strict_v_slots=args.strict_v_slots,
        progress_every=args.progress_every
    )
