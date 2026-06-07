#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
把 PT 数据，通过（tools.parse RQVAE）检索得到 frame_id，按“clip table”格式展开。
支持基于 chunk 号自动拼接 I/O 路径：
  input  = {base_dir}/raw_chunk_{chunk}.jsonl
  output = {base_dir}/ct_chunk_{chunk}.jsonl
  ckpt   = {output}.ckpt
"""

import os
import json
import re
import argparse
from pathlib import Path
from typing import Dict, Any, Iterable, List, Optional

from tools.parse import parse_multimodal
from tools.video import init_video_model
from tools.audio import init_audio_model
from tqdm import tqdm  # 进度条

# ---------- IO ----------

def read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def count_lines(path: str) -> int:
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)

def scan_max_ad_id(out_path: str) -> int:
    if not os.path.exists(out_path):
        return 0
    max_id = 0
    with open(out_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                aid = int(obj.get("ad_id", 0))
                if aid > max_id:
                    max_id = aid
            except Exception:
                continue
    return max_id

def load_ckpt(ckpt_path: str, out_path: str) -> int:
    if os.path.exists(ckpt_path):
        try:
            with open(ckpt_path, "r", encoding="utf-8") as f:
                val = f.read().strip()
                return int(val) if val else 0
        except Exception:
            pass
    return scan_max_ad_id(out_path)

def save_ckpt(ckpt_path: str, ad_id: int) -> None:
    tmp = ckpt_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(str(ad_id))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, ckpt_path)

# ---------- helpers ----------

def safe_first(ls):
    if isinstance(ls, (list, tuple)) and len(ls) > 0:
        return str(ls[0])
    return "null"

def safe_text(t):
    if isinstance(t, list):
        return str(t[0]) if t else "null"
    if isinstance(t, str):
        return t
    return "null"

def get_aud_id(parsed_sample: Dict[str, Any]) -> str:
    auds = parsed_sample.get("audios")
    if not auds:
        return "null"
    first = auds[0]
    if isinstance(first, (list, tuple)):
        return safe_first(first)
    return str(first)

_clip_pat = re.compile(r"<\|clip_start\|>(.*?)<\|clip_end\|>", flags=re.DOTALL)
_video_pat = re.compile(r"<\|video_start\|>(.*?)<\|video_end\|>", flags=re.DOTALL)
_audio_pat = re.compile(r"<\|audio_start\|>(.*?)<\|audio_end\|>", flags=re.DOTALL)

def extract_clip_vtoks(s: str) -> List[str]:
    vtoks: List[str] = []
    for m in _clip_pat.finditer(s):
        seg = m.group(1)
        vm = _video_pat.search(seg)
        vtoks.append(vm.group(1) if vm else "null")
    return vtoks

def extract_aud_tok(s: str) -> str:
    am = _audio_pat.search(s)
    return am.group(1) if am else "null"

# ---------- path resolver ----------

def resolve_paths(
    base_dir: str,
    chunk: int,
    input_path: Optional[str],
    output_path: Optional[str]
) -> (str, str, str):
    """根据 chunk 号与 base_dir 生成默认 I/O 路径；若提供 input/output 则使用提供的。"""
    base = base_dir.rstrip("/")
    default_input  = f"{base}/raw_chunk_{chunk}.jsonl"
    default_output = f"{base}/ct_chunk_{chunk}.jsonl"

    in_path  = input_path  if input_path  else default_input
    out_path = output_path if output_path else default_output
    ckpt_path = f"{out_path}.ckpt"
    return in_path, out_path, ckpt_path

# ---------- main ----------

def main():
    parser = argparse.ArgumentParser(description="Build clip table from PT data (RQVAE parse), chunk-aware auto I/O")
    # 基本参数（都给默认值）
    parser.add_argument("--base_dir", default="/data/phd/miltonzhou/sft/data_preprocess/filter_data/data_chunk_1102",
                        help="数据分块所在目录")
    parser.add_argument("--chunk", type=int, default=0, help="输入 chunk 号（将读取 raw_chunk_{chunk}.jsonl）")

    # 可覆盖的手动路径（可选）
    parser.add_argument("--input",  default=None, help="可选：手动指定输入 JSONL 路径，覆盖基于 chunk 的默认路径")
    parser.add_argument("--output", default=None, help="可选：手动指定输出 JSONL 路径，覆盖基于 chunk 的默认路径")
    parser.add_argument("--ckpt",   default=None, help="可选：断点 ckpt 路径（默认=输出文件名+.ckpt）")

    # 其它运行参数
    parser.add_argument("--video_model_name", default="video_8_256_0729")
    parser.add_argument("--audio_model_name", default="audio_8_256_0729")
    args = parser.parse_args()

    # 解析路径
    input_path, out_path, default_ckpt = resolve_paths(
        base_dir=args.base_dir,
        chunk=args.chunk,
        input_path=args.input,
        output_path=args.output
    )
    ckpt_path = args.ckpt or default_ckpt

    # 基本存在性检查
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input not found: {input_path}")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    # 进入 parse 之前初始化 RQVAE / FAISS
    _video_rqvae_model, _video_index, _frame_index, _clip_index = init_video_model(
        load_faiss=True, model_name=args.video_model_name
    )
    init_audio_model(load_faiss=True, model_name=args.audio_model_name)

    # 进度条总数 = 原始 JSONL 行数（每行一个 ad）
    total_ads = count_lines(input_path)

    # 起始 ad_id（已完成的最大 ad_id）
    last_done = load_ckpt(ckpt_path, out_path)  # 0 表示从第 1 条开始
    start_from = last_done + 1

    # 以追加方式打开输出 & 初始化进度条
    out_fh = open(out_path, "a", encoding="utf-8")
    pbar = tqdm(total=total_ads, initial=last_done,
                desc=f"Processing chunk {args.chunk}", unit="ad")

    try:
        for ad_id, obj in enumerate(read_jsonl(input_path), start=1):
            if ad_id < start_from:
                continue  # 跳过已完成

            s = str(obj.get("text", ""))

            # 解析结构
            parsed_sample, _ = parse_multimodal(s)
            clips = parsed_sample.get("clips", []) if isinstance(parsed_sample, dict) else []

            # token 段
            v_toks = extract_clip_vtoks(s)
            aud_tok = extract_aud_tok(s)
            aud_id = get_aud_id(parsed_sample)

            # 收集当前 ad 的所有输出行
            lines: List[str] = []
            if not clips:
                rec = {
                    "ad_id": ad_id,
                    "clip_id": 1,
                    "frame_id": "null",
                    "clip_seq": "null",
                    "text": "null",
                    "aud_id": aud_id,
                    "aud_tok": aud_tok,
                    "v_tok": v_toks[0] if v_toks else "null",
                }
                lines.append(json.dumps(rec, ensure_ascii=False) + "\n")
            else:
                for clip_idx, clip in enumerate(clips, start=1):
                    text_str = safe_text(clip.get("text"))
                    frame_id = safe_first(clip.get("frame"))
                    clip_seq = safe_first(clip.get("clip"))
                    v_tok = v_toks[clip_idx - 1] if clip_idx - 1 < len(v_toks) else "null"

                    rec = {
                        "ad_id": ad_id,
                        "clip_id": clip_idx,
                        "frame_id": frame_id,
                        "clip_seq": clip_seq,
                        "text": text_str,
                        "aud_id": aud_id,
                        "aud_tok": aud_tok,
                        "v_tok": v_tok,
                    }
                    lines.append(json.dumps(rec, ensure_ascii=False) + "\n")

            # 写入一个 ad 的所有行
            out_fh.writelines(lines)
            out_fh.flush()
            os.fsync(out_fh.fileno())

            # 更新 ckpt 与进度条
            save_ckpt(ckpt_path, ad_id)
            pbar.update(1)
            pbar.set_postfix_str(f"last_ad={ad_id}")

    finally:
        pbar.close()
        out_fh.close()

    print("\n=== Done ===")
    print(f"Input : {input_path}")
    print(f"Output: {out_path}")
    print(f"CKPT  : {ckpt_path}")

if __name__ == "__main__":
    main()


# USAGE： python build_clip_table.py --chunk 0