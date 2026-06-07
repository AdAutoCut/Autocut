#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
把 PT 数据解析成 (ad_id, clip_id, frame_id)，并同时下载视频与抽取每个 clip 的代表帧。
输出 JSONL：{"ad_id", "clip_id", "frame_id", "video_path", "frame_path"}
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
from typing import Dict, Any, Iterable, List, Set, Tuple, Optional

from tools.parse import parse_multimodal
from tools.video import init_video_model
from tools.common.blobstore import download_video_bytes 

# ----------------------------
# 读取/解析基础
# ----------------------------

def iter_records(input_path: str) -> Iterable[Dict[str, Any]]:
    """支持 .json (数组/单条) 和 .jsonl (每行一条) 两种输入格式"""
    if input_path.endswith(".jsonl"):
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue
    else:
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    yield item
            elif isinstance(data, dict):
                yield data
            else:
                raise ValueError("Unsupported JSON structure at top level.")

def first_human_text(record: Dict[str, Any]) -> str:
    """从 conversations 中取第一条 human 的 value"""
    conv = record.get("conversations", [])
    for m in conv:
        if m.get("from") == "human" and isinstance(m.get("value"), str):
            return m["value"]
    if conv and isinstance(conv[0].get("value"), str):
        return conv[0]["value"]
    return ""

def extract_first_frames(parsed_sample: Dict[str, Any]) -> List[int]:
    """从 parsed_sample['frames'] 中，每个子列表取第一个 frame_id"""
    out: List[int] = []
    frames = parsed_sample.get("frames", [])
    if not isinstance(frames, list):
        return out
    for sub in frames:
        if isinstance(sub, list) and len(sub) > 0:
            out.append(sub[0])
    return out

def load_processed_pairs(out_path: str) -> Set[Tuple[int, int]]:
    """
    从已存在的输出 JSONL 里恢复 (ad_id, clip_id) 集合，用于断点续跑去重。
    文件不存在则返回空集合。
    """
    processed: Set[Tuple[int, int]] = set()
    if not os.path.exists(out_path):
        return processed
    with open(out_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                ad_id = int(obj["ad_id"])
                clip_id = int(obj["clip_id"])
                processed.add((ad_id, clip_id))
            except Exception:
                # 脏行/部分写入失败时的容错
                continue
    return processed

def safe_write_line(fp, obj):
    """立即写盘，尽量减少中断造成的数据丢失。"""
    fp.write(json.dumps(obj, ensure_ascii=False) + "\n")
    fp.flush()
    os.fsync(fp.fileno())

# ----------------------------
# 下载视频与抽帧（整合你的函数，可配置目录/质量/尺寸）
# ----------------------------

def extract_frame_resize_ffmpeg(
    video_path: Path,
    t_sec: int,
    out_path: Path,
    height: int = 512,
    quality: int = 8,
    exact: bool = False
) -> None:
    """
    从 video_path 的 t_sec 秒提一帧，并等比例缩放到 height，保存到 out_path。
    - quality: 仅对 JPG 有效，范围 2(高质大文件) ~ 31(低质小文件)，默认 8
    - exact: True 时把 -ss 放到 -i 之后（更精确更慢）；False 更快（默认）
    """
    video_path = str(video_path)
    out_path = str(out_path)

    # -2 确保宽度为偶数，避免编码器/滤镜报错；高度固定为 height
    vf = f"scale=-2:{height}"

    if exact:
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-i", video_path,
            "-ss", str(t_sec),
            "-vf", vf,
            "-frames:v", "1",
            "-y", out_path,
        ]
    else:
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-ss", str(t_sec),
            "-i", video_path,
            "-vf", vf,
            "-frames:v", "1",
            "-y", out_path,
        ]

    # 仅当输出是 .jpg/.jpeg 时设置压缩质量
    if Path(out_path).suffix.lower() in {".jpg", ".jpeg"}:
        cmd.extend(["-q:v", str(quality)])

    subprocess.run(cmd, check=True)

def download_vid(photo_id: int, video_dir: Path) -> Optional[Path]:
    """
    下载视频；成功返回本地路径，失败返回 None（不中断流程）
    """
    try:
        video_dir.mkdir(parents=True, exist_ok=True)
        video_bytes = download_video_bytes(photo_id)
        if not video_bytes:
            print(f"Warning: download_video_bytes returned empty for photo_id {photo_id}", file=sys.stderr)
            return None

        output_path = video_dir / f"{photo_id}.mp4"
        with open(output_path, "wb") as f:
            f.write(video_bytes)
        return output_path
    except Exception as e:
        print(f"Warning: Failed to download video for photo_id {photo_id}: {e}", file=sys.stderr)
        return None

def download_frame(
    frame_id: int,
    video_dir: Path,
    frame_dir: Path,
    height: int = 512,
    quality: int = 8,
    exact: bool = False
) -> Optional[Tuple[Path, Path]]:
    """
    根据 frame_id 推断所属视频与时间，尝试下载视频并抽帧。
    成功返回 (video_path, frame_path)；任一环节失败返回 None（不中断流程）
    约定：frame_id 的前 len-4 位是 photo_id，后 4 位是整秒时间戳
    """
    try:
        fid_str = str(frame_id)
        f_photo_id = int(fid_str[:-4])  # 所属 photo_id
        frame_time = int(fid_str[-4:])  # 秒
    except Exception as e:
        print(f"Warning: Invalid frame_id={frame_id}: {e}", file=sys.stderr)
        return None

    frame_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)

    # 若本地已有视频则复用，否则尝试下载
    video_path = video_dir / f"{f_photo_id}.mp4"
    if not video_path.exists():
        video_path = download_vid(f_photo_id, video_dir)
        if video_path is None:
            return None  # 下载失败

    # 抽帧
    frame_path = frame_dir / f"{frame_id}.jpg"
    try:
        extract_frame_resize_ffmpeg(video_path, frame_time, frame_path, height=height, quality=quality, exact=exact)
        return (video_path, frame_path)
    except Exception as e:
        print(f"Warning: Failed to extract frame for frame_id={frame_id}: {e}", file=sys.stderr)
        return None

# ----------------------------
# 主流程
# ----------------------------

def process(
    input_path: str,
    output_jsonl: str,
    start_ad_id: int = 0,
    video_dir: str = "downloaded_vids",
    frame_dir: str = "extracted_frames",
    height: int = 512,
    quality: int = 8,
    exact: bool = False,
):

    # （可选）初始化你的视频模型与索引；目前逻辑未直接用到
    try:
        _video_rqvae_model, _video_index, _frame_index, _clip_index = init_video_model(
            load_faiss=True, model_name="video_8_256_0729"
        )
    except Exception:
        # 初始化失败不影响下载与抽帧主流程
        pass

    vd = Path(video_dir)
    fd = Path(frame_dir)

    processed_pairs = load_processed_pairs(output_jsonl)
    total_records = 0
    written = 0
    skipped_records = 0

    os.makedirs(os.path.dirname(output_jsonl) or ".", exist_ok=True)
    with open(output_jsonl, "a", encoding="utf-8") as fout:
        for ad_id, record in enumerate(iter_records(input_path)):
            print("Processing ad_id:", ad_id)
            if ad_id < start_ad_id:
                continue

            total_records += 1
            try:
                inp = first_human_text(record)
                if not inp:
                    skipped_records += 1
                    continue

                parsed_sample, _ = parse_multimodal(inp)
                first_ids = extract_first_frames(parsed_sample)
                if not first_ids:
                    skipped_records += 1
                    continue

                for clip_id, frame_id in enumerate(first_ids):
                    key = (ad_id, clip_id)
                    if key in processed_pairs:
                        # 已经写过，跳过
                        continue

                    # --- 新增：下载视频 + 抽帧 ---
                    dl = download_frame(
                        frame_id=frame_id,
                        video_dir=vd,
                        frame_dir=fd,
                        height=height,
                        quality=quality,
                        exact=exact,
                    )
                    if dl is None:
                        # 下载或抽帧失败，跳过该 clip
                        continue
                    video_path, frame_path = dl
                    print('DOWNLOADED Frame_path: ',frame_path)

                    line = {
                        "ad_id": ad_id,
                        "clip_id": clip_id,
                        "frame_id": frame_id,
                        # "video_path": str(video_path),
                        # "frame_path": str(frame_path),
                    }
                    safe_write_line(fout, line)
                    processed_pairs.add(key)
                    written += 1

            except KeyboardInterrupt:
                print("\nInterrupted by user. Progress saved so far.")
                break
            except Exception as e:
                # 保持健壮：出错的样本整体跳过
                print(f"Warning: error on ad_id={ad_id}: {e}", file=sys.stderr)
                skipped_records += 1
                continue

    print(f"Done. total_records_seen={total_records}, new_lines_written={written}, skipped_records={skipped_records}")
    print(f"Output: {output_jsonl}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="/data/phd/qinsizhong/llm_factory_test/baselines/data/sft_1006_eval.json",
                    help="输入文件路径（.json 或 .jsonl）")
    ap.add_argument("--output", default="/data/phd/miltonzhou/sft/data_preprocess/filter_data/temp/frame_id_list_1006_new.jsonl",
                    help="输出 JSONL 路径")
    ap.add_argument("--start-ad-id", type=int, default=0,
                    help="可选：从指定 ad_id 开始处理（用于手动断点）")
    ap.add_argument("--video-dir", default="downloaded_vids", help="下载视频保存目录")
    ap.add_argument("--frame-dir", default="extracted_frames", help="抽取帧保存目录")
    ap.add_argument("--height", type=int, default=512, help="抽帧图像高度（等比例缩放）")
    ap.add_argument("--quality", type=int, default=8, help="JPG 质量（2~31，数值越小质量越高）")
    # ap.add_argument("--exact", action="store_true", help="抽帧更精确但更慢（-ss 放到 -i 之后）")
    args = ap.parse_args()

    process(
        input_path=args.input,
        output_jsonl=args.output,
        start_ad_id=args.start_ad_id,
        video_dir=args.video_dir,
        frame_dir=args.frame_dir,
        height=args.height,
        quality=args.quality,
        # exact=args.exact,
    )

if __name__ == "__main__":
    main()
