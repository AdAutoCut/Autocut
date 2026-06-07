#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
add_frames_to_CT.py

根据 clip_table 中的 frame_id：
- 推断 video_id（去掉后4位）
- 推断在视频中的时间（后4位秒数）
- 从 blobstore 下载对应视频（若本地已存在则跳过下载）
- 用 ffmpeg 在指定时间抽取单帧，保存到 downloaded_frames/
- 输出增强后的 JSONL，每行包含原始字段 + video_id + frame_time + video_path + frame_path

输入示例（JSONL 每行一个 clip）：
{
  "chunk_id": 99,
  "ad_id": 14,
  "clip_id": 1,
  "frame_id": 1532426326480000,
  "clip_seq": 1532426326480001024,
  "text": "...",
  ...
}
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
from typing import Dict, Any, Iterable, Tuple, Set, Optional

from tools.common.blobstore import download_video_bytes

# ----------------------------
# 基础工具
# ----------------------------

def iter_clip_records(input_path: str) -> Iterable[Dict[str, Any]]:
    """
    按行读取 JSONL，每行一个 clip 记录。
    空行或解析失败的行会被跳过。
    """
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                print("Warning: skip invalid JSON line", file=sys.stderr)
                continue


def safe_write_line(fp, obj: Dict[str, Any]) -> None:
    """
    立即写盘，减少中断带来的数据丢失。
    """
    fp.write(json.dumps(obj, ensure_ascii=False) + "\n")
    fp.flush()
    os.fsync(fp.fileno())


def load_processed_keys(out_path: str) -> Set[Tuple[int, int, int]]:
    """
    从已经存在的输出 JSONL 中恢复已处理的 (chunk_id, ad_id, clip_id) 集合，
    用于断点续跑和避免重复处理。
    如果某行缺少这些字段则跳过。
    """
    processed: Set[Tuple[int, int, int]] = set()
    if not os.path.exists(out_path):
        return processed

    with open(out_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                ck = int(obj.get("chunk_id"))
                ad = int(obj.get("ad_id"))
                cl = int(obj.get("clip_id"))
                processed.add((ck, ad, cl))
            except Exception:
                # 容忍脏行
                continue

    return processed


def parse_frame_id(frame_id: int) -> Tuple[int, int]:
    """
    将 frame_id 拆分为 (video_id, frame_time_sec)
    约定：frame_id 的后4位为秒数，前面部分为 video_id。
    """
    fid_str = str(int(frame_id))
    if len(fid_str) <= 4:
        raise ValueError(f"Invalid frame_id length: {frame_id}")
    video_id = int(fid_str[:-4])
    frame_time = int(fid_str[-4:])
    return video_id, frame_time

# ----------------------------
# 下载视频与抽帧
# ----------------------------

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def download_video_if_needed(video_id: int, video_dir: Path) -> Optional[Path]:
    """
    检查本地是否已有 {video_id}.mp4；
    若无，则调用 download_video_bytes(video_id) 下载。
    成功返回本地路径，失败返回 None。
    """
    ensure_dir(video_dir)
    video_path = video_dir / f"{video_id}.mp4"

    if video_path.exists():
        return video_path

    try:
        video_bytes = download_video_bytes(video_id)
    except Exception as e:
        print(f"Warning: download_video_bytes error for video_id={video_id}: {e}", file=sys.stderr)
        return None

    if not video_bytes:
        print(f"Warning: empty video_bytes for video_id={video_id}", file=sys.stderr)
        return None

    try:
        with open(video_path, "wb") as f:
            f.write(video_bytes)
    except Exception as e:
        print(f"Warning: failed to write video file for video_id={video_id}: {e}", file=sys.stderr)
        return None

    return video_path


def extract_frame_ffmpeg(
    video_path: Path,
    t_sec: int,
    out_path: Path,
    height: int = 512,
    quality: int = 8,
    exact: bool = False,
) -> None:
    """
    从 video_path 的 t_sec 秒抽一帧，缩放到指定高度，保存到 out_path。
    - height: 输出图像高度，等比例缩放，宽度用 -2（保证为偶数）。
    - quality: 仅对 JPG 有效，2(高质)~31(低质)，默认8。
    - exact=False: 使用 -ss 在 -i 前，更快（近似）。
    """
    ensure_dir(out_path.parent)

    vf = f"scale=-2:{height}"

    if exact:
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-i", str(video_path),
            "-ss", str(t_sec),
            "-vf", vf,
            "-frames:v", "1",
            "-y", str(out_path),
        ]
    else:
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-ss", str(t_sec),
            "-i", str(video_path),
            "-vf", vf,
            "-frames:v", "1",
            "-y", str(out_path),
        ]

    if out_path.suffix.lower() in {".jpg", ".jpeg"}:
        cmd.extend(["-q:v", str(quality)])

    subprocess.run(cmd, check=True)


def get_or_create_frame(
    frame_id: int,
    video_dir: Path,
    frame_dir: Path,
    height: int = 512,
    quality: int = 8,
    exact: bool = False,
) -> Optional[Tuple[Path, Path, int, int]]:
    """
    根据 frame_id：
    - 拆出 video_id 与 frame_time
    - 确保视频存在（本地已有则复用，否则下载）
    - 用 ffmpeg 抽取对应帧到 {frame_dir}/{frame_id}.jpg

    返回: (video_path, frame_path, video_id, frame_time)
    任一环节失败则返回 None。
    """
    try:
        video_id, frame_time = parse_frame_id(frame_id)
    except Exception as e:
        print(f"Warning: invalid frame_id={frame_id}: {e}", file=sys.stderr)
        return None

    video_path = download_video_if_needed(video_id, video_dir)
    if video_path is None:
        return None

    frame_path = frame_dir / f"{frame_id}.jpg"

    # 如果已经存在该帧文件，可以直接复用；不存在则抽取
    if not frame_path.exists():
        try:
            extract_frame_ffmpeg(
                video_path=video_path,
                t_sec=frame_time,
                out_path=frame_path,
                height=height,
                quality=quality,
                exact=exact,
            )
        except subprocess.CalledProcessError as e:
            print(f"Warning: ffmpeg failed for frame_id={frame_id}: {e}", file=sys.stderr)
            return None
        except Exception as e:
            print(f"Warning: failed to extract frame for frame_id={frame_id}: {e}", file=sys.stderr)
            return None

    return video_path, frame_path, video_id, frame_time

# ----------------------------
# 主流程
# ----------------------------

def process(
    input_path: str,
    output_path: str,
    video_dir: str = "downloaded_vids",
    frame_dir: str = "downloaded_frames",
    height: int = 512,
    quality: int = 8,
    exact: bool = False,
) -> None:
    """
    主处理函数：
    - 遍历 clip_table.jsonl
    - 对每个 (chunk_id, ad_id, clip_id) 对应的 frame_id 抽帧
    - 输出增强后的 JSONL
    """
    vd = Path(video_dir)
    fd = Path(frame_dir)

    # 断点续跑：已处理 key 集合
    processed_keys = load_processed_keys(output_path)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    total = 0
    written = 0
    skipped = 0

    with open(output_path, "a", encoding="utf-8") as fout:
        for record in iter_clip_records(input_path):
            try:
                chunk_id = int(record.get("chunk_id"))
                ad_id = int(record.get("ad_id"))
                clip_id = int(record.get("clip_id"))
            except Exception:
                print("Warning: record missing chunk_id/ad_id/clip_id, skip", file=sys.stderr)
                skipped += 1
                continue

            key = (chunk_id, ad_id, clip_id)
            if key in processed_keys:
                continue

            total += 1

            if "frame_id" not in record:
                print(f"Warning: missing frame_id for {key}, skip", file=sys.stderr)
                skipped += 1
                continue

            try:
                frame_id = int(record["frame_id"])
            except Exception:
                print(f"Warning: invalid frame_id for {key}, skip", file=sys.stderr)
                skipped += 1
                continue

            res = get_or_create_frame(
                frame_id=frame_id,
                video_dir=vd,
                frame_dir=fd,
                height=height,
                quality=quality,
                exact=exact,
            )

            if res is None:
                skipped += 1
                continue

            video_path, frame_path, video_id, frame_time = res

            out_obj = dict(record)
            out_obj["video_id"] = video_id
            out_obj["frame_time"] = frame_time
            out_obj["video_path"] = str(video_path)
            out_obj["frame_path"] = str(frame_path)

            safe_write_line(fout, out_obj)
            processed_keys.add(key)
            written += 1

    print(f"Done. total_seen={total}, new_written={written}, skipped={skipped}")
    print(f"Output: {output_path}")


def main():
    ap = argparse.ArgumentParser(description="Add video/frame info to clip_table using frame_id.")
    ap.add_argument("--input", default="/data/phd/miltonzhou/sft/data_preprocess/filter_data/data_chunk_1102/Final_CT_eval_1109.jsonl",
                    help="输入 clip_table JSONL 路径")
    ap.add_argument("--output", default="/data/phd/miltonzhou/sft/data_preprocess/filter_data/BLCT_eval_1109.jsonl",
                    help="输出增强后 JSONL 路径（支持断点续跑追加）")
    ap.add_argument("--video-dir", default="downloaded_vids",
                    help="视频保存目录（默认：downloaded_vids）")
    ap.add_argument("--frame-dir", default="downloaded_frames",
                    help="帧图像保存目录（默认：downloaded_frames）")
    ap.add_argument("--height", type=int, default=512,
                    help="抽帧高度（默认：512）")
    ap.add_argument("--quality", type=int, default=8,
                    help="JPG 质量 2-31，越小质量越高（默认：8）")
    ap.add_argument("--exact", action="store_true",
                    help="使用更精确但更慢的 ffmpeg 抽帧方式")
    args = ap.parse_args()

    process(
        input_path=args.input,
        output_path=args.output,
        video_dir=args.video_dir,
        frame_dir=args.frame_dir,
        height=args.height,
        quality=args.quality,
        exact=args.exact,
    )


if __name__ == "__main__":
    main()
