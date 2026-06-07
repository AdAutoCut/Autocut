#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
add_frames_to_CT.py (10w条数据版本)

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

# Progress bar
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


# ----------------------------
# 基础工具
# ----------------------------

def iter_clip_records_range(input_path: str, start: int, end: int) -> Iterable[Tuple[int, Dict[str, Any]]]:
    """
    读取 JSONL 的 [start, end) 行（0-indexed），返回 (lineno, record)。
    - 行号按物理行计数（空行/坏行也计数），但空行/坏行会跳过处理。
    """
    if start < 0:
        raise ValueError(f"start must be >= 0, got {start}")
    if end < start:
        raise ValueError(f"end must be >= start, got start={start}, end={end}")

    with open(input_path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f):
            if lineno < start:
                continue
            if lineno >= end:
                break

            s = line.strip()
            if not s:
                continue

            try:
                yield lineno, json.loads(s)
            except json.JSONDecodeError:
                print(f"Warning: skip invalid JSON line at lineno={lineno}", file=sys.stderr)
                continue


class BufferedJSONLWriter:
    """
    缓冲写入 JSONL：
    - flush_every: 每写入多少行 flush 一次
    - fsync_every: 每写入多少行 fsync 一次（0 表示不 fsync）
    说明：
    - flush 是把 Python 缓冲刷到 OS
    - fsync 是把 OS 缓冲刷到磁盘（最慢）
    """
    def __init__(self, fp, flush_every: int = 100, fsync_every: int = 0):
        self.fp = fp
        self.flush_every = max(int(flush_every), 1)
        self.fsync_every = max(int(fsync_every), 0)
        self._n = 0

    def write_obj(self, obj: Dict[str, Any]) -> None:
        self.fp.write(json.dumps(obj, ensure_ascii=False) + "\n")
        self._n += 1

        if self._n % self.flush_every == 0:
            self.fp.flush()

        if self.fsync_every > 0 and (self._n % self.fsync_every == 0):
            try:
                os.fsync(self.fp.fileno())
            except Exception as e:
                print(f"Warning: fsync failed: {e}", file=sys.stderr)

    def close(self) -> None:
        try:
            self.fp.flush()
            if self.fsync_every > 0:
                os.fsync(self.fp.fileno())
        except Exception:
            pass


def load_processed_keys(out_path: str) -> Set[Tuple[int, int, int]]:
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
                continue
    return processed


def parse_frame_id(frame_id: int) -> Tuple[int, int]:
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
    try:
        video_id, frame_time = parse_frame_id(frame_id)
    except Exception as e:
        print(f"Warning: invalid frame_id={frame_id}: {e}", file=sys.stderr)
        return None

    video_path = download_video_if_needed(video_id, video_dir)
    if video_path is None:
        return None

    frame_path = frame_dir / f"{frame_id}.jpg"
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

def parse_input_rows(s: str) -> Tuple[int, int]:
    """
    解析命令行参数 --input-rows "start,end" -> (start, end)
    语义：[start, end)，end 不处理。
    """
    try:
        a, b = s.split(",")
        start = int(a.strip())
        end = int(b.strip())
        if start < 0 or end < 0 or end < start:
            raise ValueError
        return start, end
    except Exception:
        raise argparse.ArgumentTypeError('Invalid --input-rows. Use format "start,end", e.g. "0,1000".')


def process(
    input_path: str,
    output_path: str,
    video_dir: str = "downloaded_vids",
    frame_dir: str = "downloaded_frames",
    height: int = 512,
    quality: int = 8,
    exact: bool = False,
    input_rows: Optional[Tuple[int, int]] = None,
    flush_every: int = 100,
    fsync_every: int = 0,
) -> None:
    vd = Path(video_dir)
    fd = Path(frame_dir)

    processed_keys = load_processed_keys(output_path)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    if input_rows is None:
        raise ValueError("Please provide --input-rows start,end to run in batches safely.")
    start, end = input_rows
    total_target = end - start

    total_new_seen = 0
    written = 0
    skipped = 0
    already_done = 0

    it = iter_clip_records_range(input_path, start=start, end=end)

    if tqdm is None:
        print("Warning: tqdm is not installed. Install with `pip install tqdm` to see progress bar.", file=sys.stderr)
        iterator = it
    else:
        iterator = tqdm(
            it,
            total=total_target,
            desc=f"add_frames_to_CT [{start},{end})",
            unit="lines",
            dynamic_ncols=True,
            mininterval=0.5,
        )

    with open(output_path, "a", encoding="utf-8") as fout:
        writer = BufferedJSONLWriter(fout, flush_every=flush_every, fsync_every=fsync_every)

        for lineno, record in iterator:
            if tqdm is not None:
                iterator.set_postfix({
                    "written": written,
                    "skipped": skipped,
                    "done": already_done,
                    "flushN": flush_every,
                    "fsyncN": fsync_every,
                })

            try:
                chunk_id = int(record.get("chunk_id"))
                ad_id = int(record.get("ad_id"))
                clip_id = int(record.get("clip_id"))
            except Exception:
                skipped += 1
                continue

            key = (chunk_id, ad_id, clip_id)
            if key in processed_keys:
                already_done += 1
                continue

            total_new_seen += 1

            if "frame_id" not in record:
                skipped += 1
                continue

            try:
                frame_id = int(record["frame_id"])
            except Exception:
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
            out_obj["_src_lineno"] = lineno  # 方便回溯来源行号

            writer.write_obj(out_obj)
            processed_keys.add(key)
            written += 1

        writer.close()

    print(
        f"Done. input_rows=[{start},{end}), total_new_seen={total_new_seen}, new_written={written}, "
        f"skipped={skipped}, already_done={already_done}"
    )
    print(f"Output: {output_path}")


def main():
    ap = argparse.ArgumentParser(description="Add video/frame info to clip_table using frame_id (batched + progress bar).")
    ap.add_argument(
        "--input",
        default="/data/phd/qinsizhong/miltonzhou/sft/data_preprocess/filter_data/data_chunk_1102/Final_CT_train_1109.jsonl",
        help="输入 clip_table JSONL 路径"
    )
    ap.add_argument(
        "--output",
        default="/data/phd/qinsizhong/miltonzhou/sft/data_preprocess/filter_data/BLCT_train_0126_chunk1.jsonl",
        help="输出增强后 JSONL 路径（支持断点续跑追加）"
    )
    ap.add_argument("--video-dir", default="downloaded_vids_0127", help="视频保存目录（默认：downloaded_vids）")
    ap.add_argument("--frame-dir", default="downloaded_frames_0127", help="帧图像保存目录（默认：downloaded_frames）")
    ap.add_argument("--height", type=int, default=512, help="抽帧高度（默认：512）")
    ap.add_argument("--quality", type=int, default=8, help="JPG 质量 2-31，越小质量越高（默认：8）")
    ap.add_argument("--exact", action="store_true", help="使用更精确但更慢的 ffmpeg 抽帧方式")

    # NEW: batch rows
    ap.add_argument(
        "--input-rows",
        type=parse_input_rows,
        required=True,
        help='只处理输入 JSONL 的行范围 [start,end)，格式 "start,end"，例如 "0,1000"'
    )

    # NEW: speed knobs
    ap.add_argument("--flush-every", type=int, default=200,
                    help="每写入多少行 flush 一次（默认：200）。越大越快但中断丢失的行会更多（可用断点续跑补回）。")
    ap.add_argument("--fsync-every", type=int, default=0,
                    help="每写入多少行 fsync 一次（默认：0=不fsync）。fsync 很慢，除非你非常需要强一致性。")

    args = ap.parse_args()

    process(
        input_path=args.input,
        output_path=args.output,
        video_dir=args.video_dir,
        frame_dir=args.frame_dir,
        height=args.height,
        quality=args.quality,
        exact=args.exact,
        input_rows=args.input_rows,
        flush_every=args.flush_every,
        fsync_every=args.fsync_every,
    )


if __name__ == "__main__":
    main()