import os
import sys
import json
import argparse
import subprocess
from typing import Set

from tools.common.blobstore import download_video_bytes


def frame_to_pid(frame_id) -> int:
    """去掉后四位 -> pid（兼容字符串或整数）"""
    return int(str(frame_id)) // 10000


def collect_pids(jsonl_path: str) -> Set[int]:
    """逐行读取 jsonl，提取 frame_id -> pid，去重"""
    pids: Set[int] = set()
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if "frame_id" in obj:
                    pids.add(frame_to_pid(obj["frame_id"]))
            except Exception:
                # 跳过坏行
                continue
    return pids


def download_one(pid: int, out_dir: str, overwrite: bool = False) -> bool:
    """下载单个视频到 out_dir/{pid}.mp4，返回是否成功"""
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{pid}.mp4")
    if (not overwrite) and os.path.exists(out_path):
        print(f"[skip] exists: {out_path}")
        return True
    data = download_video_bytes(pid)
    if not data:
        print(f"[fail] download pid={pid}", file=sys.stderr)
        return False
    with open(out_path, "wb") as f:
        f.write(data)
    print(f"[ok] {out_path}")
    return True


def extract_audio_one(pid: int, video_dir: str, audio_dir: str, sr: int = 44100, ch: int = 2) -> bool:
    """用 ffmpeg 从 {pid}.mp4 抽取音频为 audio_dir/{pid}.wav"""
    os.makedirs(audio_dir, exist_ok=True)
    vpath = os.path.join(video_dir, f"{pid}.mp4")
    apath = os.path.join(audio_dir, f"{pid}.wav")
    if not os.path.exists(vpath):
        print(f"[skip] no video: {vpath}")
        return False
    if os.path.exists(apath):
        print(f"[skip] exists: {apath}")
        return True
    cmd = [
        "ffmpeg", "-y", "-i", vpath,
        "-vn", "-acodec", "pcm_s16le",
        "-ar", str(sr), "-ac", str(ch),
        apath
    ]
    proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if proc.returncode == 0 and os.path.exists(apath):
        print(f"[ok] {apath}")
        return True
    print(f"[fail] extract pid={pid}", file=sys.stderr)
    return False


def main():
    ap = argparse.ArgumentParser(description="从 JSONL 的 frame_id 批量下载视频并抽取音频（frame_id 去掉后4位作为 pid）")
    ap.add_argument("--jsonl", default="/data/phd/miltonzhou/sft/data_preprocess/filter_data/BLCT_TEST_1112.jsonl",
                    help="输入 JSONL 文件路径（每行含 frame_id）")
    ap.add_argument("--output-dir", default="downloaded_vids", help="视频保存目录")
    ap.add_argument("--audio-dir", default="audio_wavs", help="音频输出目录（WAV）")
    ap.add_argument("--overwrite", action="store_true", help="如已存在视频，是否覆盖下载")
    ap.add_argument("--skip-download", action="store_true", help="跳过下载，仅抽取音频")
    ap.add_argument("--skip-audio", action="store_true", help="仅下载视频，不抽取音频")
    ap.add_argument("--sr", type=int, default=44100, help="WAV 采样率")
    ap.add_argument("--ch", type=int, default=2, help="WAV 声道数")
    args = ap.parse_args()

    pids = collect_pids(args.jsonl)
    if not pids:
        print("未从 JSONL 中解析到任何 pid（检查 frame_id 字段）", file=sys.stderr)
        sys.exit(1)

    print(f"共解析到 {len(pids)} 个去重后的 pid")

    # 1) 下载视频
    ok_dl = 0
    if not args.skip_download:
        print("开始下载视频...")
        for pid in sorted(pids):
            ok_dl += download_one(pid, args.output_dir, args.overwrite)
        print(f"下载完成：成功 {ok_dl}/{len(pids)}")

    # 2) 抽取音频
    ok_wav = 0
    if not args.skip_audio:
        print("开始抽取音频(WAV)...")
        for pid in sorted(pids):
            ok_wav += extract_audio_one(pid, args.output_dir, args.audio_dir, args.sr, args.ch)
        print(f"音频抽取完成：成功 {ok_wav}/{len(pids)}")

    print("完成。")


if __name__ == "__main__":
    main()
