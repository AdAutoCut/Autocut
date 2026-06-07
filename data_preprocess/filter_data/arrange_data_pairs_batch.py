import re
import json
import os
import time
import base64
import subprocess
import sys
import argparse
import threading
from loguru import logger
from pathlib import Path
from typing import List, Dict
from tools.parse import parse_multimodal
from tools.video import vtoken2vemb, vemb2frame, vemb2clip, init_video_model
from tools.common.blobstore.client import BlobStoreClient
from tools.common.blobstore import download_video_bytes
from tools.infer import predict, predict_online
from tools.preprocess import process_photo_id_videos
from concurrent.futures import ThreadPoolExecutor, as_completed

# 日志设置
logger.remove()
logger.add(lambda msg: print(msg, end=""), format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>\n")

def read_jsonl(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Warning: 跳过无法解析的行: {line[:50]}... 错误信息: {e}")
    return data

def extract_frame_resize_ffmpeg(video_path, t_sec, out_path, height=512, quality=8, exact=False):
    vf = f"scale=-2:{height}"
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error"]
    if exact:
        cmd += ["-i", video_path, "-ss", str(t_sec)]
    else:
        cmd += ["-ss", str(t_sec), "-i", video_path]
    cmd += ["-vf", vf, "-frames:v", "1", "-y", out_path]
    if Path(out_path).suffix.lower() in {".jpg", ".jpeg"}:
        cmd.extend(["-q:v", str(quality)])
    subprocess.run(cmd, check=True)

def download_vid(photo_id, video_dir):
    os.makedirs(video_dir, exist_ok=True)
    video_bytes = download_video_bytes(photo_id)
    if not video_bytes:
        raise RuntimeError(f"Failed to download video for photo_id {photo_id}")
    output_path = os.path.join(video_dir, f"{photo_id}.mp4")
    with open(output_path, "wb") as f:
        f.write(video_bytes)
    return output_path

def download_frame(frame_id, video_dir, frame_dir):
    f_photo_id = int(str(frame_id)[:-4])
    frame_time = int(str(frame_id)[-4:])
    video_path = os.path.join(video_dir, f"{f_photo_id}.mp4")
    frame_path = os.path.join(frame_dir, f"{frame_id}.jpg")

    if not os.path.exists(video_path):
        video_path = download_vid(f_photo_id, video_dir)

    extract_frame_resize_ffmpeg(video_path, frame_time, frame_path)
    os.remove(video_path)

def check_disk_space(path="/"):
    stat = os.statvfs(path)
    free_gb = stat.f_bavail * stat.f_frsize / 1024**3
    logger.info(f" Available disk space: {free_gb:.2f} GB")
    if free_gb < 50:
        logger.warning("Low disk space! < 50 GB remaining.")

def process_one_sample(sample, ad_idx, chunk_id, processed_global_ids, fout, fout_lock, video_dir, frame_dir):
    global_id = chunk_id * 10000 + ad_idx
    if global_id in processed_global_ids:
        return 0

    try:
        parsed_sample, _ = parse_multimodal(sample["text"])
        clips = parsed_sample["clips"]
    except Exception as e:
        logger.error(f"Parse error for ad {ad_idx}: {e}")
        return 0

    local_written = 0
    for clip_idx, clip in enumerate(clips):
        try:
            texts = clip.get("text", [])
            text = texts[0] if texts else None
            frames = clip.get("frame", [])
            frame_id = frames[0] if frames else None
            if frame_id:
                download_frame(frame_id, video_dir, frame_dir)

            rec = {
                "chunk_id": chunk_id,
                "ad_id": ad_idx,
                "global_id": global_id,
                "clip_id": clip_idx,
                "text": text,
                "frame_id": frame_id
            }
            with fout_lock:
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()
            local_written += 1
        except Exception as e:
            logger.error(f"Failed to process clip {clip_idx} of ad {ad_idx}: {e}")

    if ad_idx % 100 == 0:
        check_disk_space()

    return local_written

def arrange_data_pair(samples: List[Dict], out_dir: str, chunk_id: int, max_workers: int = 64) -> int:
    os.makedirs(out_dir, exist_ok=True)
    video_dir = os.path.join(out_dir, "downloaded_vids")
    frame_dir = os.path.join(out_dir, "extracted_frames")
    os.makedirs(video_dir, exist_ok=True)
    os.makedirs(frame_dir, exist_ok=True)
    out_jsonl = os.path.join(out_dir, "clips_pairs.jsonl")

    processed_global_ids = set()
    if os.path.exists(out_jsonl):
        with open(out_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    record = json.loads(line)
                    processed_global_ids.add(record["global_id"])
                except Exception:
                    continue
        logger.info(f"[Chunk {chunk_id}] Found {len(processed_global_ids)} already processed global_ids.")

    video_rqvae_model, video_index, frame_index, clip_index = init_video_model(
        load_faiss=True, model_name="video_8_256_0729"
    )

    fout = open(out_jsonl, "a", encoding="utf-8")
    fout_lock = threading.Lock()
    total_written = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                process_one_sample,
                sample, ad_idx, chunk_id,
                processed_global_ids,
                fout, fout_lock,
                video_dir, frame_dir
            )
            for ad_idx, sample in enumerate(samples)
        ]

        for i, future in enumerate(as_completed(futures), 1):
            try:
                written = future.result()
                total_written += written
                logger.info(f"[{i}/{len(futures)}] Sample written: {written} records")
            except Exception as e:
                logger.error(f"Thread failed: {e}")

    fout.close()
    return total_written

def main():
    parser = argparse.ArgumentParser(description="Process one data chunk.")
    parser.add_argument("--chunk_id", type=int, required=True)
    parser.add_argument("--max_workers", type=int, default=64)
    args = parser.parse_args()

    chunk_id = args.chunk_id
    max_workers = args.max_workers
    input_path = f"raw_data/raw_chunk_{chunk_id}.jsonl"
    output_dir = f"cp_results/cp_chunk{chunk_id}"

    raw_samples = read_jsonl(input_path)
    logger.info(f"Loaded {len(raw_samples)} samples from chunk {chunk_id}")
    cnt = arrange_data_pair(raw_samples, output_dir, chunk_id, max_workers)
    logger.success(f"Done processing chunk {chunk_id}, {cnt} clips written.")

if __name__ == "__main__":
    main()
