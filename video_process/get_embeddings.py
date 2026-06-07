#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：test1 
@File    ：call_transnet.py
@Author  ：wangminquan <wangminquan@kuaishou.com>
@Date    ：2025-03-26
'''
import glob
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import os
import shutil
import uuid
import sys
import logging
import gc
import time

import numpy as np
import pandas as pd
from kess.framework import ClientOption, GrpcClient, KessOption
from mars.protos.model_serving_pb2 import PredictRequest, MetaInfo
from mars.protos.model_serving_pb2_grpc import ModelServingStub
from video.utils.blobstore import BlobStoreClient
import time
import json
import argparse
import cv2
import subprocess
import torch

import ffmpeg
from PIL import Image
import io
import tempfile


from google.protobuf.json_format import MessageToDict

logger = logging.getLogger(__name__)

BLOB_CLIENT = None
EMB_CLIENT = None


class Client:
    def __init__(self):
        self.client_option = ClientOption(
            biz_def='ad',
            grpc_service_name='grpc-mmu-cnnv2-emb-example-T4-ad',  # grpc-mmu-cnnv2-emb-example-4090, grpc-mmu-cnnv2-emb-example-T4-ad
            grpc_stub_class=ModelServingStub,
        )
        self.client = GrpcClient(self.client_option)

    def sync_run(self, img, video_name):
        max_retries = 5
        for attempt in range(1, max_retries + 1):
            try:
                uid = str(uuid.uuid4())
                req = PredictRequest(id=uid)
                req.media.data = img
                resp = self.client.Predict(req, timeout=5)
                resp_dict = MessageToDict(resp)
                if resp_dict.get("status") == "SUCCESS":
                    return resp_dict["feature"]["floatArray"]["floatElems"]
                else:
                    # 状态不成功，也算一次失败
                    logger.warning(f"[Attempt {attempt}] status={resp_dict.get('status')}, retrying...")
            except Exception as e:
                logger.warning(f"[Attempt {attempt}] Exception: {e}, retrying...")
            time.sleep(attempt * 0.1)
        # 所有重试都失败
        logger.warning(f"sync_run: all {max_retries} attempts failed for {video_name}.mp4")
        return None


def get_time_str(start_time):
    current_time = time.time()
    duration = current_time - start_time
    time_str = time.strftime("%H:%M:%S", time.gmtime(duration))
    return time_str


def initialize_client():
    global BLOB_CLIENT, EMB_CLIENT
    BLOB_CLIENT = BlobStoreClient('video-def')
    EMB_CLIENT = Client()


def extract_frames(video_path, interval_ms=1000):
    """
    读取一段视频，按 interval_ms 截帧，
    返回 (video_name, [png_bytes, ...])
    """
    video_name = os.path.basename(video_path).rsplit('.', 1)[0]
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = total / fps * 1000  # 毫秒
    frames = []
    t = 0
    while t < duration:
        cap.set(cv2.CAP_PROP_POS_MSEC, t)
        ok, frame = cap.read()
        if not ok:
            break
        _, buf = cv2.imencode('.png', frame)
        frames.append(buf.tobytes())
        t += interval_ms
    cap.release()
    return video_name, frames


def split_png_stream(data):
    """
    将连续的 PNG 流（多个 PNG 文件首尾相连）切分成独立的 PNG 字节块。
    """
    PNG_SIG = b'\x89PNG\r\n\x1a\n'
    # 找到所有 PNG 文件头的位置
    offsets = []
    idx = data.find(PNG_SIG)
    while idx != -1:
        offsets.append(idx)
        idx = data.find(PNG_SIG, idx + 1)

    frames = []
    for i, start in enumerate(offsets):
        end = offsets[i + 1] if i + 1 < len(offsets) else len(data)
        frames.append(data[start:end])
    return frames


def extract_frames_from_bytes(video_bytes, interval_s=1):
    # 写 tmp 文件
    tmp = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
    tmp.write(video_bytes)
    tmp.close()
    path = tmp.name
    cmd = [
        'ffmpeg', '-threads', '8',   # 用 4 线程解码
        '-i', path,
        '-vf', f'fps=1/{interval_s}',  # 每秒抽 1 帧
        '-f', 'image2pipe', '-vcodec', 'png', 'pipe:1'
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    data, err = proc.communicate()   # 一次性读完所有输出
    if data is None:
        raise Exception("ffmpeg error")
    os.remove(path)
    frames = split_png_stream(data)
    return frames


# def extract_frames_from_bytes(video_bytes, interval_s=1):
#     cmd = [
#         'ffmpeg', '-threads', '8',
#         '-i', 'pipe:0',
#         '-vf', f'fps=1/{interval_s}',  # 每秒抽 1 帧
#         '-f', 'image2pipe', '-vcodec', 'png', 'pipe:1'
#     ]
#     proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#     data, err = proc.communicate(video_bytes)   # 一次性读完所有输出
#     if data is None:
#         raise Exception("ffmpeg error")
#     if len(data)<2000:
#         logger.warning(f"data:{data}")
#     frames = split_png_stream(data)
#     return frames

def download_and_extract_frames(photo_id):
    """
    给定photo_id，从blobstore下载视频，按 interval_ms 截帧，
    返回 (video_name, [png_bytes, ...])
    """
    photo_key = f"{photo_id}.mp4"
    is_downloaded, video_bytes = BLOB_CLIENT.download_bytes_from_s3(photo_key)
    if video_bytes is None:
        raise RuntimeError("video bytes is None")
    video_name = str(photo_id)
    frames = extract_frames_from_bytes(video_bytes)
    return video_name, frames


def embed_frames(args):
    """
    args 是 (video_name, [png_bytes, ...])
    返回 {embed_id: tensor(embed), ...}
    """
    video_name, png_list = args
    embed_dict = {}
    for idx, img_bytes in enumerate(png_list):
        # logger.warning(uid)
        emb = EMB_CLIENT.sync_run(img_bytes, video_name)
        if emb is None:
            break
        embed_id = int(f"{video_name}{idx:04d}")
        embed_dict[embed_id] = torch.tensor(emb)
    return embed_dict


def process_video(video_path, download=False):
    """在子进程里：抽帧 -> 立刻嵌入 -> 返回 embedding dict"""
    try:
        if download:
            name, pngs = download_and_extract_frames(video_path)
        else:
            name, pngs = extract_frames(video_path)
    except Exception as e:
        raise Exception(f"[处理失败_download_and_extract] photo_id={video_path} → {e}")
    if not pngs:
        raise Exception(f"[处理失败_no_pngs] photo_id={video_path}")
    try:
        part = embed_frames((name, pngs))
    except Exception as e:
        raise Exception(f"[处理失败_embed_frames] photo_id={video_path} → {e}")
    finally:
        # 释放子进程里的内存
        del pngs
        gc.collect()
    return part


def chunkify(lst, chunk_size):
    """将列表按 chunk_size 切分成多个子列表"""
    return [lst[i: i + chunk_size] for i in range(0, len(lst), chunk_size)]


if __name__ == "__main__":
    # #####IF###### 视频已经下载好了，找视频
    # input_glob = "/home/qinsizhong/data/downloaded_videos/*.mp4"
    # videos = glob.glob(input_glob)
    # videos.sort()
    # videos = videos[:9000]
    # if not videos:
    #     logger.warning("未找到视频文件，退出。")
    #     exit()
    # processed_videos = {os.path.dirname(
    #     input_glob)+f'/{int(i/10000)}.mp4' for i in merged.keys()}
    # videos = [v for v in videos if not v in processed_videos]
    # chunks = chunkify(videos, 1000)
    # pbar = tqdm(total=len(videos), desc="Overall",
    #             bar_format='{l_bar}{bar}{r_bar}\n')
    # for chunk_idx, chunk_ids in enumerate(chunks, start=1):
    #     logger.warning(f"Processing chunk {chunk_idx}/{len(chunks)} ({len(chunk_ids)} videos)...")
    #     # 创建进程池
    #     disk_pool = ProcessPoolExecutor(max_workers=parallel_video_num)
    #     futures = {
    #         disk_pool.submit(process_video, v): v for v in chunk_ids
    #     }

    # #####ELSEIF##### 需要直接下载视频
    input_file = "/data/phd/qinsizhong/video_process/data/test.csv"
    data = pd.read_csv(input_file)
    photo_ids = data['photo_id'].to_list()
    # photo_ids = photo_ids[:45000]
    if not photo_ids:
        logger.warning("未找到视频文件，退出。")
        exit()
    # processed_photo_ids = {int(i/10000) for i in merged.keys()}
    # photo_ids = [pid for pid in photo_ids if not pid in processed_photo_ids]
    chunk_len = 500
    chunks = chunkify(photo_ids, chunk_len)
    process = 0
    total_len = len(photo_ids)
    start_time = time.time()
    disk_pool = ProcessPoolExecutor(max_workers=32, initializer=initialize_client)
    # pbar = tqdm(total=len(photo_ids), desc="Overall", bar_format='{l_bar}{bar}{r_bar}\n')
    for chunk_idx, chunk_ids in enumerate(chunks, start=1):
        # with ProcessPoolExecutor(max_workers=parallel_video_num, initializer=initialize_client) as disk_pool:
        out_file = f"/home/qinsizhong/data/test_fps1/test_fps1_chunk{chunk_idx}.pt"
        # 处理缓存数据
        if os.path.exists(out_file):
            process += len(chunk_ids)
            logger.warning(f"chunk{chunk_idx} is processed.")
            time_str = get_time_str(start_time)
            logger.warning(f"{process}/{total_len} [{time_str}]")
            continue
        else:
            merged = {}
        logger.warning(f"Processing chunk {chunk_idx}/{len(chunks)} ({len(chunk_ids)} videos)...")
        futures = {
            disk_pool.submit(process_video, pid, True): pid for pid in chunk_ids
        }
        for fut in as_completed(futures):
            try:
                video = futures[fut]
                part = fut.result()
                merged.update(part)
                logger.warning(f"video: {video}.mp4 || frame number:{len(part)}")
            except Exception as e:
                logger.warning(f"[处理失败] {video} → {e}")
            finally:
                process += 1
                time_str = get_time_str(start_time)
                logger.warning(f"{process}/{total_len} [{time_str}]")
                gc.collect()
        torch.save(merged, out_file)
        logger.warning(f"完成chunk{chunk_idx}，保存 {len(merged)} 条 embedding 到 {out_file}")
    disk_pool.shutdown(wait=True)
