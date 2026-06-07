#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：test1 
@File    ：call_transnet.py
@Author  ：wangminquan <wangminquan@kuaishou.com>
@Date    ：2025-03-26
'''
import sys
from loguru import logger
import glob
import subprocess
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed, TimeoutError
import os
import uuid
import time
import json
import argparse
import torch
import re
import tempfile
from io import BytesIO
import pandas as pd
from pydub import AudioSegment
import torchaudio
from google.protobuf.json_format import MessageToDict

from kess.framework import ClientOption, GrpcClient, KessOption
from mars.protos.model_serving_pb2 import PredictRequest, MetaInfo
from mars.protos.model_serving_pb2_grpc import ModelServingStub
from video.utils.blobstore import BlobStoreClient
from video.client.photo_asr_mmu_client import PhotoAsrMmuClient
from utils.llm import process_one_asr
from audio.client import VocalSplitClient
from audio.Vocal_split_pb2 import VocalSplitRequest
from panns_inference_local import AudioTagging

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


logger.remove()
logger.add(sys.stderr, enqueue=True, level='WARNING')
logger.add(f"{time.time()}.log", enqueue=True, level='INFO')

BLOB_VIDEO_CLIENT = None
BLOB_AUDIO_CLIENT = None
VOCAL_SPLIT_CLIENT = None
EMB_CLIENT = None
ASR_CLIENT = None
CONFIDENCE_THRESHOLD = 0.85


class Client:
    def __init__(self):
        self.client_option = ClientOption(
            biz_def='ad',
            grpc_service_name='grpc-mmu-cnnv2-emb-example-T4',   #被调服务（是否到瓶颈） grpc-mmu-cnnv2-emb-example-4090, grpc-mmu-cnnv2-emb-example-T4-ad, grpc-mmu-cnnv2-emb-example-T4
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
                    logger.warning(f"{video_name} [Attempt {attempt}] status={resp_dict.get('status')}, retrying...")
            except Exception as e:
                logger.warning(f"{video_name} [Attempt {attempt}] Exception: {e}, retrying...")
            time.sleep(0.1*attempt)
        # 所有重试都失败
        logger.warning(f"sync_run: all {max_retries} attempts failed for {video_name}.mp4")
        return None


def get_time_str():
    current_time = time.time()
    duration = current_time - START_TIME
    time_str = time.strftime("%H:%M:%S", time.gmtime(duration))
    return time_str


def initialize_client():
    global EMB_CLIENT, ASR_CLIENT, VOCAL_SPLIT_CLIENT
    global BLOB_VIDEO_CLIENT, BLOB_AUDIO_CLIENT
    global AUDIOTAGGING
    BLOB_VIDEO_CLIENT = BlobStoreClient('video-def')
    BLOB_AUDIO_CLIENT = BlobStoreClient('ad-nieuwland-material')
    VOCAL_SPLIT_CLIENT = VocalSplitClient()
    EMB_CLIENT = Client()
    ASR_CLIENT = PhotoAsrMmuClient()
    AUDIOTAGGING = AudioTagging(checkpoint_path='./panns_data/audiotagging.pth', device='cuda')


def extract_json_list_from_text(text):
    matches = re.findall(r"\[.*?\]", text, re.DOTALL)
    if matches:
        return json.loads(matches[0])
    else:
        return []


def get_asr(photo_id):
    """
    获取ASR并调用LLM修正
    """
    try:
        raw_asr = ASR_CLIENT._sync_run(photo_id=int(photo_id))
        fixed_asr_ori = list(raw_asr.get('speechResult', {}).values())[0].get('speechText')
    except Exception as e:
        # logger.warning(f"{photo_id} [处理失败_asr_client] {e}")
        # logger.warning(raw_asr)
        # return fixed_asr_ori, []
        raise Exception(f"{photo_id} [处理失败_asr_client] {e} {raw_asr}")
    attempt = 3
    while attempt:
        try:
            attempt -= 1
            fixed_asr = process_one_asr(raw_asr)
            fixed_asr = extract_json_list_from_text(fixed_asr)
            fixed_asr = [asr for asr in fixed_asr if asr.get("confidence", 0) > CONFIDENCE_THRESHOLD]
            logger.info(f'asr-{photo_id}:' + get_time_str())
            return fixed_asr_ori, fixed_asr
        except Exception as e:
            logger.warning(f"{photo_id} [asr 剩余ATTEMPT次数： {attempt}]")
    raise Exception(f"[处理失败_asr_llm]")


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


def extract_audio_from_bytes(video_bytes, photo_id):
    try:
        video_buf = BytesIO(video_bytes)
        audio_buf = BytesIO()
        AudioSegment.from_file(video_buf).export(audio_buf, format='mp3')
        audio = audio_buf.getvalue()
    except Exception as e:
        raise Exception(f"[处理失败_extract_audio] {e}")
    logger.info(f'get_audio-{photo_id}:' + get_time_str())
    return audio


def extract_frames_from_bytes(video_bytes, photo_id, interval_s=1):
    # 写 tmp 文件
    try:
        tmp = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
        path = tmp.name
        tmp.write(video_bytes)
        tmp.close()
        cmd = [
            'ffmpeg', '-threads', '8',   # 用 4 线程解码
            '-i', path,
            '-vf', f'fps=1/{interval_s}',  # 每秒抽 1 帧
            '-f', 'image2pipe', '-vcodec', 'png', 'pipe:1'
        ]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        data, err = proc.communicate()
    except:
        raise Exception("[处理失败_extract_frames] ffmpeg error")
    finally:
        if path and os.path.exists(path):
            os.remove(path)
    if data is None or len(data) < 10:
        raise Exception("[处理失败_extract_frames] ffmpeg result error")
    frames = split_png_stream(data)
    if not frames:
        raise Exception(f"[处理失败_not_pngs]")
    logger.info(f'get_frames-{photo_id}:' + get_time_str())
    return frames


def download_video(photo_id):
    """
    给定photo_id，从blobstore下载视频，按 interval_ms 截帧，
    返回 (video_name, [png_bytes, ...])
    """
    photo_key = f"{photo_id}.mp4"
    is_downloaded, video_bytes = BLOB_VIDEO_CLIENT.download_bytes_from_s3(photo_key)
    if video_bytes is None or len(video_bytes) < 10:
        raise Exception("[处理失败_download_video] video bytes is None")
    logger.info(f'download-{photo_id}:' + get_time_str())
    return video_bytes


def get_frame_embs(photo_id, png_list):
    """
    args 是 (video_name, [png_bytes, ...])
    返回 {embed_id: tensor(embed), ...}
    """
    try:
        embed_dict = {}
        for idx, img_bytes in enumerate(png_list):
            emb = EMB_CLIENT.sync_run(img_bytes, photo_id)
            if emb is None:
                break
            embed_id = int(f"{idx:04d}")
            embed_dict[embed_id] = torch.tensor(emb)
    except Exception as e:
        raise Exception(f"[处理失败_get_frame_embs {e}")
    logger.info(f'get_frame_embs-{photo_id}:' + get_time_str())
    return embed_dict


def if_cached(photo_id):
    return CHUNK_INDEX.get(photo_id, None)


def get_cached_frame_embs(photo_id, chunk_idx):
    chunk_data = torch.load(os.path.join("data/incycle_45000_fps1_new", chunk_idx), map_location=MAP_LOCATION)
    embs = chunk_data[photo_id]
    return embs


def get_bgm_embs(wav_bytes, thr=0.5):
    buf = BytesIO(wav_bytes)
    y, sr = torchaudio.load(buf)
    y = y.mean(dim=0, keepdim=True)
    (clipwise_output, embedding) = AUDIOTAGGING.inference(y)
    is_music = float(clipwise_output[0, 137])  # 137是music
    return is_music > 0.5, embedding


def get_audio_embs(photo_id, audio):
    try:
        BLOB_AUDIO_CLIENT.upload_bytes_to_s3(audio, f"{photo_id}.mp3")
        req = VocalSplitRequest(vocal_blob_key={"db": "ad", "table": "nieuwland-material", "key": f"{photo_id}.mp3"})
        resp = VOCAL_SPLIT_CLIENT.sync_run(req)
    except Exception as e:
        raise Exception(f"[处理失败_vocal_split] {e}")
    if resp:
        no_vocal_part = resp.get('res', {}).get('noVocalPart', {}).get('key', {})
    else:
        raise Exception("[处理失败_vocal_split] no resp")
    downloaded, bgm_bytes = BLOB_AUDIO_CLIENT.download_bytes_from_s3(no_vocal_part)
    if downloaded is False or bgm_bytes is None:
        raise Exception("[处理失败_no_vocal_part download]")
    logger.info(f'vocal_split-{photo_id}:' + get_time_str())
    try:
        is_bgm, embedding = get_bgm_embs(bgm_bytes)
        logger.info(f'get_bgm_embs-{photo_id}:' + get_time_str())
    except Exception as e:
        raise Exception(f"[处理失败_get_bgm_embs] {e}")
    if is_bgm:
        return embedding, no_vocal_part
    else:
        return None, no_vocal_part


def process_all(photo_id, chunk_id):
    """同时处理asr，音频，视频"""
    info_path = os.path.join(OUTPUT_DIR, f'chunk{chunk_id}', f'{photo_id}.pt')
    if os.path.exists(info_path):
        return info_path
    try:
        video_info = {"photo_id": photo_id}
        tpool = ThreadPoolExecutor(max_workers=4)
        logger.info(f'start-{photo_id}:' + get_time_str())
        # f_cache = tpool.submit(if_cached, photo_id)
        f_asr = tpool.submit(get_asr, photo_id)
        f_download = tpool.submit(download_video, photo_id)
        # chunk_idx = f_cache.result()
        chunk_idx = None
        video_bytes = f_download.result()
        f_extract_audio = tpool.submit(extract_audio_from_bytes, video_bytes, photo_id)
        audio = f_extract_audio.result()
        if chunk_idx:
            f_femb = tpool.submit(get_cached_frame_embs, photo_id, chunk_idx)
            f_aemb = tpool.submit(get_audio_embs, photo_id, audio)
        else:
            f_extract_frame = tpool.submit(extract_frames_from_bytes, video_bytes, photo_id)
            f_aemb = tpool.submit(get_audio_embs, photo_id, audio)
            pngs = f_extract_frame.result()
            f_femb = tpool.submit(get_frame_embs, photo_id, pngs)
        frame_embs_dict = f_femb.result()
        audio_embs, no_vocal_part = f_aemb.result()
        ori_asr, asr = f_asr.result()    # 阻塞直到 ASR 完成
        video_info["ori_asr"] = ori_asr
        video_info["asr"] = asr
        video_info["audio"] = audio_embs if audio_embs is not None else {}
        video_info["no_vocal_part_path"] = no_vocal_part
        video_info["frames"] = frame_embs_dict
        tpool.shutdown()
        torch.save(video_info, info_path)
        return info_path
    except TimeoutError as e:
        logger.warning(f"[处理失败] {photo_id} → {e}线程超时")
        tpool.shutdown()
        return None
    except Exception as e:
        logger.warning(f"[处理失败] {photo_id} → {e}")
        tpool.shutdown()
        return None


def chunkify(lst, chunk_size):
    """将列表按 chunk_size 切分成多个子列表"""
    return {i: lst[i * chunk_size: (i + 1) * chunk_size] for i in range(0, int((len(lst) - 1) / chunk_size) + 1)}


if __name__ == "__main__":
    for var in ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy", "NO_PROXY", "no_proxy"):
        os.environ.pop(var, None)
    global START_TIME, CHUNK_INDEX, OUTPUT_DIR, DEVICE
    MAP_LOCATION = None if torch.cuda.is_available() else torch.device('cpu')
    START_TIME = time.time()
    CHUNK_INDEX = torch.load("data/incycle_45000_fps1_new/chunk_index.pt", map_location=MAP_LOCATION)
    # OUTPUT_DIR = "/data/phd/qinsizhong/video_process/inex_7_8/" ###### （改这个
    OUTPUT_DIR = "/data/phd/qinsizhong/video_process/inex_us/" ###### （改这个

    # global BLOB_VIDEO_CLIENT, BLOB_AUDIO_CLIENT, AUDIOTAGGING
    # BLOB_VIDEO_CLIENT = BlobStoreClient('video-def')
    # BLOB_AUDIO_CLIENT = BlobStoreClient('ad-nieuwland-material')
    # AUDIOTAGGING = AudioTagging(checkpoint_path='./panns_data/audiotagging.pth', device='cuda')

    # input_file = "/data/phd/qinsizhong/video_process/data/inex_7_8.csv" ##### 改（连续跑4个+）
    input_file = "/data/phd/qinsizhong/video_process/data/inex_us.csv" ##### 改（连续跑4个+）
    # output_dir = "/data/phd/qinsizhong/video_process/incyle/"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    data = pd.read_csv(input_file)
    photo_ids_all = data['photo_id'].to_list()
    chunk_size = 1000
    chunk_idx = chunkify(photo_ids_all, chunk_size)
    torch.save(chunk_idx, os.path.join(OUTPUT_DIR, "chunk_index.pt"))
    photo_id_length_ori = len(photo_ids_all)
    for chunk_id, chunk_ids in sorted(chunk_idx.items(), key=lambda pair: pair[0]):
        chunk_dict_dir = os.path.join(OUTPUT_DIR, f'chunk{chunk_id}')
        chunk_dict_path = os.path.join(OUTPUT_DIR, f'chunk{chunk_id}', f'chunk{chunk_id}.pt')
        if os.path.exists(chunk_dict_path) or os.path.exists(chunk_dict_dir):
            continue
            # if chunk_id!=7:
            # if os.path.exists(chunk_dict_path):
        # if os.path.exists(chunk_dict_path):
        #     continue
        logger.warning(f"chunk {chunk_id}/{len(chunk_idx)-1} ---- start")
        chunk_dict = {}
        os.makedirs(os.path.join(OUTPUT_DIR, f'chunk{chunk_id}'), exist_ok=True)
        suc_processed = 0
        processed = 0
        with ProcessPoolExecutor(max_workers=64, initializer=initialize_client) as disk_pool:
            futures = {disk_pool.submit(process_all, pid, chunk_id): pid for pid in chunk_ids}
            logger.warning(f"chunk {chunk_id} 累计任务数量{len(futures)}")
            for fut in as_completed(futures):
                pid = futures[fut]
                video_info_path = fut.result()
                if video_info_path:
                    video_info = torch.load(video_info_path, map_location=MAP_LOCATION)
                    chunk_dict[pid] = video_info
                    suc_processed += 1
                processed += 1
                current_time = get_time_str()
                logger.warning(f"chunk {chunk_id}/{len(chunk_idx)-1} ---- {suc_processed}/{processed}/{len(chunk_ids)} ---- [{current_time}] ---- {pid}")
        logger.warning(f"chunk {chunk_id}/{len(chunk_idx)-1} ---- shut down")
        if len(os.listdir(chunk_dict_dir)) == 0:
            os.remove(chunk_dict_dir)
        else:
            torch.save(chunk_dict, chunk_dict_path)
            logger.warning(f"chunk {chunk_id}/{len(chunk_idx)-1} ---- saved")
