import subprocess
import tempfile
import torch
import os
from loguru import logger
import numpy as np
import faiss
import random
from .video import video2vemb, vemb2vtoken
from .common.blobstore import download_video_bytes
from .common.utils import threshold_cos_partition, tensor_to_str_matrix


def get_video_clips(video_bytes, interval_s=0.1, threshold=0.85):
    frame_dict = video2vemb(video_bytes, interval_s=interval_s)
    frame_dict_integer = {k: v for k, v in frame_dict.items() if k.is_integer()}
    vtokens = vemb2vtoken(frame_dict_integer)
    vtoken_str_list = tensor_to_str_matrix(vtokens)
    # print(vtoken_str_list)
    # print(len(vtoken_str_list))
    frame_tuple_list = [(k, v) for k, v in frame_dict.items()]
    frame_tuple_list = sorted(frame_tuple_list, key=lambda x: x[0])
    time_list, frame_list = zip(*frame_tuple_list)
    time_list = list(time_list)
    frame_list = list(frame_list)
    frame_tensor = torch.stack([torch.tensor(i) for i in frame_list])
    cuts = threshold_cos_partition(frame_tensor, thr=threshold)
    cuts.append(len(time_list))
    time_list.append(time_list[-1] + interval_s)
    start_cut = 0
    clips = []
    input_str_list = []
    idx = 0
    for cut in cuts:
        start_time = time_list[start_cut]
        end_time = time_list[cut]
        duration_time = end_time - start_time
        clip_tensor = frame_tensor[start_cut:cut].mean(dim=0)
        start_cut = cut
        clips.append((start_time, duration_time, clip_tensor))
        if idx > end_time:
            continue
        input_str = ""
        input_str += "<|clip_start|><|video_start|>"
        while idx < end_time:
            input_str += "<|frame_start|>"
            input_str += vtoken_str_list[idx]
            input_str += "<|frame_end|>"
            idx += 1
        input_str += "<|video_end|><|clip_end|>"
        input_str_list.append(input_str)
    return clips, input_str_list


def process_photo_id_videos(photo_id_list, temp_dir="temp_video", use_gpu=False, shuffle=False, interval_s=0.1, threshold=0.85):
    os.makedirs(temp_dir, exist_ok=True)
    clip_keys = []
    embs = []
    all_str_list = []
    for i, photo_id in enumerate(photo_id_list):
        video_bytes = download_video_bytes(photo_id)
        with open(os.path.join(temp_dir, f"{i}.mp4"), "wb") as f:
            f.write(video_bytes)
        clips, input_str_list = get_video_clips(video_bytes, interval_s=interval_s, threshold=threshold)
        all_str_list.extend(input_str_list)
        for clip in clips:
            clip_key = int(i * (10 ** 11) + clip[0] * (10 ** 7) + clip[1] * (10 ** 2))
            clip_keys.append(clip_key)
            embs.append(clip[2])
    emb = torch.stack(embs, dim=0)
    ids = np.array(clip_keys)
    emb_np = emb.cpu().numpy().astype('float32')
    norms = np.linalg.norm(emb_np, axis=1, keepdims=True)
    emb_np = emb_np / (norms + 1e-10)
    d = emb_np.shape[1]
    index = faiss.IndexFlatIP(d)         # IP = inner product
    index = faiss.IndexIDMap(index)      # to keep track of your original IDs
    if use_gpu:
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)
    index.add_with_ids(emb_np, ids)
    if use_gpu:
        index = faiss.index_gpu_to_cpu(index)
    if shuffle:
        random.shuffle(all_str_list)
    all_str = "".join(all_str_list)
    return index, all_str
