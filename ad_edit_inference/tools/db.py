import os
import sys
import torch
import torch.nn.functional as F
import json
import jsonlines
import re
import faiss
import numpy as np
import random
from tqdm import tqdm
from loguru import logger
from .video import vemb2vtoken, init_video_model
from .audio import aemb2atoken, init_audio_model
from .common.utils import threshold_cos_partition, tensor_to_str_matrix
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
from typing import List, Dict, Any, Tuple, Union
# import multiprocessing as mp
# mp.set_start_method('spawn', force=True)
# from torch.utils.data import Dataset, DataLoader, DistributedSampler

logger.remove()
logger.add(sys.stderr, enqueue=True, level="INFO")


def drop_all_zero(tensor: torch.Tensor, dim: int):
    """
    删除 tensor 在第 dim 维度上，那些“整个切片都是零”的条目。
    例如：
      x.shape = (N, D), dim=0 时删掉全零的行；dim=1 时删掉全零的列。
    """
    # 1) 在 dim 轴上检测全零：对“除 dim 外所有维度”求 all()
    reduce_dims = tuple(i for i in range(tensor.ndim) if i != dim)
    # mask.shape == (tensor.shape[dim],)
    mask = ~torch.all(tensor == 0, dim=reduce_dims)

    # 2) 构造索引：在 dim 位置用 mask，其它维度全部用 slice(None)
    idx = [slice(None)] * tensor.ndim
    idx[dim] = mask

    # 3) 用 tuple(idx) 索引
    return tensor[tuple(idx)]


def split_asr(asr):
    start_time = float(asr.get('startTime', None))
    end_time = float(asr.get('endTime', None))
    text = asr.get('text', None)
    if start_time is not None and end_time is not None and text:
        parts = re.split(r'([，。？！；,!?;])', text)
        # 将分片重组为带标点的句子
        segments = []
        for i in range(0, len(parts) - 1, 2):
            seg = parts[i].strip() + parts[i + 1]
            if seg:
                segments.append(seg)
        # 如果最后一段没有尾标点，也加进去
        if len(parts) % 2 == 1 and parts[-1].strip():
            segments.append(parts[-1].strip())

        # 2. 计算总字符数（可根据需求排除标点）
        total_chars = sum(len(seg) for seg in segments)
        if total_chars == 0:
            return []

        # 3. 按比例分配时间
        total_duration = end_time - start_time
        results = []
        start_time_list = []
        end_time_list = []
        current_time = start_time
        for seg in segments:
            seg_duration = total_duration * (len(seg) / total_chars)
            seg_start = current_time
            seg_end = current_time + seg_duration
            results.append((seg, seg_start, seg_end))
            current_time = seg_end
        return results
    else:
        return []


def build_str(v):
    text = "<|ad_start|>"
    asr_split_list = []
    if v['audio'] == {}:
        return None
    if v['asr'] != [] and v.get('ori_asr', [1]) != []:
        asr_list = v['asr']
        try:
            ch = asr_list[0]['text'][0]
            if 'A' <= ch <= 'Z' or 'a' <= ch <= 'z':
                return None
        except Exception as e:
            photo_id = v["photo_id"]
            print(f"{photo_id},{e}")
            print(asr_list)
            return None
        for asr in asr_list:
            try:
                asr_split_list.extend(split_asr(asr))
            except Exception as e:
                photo_id = v["photo_id"]
                print(f"{photo_id},{e}")
    if len(asr_split_list) == 0:
        return None
    try:
        atokens = aemb2atoken(v['audio'])
        atoken_str = tensor_to_str_matrix(atokens.unsqueeze(0), modality="audio")[0]
        frame_tokens = vemb2vtoken(v['frames'])
        frame_tokens_str_list = tensor_to_str_matrix(frame_tokens)
    except Exception as e:
        photo_id = v['photo_id']
        logger.warning(f"{photo_id}: Error {e}")
        return None
    ct = 0
    for asr, st, et in asr_split_list:
        if int(ct) < int(st) and int(st) != 0:  # 如果没有匹配文本
            text += "<|clip_start|>"
            text += "<|video_start|>"
            for frame_tokens_str in frame_tokens_str_list[int(ct) + 1: int(st) + 1]:
                text += "<|frame_start|>"
                text += frame_tokens_str
                text += "<|frame_end|>"
            text += "<|video_end|>"
            text += "<|clip_end|>"
        if int(st) == 0:  # 如果初始st，从第0帧开始
            st = -1
        text += "<|clip_start|>"
        if int(st) < int(et):
            text += "<|text_start|>"
            text += asr
            text += "<|text_end|>"
            text += "<|video_start|>"
            for frame_tokens_str in frame_tokens_str_list[int(st) + 1: int(et) + 1]:
                text += "<|frame_start|>"
                text += frame_tokens_str
                text += "<|frame_end|>"
            text += "<|video_end|>"
        elif int(st) == int(et):
            text += "<|text_start|>"
            text += asr
            text += "<|text_end|>"
        else:
            logger.warning(f"{photo_id}: Error st > et")
            return None
        text += "<|clip_end|>"
        ct = et
    # 一直到视频末尾
    et = len(v['frames']) - 1
    if ct < et:
        text += "<|clip_start|>"
        text += "<|video_start|>"
        for frame_tokens_str in frame_tokens_str_list[int(ct) + 1:int(et) + 1]:
            text += "<|frame_start|>"
            text += frame_tokens_str
            text += "<|frame_end|>"
        text += "<|video_end|>"
        text += "<|clip_end|>"
    text += "<|audio_start|>"
    text += atoken_str
    text += "<|audio_end|>"
    text += "<|ad_end|>"
    return text


def build_chunk_str(retrieve_dir, chunk_id, eval_photo_ids={}, saved_path="excycle/txt-0722"):
    name_id = os.path.basename(retrieve_dir)
    train_path = os.path.join(saved_path, f"{name_id}_{chunk_id}_train.jsonl")
    eval_path = os.path.join(saved_path, f"{name_id}_{chunk_id}_eval.jsonl")
    temp_flag = os.path.join(saved_path, f"{name_id}_{chunk_id}")
    try:
        if os.path.exists(temp_flag) or os.path.exists(train_path):
            return
        else:
            os.makedirs(temp_flag)
    except:
        return
    text_list_train = []
    text_list_eval = []
    chunk_dir = os.path.join(retrieve_dir, f'chunk{chunk_id}')
    chunk_info_path = os.path.join(chunk_dir, f'chunk{chunk_id}.pt')
    chunk_info = torch.load(chunk_info_path, map_location=torch.device('cpu'))
    for photo_id, v in chunk_info.items():
        text = build_str(v)
        if text and photo_id not in eval_photo_ids:
            text_list_train.append({"text": text})
        if text and photo_id in eval_photo_ids:
            text_list_eval.append({"text": text})
    with jsonlines.open(train_path, mode='w') as writer:
        writer.write_all(text_list_train)
    with jsonlines.open(eval_path, mode='w') as writer:
        writer.write_all(text_list_eval)
    logger.info(f"{name_id}_{chunk_id} done")


def build_faiss_cosine_index(tensor_path, mapping_json, index_path, use_gpu=False, frame=False, norm=False):
    # 1. Load and normalize
    db = torch.load(tensor_path, map_location="cpu")              # shape (N, D)
    db_np = db.numpy().astype('float32')
    # normalize each row to unit length
    if norm:
        norms = np.linalg.norm(db_np, axis=1, keepdims=True)
        db_np = db_np / (norms + 1e-10)

    # 2. Load ID mapping
    with open(mapping_json, 'r') as f:
        id_map = json.load(f)
    if frame:
        ids = np.array([i[0] * 10000 + i[1] for i in id_map.values()], dtype='int64')
    else:
        ids = np.array([int(i) for i in id_map.values()], dtype='int64')

    # 3. Build an inner‑product index
    d = db_np.shape[1]
    index = faiss.IndexFlatIP(d)         # IP = inner product
    index = faiss.IndexIDMap(index)      # to keep track of your original IDs

    # 4. (Optional) move to GPU
    if use_gpu:
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)

    # 5. Add normalized vectors + IDs
    index.add_with_ids(db_np, ids)

    # 6. Save back on CPU
    if use_gpu:
        index = faiss.index_gpu_to_cpu(index)
    faiss.write_index(index, index_path)
    print(f"Cosine‑index built, {index.ntotal} vectors, saved to {index_path}")
    return index


class RetrieveDataset():
    def __init__(
            self,
            retrieve_dir: Union[str, List[str]] = ['/data/phd/qinsizhong/video_process/excycle', '/data/phd/qinsizhong/video_process/incycle', '/data/phd/qinsizhong/video_process/incycle_2'],
            save_dir="ex_incycle"
    ):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        if type(retrieve_dir) is str:
            chunk_index_path = os.path.join(retrieve_dir, "chunk_index.pt")
            self.retrieve_dir_list = [retrieve_dir]
            self.chunk_index_list = [torch.load(chunk_index_path, map_location=torch.device('cpu'))]
        elif type(retrieve_dir) is list:
            chunk_index_path = [os.path.join(r, "chunk_index.pt") for r in retrieve_dir]
            self.retrieve_dir_list = retrieve_dir
            self.chunk_index_list = [torch.load(p, map_location=torch.device('cpu')) for p in chunk_index_path]

    def get_db(self, threshold=0.85):
        videoidx2photoid = {}
        clipidx2photo_clipid = {}
        frameidx2photo_frameid = {}
        frame_list = []
        clip_list = []
        video_list = []
        audio_list = []
        text_list = []
        video_idx = 0
        clip_idx = 0
        frame_idx = 0
        for retrieve_dir, chunk_index in zip(self.retrieve_dir_list, self.chunk_index_list):
            for chunk_id in chunk_index:
                chunk_dir = os.path.join(retrieve_dir, f'chunk{chunk_id}')
                chunk_info_path = os.path.join(chunk_dir, f'chunk{chunk_id}.pt')
                chunk_info = torch.load(chunk_info_path, map_location=torch.device('cpu'))
                for photo_id, v in chunk_info.items():
                    try:
                        frame_dict = v['frames']
                        if not v['frames']:
                            logger.warning(f"{chunk_id}-{photo_id}")
                            continue
                        frames = []
                        frame_ids = sorted(frame_dict.keys())
                        for frame_id in frame_ids:
                            frames.append(frame_dict[frame_id])
                            frameidx2photo_frameid[frame_idx] = [photo_id, frame_id]
                            frame_idx += 1
                        video_tensor = torch.stack(frames)
                        cut_time_list = threshold_cos_partition(video_tensor, thr=threshold)
                        cut_time_list.append(len(frame_ids))
                        start = 0
                        for end in cut_time_list:
                            clip_key = photo_id * 10000000 + start * 1000 + (end - start)
                            if (end - start) >= 1000:
                                raise Exception("Clips Too Long!!!!")
                            clip_emb = video_tensor[start:end].mean(dim=0)
                            clip_list.append(clip_emb)
                            clipidx2photo_clipid[clip_idx] = clip_key
                            clip_idx += 1
                            start = end
                        video_mean = torch.mean(video_tensor, dim=0)
                        video_list.append(video_mean)
                        frame_list.extend(frames)
                        if v['audio'] != {}:
                            audio_list.append(v['audio'])
                        else:
                            audio_list.append(torch.zeros(2048))
                        videoidx2photoid[video_idx] = photo_id
                        video_idx += 1
                    except Exception as e:
                        logger.warning(f"{retrieve_dir}-{chunk_id}-{photo_id}")
                        logger.warning(e)
                logger.info(f"{retrieve_dir}-{chunk_id} done")
        with open(os.path.join(self.save_dir, 'v.json'), 'w') as f:
            json.dump(videoidx2photoid, f, indent=4)
        with open(os.path.join(self.save_dir, 'f.json'), 'w') as f:
            json.dump(frameidx2photo_frameid, f, indent=4)
        thr_str = str(int(threshold * 100))
        with open(os.path.join(self.save_dir, f'c_{thr_str}.json'), 'w') as f:
            json.dump(clipidx2photo_clipid, f, indent=4)
        try:
            frame_db = torch.stack(frame_list)
            video_db = torch.stack(video_list)
            audio_db = torch.stack(audio_list)
            clip_db = torch.stack(clip_list)
            print(frame_db.shape)
            print(video_db.shape)
            print(audio_db.shape)
            print(clip_db.shape)
            torch.save(frame_db, os.path.join(self.save_dir, 'frame_db.pt'))
            torch.save(video_db, os.path.join(self.save_dir, 'video_db.pt'))
            torch.save(audio_db, os.path.join(self.save_dir, 'audio_db.pt'))
            torch.save(clip_db, os.path.join(self.save_dir, f'clip_db_{thr_str}.pt'))
        except Exception as e:
            print(e)

    def build_faiss(self, threshold=0.85):
        thr_str = str(int(threshold * 100))
        build_faiss_cosine_index(
            tensor_path=os.path.join(self.save_dir, 'frame_db.pt'),
            mapping_json=os.path.join(self.save_dir, 'f.json'),
            index_path=os.path.join(self.save_dir, 'frame_cosine.idx'),
            use_gpu=False,
            frame=True
        )
        # video‑mean cosine index
        build_faiss_cosine_index(
            tensor_path=os.path.join(self.save_dir, 'video_db.pt'),
            mapping_json=os.path.join(self.save_dir, 'v.json'),
            index_path=os.path.join(self.save_dir, 'video_cosine.idx'),
            use_gpu=False
        )
        # audio cosine index (if using same v.json mapping)
        build_faiss_cosine_index(
            tensor_path=os.path.join(self.save_dir, 'audio_db.pt'),
            mapping_json=os.path.join(self.save_dir, 'v.json'),
            index_path=os.path.join(self.save_dir, 'audio_cosine.idx'),
            use_gpu=False,
            norm=True
        )
        # clip
        build_faiss_cosine_index(
            tensor_path=os.path.join(self.save_dir, f'clip_db_{thr_str}.pt'),
            mapping_json=os.path.join(self.save_dir, f'c_{thr_str}.json'),
            index_path=os.path.join(self.save_dir, f'clip_cosine_{thr_str}.idx'),
            use_gpu=False,
            norm=True
        )

    def split_train_and_eval_step1(self, ratio=0.98):
        random.seed(1357)
        with open(os.path.join(self.save_dir, "v.json"), 'r') as f:
            video_map = json.load(f)
        n = len(video_map)
        k = int(n * (1 - ratio))
        all_keys = list(video_map.keys())
        sample_keys = set(random.sample(all_keys, k))
        v_eval = {k: video_map[k] for k in sample_keys}
        v_train = {k: video_map[k] for k in all_keys if k not in sample_keys}
        with open(os.path.join(self.save_dir, "v_train.json"), 'w') as f:
            json.dump(v_train, f)
        with open(os.path.join(self.save_dir, "v_eval.json"), 'w') as f:
            json.dump(v_eval, f)
        train_photo_ids = set(v_train.values())
        eval_photo_ids = set(v_eval.values())
        with open(os.path.join(self.save_dir, "f.json"), 'r') as f:
            frame_map = json.load(f)
        f_train = {}
        f_eval = {}
        for k, v in tqdm(frame_map.items()):
            if v[0] in eval_photo_ids:
                f_eval[k] = v
            else:
                f_train[k] = v
        with open(os.path.join(self.save_dir, "f_train.json"), 'w') as f:
            json.dump(f_train, f)
        with open(os.path.join(self.save_dir, "f_eval.json"), 'w') as f:
            json.dump(f_eval, f)

    def split_train_and_eval_step2(self):
        with open(os.path.join(self.save_dir, "f_train.json"), 'r') as f:
            f_train = json.load(f)
        with open(os.path.join(self.save_dir, "f_eval.json"), 'r') as f:
            f_eval = json.load(f)
        with open(os.path.join(self.save_dir, "v_train.json"), 'r') as f:
            v_train = json.load(f)
        with open(os.path.join(self.save_dir, "v_eval.json"), 'r') as f:
            v_eval = json.load(f)
        f_train_idx = sorted(map(int, list(f_train.keys())))
        f_eval_idx = sorted(map(int, list(f_eval.keys())))
        v_train_idx = sorted(map(int, list(v_train.keys())))
        v_eval_idx = sorted(map(int, list(v_eval.keys())))
        frame_db = torch.load(os.path.join(self.save_dir, "frame_db.pt"))
        frame_db_train = frame_db[f_train_idx].contiguous().clone()
        frame_db_eval = frame_db[f_eval_idx].contiguous().clone()
        audio_db = torch.load(os.path.join(self.save_dir, "audio_db.pt"))
        audio_db_train = audio_db[v_train_idx].contiguous().clone()
        audio_db_eval = audio_db[v_eval_idx].contiguous().clone()
        torch.save(frame_db_train, os.path.join(self.save_dir, "frame_db_train.pt"))
        torch.save(frame_db_eval, os.path.join(self.save_dir, "frame_db_eval.pt"))
        audio_db_train = drop_all_zero(audio_db_train, 0)
        audio_db_eval = drop_all_zero(audio_db_eval, 0)
        torch.save(audio_db_train, os.path.join(self.save_dir, "audio_db_train.pt"))
        torch.save(audio_db_eval, os.path.join(self.save_dir, "audio_db_eval.pt"))

    def get_str(self, saved_path="test"):
        os.makedirs(saved_path, exist_ok=True)
        init_audio_model(model_name="audio_8_256_0729")
        init_video_model(model_name="video_8_256_0729")
        with open(os.path.join(self.save_dir, "v_eval.json"), 'r') as f:
            v_eval = json.load(f)
        eval_photo_ids = set(v_eval.values())
        for retrieve_dir, chunk_index in zip(self.retrieve_dir_list, self.chunk_index_list):
            for chunk_id in chunk_index:
                build_chunk_str(retrieve_dir, chunk_id, eval_photo_ids, saved_path=saved_path)
