import subprocess
import tempfile
import os
import time
import uuid
import torch
import json
import faiss
from loguru import logger
from concurrent.futures import ThreadPoolExecutor


from .emb_vqvae import emb2token, token2emb, get_model
from .share.find_topk import find_top_k_results


def init_video_model(load_faiss=False, model_name="video_8_256_0729"):
    global video_rqvae_model, video_index, frame_index, clip_index
    current_file = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file)
    config_path = os.path.join(current_dir, f'emb_vqvae/model/{model_name}/config.yaml')
    video_rqvae_model = get_model(config_path)
    if load_faiss:
        video_index = faiss.read_index('/data/phd/qinsizhong/ad_edit_inference/inex_f831/video_cosine.idx')
        frame_index = faiss.read_index('/data/phd/qinsizhong/ad_edit_inference/inex_f831/frame_cosine.idx')
        clip_index = faiss.read_index('/data/phd/qinsizhong/ad_edit_inference/inex_f831/clip_cosine_85.idx')
        return video_rqvae_model, video_index, frame_index, clip_index
    else:
        return video_rqvae_model


class FrameEmbeddingClient:
    def __init__(self):
        from kess.framework import ClientOption, GrpcClient
        from mars.protos.model_serving_pb2 import PredictRequest
        from mars.protos.model_serving_pb2_grpc import ModelServingStub
        from google.protobuf.json_format import MessageToDict
        self.client_option = ClientOption(
            biz_def='ad',
            grpc_service_name='grpc-mmu-cnnv2-emb-example-T4',
            grpc_stub_class=ModelServingStub,
        )
        self.client = GrpcClient(self.client_option)
        self.PredictRequest = PredictRequest
        self.MessageToDict = MessageToDict

    def sync_run(self, img, video_name="XXX"):
        max_retries = 5
        for attempt in range(1, max_retries + 1):
            try:
                uid = str(uuid.uuid4())
                req = self.PredictRequest(id=uid)
                req.media.data = img
                resp = self.client.Predict(req, timeout=5)
                resp_dict = self.MessageToDict(resp)
                if resp_dict.get("status") == "SUCCESS":
                    return resp_dict["feature"]["floatArray"]["floatElems"]
                else:
                    # # 状态不成功，也算一次失败
                    logger.warning(f"[Attempt {attempt}] status={resp_dict.get('status')}, retrying...")
            except Exception as e:
                pass
                logger.warning(f"[Attempt {attempt}] Exception: {e}, retrying...")
            time.sleep(attempt * 0.1)
        # 所有重试都失败
        # logger.warning(f"sync_run: all {max_retries} attempts failed for {video_name}.mp4")
        return None


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


def frame2vemb(frame):
    """返回frame_embs"""
    frame_emb_client = FrameEmbeddingClient()
    frame_embs = frame_emb_client.sync_run(frame)
    return frame_embs


def video2vemb(video_bytes, interval_s=1):
    """ffmpeg不支持直接读取mp4的bytes，因此需要先保存到一个temp flie再处理，返回frame_embs_dict"""
    tmp = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
    tmp.write(video_bytes)
    tmp.close()
    path = tmp.name
    try:
        cmd = [
            'ffmpeg',
            '-f', 'mp4',
            '-i', path,
            '-vf', f'fps=1/{interval_s}',  # 每 x 秒抽 1 帧
            '-f', 'image2pipe', '-vcodec', 'png', 'pipe:1'
        ]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        data, err = proc.communicate()
    finally:
        os.remove(path)
    if len(data) < 10:
        print(err)
        raise Exception("ffmpeg error")
    frames = split_png_stream(data)
    frame_emb_client = FrameEmbeddingClient()

    def process(i, frame):
        return round(i * interval_s, 2), frame_emb_client.sync_run(frame)

    with ThreadPoolExecutor(max_workers=16) as executor:
        results = list(executor.map(lambda args: process(*args), enumerate(frames)))
    frame_embs_dict = dict(results)
    # frame_embs_dict = {i * interval_s: frame_emb_client.sync_run(frame) for i, frame in enumerate(frames)}
    return frame_embs_dict


def vemb2vtoken(vemb):
    """vemb (B, D) -> vtoken (B, H)"""
    if type(vemb) is dict:
        vemb_list = [torch.tensor(i) for i in vemb.values()]
        vemb_tensor = torch.stack(vemb_list)
        vtoken = emb2token(video_rqvae_model, vemb_tensor)
    elif type(vemb) is list:
        vemb_tensor = torch.tensor(vemb[0]).unsqueeze(0)
        vtoken = emb2token(video_rqvae_model, vemb_tensor)
    return vtoken


def vtoken2vemb(vtoken):
    """vtoken (B, H) -> vemb (B, D)"""
    vemb = token2emb(video_rqvae_model, vtoken)
    return vemb


# def vemb2video(vemb, topk=5):
#     video_db = torch.load('incycle-1d/video_db.pt')
#     with open('incycle-1d/v.json', 'r') as f:
#         vidx2photoid=json.load(f)
#     if vemb.ndim==1:
#         batch_vemb = vemb.unsqueeze(dim=0)
#     else:
#         batch_vemb = vemb
#     topk_idx, topk_val = find_top_k_results(batch_vemb, video_db, topk)
#     batch, _ = batch_vemb.shape
#     result_list = []
#     print("video-------------------")
#     for b in range(batch):
#         print(f"{b}------------")
#         top_result = vidx2photoid[str(int(topk_idx[b][0]))]
#         for k in range(topk):
#             result = vidx2photoid[str(int(topk_idx[b][k]))]
#             print(f"{k}:", end=" ")
#             print(result, end=", ")
#             print(float(topk_val[b][k]))
#         result_list.append(top_result)
#     return result_list

def vemb2video(vemb, topk=5):
    """FAISS"""
    # video_index = faiss.read_index('video_cosine.idx')
    if vemb.ndim == 1:
        batch_vemb = vemb.unsqueeze(dim=0)
    else:
        batch_vemb = vemb
    norms = batch_vemb.norm(p=2, dim=1, keepdim=True)
    batch_vemb_normed = batch_vemb / (norms + 1e-10)
    batch_vemb_normed = batch_vemb_normed.to('cpu')
    vemb_np = batch_vemb_normed.numpy().astype('float32')
    topk_val, topk_idx = video_index.search(vemb_np, topk)
    batch, _ = batch_vemb.shape
    result_list = []
    print("video-------------------")
    for b in range(batch):
        print(f"{b}------------")
        top_result = int(topk_idx[b][0])
        for k in range(topk):
            result = int(topk_idx[b][k])
            print(f"{k}:", end=" ")
            print(result, end=", ")
            print(float(topk_val[b][k]))
        result_list.append(top_result)
    return result_list


# def vemb2frame(vemb, topk=5):
#     # frame_db = torch.load('incycle-1d/frame_db.pt')
#     # with open('incycle-1d/f.json', 'r') as f:
#     frame_db = torch.load('frame_db.pt')
#     with open('f.json', 'r') as f:
#         fidx2frameid=json.load(f)
#     if vemb.ndim==1:
#         batch_vemb = vemb.unsqueeze(dim=0)
#     else:
#         batch_vemb = vemb
#     topk_idx, topk_val = find_top_k_results(batch_vemb, frame_db, topk)
#     batch, _ = batch_vemb.shape
#     result_list = []
#     print("frame-------------------")
#     for b in range(batch):
#         print(f"{b}------------")
#         top_result = fidx2frameid[str(int(topk_idx[b][0]))]
#         for k in range(topk):
#             result = fidx2frameid[str(int(topk_idx[b][k]))]
#             print(f"{k}:", end=" ")
#             print(result, end=", ")
#             print(float(topk_val[b][k]))
#         result_list.append(top_result)
#     return result_list

def vemb2frame(vemb, topk=5, cur_frame_index=None):
    """FAISS"""
    # frame_index = faiss.read_index('frame_cosine.idx')
    if vemb.ndim == 1:
        batch_vemb = vemb.unsqueeze(dim=0)
    else:
        batch_vemb = vemb
    norms = batch_vemb.norm(p=2, dim=1, keepdim=True)
    batch_vemb_normed = batch_vemb / (norms + 1e-10)
    batch_vemb_normed = batch_vemb_normed.to('cpu')
    vemb_np = batch_vemb_normed.numpy().astype('float32')
    if cur_frame_index is None:
        topk_val, topk_idx = frame_index.search(vemb_np, topk)
    else:
        topk_val, topk_idx = cur_frame_index.search(vemb_np, topk)
    batch, _ = batch_vemb.shape
    result_list = []
    # print("frame-------------------")
    for b in range(batch):
        # print(f"{b}------------")
        top_result = int(topk_idx[b][0])
        for k in range(topk):
            result = int(topk_idx[b][k])
            # print(f"{k}:", end=" ")
            # print(result, end=", ")
            # print(float(topk_val[b][k]))
        result_list.append(top_result)
    return result_list


def vemb2clip(vemb, topk=5, cur_clip_index=None):
    """FAISS"""
    if vemb.ndim == 1:
        batch_vemb = vemb.unsqueeze(dim=0)
    else:
        batch_vemb = vemb
    norms = batch_vemb.norm(p=2, dim=1, keepdim=True)
    batch_vemb_normed = batch_vemb / (norms + 1e-10)
    batch_vemb_normed = batch_vemb_normed.to('cpu')
    vemb_np = batch_vemb_normed.numpy().astype('float32')
    if cur_clip_index is None:
        topk_val, topk_idx = clip_index.search(vemb_np, topk)
    else:
        topk_val, topk_idx = cur_clip_index.search(vemb_np, topk)
    batch, _ = batch_vemb.shape
    result_list = []
    # print("clip-------------------")
    for b in range(batch):
        # print(f"{b}------------")
        top_result = int(topk_idx[b][0])
        for k in range(topk):
            result = int(topk_idx[b][k])
            # print(f"{k}:", end=" ")
            # print(result, end=", ")
            # print(float(topk_val[b][k]))
        result_list.append(top_result)
    return result_list
