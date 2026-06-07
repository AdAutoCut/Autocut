from io import BytesIO
import re
import subprocess
import tempfile
import os
import time
import uuid
import torch
import json
import faiss

from .emb_vqvae import emb2token, token2emb, get_model
from .share.find_topk import find_top_k_results


def init_audio_model(load_faiss=False, model_name="audio_8_256_0729"):
    global audio_rqvae_model, audio_index
    current_file = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file)
    # print(current_dir)
    config_path = os.path.join(current_dir, f'emb_vqvae/model/{model_name}/config.yaml')
    audio_rqvae_model = get_model(config_path)
    if load_faiss:
        # audio_index = faiss.read_index('/data/phd/qinsizhong/ad_edit_inference/ex_incycle2/audio_cosine.idx')
        audio_index = faiss.read_index('/data/phd/qinsizhong/ad_edit_inference/inex_us/audio_cosine.idx')
        return audio_rqvae_model, audio_index
    else:
        return audio_rqvae_model


def audio2aemb(wav_bytes, music_thr=0.5):
    """返回is_music, audio_embs"""
    import torchaudio
    from .panns_inference_local import AudioTagging
    at = AudioTagging()
    buf = BytesIO(wav_bytes)
    y, sr = torchaudio.load(buf)
    y = y.mean(dim=0, keepdim=True)
    (clipwise_output, embedding) = at.inference(y)
    is_music = float(clipwise_output[0, 137])  # 137是music
    return is_music > music_thr, embedding


def aemb2atoken(aemb):
    """aemb (D,) -> atoken (H,)"""
    aemb_tensor = aemb.unsqueeze(0)
    atoken = emb2token(audio_rqvae_model, aemb_tensor)
    atoken = atoken.squeeze(0)
    return atoken


def atoken2aemb(atoken):
    """atoken (H,) -> aemb (D,)"""
    atoken = atoken.unsqueeze(0)
    aemb = token2emb(audio_rqvae_model, atoken)
    aemb = aemb.squeeze(0)
    return aemb


# def aemb2audio(aemb, topk=5):
#     aemb = aemb.unsqueeze(dim=0)
#     audio_db = torch.load('audio_db.pt')
#     topk_idx, topk_val = find_top_k_results(aemb, audio_db, topk)
#     print(topk_val)
#     with open('v.json', 'r') as f:
#         vidx2photoid = json.load(f)
#     for i in range(topk):
#         print(vidx2photoid[str(int(topk_idx[0][i]))])


def aemb2audio(aemb, topk=5):
    """FAISS"""
    # audio_index = faiss.read_index('audio_cosine.idx')
    if aemb.ndim==1:
        batch_aemb = aemb.unsqueeze(dim=0)
    else:
        batch_aemb = aemb
    norms = batch_aemb.norm(p=2, dim=1, keepdim=True)
    batch_aemb_normed = batch_aemb / (norms+1e-10)
    batch_aemb_normed = batch_aemb_normed.to('cpu')
    aemb_np = batch_aemb_normed.numpy().astype('float32')
    topk_val, topk_idx = audio_index.search(aemb_np, topk)
    batch, _ = batch_aemb.shape
    result_list = []
    # print("audio-------------------")
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

### ADDED

def astr2tok(s: str) -> list[int]:
    """
    提取 audio token 串中的 <a_i_j>，返回 [j1, j2, ...]（按出现顺序）。
    若找不到 audio 块，则在整串上搜索；若没有任何 <a_._.>，返回 []。
    """
    _AUDIO_BLOCK = re.compile(r"<\|audio_start\|>(.*?)<\|audio_end\|>", flags=re.DOTALL)
    _A_PAIR      = re.compile(r"<a_(\d+)_(\d+)>")

    if not isinstance(s, str) or not s:
        return []

    m = _AUDIO_BLOCK.search(s)
    seg = m.group(1) if m else s

    out: List[int] = []
    for m in _A_PAIR.finditer(seg):
        out.append(int(m.group(2)))
    return out


def aemb2audio_t5(aemb, topk=5):
    """FAISS"""
    # audio_index = faiss.read_index('audio_cosine.idx')
    if aemb.ndim==1:
        batch_aemb = aemb.unsqueeze(dim=0)
    else:
        batch_aemb = aemb
    norms = batch_aemb.norm(p=2, dim=1, keepdim=True)
    batch_aemb_normed = batch_aemb / (norms+1e-10)
    batch_aemb_normed = batch_aemb_normed.to('cpu')
    aemb_np = batch_aemb_normed.numpy().astype('float32')
    topk_val, topk_idx = audio_index.search(aemb_np, topk)
    batch, _ = batch_aemb.shape
    result_list = []
    # print("audio-------------------")
    for b in range(batch):
        # print(f"{b}------------")
        top_result = int(topk_idx[b][0])
        for k in range(topk):
            result = int(topk_idx[b][k])
            # print(f"{k}:", end=" ")
            # print(result, end=", ")
            # print(float(topk_val[b][k]))
            result_list.append(result)
    return result_list    