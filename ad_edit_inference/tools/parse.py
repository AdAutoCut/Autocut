import re
from typing import List, Dict, Any, Tuple, Union
import torch
from .video import vtoken2vemb, vemb2clip, vemb2frame
from .audio import atoken2aemb, aemb2audio

# ------------------------------------------------------------------
# 1. 定义所有 sentinel 的“字面量字符串”
# ------------------------------------------------------------------
S = {
    "CLIP_START": "<|clip_start|>",
    "CLIP_END": "<|clip_end|>",
    "VIDEO_START": "<|video_start|>",
    "VIDEO_END": "<|video_end|>",
    "FRAME_START": "<|frame_start|>",
    "FRAME_END": "<|frame_end|>",
    "TEXT_START": "<|text_start|>",
    "TEXT_END": "<|text_end|>",
    "AUDIO_START": "<|audio_start|>",
    "AUDIO_END": "<|audio_end|>",
}

# ------------------------------------------------------------------
# 2. 构造正则：将所有 sentinel 逐个 escape 后 OR 起来
# ------------------------------------------------------------------
sentinel_pat = "|".join(re.escape(x) for x in S.values())

# 原子多模态 token（视频 code & 音频 code）
V_PAT = r"<v_\d+_\d+>"
A_PAT = r"<a_\d+_\d+>"

TOKEN_RE = re.compile(f"({sentinel_pat}|{V_PAT}|{A_PAT})")

# 捕获并解析具体数字
V_TOKEN_RE = re.compile(r"^<v_(\d+)_(\d+)>$")
A_TOKEN_RE = re.compile(r"^<a_(\d+)_(\d+)>$")


def parse_multimodal(
    seq: str,
    *,
    parse_v_pair: bool = True,
    parse_a_pair: bool = True,
    strip_text: bool = True,
    cur_clip_index = None,
    cur_frame_index = None
):
    """
    将打平的多模态串解析为结构化 {texts, audios, frames, clips}.
    """
    parts = TOKEN_RE.split(seq)  # split 并保留匹配项
    # print('\n Parts:', parts)

    texts = []
    audios = []
    frames = []
    clips = []
    frame_embs = []
    clip_embs = []

    # 状态
    in_clip = False
    in_video = False
    in_frame = False
    in_text = False
    in_audio = False

    # 当前累积
    cur_video: List[List[Any]] = []
    cur_frame: List[Any] = []
    cur_text_chunks: List[str] = []
    cur_audio_tokens: List[Any] = []
    cur_clip = {
        "text": [],
        "clip": [],
        "frame": [],
    }

    # flush helpers ---------------------------------------------------
    def flush_frame():
        nonlocal cur_frame, cur_video
        if cur_frame is not None:
            cur_video.append(cur_frame)
        cur_frame = []

    def flush_video():
        nonlocal cur_video, cur_clip
        try:
            if cur_video:
                # print('\n CURVID:', cur_video)
                video_tensor = torch.tensor(cur_video)
                # print('\n video_tensor:', video_tensor)
                vemb = vtoken2vemb(video_tensor)
                vemb_mean = vemb.mean(dim=0, keepdim=True)
                clip = vemb2clip(vemb_mean, cur_clip_index=cur_clip_index)
                print('\n clip:', clip)
                frame = vemb2frame(vemb, cur_frame_index=cur_frame_index)
                # print('\n frame:', frame)
                frames.append(frame)
                frame_embs.append(vemb)
                clip_embs.append(vemb_mean)
                cur_clip["frame"].extend(frame)
                cur_clip["clip"].extend(clip)
            else:
                frames.append([])
        except Exception as e:
            print(e)
            frames.append([])
        cur_video = []

    def flush_text():
        nonlocal cur_text_chunks, texts, cur_clip
        txt = "".join(cur_text_chunks)
        if strip_text:
            txt = txt.strip()
        texts.append(txt)
        cur_clip["text"].append(txt)
        cur_text_chunks = []

    def flush_audio():
        nonlocal cur_audio_tokens, audios
        try:
            if cur_audio_tokens:
                # print('cur_audio_tokens:', cur_audio_tokens)
                audio_tensor = torch.tensor(cur_audio_tokens)
                # print('audio_tensor:', audio_tensor)
                aemb = atoken2aemb(audio_tensor)
                audio = aemb2audio(aemb)
                # print('audio:', audio)
                audios.append(audio)
            else:
                audios.append([])
        except Exception as e:
            print(e)
            audios.append([])
        cur_audio_tokens = []

    def flush_clip():
        nonlocal cur_clip, clips
        if cur_clip:
            clips.append(cur_clip)
        else:
            clips.append({})
        cur_clip = {
            "text": [],
            "clip": [],
            "frame": [],
        }

    # ----------------------------------------------------------------
    # 扫描
    # ----------------------------------------------------------------
    for p in parts:
        if not p:
            continue

        # --- sentinels (用字面量匹配) ---
        if p == S["VIDEO_START"]:
            in_video = True
            cur_video = []
            continue

        if p == S["VIDEO_END"]:
            if in_frame:
                flush_frame()
                in_frame = False
            if in_video:
                flush_video()
                in_video = False
            continue

        if p == S["FRAME_START"]:
            in_frame = True
            cur_frame = []
            continue

        if p == S["FRAME_END"]:
            if in_frame:
                flush_frame()
            in_frame = False
            continue

        if p == S["TEXT_START"]:
            in_text = True
            cur_text_chunks = []
            continue

        if p == S["TEXT_END"]:
            if in_text:
                flush_text()
            in_text = False
            continue

        if p == S["AUDIO_START"]:
            in_audio = True
            cur_audio_tokens = []
            continue

        if p == S["AUDIO_END"]:
            if in_audio:
                flush_audio()
            in_audio = False
            continue

        if p == S["CLIP_START"]:
            in_clip = True
            cur_clip = {
                "text": [],
                "clip": [],
                "frame": [],
            }
            continue

        if p == S["CLIP_END"]:
            if in_text:
                flush_text()
                in_text = False
            if in_frame:
                flush_frame()
                in_frame = False
            if in_video:
                flush_video()
                in_video = False
            if in_clip:
                flush_clip()
                in_clip = False
            continue

        # --- video atomic token? ---
        m_v = V_TOKEN_RE.match(p)
        if m_v:
            token = int(m_v.group(2)) if parse_v_pair else p
            if in_frame:
                cur_frame.append(token)
            else:
                # 容错：视频 token 出现在 frame 外 => 自动起一帧
                in_frame = True
                cur_frame = [token]
            continue

        # --- audio atomic token? ---
        m_a = A_TOKEN_RE.match(p)
        if m_a:
            token = int(m_a.group(2)) if parse_a_pair else p
            if in_audio:
                cur_audio_tokens.append(token)
            else:
                in_audio = True
                cur_audio_tokens = [token]
            continue

        # --- 普通裸文本块 ---
        if in_text:
            cur_text_chunks.append(p)
        else:
            # 未处于任何块：忽略或记录为警告
            pass

    # ----------------------------------------------------------------
    # 扫尾
    # ----------------------------------------------------------------
    if in_frame:
        flush_frame()
        in_frame = False
    if in_video:
        flush_video()
        in_video = False
    if in_clip:
        flush_clip()
        in_clip = False
    if in_text and cur_text_chunks:
        flush_text()
        in_text = False
    if in_audio:
        flush_audio()
        in_audio = False

    return {
        "clips": clips,
        "texts": texts,
        "audios": audios,
        "frames": frames
    }, {
        "frame_emds": frame_embs,
        "clip_embs": clip_embs
    }

    # return {
    #     "clips": clips,
    #     "texts": texts,
    #     "audios": audios,
    #     "frames": frames
    # }, {
    #     "frame_emds": torch.cat(frame_embs),
    #     "clip_embs": torch.cat(clip_embs)
    # }
