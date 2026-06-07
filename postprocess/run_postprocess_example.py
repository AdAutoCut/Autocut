'''
Example of postprocessing video outline
'''
from tools.parse import parse_multimodal
from tools.video import init_video_model
from tools.audio import init_audio_model
from tools.infer import predict_online
from tools.postprocess import generate_video, get_input_faiss
from tools.preprocess import get_video_clips
from tools.common.blobstore import download_video_bytes
from tools.common.extract_audio_from_video import extract_audio_from_video
from batch_audio_split import upload_all_audio, run_vocal_split_batch, download_all_bgm
from tools.common.blobstore.client import BlobStoreClientManager
import random
import re
import os
import subprocess
from typing import List, Dict, Any, Optional, Tuple
import torch
import torchaudio
import ChatTTS


def merge_video_and_audio(video_path: str, audio_path: str, output_path: str, overwrite: bool = True) -> None:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-i", video_path,   # 0: 视频 + 可能有原音
        "-i", audio_path,   # 1: 新 BGM
        "-filter_complex", "[0:a][1:a]amix=inputs=2:duration=first:dropout_transition=0[aout]",
        "-map", "0:v:0",    # 保留原视频画面
        "-map", "[aout]",   # 使用混合后的音频
        "-c:v", "copy",
        "-c:a", "aac",
        "-shortest",
        output_path,
    ]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)

def build_sort_task_input(clip_tokens: List[str],seed: Optional[int] = None,) -> Tuple[str, Dict[str, Any]]:
    """
    输入：
        clip_tokens: 原始的 clip token 列表（包含空 clip）
        seed: 随机种子（可选，便于复现打乱）

    处理：
      1) 过滤掉空 clip（严格等于 EMPTY_CLIP）
      2) 不修改原始列表的前提下，打乱顺序
      3) 为打乱后的每个 clip 分配 local_id = 1..N
      4) 构造 sort 任务输入字符串：
         "1)<clip1_tokens>\\n2)<clip2_tokens>\\n..."

    返回：
      - input_str: 给模型的 sort 输入字符串
      - index: 一个 dict，包含各种索引映射，结构：
          {
            "local_id_to_token": {local_id: token_str, ...},
            "local_id_to_orig_idx": {local_id: orig_idx, ...},
            "orig_idx_to_local_id": {orig_idx: local_id, ...},
            "perm": [orig_idx0, orig_idx1, ...]  # 当前乱序列表对应的原始下标顺序
          }
    """

    # ---------- 1) 过滤空 clip，同时保留原始下标 ----------
    EMPTY_CLIP = "<|clip_start|><|video_start|><|video_end|><|clip_end|>"

    filtered = [
        (i, s)
        for i, s in enumerate(clip_tokens)
        if s != EMPTY_CLIP
        # 如果想更鲁棒：只保留有 frame 的
        # if "<|frame_start|>" in s
    ]

    # ---------- 2) 拷贝一份并打乱（不改原 filtered/clip_tokens） ----------
    shuffled = filtered[:]  # 拷贝
    rng = random.Random(seed) if seed is not None else random
    rng.shuffle(shuffled)

    # ---------- 3) 构造 input_str + 索引映射 ----------
    lines: List[str] = []
    local_id_to_token: Dict[int, str] = {}
    local_id_to_orig_idx: Dict[int, int] = {}
    orig_idx_to_local_id: Dict[int, int] = {}
    perm: List[int] = []

    for local_id, (orig_idx, token) in enumerate(shuffled, start=1):
        # 行文本： "1)<tokens>"
        lines.append(f"{local_id}){token}")

        local_id_to_token[local_id] = token
        local_id_to_orig_idx[local_id] = orig_idx
        orig_idx_to_local_id[orig_idx] = local_id
        perm.append(orig_idx)

    input_str = "\n".join(lines)

    index: Dict[str, Any] = {
        "local_id_to_token": local_id_to_token,
        "local_id_to_orig_idx": local_id_to_orig_idx,
        "orig_idx_to_local_id": orig_idx_to_local_id,
        "perm": perm,
    }

    return input_str, index

def build_vid2aud_clips_block(script: str, sorted_tokens: list[str]) -> str:
    # 1. 按行切台词，去掉全空行（避免最后一个多余的空行影响对齐）
    lines = [line for line in script.split("\n") if line.strip()]

    num_clips = len(sorted_tokens)
    num_lines = len(lines)

    # 2. 根据规则对齐台词和 clips
    if num_lines >= num_clips:
        # 台词多：只要前 num_clips 行
        used_lines = lines[:num_clips]
    else:
        # 台词少：后面的 clip 用空字符串占位
        used_lines = lines + [""] * (num_clips - num_lines)

    # 3. 把台词插入到每个 clip 的 tokens 中
    clips_with_text = []
    for text, vtok in zip(used_lines, sorted_tokens):
        # 不管 text 是否为空，都插入 <|text_start|>text<|text_end|>
        # 保留 <|video_start|>... 的原始结构
        new_vtok = vtok.replace(
            "<|clip_start|><|video_start|>",
            f"<|clip_start|><|text_start|>{text}<|text_end|><|video_start|>",
            1,
        )
        clips_with_text.append(new_vtok)

    # 4. 用换行拼成一个 block，后面可以直接塞到 prompt 里
    clips_block = "\n".join(clips_with_text)
    return clips_block

def synthesize_tts_full(text: str,
                        seed: int,
                        out_dir: str,
                        name: str = "tts_full.wav") -> str:
    """
    使用 ChatTTS 将整段台词 text 合成为一条 wav 音频。
    返回生成的 wav 路径。
    """
    os.makedirs(out_dir, exist_ok=True)

    # 固定种子，控制音色
    torch.manual_seed(seed)

    chat = ChatTTS.Chat()
    chat.load(compile=False,
              custom_path="/data/phd/qinsizhong/ad_edit_inference/tools")

    # 在这个 seed 下采一个说话人
    spk_emb = chat.sample_random_speaker()

    params_infer_code = ChatTTS.Chat.InferCodeParams(
        spk_emb=spk_emb,
        # 需要的话可以开放更多控制参数：
        # temperature=[0.3, 0.7],
    )

    # 一次性合成整段文本
    wavs = chat.infer(
        [text],
        use_decoder=True,
        params_infer_code=params_infer_code,
    )
    wav = wavs[0]          # numpy array, shape [T]
    sample_rate = 24000

    # 输出文件名
    wav_name = name if name.endswith(".wav") else name + ".wav"
    wav_path = os.path.join(out_dir, wav_name)

    wav_tensor = torch.from_numpy(wav).unsqueeze(0)  # [1, T]
    torchaudio.save(wav_path, wav_tensor, sample_rate)
    print(f"[OK] 保存整段 TTS 到: {wav_path}")

    # 如有需要，也可以把 spk_emb 存下来复用
    spk_path = os.path.join(out_dir, f"spk_seed{seed}.pt")
    torch.save(spk_emb, spk_path)
    print(f"[OK] 保存说话人音色到: {spk_path}")

    return wav_path

def merge_video_with_tts_overlay(
    base_video_path: str,
    tts_wav_path: str,
    output_path: str,
) -> None:
    """
    在已有视频音频的基础上叠加一条 TTS：
    - 保留 base_video_path 中原有音频（例如 BGM）
    - 与 tts_wav_path 做 amix 叠加
    - 输出到 output_path
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",
        "-i", base_video_path,   # 0: 原视频 (含当前音频：BGM 等)
        "-i", tts_wav_path,      # 1: 新 TTS 音频
        "-filter_complex",
        "[0:a][1:a]amix=inputs=2:duration=first:dropout_transition=0[aout]",
        "-map", "0:v:0",         # 保留原视频画面
        "-map", "[aout]",        # 使用混合后的音频
        "-c:v", "copy",
        "-c:a", "aac",
        "-shortest",
        output_path,
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    print(f"[OK] 最终带 TTS 的视频已保存到: {output_path}")

product_type = "无限耳机"
brand = "漫步者EVO PRO"
selling_points = ["U形入耳曲线","机身皮革质感","开盖轻松连接","EQ自定义调节","多噪声模式","多耳帽设计"]
video_path = "" # The video material (only one video for this example)
script = "漫步者EVO PRO，\n这款无线耳机，\nU形入耳曲线，\n机身皮革质感，\n开盖轻松连接，\nEQ自定义调节，\n多种噪声模式随意切换，\n超多耳帽随心用，\n值得你拥有。"

_video_rqvae_model, _video_index, _frame_index, _clip_index = init_video_model(load_faiss=True, model_name="video_8_256_0729")
init_audio_model(load_faiss=True, model_name="audio_8_256_0729")

###--------------- 切clips------------------
with open(video_path, "rb") as f:
    video_bytes = f.read()
clips, clips_token_list = get_video_clips(video_bytes=video_bytes, threshold=0.8)
print('CLIPS:',clips)
### 检查一下id是不是一致的，如果不一致 pass掉

###--------------- 打乱再sort------------------
input_str, idx = build_sort_task_input(clips_token_list, seed=42)
# 临时的检索库
# cur_clip_index, cur_frame_index = get_input_faiss(input_str, _clip_index, _frame_index)
print('IDX:',idx)
sentence_num = len(script.strip().split("\n")) #算行数
print('CLIP LEN: ',len(clips))
print('SEN LEN: ',sentence_num)

SORT_PROMPT = f'''
你是专业的视频剪辑师。请你根据提供的广告台词顺序，将视频片段进行排序。\n【输出要求】\n 1）只输出与下方台词句数相同数量的视频片段编号；\n 2）输出的视频片段编号不能有重复；\n【本次台词总句数】：{sentence_num} 段；\n【商品】：{product_type}\n【品牌】：{brand}\n【卖点】：{selling_points}\n【广告台词如下】\n{script}\n【视频片段如下】{input_str}\n【现在请输出按正确顺序排列的视频片段编号序列】
'''

sort_answer = predict_online(prompt = SORT_PROMPT, t = 0.01) 
print('SORT ANSWER: ',sort_answer)
sorted_local_ids = [int(x) for x in sort_answer.split(",")]
sorted_tokens = [idx["local_id_to_token"][i] for i in sorted_local_ids]
print('SORTED TOKENS:',sorted_tokens)
sorted2orig_ids = [idx["local_id_to_orig_idx"][i] for i in sorted_local_ids]

### 音频检索功能
clips_block = build_vid2aud_clips_block(script, sorted_tokens)

VID2AUD_PROMPT = f'''
你是专业的广告背景音乐生成助手。请根据广告的商品信息、台词、以及视频片段，输出对应广告的背景音频。\n【输出要求】\n1) 只输出一行，且仅包含 <|audio_start|>…<|audio_end|>；\n2) 禁止输出任何解释、编号、引号或额外符号。\n\n【商品】：{product_type}\n【品牌】：{brand}\n【卖点】：{selling_points}\n【台词和视频片段如下】\n{clips_block}\n【现在请输出相关的背景音乐】"
'''

aud_answer = predict_online(prompt = VID2AUD_PROMPT, t = 0.01) 
audio_block = f"<|audio_start|>{aud_answer}<|audio_end|>"
clips_flat = "".join(clips_block.splitlines())
full_str = clips_flat + audio_block
print(full_str)

# parsed_sample, _ = parse_multimodal(full_str,cur_clip_index=cur_clip_index, cur_frame_index=cur_frame_index)
parsed_sample, _ = parse_multimodal(full_str)
final_clips = parsed_sample['clips']

# 生成视频
temp_video_path = "/data/phd/miltonzhou/postprocess/temp_video/test4.mp4"
generate_video(final_clips, extract_method="clip", out_path=temp_video_path)
# generate_video(final_clips, extract_method="frame", out_path=temp_video_path)

audio_pid = parsed_sample['audios'][0][0] 
audio_path = "/data/phd/miltonzhou/postprocess/temp_audio"
BGM_path = "/data/phd/miltonzhou/postprocess/temp_BGM"
cur_BGM_path = BGM_path + f"/{audio_pid}.wav"
result_video_path = "/data/phd/miltonzhou/postprocess/result_video"
cur_result_video_path = result_video_path+f"/result.mp4"

os.makedirs(audio_path, exist_ok=True)
os.makedirs(BGM_path, exist_ok=True)

video_bytes = download_video_bytes(audio_pid)
audio_bytes = extract_audio_from_video(video_bytes)

apath = os.path.join(audio_path, f"{audio_pid}.wav")
with open(apath, "wb") as f:
    f.write(audio_bytes)

# --- split vocal ---

BUCKET = "ad-nieuwland-material"
ADBUCKET = "nieuwland-material"
SERVICE_NAME = "ad-Vocal-Split"
RESULT_JSON = "split_results_DEMO.json"
NUM_WORKERS = 4

upload_all_audio(audio_path, BUCKET, NUM_WORKERS)
run_vocal_split_batch(
        audio_dir=audio_path,
        bucket=ADBUCKET,
        service_name=SERVICE_NAME,
        output_json=RESULT_JSON,
        num_workers=NUM_WORKERS
    )
download_all_bgm(RESULT_JSON, BGM_path, BUCKET, NUM_WORKERS)

merge_video_and_audio(video_path = temp_video_path, audio_path = cur_BGM_path, output_path = cur_result_video_path)

# -------- 生成整段脚本的 TTS 并叠加到已有视频上 --------
tts_out_dir = "/data/phd/miltonzhou/postprocess/temp_tts"
tts_seed = 3335  # voice tone

# 1) 用整段 script 生成一条 TTS 音频
clean_script = script.replace("\n", "")
tts_wav_path = synthesize_tts_full(
    text=clean_script,
    seed=tts_seed,
    out_dir=tts_out_dir,
    name="full_script_tts.wav",
)

# 2) 在 cur_result_video_path 的音频基础上叠加 TTS
final_video_with_tts = os.path.join(result_video_path, "result_with_tts.mp4")
merge_video_with_tts_overlay(
    base_video_path=cur_result_video_path,
    tts_wav_path=tts_wav_path,
    output_path=final_video_with_tts,
)

print("Final video path:", final_video_with_tts)
