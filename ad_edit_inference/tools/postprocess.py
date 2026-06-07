import faiss
import torch
import numpy as np
import json
import os
import io
import re
import shutil
import subprocess
import threading
import tempfile
from loguru import logger
import ChatTTS
import torchaudio
from typing import List, Dict
from collections import OrderedDict
from .parse import parse_multimodal
from .video import vtoken2vemb, vemb2frame, vemb2clip, init_video_model
from .audio import init_audio_model
from .common.blobstore.client import BlobStoreClient
from .common.blobstore import download_video_bytes
from .infer import predict, predict_online
from .preprocess import process_photo_id_videos


torch.manual_seed(3335)

chat = ChatTTS.Chat()
chat.load(compile=False, custom_path="/data/phd/qinsizhong/ad_edit_inference/tools")  # Set to True for better performance
rand_spk = chat.sample_random_speaker()

params_infer_code = ChatTTS.Chat.InferCodeParams(
    spk_emb=rand_spk,
    # temperature=0.3,
)


def build_faiss(embs, faiss_index, use_gpu=False):
    ### 输入embs为List[Tensor]，这里报错：TypeError: unique_dim(): argument 'input' (position 1) must be Tensor, not list
    if isinstance(embs, list):
        # embs: List[Tensor[1, D]] 或 List[Tensor[D]]
        embs = torch.cat(
            [e if e.ndim == 2 else e.unsqueeze(0) for e in embs],
            dim=0
        )

    emb_unique = torch.unique(embs, dim=0)
    emb_np = emb_unique.cpu().numpy().astype('float32')
    norms = np.linalg.norm(emb_np, axis=1, keepdims=True)
    emb_np = emb_np / (norms + 1e-10)
    d = emb_np.shape[1]
    index = faiss.IndexFlatIP(d)         # IP = inner product
    index = faiss.IndexIDMap(index)      # to keep track of your original IDs
    if use_gpu:
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)
    topk_val, topk_idx = faiss_index.search(emb_np, 1)
    ids = topk_idx.reshape(-1)
    index.add_with_ids(emb_np, ids)
    if use_gpu:
        index = faiss.index_gpu_to_cpu(index)
    return index


def get_input_faiss(input_videos, clip_index, frame_index, save_path=None):
    if type(input_videos) is str:
        ori, result = parse_multimodal(input_videos)
        clip_embs = result['clip_embs']
        # print(clip_embs)
        frame_embs = result['frame_emds']
    cur_clip_index = build_faiss(clip_embs, clip_index)
    cur_frame_index = build_faiss(frame_embs, frame_index)
    if save_path:
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(ori, f, ensure_ascii=False)
    return cur_clip_index, cur_frame_index


def eval_sft_data_input_photo_id_videos(
    sft_data,
    photo_ids,
    model_name="saves/qwen-8b-sft-0726-full-epoch10",
    output_path="ans.json",
    gt_path="gt.json",
    temp_dir="temp_video",
    video_rqvae_model="video_8_256_0729",
    audio_rqvae_model="audio_8_256_0729"
):
    video_rqvae_model, video_index, frame_index, clip_index = init_video_model(load_faiss=True, model_name=video_rqvae_model)
    init_audio_model(load_faiss=True, model_name=audio_rqvae_model)
    index_path = os.path.join(temp_dir, "index.idx")
    input_str_path = os.path.join(temp_dir, "input.txt")
    if os.path.exists(index_path) and os.path.exists(input_str_path):
        index = faiss.read_index(index_path)
        with open(input_str_path, "r") as f:
            input_str = f.read()
    else:
        index, input_str = process_photo_id_videos(photo_ids, temp_dir=temp_dir, shuffle=True, interval_s=0.05, threshold=0.7)
        faiss.write_index(index, index_path)
        with open(input_str_path, 'w') as f:
            f.write(input_str)
        # return
    if gt_path:
        gt, _ = parse_multimodal(sft_data['output'], cur_clip_index=index)
        with open(gt_path, 'w') as f:
            json.dump(gt, f, ensure_ascii=False)
    prompt = sft_data["instruction"] + sft_data["input"].split('素材视频')[0] + "素材视频：" + input_str
    # result = predict(prompt, model_name=model_name)
    result = predict_online(prompt)
    parsed_dict, _ = parse_multimodal(result, cur_clip_index=index)
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(parsed_dict, f, ensure_ascii=False)
    return parsed_dict


def eval_sft_data(
    sft_data,
    model_name="saves/qwen-8b-sft-0730",
    output_path="ans.json",
    input_path="ori.json",
    gt_path="gt.json",
    video_rqvae_model="video_8_256_0729",
    audio_rqvae_model="audio_8_256_0729"
):
    video_rqvae_model, video_index, frame_index, clip_index = init_video_model(load_faiss=True, model_name=video_rqvae_model)
    init_audio_model(load_faiss=True, model_name=audio_rqvae_model)
    if gt_path:
        gt, _ = parse_multimodal(sft_data['output'])
        with open(gt_path, 'w') as f:
            json.dump(gt, f, ensure_ascii=False)
    prompt = sft_data["instruction"] + sft_data["input"]
    try:
        cur_clip_index, cur_frame_index = get_input_faiss(prompt, clip_index, frame_index, input_path)
    except:
        print("using all databases")
        cur_clip_index = None
        cur_frame_index = None
    # result = predict(prompt, model_name=model_name)
    result = predict_online(prompt)
    parsed_dict, _ = parse_multimodal(result, cur_clip_index=cur_clip_index, cur_frame_index=cur_frame_index)
    # with open("hello.log", 'w') as f:
    #     f.write(result)
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(parsed_dict, f, ensure_ascii=False)
    return parsed_dict

def build_mapping_table(input_text):
    """
    建立视频片段ID到token序列的映射表
    
    Args:
        input_text: 包含视频片段的完整输入字符串
        
    Returns:
        dict: {"id号码": "token序列"}
    """
    # 找到视频片段素材开始位置
    video_start = input_text.find('视频片段素材如下：')
    if video_start == -1:
        return {}
    
    video_content = input_text[video_start:]
    mapping_table = {}
    
    # 按<|clip_start|>分割片段
    clips = video_content.split('<|clip_start|>')[1:]  # 去掉第一个空元素
    
    for clip in clips:
        if not clip.strip():
            continue
            
        # 找到<|clip_end|>位置
        clip_end_pos = clip.find('<|clip_end|>')
        if clip_end_pos == -1:
            continue
            
        # 提取token序列
        tokens = clip[:clip_end_pos]
        
        # 提取ID（在<|clip_end|>后面，逗号前面）
        remaining = clip[clip_end_pos + len('<|clip_end|>'):]
        id_match = remaining.split(',')[0].strip()
        
        if id_match and tokens:
            mapping_table[id_match] = tokens
    
    return mapping_table

def clipid2token(test_output, id_clip_map):
    """
    根据ID序列查询映射表，还原tokens序列
    
    Args:
        test_output: ID序列字符串，如 "08,19,05,04"
        id_clip_map: ID到tokens的映射字典
        
    Returns:
        str: 拼接后的完整tokens序列
    """
    id_list = test_output.split(',')
    result_tokens = []
    
    for clip_id in id_list:
        clip_id = clip_id.strip()
        if clip_id in id_clip_map:
            tokens = id_clip_map[clip_id]
            result_tokens.append(f"<|clip_start|>{tokens}<|clip_end|>")
    
    return ''.join(result_tokens)

def remove_clip_id(input_text: str) -> str:
    """
    去除每个 <|clip_end|> 后紧跟的数字 ID。
    """
    # 匹配 <|clip_end|> 后面跟的一或多个数字，并替换为 <|clip_end|>
    return re.sub(r'(<\|clip_end\|>)(\d+),?', r'\1', input_text)


def eval_sft_data_CID(
    sft_data,
    model_name="saves/qwen-8b-sft-0730",
    output_path="ans.json",
    input_path="ori.json",
    gt_path="gt.json",
    video_rqvae_model="video_8_256_0729",
    audio_rqvae_model="audio_8_256_0729"
):
    ### sft data 里面加了clipid
    video_rqvae_model, video_index, frame_index, clip_index = init_video_model(load_faiss=True, model_name=video_rqvae_model)
    init_audio_model(load_faiss=True, model_name=audio_rqvae_model)

    # 先建立clipid-tokens映射表格
    id_token_map = build_mapping_table(sft_data["input"])
    
    if gt_path:
        ### 把gt的id序列转化成tokens
        gt_tokens = clipid2token(sft_data['output'],id_token_map)
        gt, _ = parse_multimodal(gt_tokens)
        with open(gt_path, 'w') as f:
            json.dump(gt, f, ensure_ascii=False)
    
    prompt = sft_data["instruction"] + sft_data["input"]
    removed_id_input = remove_clip_id(sft_data["input"])
    prompt_rmid = sft_data["instruction"] + removed_id_input

    try:
        cur_clip_index, cur_frame_index = get_input_faiss(prompt_rmid, clip_index, frame_index, input_path)
    except:
        print("using all databases")
        cur_clip_index = None
        cur_frame_index = None
    # result = predict(prompt, model_name=model_name)
    result_ids = predict_online(prompt)
    print('ResultID:', result_ids)
    result_tokens = clipid2token(result_ids,id_token_map)
    parsed_dict, _ = parse_multimodal(result_tokens, cur_clip_index=cur_clip_index, cur_frame_index=cur_frame_index)
    # with open("hello.log", 'w') as f:
    #     f.write(result)
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(parsed_dict, f, ensure_ascii=False)
    return parsed_dict

def eval_full_data(
    full_data,
    model_name="saves/qwen-8b-full-0726-epoch10",
    output_path="ans.json",
    gt_path="gt.json",
    video_rqvae_model="video_8_256_0729",
    audio_rqvae_model="audio_8_256_0729"
):
    video_rqvae_model, video_index, frame_index, clip_index = init_video_model(load_faiss=True, model_name=video_rqvae_model)
    init_audio_model(load_faiss=True, model_name=audio_rqvae_model)
    if gt_path:
        gt, _ = parse_multimodal(full_data['text'])
        with open(gt_path, 'w') as f:
            json.dump(gt, f, ensure_ascii=False)
    prompt = full_data['text'][:600]
    print(prompt)
    result = predict("将下面的视频片段补充完整" + prompt, model_name=model_name)
    print(result)
    ans = prompt + result
    parsed_dict, _ = parse_multimodal(ans)
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(parsed_dict, f, ensure_ascii=False)
    return parsed_dict


def extract_and_fix_clip(
    video_bytes: bytes,
    start_s: float,
    duration_s: float,
    *,
    resolution: str = "1080:1920",   # "width:height"
    fps: int = 30,
    video_codec: str = "libx264",
    crf: int = 23,
    container_ext: str = "mp4"
):
    """
    从 video_bytes 中截取 start_s 开始、持续 duration_s 的片段，
    同时将其转码为统一参数：
      - 分辨率 resolution
      - 帧率 fps
      - 视频编码 video_codec + crf
      - 音频编码 audio_codec + audio_bitrate
    返回转码后的 mp4 bytes。
    """
    # 写入临时输入文件
    with tempfile.NamedTemporaryFile(suffix=f'.{container_ext}', delete=False) as in_tmp:
        in_tmp.write(video_bytes)
        in_path = in_tmp.name

    try:
        cmd = [
            "ffmpeg",
            "-y",
            "-ss", str(start_s),
            "-t", str(duration_s),
            "-i", in_path,
            "-vf", f"scale={resolution},fps={fps},setsar=1",
            "-c:v", video_codec,
            "-crf", str(crf),
            "-preset", "veryfast",
            "-an",
            "-f", "mpegts", "pipe:1"
        ]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        ts_bytes, err = proc.communicate(video_bytes)
        if proc.returncode != 0:
            raise RuntimeError(err.decode())
        # logger.info(f"{err}")
        return ts_bytes
    finally:
        # 清理临时文件
        os.remove(in_path)


def concat_clips(clip_bytes_list: List[bytes]):
    """
    将多个已经“抽取并统一参数”的片段拼接成一个完整视频，直接无损 copy。
    """
    cmd = [
        "ffmpeg",
        "-y",
        "-f", "mpegts",
        "-i", "pipe:0",
        "-c:v", "copy",           # 0805新加
        "-c:a", "copy",
        "-vsync", "2",
        "-max_delay", "5000000",  # 增加音频延迟缓冲
        "-bufsize", "1000k",      # 增加缓冲区大小
        "-f", "mpegts", "pipe:1"
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    all_ts = b"".join(clip_bytes_list)
    ts_bytes, err = proc.communicate(all_ts)
    if proc.returncode != 0:
        raise RuntimeError(f"concat_clips 失败：{err}")
    # logger.info(f"{err}")
    return ts_bytes


def merge_clips(clips):
    """
    按照 clips 原始顺序分组合并（仅合并连续相同 key 的段）：
      - 相同 clip 键（用 tuple 表示）且连续出现的会合并到同一组
      - clip 为空的，也会当成一个 key=() 的组，每次遇到都单独成组
      - 各组内部的 text 和 frame，按遇到顺序 append
    """
    merged: List[Dict[str, Any]] = []
    current_key: Tuple = None
    current_group: Dict[str, Any] = None
    for item in clips:
        key = tuple(item.get('clip', []))  # 当前条目的 group key
        # 若 key 不同，则先把上一个组推入结果（若存在），再新开组
        if key and key != current_key:
            if current_group is not None:
                merged.append(current_group)
            # 初始化新组
            current_group = {
                'clip': list(key),
                'text': [],
                'frame': []
            }
            current_key = key
        # 向当前组累加 text 和 frame
        current_group['text'].extend(item.get('text', []))
        current_group['frame'].extend(item.get('frame', []))
    # 把最后一个组也加上
    if current_group is not None and current_group.get('clip')!=[]:
        merged.append(current_group)
    return merged


def wav_bytes_to_aac_bytes(wav_bytes: bytes):
    proc = subprocess.Popen(
        ["ffmpeg", "-y", "-f", "wav", "-i", "pipe:0", "-c:a", "aac", "-b:a", "128k", "-f", "adts", "pipe:1"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    out, err = proc.communicate(input=wav_bytes)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg AAC encode failed:\n{err.decode()}")
    return out


def tts(text):
    texts = [text]
    wavs = chat.infer(texts, params_infer_code=params_infer_code)
    audio_buffer = io.BytesIO()
    sample_rate = 24000
    torchaudio.save(audio_buffer, torch.from_numpy(wavs[0]).unsqueeze(0), sample_rate, format="wav")
    duration_sec = len(wavs[0]) / sample_rate
    audio_bytes = wav_bytes_to_aac_bytes(audio_buffer.getvalue())
    return audio_bytes, duration_sec


def silence(duration_sec):
    sample_rate = 24000
    silence = torch.zeros(int(duration_sec * sample_rate))
    audio_buffer = io.BytesIO()
    torchaudio.save(audio_buffer, silence.unsqueeze(0), sample_rate, format="wav")
    audio_bytes = wav_bytes_to_aac_bytes(audio_buffer.getvalue())
    return audio_bytes, duration_sec


def write_audio_pipe(wfd, tts_bytes):
    """
    写入音频数据到管道
    """
    os.write(wfd, tts_bytes)
    os.close(wfd)  # 关闭管道的写端


def merge_video_tts_mpegts(video_bytes: bytes, tts_bytes: bytes, video_time: float, audio_time: float):
    """
    合并 MPEG-TS 格式的视频字节流和 TTS 音频字节流
    """

    # speed = video_time / audio_time  # 视频加速/减速倍速因子
    # logger.info(f"{video_time}-{audio_time}-{speed}")

    # 创建管道用于音频数据
    rfd, wfd = os.pipe()

    # 启动一个线程，将音频数据写入管道
    write_thread = threading.Thread(target=write_audio_pipe, args=(wfd, tts_bytes))
    write_thread.start()

    # 构造 FFmpeg 命令
    cmd = [
        "ffmpeg",
        "-y",
        "-f", "mpegts", "-i", "pipe:0",  # 输入0：视频
        "-f", "aac", "-i", f"pipe:{rfd}",  # 输入1：音频
        # "-filter_complex", f"[0:v]setpts=(PTS-STARTPTS)/{speed}[v]",  # 缩放视频时间轴
        # "-map", "[v]",  # 映射处理后的视频
        "-map", "0:v:0", 
        "-map", "1:a:0",    # 使用 pipe 中的音频
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "23",
        "-c:a", "copy",  # 音频不重新编码
        "-f", "mpegts", "pipe:1"  # 输出到管道
    ]

    # 使用 Popen 启动 FFmpeg 进程
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        pass_fds=(rfd,)
    )

    # 将视频字节流作为输入传递给 FFmpeg
    out_bytes, err = proc.communicate(input=video_bytes)

    # 等待写入线程完成
    write_thread.join()

    # 关闭音频管道的读取端
    os.close(rfd)

    # 检查 FFmpeg 进程是否执行成功
    if proc.returncode != 0:
        raise RuntimeError(f"FFmpeg failed:\n{err.decode()}")

    return out_bytes


def process_clip(clip, client, extract_method="clip", temp_dir="temp_video"):
    try:
        text = "\n".join(clip['text'])
        if extract_method == 'clip':
            clip_key = clip['clip'][0]
            duration_time = clip_key % 1000
            clip_key = int(clip_key / 1000)
            start_time = clip_key % 10000
            photo_id = int(clip_key / 10000)
            _, video_bytes = client.download_bytes_from_s3(f"{photo_id}.mp4")
            fixed_clip = extract_and_fix_clip(video_bytes, start_time, duration_time)
        elif extract_method == 'clip_local':
            clip_key = clip['clip'][0]
            duration_time = (clip_key % 100000) / 100.0
            clip_key = int(clip_key / 100000)
            start_time = (clip_key % 1000000) / 100.0
            video_index = int(clip_key / 1000000)
            with open(os.path.join(temp_dir, f"{video_index}.mp4"), "rb") as f:
                video_bytes = f.read()
            fixed_clip = extract_and_fix_clip(video_bytes, start_time, duration_time)
        elif extract_method == 'frame':
            frame_keys = clip['frame']
            fixed_frames = []
            duration_time = 0
            for frame_key in frame_keys:
                start_time = frame_key % 10000
                photo_id = int(frame_key / 10000)
                _, video_bytes = client.download_bytes_from_s3(f"{photo_id}.mp4")
                fixed_frame = extract_and_fix_clip(video_bytes, start_time, 1)
                fixed_frames.append(fixed_frame)
                duration_time += 1
            fixed_clip = concat_clips(fixed_frames)
        # TODO TTS
        raw_text = text.replace('\n', '')
        if raw_text != "":
            tts_bytes, audio_time = tts(raw_text)
        else:
            tts_bytes, audio_time = silence(duration_time)
        # tts_bytes, audio_time = silence(duration_time)
        # fixed_clip = merge_video_tts_mpegts(fixed_clip, tts_bytes, duration_time, audio_time)
        logger.info(f"<<<{text}>>> done")
    except Exception as e:
        logger.warning(f"{e}")
        return b"", text, 0
    return fixed_clip, text, audio_time


def fmt_ts(t: float):
    """把秒数转成 SRT 时间戳格式 HH:MM:SS,mmm"""
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    ms = int((t - int(t)) * 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def burn_in_subtitles(
    input_video,
    srt_file: str,
    output_video: str,
    font_size: int = 9,
    outline: int = 1,
    crf: int = 23,
    preset: str = "veryfast"
):
    """
    Uses FFmpeg to hardcode subtitles from an SRT file into a video.

    Parameters:
    - input_video: Path to the input video file.
    - srt_file: Path to the SRT subtitle file.
    - output_video: Path for the output video with burned-in subtitles.
    - font_size: Font size for subtitles.
    - outline: Outline thickness for subtitle text.
    - crf: Constant Rate Factor for video quality.
    - preset: Encoding preset for speed/quality tradeoff.
    """
    # vf_filter = (
    #     f"subtitles={srt_file}:"
    #     f"force_style='fontfile=/data/phd/qinsizhong/popular-fonts/yahei.ttf,FontSize={font_size},Outline={outline}'"
    # )
    vf_filter = (
        f"subtitles={srt_file}:fontsdir=/data/phd/qinsizhong/popular-fonts:"
        f"force_style='Fontname=Microsoft YaHei,FontSize={font_size},Outline={outline}'"
    )
    cmd = [
        "ffmpeg", "-y",
        "-f", "mpegts", "-i", "pipe:0",
        "-vf", vf_filter,
        "-c:v", "libx264", "-crf", str(crf), "-preset", preset,
        "-c:a", "copy",
        "-movflags", "+faststart",
        output_video
    ]

    result = subprocess.run(cmd, input=input_video, capture_output=True)
    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError("Failed to burn in subtitles.")
    else:
        print(f"Subtitled video saved to {output_video}.")


def generate_video(clips, extract_method="frame", out_path="test.mp4", temp_dir="temp_video"):
    logger.info(f"{extract_method} generation begin!")
    clips = merge_clips(clips)
    client = BlobStoreClient('video-def')
    clip_info_list = [process_clip(c, client, extract_method, temp_dir) for c in clips]
    generated_clip_list = []
    t_start = 0
    srt_path = out_path.replace(".mp4", ".srt")
    with open(srt_path, "w", encoding="utf-8") as f:
        for idx, (clip, text, dur) in enumerate(clip_info_list, start=1):
            if dur == 0 or len(clip) < 1000:
                continue
            generated_clip_list.append(clip)
            t_end = t_start + dur
            f.write(f"{idx}\n")
            f.write(f"{fmt_ts(t_start)} --> {fmt_ts(t_end)}\n")
            f.write(text + "\n\n")
            t_start = t_end
    len_clip = [len(clip) for clip in generated_clip_list]
    print(len_clip)
    result = concat_clips(generated_clip_list)
    burn_in_subtitles(result, srt_path, out_path)
