import json
import os
import subprocess
from typing import Optional, List

# 路径根据你自己的实际情况修改
PRED_AUD_PATH = "/data/phd/qinsizhong/llm_factory_test/baselines/results/F2___audrank_atc_embsft_wgt.jsonl"   # 含有 sample_id, photo_id, predict_aud_id 的 jsonl
VIDEO_DIR = "/data/phd/qinsizhong/llm_factory_test/user_study_vids3"      # 存放 atc_c{chunk}_sid{sample_id}_model_frame.mp4
AUDIO_DIR = "/data/phd/miltonzhou/audio_process/bgm_UserStudy"           # 存放 {aud_id}.wav
OUTPUT_DIR = "/data/phd/qinsizhong/llm_factory_test/Final_user_study_atc_2"    # 输出新视频目录

os.makedirs(OUTPUT_DIR, exist_ok=True)


def merge_video_audio(video_path: str, audio_path: str, output_path: str) -> None:
    """
    用 ffmpeg 把“视频原音频 + 新音频”混合成一个音轨：
    - 视频流直接 copy
    - 音频用 amix 混合后转 aac
    - 时长取较短一方
    注意：如果视频没有音频轨道，这个命令会失败，到时候我们再加兜底分支。
    """
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


def find_video_for_sample(sample_id: int) -> Optional[str]:
    """
    根据 sample_id 在 VIDEO_DIR 里找到对应的视频：
    只匹配形如 atc_c0_sid0_model_frame.mp4 这种以 atc 开头的文件，
    且要求文件名中包含精确的 `_sid{sample_id}_` 模式，避免 11 匹配到 110.
    """
    sid_token = f"_sid{sample_id}_"
    candidates: List[str] = []

    for name in os.listdir(VIDEO_DIR):
        # 只要 atc 开头 + .mp4 结尾
        if not name.lower().endswith(".mp4"):
            continue
        if not name.startswith("atc"):
            continue

        # 精确匹配 sid：要求文件名里有 `_sid{sample_id}_`
        if sid_token in name:
            candidates.append(os.path.join(VIDEO_DIR, name))

    if not candidates:
        return None

    if len(candidates) > 1:
        print(f"[warning] sample_id={sample_id} 找到多个视频文件，使用第一个：")
        for c in candidates:
            print("   ", c)

    return candidates[0]


def main():
    total = 0
    done = 0
    skipped_no_audio = 0
    skipped_missing_file = 0

    with open(PRED_AUD_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            obj = json.loads(line)
            sample_id = obj.get("sample_id")
            # photo_id 现在不再用于找视频，但保留在命名里也可以追踪
            photo_id = obj.get("photo_id")
            pred_list = obj.get("predict_aud_id", [])

            total += 1

            # 基本防御：没有 sample_id 或没有预测列表，跳过
            if sample_id is None or not pred_list:
                skipped_no_audio += 1
                continue

            # 只取 Top1
            top1_aud_id = pred_list[0]
            if top1_aud_id in (-1, None):
                skipped_no_audio += 1
                continue

            # 用 sample_id 找视频（通过 sid{sample_id}）
            video_path = find_video_for_sample(sample_id)
            if video_path is None:
                print(f"[warning] video not found for sample_id={sample_id}")
                skipped_missing_file += 1
                continue

            audio_path = os.path.join(AUDIO_DIR, f"{top1_aud_id}.wav")
            if not os.path.exists(audio_path):
                print(f"[warning] audio not found: {audio_path}")
                skipped_missing_file += 1
                continue

            # 输出文件名：用 sid + aud_id（photo_id 只是可选信息）
            if photo_id is not None:
                output_name = f"sid{sample_id}__p{photo_id}__a{top1_aud_id}_atc.mp4"
            else:
                output_name = f"sid{sample_id}__a{top1_aud_id}_atc.mp4"

            output_path = os.path.join(OUTPUT_DIR, output_name)

            print(f"processing sample_id={sample_id}, video={os.path.basename(video_path)}, aud_id={top1_aud_id}")
            merge_video_audio(video_path, audio_path, output_path)
            done += 1

    print(f"总样本数: {total}")
    print(f"成功合成: {done}")
    print(f"跳过(预测为空或 top1 无效): {skipped_no_audio}")
    print(f"跳过(找不到视频或音频文件): {skipped_missing_file}")


if __name__ == "__main__":
    main()





######################################## 下面是gpt4o的合成视频 #####################################
# !/usr/bin/env python3
# -*- coding: utf-8 -*-

# import json
# import os
# import re
# import subprocess
# from typing import Optional

# # -------- 配置：根据你自己的路径修改 --------
# MATCHES_PATH = "/data/phd/qinsizhong/llm_factory_test/baselines/results/video_audio_matches.jsonl"
# VIDEO_DIR    = "/data/phd/qinsizhong/llm_factory_test/user_study_vids3"        # 存放 mp4
# AUDIO_DIR    = "/data/phd/miltonzhou/audio_process/bgm_UserStudy"              # 存放 wav
# OUTPUT_DIR   = "/data/phd/qinsizhong/llm_factory_test/Final_user_study_gpt4o_2" # 输出目录

# os.makedirs(OUTPUT_DIR, exist_ok=True)

# # -------- 小工具函数 --------

# _sid_pattern = re.compile(r"_sid(\d+)_")

# def parse_sample_id_from_video(filename: str) -> Optional[int]:
#     """
#     从类似 gpt4o_c1_sid23_model_frame.mp4 解析 sample_id = 23
#     要求文件名中存在 `_sid{num}_`
#     """
#     m = _sid_pattern.search(filename)
#     if not m:
#         return None
#     return int(m.group(1))


# def get_audio_id(audio_filename: str) -> str:
#     """
#     从 157385222572.wav -> '157385222572'
#     """
#     return os.path.splitext(os.path.basename(audio_filename))[0]


# def merge_video_audio(video_path: str, audio_path: str, output_path: str) -> None:
#     """
#     用 ffmpeg 把视频原音频和新音频混合在一起：
#     - 视频流直接 copy
#     - 音频用 amix 混音后转 aac
#     - 时长取较短一方
#     """
#     cmd = [
#         "ffmpeg",
#         "-y",
#         "-i", video_path,   # 0: 视频 + 原音
#         "-i", audio_path,   # 1: 新的音频
#         "-filter_complex", "[0:a][1:a]amix=inputs=2:duration=first:dropout_transition=0[aout]",
#         "-map", "0:v:0",    # 用原来的视频流
#         "-map", "[aout]",   # 用混好的音频
#         "-c:v", "copy",
#         "-c:a", "aac",
#         "-shortest",
#         output_path,
#     ]
#     subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)

# # -------- 主逻辑：逐行处理 jsonl，拼接 Top1 音频 --------

# def main():
#     total = 0            # 满足 "gpt 开头" 且有 matches 的样本数
#     done = 0
#     skipped_parse = 0
#     skipped_missing = 0
#     skipped_empty = 0

#     with open(MATCHES_PATH, "r", encoding="utf-8") as f:
#         for line in f:
#             line = line.strip()
#             if not line:
#                 continue

#             obj = json.loads(line)
#             video_file = obj.get("video_file")
#             matches = obj.get("matches", [])

#             # 先筛掉非 gpt 开头的
#             if not video_file or not video_file.startswith("gpt"):
#                 continue

#             # 到这里再计数（只统计 gpt 开头且有 matches 的）
#             total += 1

#             if not matches:
#                 skipped_empty += 1
#                 continue

#             # 解析 sample_id
#             sample_id = parse_sample_id_from_video(video_file)
#             if sample_id is None:
#                 print(f"[warning] 无法从文件名解析 sample_id: {video_file}")
#                 skipped_parse += 1
#                 continue

#             # Top1 audio（假设 matches 已按 score 从高到低）
#             top_match = matches[0]
#             audio_file = top_match.get("audio_file")
#             if not audio_file:
#                 skipped_empty += 1
#                 continue

#             audio_id = get_audio_id(audio_file)

#             video_path = os.path.join(VIDEO_DIR, video_file)
#             audio_path = os.path.join(AUDIO_DIR, audio_file)

#             if not os.path.exists(video_path):
#                 print(f"[warning] video not found: {video_path}")
#                 skipped_missing += 1
#                 continue

#             if not os.path.exists(audio_path):
#                 print(f"[warning] audio not found: {audio_path}")
#                 skipped_missing += 1
#                 continue

#             output_name = f"sid{sample_id}__a{audio_id}.mp4"
#             output_path = os.path.join(OUTPUT_DIR, output_name)

#             print(f"processing sid={sample_id}, video={video_file}, audio={audio_file}")
#             merge_video_audio(video_path, audio_path, output_path)
#             done += 1

#     print(f"总样本数(gpt 开头): {total}")
#     print(f"成功合成: {done}")
#     print(f"跳过：无法解析 sid         = {skipped_parse}")
#     print(f"跳过：matches 为空 / 无音频 = {skipped_empty}")
#     print(f"跳过：找不到视频或音频文件  = {skipped_missing}")


# if __name__ == "__main__":
#     main()
