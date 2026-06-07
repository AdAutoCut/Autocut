#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse

import torch
import torchaudio
import ChatTTS


def main():
    parser = argparse.ArgumentParser(description="Test ChatTTS voice with fixed seed")
    parser.add_argument("--text", required=True, help="要合成的中文台词")
    parser.add_argument("--seed", type=int, default=3333, help="控制音色的随机种子")
    parser.add_argument("--out-dir", type=str, default="./tts_test", help="输出目录")
    parser.add_argument("--name", type=str, default=None, help="输出 wav 文件名（可选）")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # 固定种子，控制音色
    torch.manual_seed(args.seed)

    chat = ChatTTS.Chat()
    chat.load(compile=False, custom_path="/data/phd/qinsizhong/ad_edit_inference/tools")

    # 在这个 seed 下采一个说话人
    spk_emb = chat.sample_random_speaker()

    # 关键：用 InferCodeParams，而不是 dict
    params_infer_code = ChatTTS.Chat.InferCodeParams(
        spk_emb=spk_emb,
        # temperature 等参数可以用默认的，也可以自己设：
        # temperature=[0.3, 0.7],
    )

    wavs = chat.infer(
        [args.text],
        use_decoder=True,
        params_infer_code=params_infer_code,
    )

    wav = wavs[0]           # numpy array, shape [T]
    sample_rate = 24000

    # 5) 保存 wav
    if args.name is None:
        wav_name = f"tts_seed{args.seed}.wav"
    else:
        wav_name = args.name if args.name.endswith(".wav") else args.name + ".wav"

    wav_path = os.path.join(args.out_dir, wav_name)
    wav_tensor = torch.from_numpy(wav).unsqueeze(0)  # [1, T]
    torchaudio.save(wav_path, wav_tensor, sample_rate)
    print(f"[OK] 保存语音到: {wav_path}")

    # 6) 顺便把音色向量也保存成 .pt，后面项目里可以直接复用
    spk_path = os.path.join(args.out_dir, f"spk_seed{args.seed}.pt")
    torch.save(spk_emb, spk_path)
    print(f"[OK] 保存说话人音色到: {spk_path}")
    print(f"seed={args.seed}, wav 长度={len(wav) / sample_rate:.2f}s")


if __name__ == "__main__":
    main()
