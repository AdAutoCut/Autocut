#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
from typing import List, Dict, Optional

import torch


def load_json_int_key(path: str) -> Dict[int, object]:
    with open(path, "r") as f:
        d = json.load(f)
    # JSON 键是字符串，这里转回 int
    return {int(k): v for k, v in d.items()}


def save_json_int_key(path: str, d: Dict[int, object]) -> None:
    # 写回时键会变成字符串；这符合 JSON 常规
    with open(path, "w") as f:
        json.dump({str(k): v for k, v in d.items()}, f, indent=4, ensure_ascii=False)


def assert_same_feature_shape(name: str, tensors: List[torch.Tensor]) -> None:
    """确保除了第0维（样本维）外，其他维度一致。"""
    if not tensors:
        raise ValueError(f"{name}: 空列表")
    base = tensors[0].shape[1:]
    for i, t in enumerate(tensors):
        if t.shape[1:] != base:
            raise ValueError(
                f"{name}: 第 {i} 个子库特征维不一致：{tuple(t.shape)} vs {tuple(tensors[0].shape)}"
            )


def parse_cast_dtype(s: Optional[str]) -> Optional[torch.dtype]:
    if s is None:
        return None
    s = s.lower()
    if s in ("fp32", "float32"):
        return torch.float32
    if s in ("fp16", "float16", "half"):
        return torch.float16
    if s in ("bf16", "bfloat16"):
        return torch.bfloat16
    raise ValueError(f"不支持的 --cast 值：{s}")


def merge_index_maps(maps: List[Dict[int, object]]) -> Dict[int, object]:
    """按传入顺序、各自内部按键升序，重建 0..N-1 的新索引。"""
    merged: Dict[int, object] = {}
    idx = 0
    for mp in maps:
        for k in sorted(mp.keys()):
            merged[idx] = mp[k]
            idx += 1
    return merged


def load_one_dir(d: str, thr_str: str, cast_dtype: Optional[torch.dtype]):
    """读取单个目录下四个 .pt 与三个 .json，并做基本一致性检查。"""
    must_files = [
        "frame_db.pt",
        "video_db.pt",
        "audio_db.pt",
        f"clip_db_{thr_str}.pt",
        "f.json",
        "v.json",
        f"c_{thr_str}.json",
    ]
    for fn in must_files:
        path = os.path.join(d, fn)
        if not os.path.exists(path):
            raise FileNotFoundError(f"{d}: 缺少文件 {fn}")

    frame_db = torch.load(os.path.join(d, "frame_db.pt"), map_location="cpu")
    video_db = torch.load(os.path.join(d, "video_db.pt"), map_location="cpu")
    audio_db = torch.load(os.path.join(d, "audio_db.pt"), map_location="cpu")
    clip_db = torch.load(os.path.join(d, f"clip_db_{thr_str}.pt"), map_location="cpu")

    # 可选降精度（仅对视觉相关向量）；音频通常保持原状更稳健
    if cast_dtype is not None:
        frame_db = frame_db.to(dtype=cast_dtype)
        video_db = video_db.to(dtype=cast_dtype)
        clip_db = clip_db.to(dtype=cast_dtype)

    f_map = load_json_int_key(os.path.join(d, "f.json"))
    v_map = load_json_int_key(os.path.join(d, "v.json"))
    c_map = load_json_int_key(os.path.join(d, f"c_{thr_str}.json"))

    # 子库内部一致性
    if len(f_map) != frame_db.shape[0]:
        raise ValueError(f"{d}: f.json 键数({len(f_map)}) != frame_db 行数({frame_db.shape[0]})")
    if len(v_map) != video_db.shape[0]:
        raise ValueError(f"{d}: v.json 键数({len(v_map)}) != video_db 行数({video_db.shape[0]})")
    if len(c_map) != clip_db.shape[0]:
        raise ValueError(f"{d}: c_{thr_str}.json 键数({len(c_map)}) != clip_db 行数({clip_db.shape[0]})")
    if audio_db.shape[0] != video_db.shape[0]:
        raise ValueError(f"{d}: audio_db 行数({audio_db.shape[0]}) != video_db 行数({video_db.shape[0]})")
    if audio_db.dim() != 2:
        raise ValueError(f"{d}: audio_db 维度异常，期望2维，得到 {audio_db.dim()} 维")
    if audio_db.shape[1] != 2048:
        raise ValueError(f"{d}: audio_db 特征维不是 2048（实际 {audio_db.shape[1]}）")

    return frame_db, video_db, audio_db, clip_db, f_map, v_map, c_map


def main():
    parser = argparse.ArgumentParser(
        description="合并多个检索库产物（四个 .pt + 三个 .json），保序拼接生成总库。"
    )
    parser.add_argument(
        "-s",
        "--src",
        nargs="+",
        required=True,
        help="源目录列表（按此顺序拼接）",
    )
    parser.add_argument(
        "-o",
        "--out",
        required=True,
        help="输出目录（不存在将自动创建）",
    )
    parser.add_argument(
        "--thr",
        default="85",
        help="片段阈值标签（如 threshold=0.85 则为 '85'；需与各源目录一致）",
    )
    parser.add_argument(
        "--cast",
        default=None,
        help="可选降精度：fp32 / fp16 / bf16（仅对 frame/video/clip 生效）",
    )
    args = parser.parse_args()

    src_dirs: List[str] = args.src
    out_dir: str = args.out
    thr_str: str = args.thr
    cast_dtype = parse_cast_dtype(args.cast)

    os.makedirs(out_dir, exist_ok=True)

    # 逐目录加载
    frame_parts: List[torch.Tensor] = []
    video_parts: List[torch.Tensor] = []
    audio_parts: List[torch.Tensor] = []
    clip_parts: List[torch.Tensor] = []

    f_maps: List[Dict[int, object]] = []
    v_maps: List[Dict[int, object]] = []
    c_maps: List[Dict[int, object]] = []

    print("即将合并以下目录（按此顺序）：")
    for d in src_dirs:
        print(" -", d)

    for d in src_dirs:
        (
            frame_db,
            video_db,
            audio_db,
            clip_db,
            f_map,
            v_map,
            c_map,
        ) = load_one_dir(d, thr_str=thr_str, cast_dtype=cast_dtype)

        frame_parts.append(frame_db)
        video_parts.append(video_db)
        audio_parts.append(audio_db)
        clip_parts.append(clip_db)

        f_maps.append(f_map)
        v_maps.append(v_map)
        c_maps.append(c_map)

    # 跨目录特征维一致性
    assert_same_feature_shape("frame_db", frame_parts)
    assert_same_feature_shape("video_db", video_parts)
    assert_same_feature_shape("clip_db", clip_parts)
    # audio 的特征维已在 load_one_dir() 内保证为 2048

    # 拼接四个库
    frame_db_merged = torch.cat(frame_parts, dim=0)
    video_db_merged = torch.cat(video_parts, dim=0)
    audio_db_merged = torch.cat(audio_parts, dim=0)
    clip_db_merged = torch.cat(clip_parts, dim=0)

    # 重建三份索引（0..N-1）
    f_merged = merge_index_maps(f_maps)
    v_merged = merge_index_maps(v_maps)
    c_merged = merge_index_maps(c_maps)

    # 终检
    if len(f_merged) != frame_db_merged.shape[0]:
        raise RuntimeError(
            f"合并后 f.json 键数({len(f_merged)}) != frame_db 行数({frame_db_merged.shape[0]})"
        )
    if len(v_merged) != video_db_merged.shape[0]:
        raise RuntimeError(
            f"合并后 v.json 键数({len(v_merged)}) != video_db 行数({video_db_merged.shape[0]})"
        )
    if len(c_merged) != clip_db_merged.shape[0]:
        raise RuntimeError(
            f"合并后 c_{thr_str}.json 键数({len(c_merged)}) != clip_db 行数({clip_db_merged.shape[0]})"
        )
    if audio_db_merged.shape[0] != video_db_merged.shape[0]:
        raise RuntimeError(
            f"合并后 audio_db 行数({audio_db_merged.shape[0]}) != video_db 行数({video_db_merged.shape[0]})"
        )

    # 写出
    torch.save(frame_db_merged, os.path.join(out_dir, "frame_db.pt"))
    torch.save(video_db_merged, os.path.join(out_dir, "video_db.pt"))
    torch.save(audio_db_merged, os.path.join(out_dir, "audio_db.pt"))
    torch.save(clip_db_merged, os.path.join(out_dir, f"clip_db_{thr_str}.pt"))

    save_json_int_key(os.path.join(out_dir, "f.json"), f_merged)
    save_json_int_key(os.path.join(out_dir, "v.json"), v_merged)
    save_json_int_key(os.path.join(out_dir, f"c_{thr_str}.json"), c_merged)

    print("合并完成：", out_dir)
    print("最终形状：")
    print("  frame_db:", tuple(frame_db_merged.shape))
    print("  video_db:", tuple(video_db_merged.shape))
    print("  audio_db:", tuple(audio_db_merged.shape))
    print("  clip_db :", tuple(clip_db_merged.shape))


if __name__ == "__main__":
    main()
