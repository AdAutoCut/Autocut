#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
多任务 SFT & Baseline 数据构造器（ShareGPT 格式，三版本对齐，含完全可复现随机性 + 题目检索表 manifest）

说明：
- 所有随机行为只依赖显式传入的 seed；
- 不使用 Python 内置 hash() 的随机化特性；
- vid2text / vid_sort / vid_select 的 atc / text / mm 三份数据严格对齐且可复现；
- vid_select 的负样本采样、vid_sort 的乱序顺序在相同输入和 seed 下完全一致；
- 额外生成 manifest.json：
    * 每条记录对应一道题（一个样本）；
    * 记录 task 类型、原始广告标识、三份输出文件中的索引；
    * 给出本题中所有「局部编号 -> 原始片段信息」映射；
    * 给出该任务的 gold label（排序/选择/台词/音频），方便评测与回溯剪辑。
- 新增：在每条样本（atc / baseline_text / baseline_mm）顶层注入 sample_id / task / ad_key，便于离线检索回溯。
"""

import argparse
import json
import sys
import hashlib
from collections import defaultdict, Counter
from typing import Any, Dict, Iterable, List, Optional, Tuple
from random import Random


# ========== 基础 IO ==========

def read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                sys.stderr.write(f"[WARN] JSON 解析失败（第%d行）: %s\n" % (ln, e))
                continue


def write_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# ========== 工具函数 ==========

NULL_SENTINELS = {"", "null", "None", "无", "(null)", "(none)"}


def norm_str(x: Optional[str]) -> Optional[str]:
    if x is None:
        return None
    if not isinstance(x, str):
        x = str(x)
    s = x.strip()
    if not s:
        return None
    if s.lower() in {z.lower() for z in NULL_SENTINELS}:
        return None
    return s


def majority_nonnull(values: List[Any]) -> str:
    cleaned: List[str] = []
    for v in values:
        if isinstance(v, str):
            s = norm_str(v)
        else:
            s = norm_str(str(v))
        if s is not None:
            cleaned.append(s)
    if not cleaned:
        return "无"
    cnt = Counter(cleaned)
    mode_val, _ = sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0]))[0]
    return mode_val


def ensure_audio_wrapper(aud_tok: Optional[str]) -> Optional[str]:
    if aud_tok is None:
        return None
    if not isinstance(aud_tok, str):
        aud_tok = str(aud_tok)
    s = aud_tok.strip()
    if not s:
        return None
    if s.lower() in {z.lower() for z in NULL_SENTINELS}:
        return None
    if "<|audio_start|>" in s and "<|audio_end|>" in s:
        return s
    return f"<|audio_start|>{s}<|audio_end|>"


def safe_line_text(t: Optional[str]) -> str:
    if t is None or t == "null":
        return "（无台词）"
    s = str(t).strip()
    return s if s else "（无台词）"


def sort_key_clip(rec: Dict[str, Any]) -> Tuple[int, Any]:
    cid = rec.get("clip_id")
    try:
        cid = int(cid)
    except Exception:
        cid = 10 ** 9
    fid = rec.get("frame_id")
    return (cid, fid if fid is not None else "")


def ensure_clip_wrapper_vtok(v_tok: Optional[str]) -> str:
    base = v_tok if isinstance(v_tok, str) and v_tok.strip() else "<|frame_start|><|frame_end|>"
    return f"<|clip_start|><|video_start|>{base}<|video_end|><|clip_end|>"


def get_caption(rec: Dict[str, Any]) -> str:
    cap = norm_str(rec.get("caption"))
    return cap if cap is not None else "（无画面描述）"


def get_frame_placeholder(rec: Dict[str, Any]) -> str:
    fid = rec.get("frame_id")
    if fid is None:
        return "[NO_FRAME_ID]"
    return f"[{fid}]"


def stable_subseed(*parts: Any) -> int:
    """
    使用 md5 基于 (parts) 生成稳定子种子，避免 Python 内置 hash 的随机化。
    取低 32 bit 作为 Random 的种子。
    """
    s = "|".join(str(p) for p in parts)
    h = hashlib.md5(s.encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def make_clip_orig(rec: Dict[str, Any]) -> Dict[str, Any]:
    """提取原始片段的关键信息，写入 manifest。"""
    return {
        "chunk_id": rec.get("chunk_id"),
        "ad_id": rec.get("ad_id"),
        "clip_id": rec.get("clip_id"),
        "frame_id": rec.get("frame_id"),
        "caption": rec.get("caption"),
        "text": rec.get("text"),
        "v_tok": rec.get("v_tok"),
        "aud_tok": rec.get("aud_tok"),
    }


def attach_meta(item: Optional[Dict[str, Any]],
                sample_id: int,
                task_name: str,
                ad_key: Tuple[Any, Any]) -> Optional[Dict[str, Any]]:
    """在样本顶层注入检索元信息；item 可能为 None。"""
    if item is None:
        return None
    item["sample_id"] = sample_id
    item["task"] = task_name
    item["ad_key"] = {"chunk_id": ad_key[0], "ad_id": ad_key[1]}
    return item


# ========== 多任务生成器 ==========

class SFTDataGenerator:
    def __init__(
        self,
        input_path: str,
        task_ratios: Dict[str, float],
        seed: int = 42,
        neg_pos_ratio: float = 1.0,
    ) -> None:
        self.input_path = input_path
        self.task_ratios = task_ratios
        self.seed = seed
        self.neg_pos_ratio = neg_pos_ratio

        self.by_ad: Dict[Tuple[Any, Any], List[Dict[str, Any]]] = defaultdict(list)
        self.all_clips: List[Tuple[Tuple[Any, Any], Dict[str, Any]]] = []

        self._load()

    # ----- 加载 -----

    def _load(self) -> None:
        for rec in read_jsonl(self.input_path):
            chunk_id = rec.get("chunk_id")
            ad_id = rec.get("ad_id")
            if chunk_id is None or ad_id is None:
                continue
            key = (chunk_id, ad_id)
            self.by_ad[key].append(rec)

        empty_keys = [k for k, g in self.by_ad.items() if not g]
        for k in empty_keys:
            self.by_ad.pop(k, None)

        for key, group in self.by_ad.items():
            for rec in group:
                self.all_clips.append((key, rec))

    # ----- 任务分配（互斥） -----

    def _assign_tasks(self) -> Dict[str, List[Tuple[Tuple[Any, Any], List[Dict[str, Any]]]]]:
        ad_items = list(self.by_ad.items())
        rng = Random(self.seed)
        rng.shuffle(ad_items)

        total = len(ad_items)
        task_to_pairs: Dict[str, List[Tuple[Tuple[Any, Any], List[Dict[str, Any]]]]] = {
            t: [] for t in self.task_ratios
        }

        cursor = 0
        for task, ratio in self.task_ratios.items():
            if ratio <= 0:
                continue
            n = int(total * ratio)
            if n <= 0 or cursor >= total:
                continue
            end = min(cursor + n, total)
            task_to_pairs[task] = ad_items[cursor:end]
            cursor = end

        return task_to_pairs

    # ================== vid2text：三份对齐 + manifest ==================

    def build_vid2text_triplet(
        self,
        key: Tuple[Any, Any],
        group: List[Dict[str, Any]],
    ):
        clips = sorted(group, key=sort_key_clip)
        num_clips = len(clips)
        if num_clips == 0:
            return None, None, None, None

        gpt_lines = [safe_line_text(rec.get("text")) for rec in clips]
        if all(line == "（无台词）" for line in gpt_lines):
            return None, None, None, None

        product = majority_nonnull([rec.get("product") for rec in clips])
        brand = majority_nonnull([rec.get("brand") for rec in clips])
        features = majority_nonnull([rec.get("features") for rec in clips])
        gpt_value = "\n".join(gpt_lines)

        # atc
        header_atc = (
            "你是专业广告台词助手。看到每个视频片段的视频帧 token 后，为该视频片段写一句中文台词，"
            "要求口语自然连贯、贴合画面、符合商品和品牌信息。\n"
            "【输出要求】\n"
            " 1) 只输出与下方视频片段数量相同的台词行数；\n"
            " 2) 每行对应一个视频片段，顺序与视频 tokens 中的视频片段顺序一致；\n"
            f"【本次片段总数】：{num_clips} 段；请严格输出 {num_clips} 行台词（每行一句）。\n"
            f"【商品】：{product}\n"
            f"【品牌】：{brand}\n"
            f"【卖点】：{features}\n"
            "【视频片段如下】"
        )
        atc_lines = [
            f"{idx}){ensure_clip_wrapper_vtok(rec.get('v_tok'))}"
            for idx, rec in enumerate(clips, 1)
        ]
        sft_item = {
            "conversations": [
                {"from": "human",
                 "value": header_atc + "\n" + "\n".join(atc_lines) + "\n【现在开始输出台词】"},
                {"from": "gpt", "value": gpt_value},
            ],
            "system": "",
            "tools": "",
        }

        # text baseline
        header_text = (
            "你是专业广告台词助手。下面给出若干视频片段的文字描述，请你为每个片段分别写一句中文广告台词。\n"
            "【生成要求（必须严格遵守）】\n"
            f" 1）你必须只输出 {num_clips} 行内容，每行对应一个视频片段；\n"
            " 2）每一行只能包含一条完整台词，不要在同一行写多句；\n"
            " 3）输出中不要包含序号、括号、项目符号或任何多余符号（例如不要写“1.”、“1)”、“第一句：”等）；\n"
            " 4）禁止输出任何解释、分析、说明文字或空行，只能输出台词本身；\n"
            " 5）第 i 行台词必须只对应下方第 i 个视频片段的文字描述。\n"
            f"【本次片段总数】：{num_clips} 段；\n"
            f"【商品】：{product}\n"
            f"【品牌】：{brand}\n"
            f"【卖点】：{features}\n"
            "【视频片段文字描述如下】"
        )
        text_lines = [f"{idx}){get_caption(rec)}" for idx, rec in enumerate(clips, 1)]
        text_item = {
            "conversations": [
                {"from": "human",
                 "value": header_text + "\n" + "\n".join(text_lines) + "\n【现在开始输出台词】"},
                {"from": "gpt", "value": gpt_value},
            ],
            "system": "",
            "tools": "",
        }

        # mm baseline
        header_mm = (
            "你是专业广告台词助手。下面给出若干视频片段的帧 ID，每个帧 ID 对应一个视频画面，请你为每个片段分别写一句中文广告台词。\n"
            "【生成要求（必须严格遵守）】\n"
            f" 1）你必须只输出 {num_clips} 行内容，每行对应一个视频片段；\n"
            " 2）每一行只能包含一条完整台词，不要在同一行写多句；\n"
            " 3）输出中不要包含序号、括号、项目符号或任何多余符号；\n"
            " 4）禁止输出任何解释、分析、说明文字或空行，只能输出台词本身；\n"
            " 5）第 i 行台词必须只对应下方第 i 个视频片段（按编号顺序）。\n"
            f"【本次片段总数】：{num_clips} 段；\n"
            f"【商品】：{product}\n"
            f"【品牌】：{brand}\n"
            f"【卖点】：{features}\n"
            "【视频片段如下】"
        )
        mm_lines = [f"{idx}){get_frame_placeholder(rec)}" for idx, rec in enumerate(clips, 1)]
        mm_item = {
            "conversations": [
                {"from": "human",
                 "value": header_mm + "\n" + "\n".join(mm_lines) + "\n【现在开始输出台词】"},
                {"from": "gpt", "value": gpt_value},
            ],
            "system": "",
            "tools": "",
        }

        # manifest partial
        manifest = {
            "task": "vid2text",
            "ad_key": {"chunk_id": key[0], "ad_id": key[1]},
            "meta": {
                "product": product,
                "brand": brand,
                "features": features,
                "num_clips": num_clips,
            },
            "clips": [
                {
                    "local_id": idx,
                    "role": "na",
                    "orig": make_clip_orig(rec),
                }
                for idx, rec in enumerate(clips, 1)
            ],
            "label": {
                "vid2text": {
                    "lines": gpt_lines
                }
            },
        }

        return sft_item, text_item, mm_item, manifest

    # ================== vid_sort：三份对齐 + manifest ==================

    def build_vid_sort_triplet(
        self,
        key: Tuple[Any, Any],
        group: List[Dict[str, Any]],
    ):
        rng = Random(stable_subseed(key[0], key[1], "vid_sort", self.seed))

        clips = sorted(group, key=sort_key_clip)
        num_clips = len(clips)
        if num_clips < 2:
            return None, None, None, None

        script_lines = [safe_line_text(rec.get("text")) for rec in clips]
        if all(line == "（无台词）" for line in script_lines):
            return None, None, None, None

        product = majority_nonnull([rec.get("product") for rec in clips])
        brand = majority_nonnull([rec.get("brand") for rec in clips])
        features = majority_nonnull([rec.get("features") for rec in clips])

        orig_indices = list(range(num_clips))
        shuffled_indices = orig_indices[:]
        rng.shuffle(shuffled_indices)

        pos_of_orig = {orig_idx: pos + 1 for pos, orig_idx in enumerate(shuffled_indices)}
        correct_local = [pos_of_orig[orig_idx] for orig_idx in orig_indices]
        gpt_value = ",".join(str(x) for x in correct_local)

        script_block = "\n".join(script_lines)

        # atc
        human_prefix_atc = (
            "你是专业的视频剪辑师。请你根据提供的广告台词顺序，将视频片段进行排序。\n"
            "【输出要求】\n"
            " 1）只输出与下方台词句数相同数量的视频片段编号；\n"
            " 2）输出的视频片段编号不能有重复；\n"
            f"【本次台词总句数】：{num_clips} 段；\n"
            f"【商品】：{product}\n"
            f"【品牌】：{brand}\n"
            f"【卖点】：{features}\n"
            "【广告台词如下】\n"
            f"{script_block}\n\n"
            "【视频片段如下】\n"
        )
        atc_lines = [
            f"{new_idx}){ensure_clip_wrapper_vtok(clips[orig_idx].get('v_tok'))}"
            for new_idx, orig_idx in enumerate(shuffled_indices, 1)
        ]
        sft_item = {
            "conversations": [
                {"from": "human",
                 "value": human_prefix_atc + "\n".join(atc_lines)
                          + "\n【现在请输出按正确顺序排列的视频片段编号序列】"},
                {"from": "gpt", "value": gpt_value},
            ],
            "system": "",
            "tools": "",
        }

        # text baseline
        human_prefix_text = (
            "你是专业的视频剪辑师。请你根据提供的广告台词顺序，将下方乱序的片段文字描述重新排序，并输出正确的编号序列。\n"
            "【输出要求（必须严格遵守）】\n"
            f" 1）只输出一行，共 {num_clips} 个编号，使用半角逗号分隔，例如：\"1,3,2,4\"；\n"
            " 2）每个编号只能出现一次，不允许多余或缺少编号；\n"
            " 3）禁止输出任何解释、分析、理由、自然语言句子或多余符号；\n"
            " 4）如果你的回答包含汉字、句子、换行解释等非编号内容，则视为错误回答。\n"
            f"【商品】：{product}\n"
            f"【品牌】：{brand}\n"
            f"【卖点】：{features}\n"
            "【广告台词如下】\n"
            f"{script_block}\n\n"
            "【片段文字描述如下】\n"
        )
        text_lines = [
            f"{new_idx}){get_caption(clips[orig_idx])}"
            for new_idx, orig_idx in enumerate(shuffled_indices, 1)
        ]
        text_item = {
            "conversations": [
                {"from": "human",
                 "value": human_prefix_text + "\n".join(text_lines)
                          + "\n【现在请输出正确顺序的编号序列】"},
                {"from": "gpt", "value": gpt_value},
            ],
            "system": "",
            "tools": "",
        }

        # mm baseline
        human_prefix_mm = human_prefix_text.replace("片段文字描述", "片段的代表帧")
        mm_lines = [
            f"{new_idx}){get_frame_placeholder(clips[orig_idx])}"
            for new_idx, orig_idx in enumerate(shuffled_indices, 1)
        ]
        mm_item = {
            "conversations": [
                {"from": "human",
                 "value": human_prefix_mm + "\n".join(mm_lines)
                          + "\n【现在请输出正确顺序的编号序列】"},
                {"from": "gpt", "value": gpt_value},
            ],
            "system": "",
            "tools": "",
        }

        # manifest partial
        clips_manifest = []
        for new_idx, orig_idx in enumerate(shuffled_indices, 1):
            rec = clips[orig_idx]
            clips_manifest.append({
                "local_id": new_idx,
                "role": "na",
                "orig": make_clip_orig(rec),
            })

        manifest = {
            "task": "vid_sort",
            "ad_key": {"chunk_id": key[0], "ad_id": key[1]},
            "meta": {
                "product": product,
                "brand": brand,
                "features": features,
                "num_clips": num_clips,
            },
            "clips": clips_manifest,
            "label": {
                "vid_sort": {
                    "correct_order": correct_local  # 基于 local_id 的正确顺序
                }
            },
        }

        return sft_item, text_item, mm_item, manifest

    # ================== vid_select：三份对齐 + manifest ==================

    def build_vid_select_triplet(
        self,
        key: Tuple[Any, Any],
        group: List[Dict[str, Any]],
    ):
        clips = sorted(group, key=sort_key_clip)
        num_pos = len(clips)
        if num_pos < 1:
            return None, None, None, None

        script_lines = [safe_line_text(rec.get("text")) for rec in clips]
        if all(line == "（无台词）" for line in script_lines):
            return None, None, None, None

        product = majority_nonnull([rec.get("product") for rec in clips])
        brand = majority_nonnull([rec.get("brand") for rec in clips])
        features = majority_nonnull([rec.get("features") for rec in clips])

        total_clips = len(self.all_clips)
        if total_clips <= num_pos:
            return None, None, None, None

        rng = Random(stable_subseed(key[0], key[1], "vid_select", self.seed))

        neg_target = int(num_pos * self.neg_pos_ratio)
        neg_target = max(1, neg_target) if num_pos >= 1 else 1
        max_neg_available = total_clips - num_pos
        if max_neg_available <= 0:
            return None, None, None, None
        neg_target = min(neg_target, max_neg_available)

        neg_indices: List[int] = []
        used_idx = set()
        max_trials = neg_target * 10 + 100
        trials = 0
        while len(neg_indices) < neg_target and trials < max_trials:
            trials += 1
            idx = rng.randrange(total_clips)
            if idx in used_idx:
                continue
            k2, _rec2 = self.all_clips[idx]
            if k2 == key:
                continue
            used_idx.add(idx)
            neg_indices.append(idx)

        if not neg_indices:
            return None, None, None, None

        sampled_negs = [self.all_clips[i] for i in neg_indices]

        pool: List[Tuple[str, Dict[str, Any]]] = []
        for rec in clips:
            pool.append(("pos", rec))
        for _k, rec in sampled_negs:
            pool.append(("neg", rec))

        rng.shuffle(pool)

        sft_lines, text_lines, mm_lines, correct_indices = [], [], [], []
        clips_manifest = []

        for idx, (tag, rec) in enumerate(pool, 1):
            sft_lines.append(f"{idx}){ensure_clip_wrapper_vtok(rec.get('v_tok'))}")
            text_lines.append(f"{idx}){get_caption(rec)}")
            mm_lines.append(f"{idx}){get_frame_placeholder(rec)}")
            if tag == "pos":
                correct_indices.append(idx)
            clips_manifest.append({
                "local_id": idx,
                "role": "pos" if tag == "pos" else "neg",
                "orig": make_clip_orig(rec),
            })

        if len(correct_indices) != num_pos:
            return None, None, None, None

        gpt_value = ",".join(str(i) for i in correct_indices)
        script_block = "\n".join(script_lines)

        human_prefix_atc = (
            "你是专业的视频剪辑师。请你根据商品信息和广告台词，从提供的素材池中选择相关的视频片段，并输出它们的编号。\n"
            "【输出要求】\n"
            " 1）只输出与下方台词句数相同数量的视频片段编号；\n"
            " 2）输出的视频片段编号不能有重复；\n"
            f"【本次台词总句数】：{num_pos} 段；\n"
            f"【商品】：{product}\n"
            f"【品牌】：{brand}\n"
            f"【卖点】：{features}\n"
            "【广告台词如下】\n"
            f"{script_block}\n\n"
            "【视频片段素材（含不相关的样本，正样本约占一半）】\n"
        )

        human_prefix_textllm = (
            "你是专业的视频剪辑师。请你根据商品信息和广告台词，从提供的素材池中选择相关的视频片段，并输出它们的编号。\n"
            "【输出要求（必须严格遵守）】\n"
            f" 1）你必须只输出一行，共 {num_pos} 个编号，使用半角逗号分隔，例如：\"2,5,1\"；\n"
            " 2）每个编号只能出现一次，不允许多余、缺少或重复编号；\n"
            " 3）禁止输出任何解释、分析、自然语言、标点符号或多余空格；\n"
            " 4）如果输出包含中文、句子、换行或额外说明内容，则视为错误回答。\n"
            "【示例正确输出】1,3,5\n"
            "【示例错误输出】正确答案是1,3,5，因为……（包含文字解释）\n"
            f"【商品】：{product}\n"
            f"【品牌】：{brand}\n"
            f"【卖点】：{features}\n"
            "【广告台词如下】\n"
            f"{script_block}\n\n"
            "【视频片段素材（含不相关的样本，正样本约占一半）】\n"
        )

        human_prefix_mmllm = human_prefix_textllm.replace(
            "【视频片段素材（含不相关的样本，正样本约占一半）】",
            "【视频片段帧 ID（含不相关的样本，正样本约占一半）】",
        )

        sft_item = {
            "conversations": [
                {"from": "human",
                 "value": human_prefix_atc + "\n".join(sft_lines)
                          + "\n【现在请输出相关的视频片段编号】"},
                {"from": "gpt", "value": gpt_value},
            ],
            "system": "",
            "tools": "",
        }
        text_item = {
            "conversations": [
                {"from": "human",
                 "value": human_prefix_textllm + "\n".join(text_lines)
                          + "\n【现在请输出相关的视频片段编号】"},
                {"from": "gpt", "value": gpt_value},
            ],
            "system": "",
            "tools": "",
        }
        mm_item = {
            "conversations": [
                {"from": "human",
                 "value": human_prefix_mmllm + "\n".join(mm_lines)
                          + "\n【现在请输出相关的视频片段编号】"},
                {"from": "gpt", "value": gpt_value},
            ],
            "system": "",
            "tools": "",
        }

        manifest = {
            "task": "vid_select",
            "ad_key": {"chunk_id": key[0], "ad_id": key[1]},
            "meta": {
                "product": product,
                "brand": brand,
                "features": features,
                "num_clips": len(pool),
                "num_pos": num_pos,
            },
            "clips": clips_manifest,
            "label": {
                "vid_select": {
                    "correct_indices": correct_indices  # 基于 local_id 的正样本集合
                }
            },
        }

        return sft_item, text_item, mm_item, manifest

    # ================== vid2aud：仅 ATC + manifest ==================

    def build_vid2aud_item(self, key, group):
        clips = sorted(group, key=sort_key_clip)
        if not clips:
            return None, None

        aud_list = [rec.get("aud_tok") for rec in clips if rec.get("aud_tok")]
        if not aud_list:
            return None, None

        aud_mode = majority_nonnull(aud_list)
        aud_wrapped = ensure_audio_wrapper(aud_mode)
        if not aud_wrapped:
            return None, None

        product = majority_nonnull([rec.get("product") for rec in clips])
        brand = majority_nonnull([rec.get("brand") for rec in clips])
        features = majority_nonnull([rec.get("features") for rec in clips])

        blocks = []
        for rec in clips:
            text = safe_line_text(rec.get("text"))
            v_tok = rec.get("v_tok")
            v_inner = v_tok if isinstance(v_tok, str) and v_tok.strip() else "<|frame_start|><|frame_end|>"
            blk = (
                "<|clip_start|>"
                f"<|text_start|>{text}<|text_end|>"
                f"<|video_start|>{v_inner}<|video_end|>"
                "<|clip_end|>"
            )
            blocks.append(blk)

        human = (
            "你是专业的广告背景音乐生成助手。请根据广告的商品信息、台词、以及视频片段，输出对应广告的背景音频。\n"
            "【输出要求】\n"
            "1) 只输出一行，且仅包含 <|audio_start|>…<|audio_end|>；\n"
            "2) 禁止输出任何解释、编号、引号或额外符号。\n\n"
            f"【商品】：{product}\n"
            f"【品牌】：{brand}\n"
            f"【卖点】：{features}\n"
            "【台词和视频片段如下】\n"
            + "\n".join(blocks)
            + "\n【现在请输出相关的背景音乐】"
        )

        sft_item = {
            "conversations": [
                {"from": "human", "value": human},
                {"from": "gpt", "value": aud_wrapped},
            ],
            "system": "",
            "tools": "",
        }

        manifest = {
            "task": "vid2aud",
            "ad_key": {"chunk_id": key[0], "ad_id": key[1]},
            "meta": {
                "product": product,
                "brand": brand,
                "features": features,
                "num_clips": len(clips),
            },
            "clips": [
                {
                    "local_id": idx,
                    "role": "na",
                    "orig": make_clip_orig(rec),
                }
                for idx, rec in enumerate(clips, 1)
            ],
            "label": {
                "vid2aud": {
                    "audio_tok": aud_wrapped
                }
            },
        }

        return sft_item, manifest

    # ================== 汇总构造（含 manifest + 顶层检索元信息） ==================

    def build_datasets(self):
        task_assign = self._assign_tasks()

        atc_dataset: List[Dict[str, Any]] = []
        baseline_text: List[Dict[str, Any]] = []
        baseline_mm: List[Dict[str, Any]] = []
        manifest: List[Dict[str, Any]] = []

        for task_name in ["vid2text", "vid_sort", "vid_select", "vid2aud"]:
            pairs = task_assign.get(task_name, [])
            if not pairs:
                continue

            for key, group in pairs:
                if not group:
                    continue

                # 记录 append 前的索引位置（写 manifest.indices 用）
                atc_idx = len(atc_dataset)
                text_idx = len(baseline_text)
                mm_idx = len(baseline_mm)
                # 预先计算本题 sample_id（等于当前 manifest 长度）
                sample_id_next = len(manifest)

                if task_name == "vid2text":
                    sft_item, text_item, mm_item, m = self.build_vid2text_triplet(key, group)
                    if not sft_item or not text_item or not mm_item or not m:
                        continue
                    # 注入检索元信息
                    sft_item  = attach_meta(sft_item,  sample_id_next, task_name, key)
                    text_item = attach_meta(text_item, sample_id_next, task_name, key)
                    mm_item   = attach_meta(mm_item,   sample_id_next, task_name, key)

                    atc_dataset.append(sft_item)
                    baseline_text.append(text_item)
                    baseline_mm.append(mm_item)

                elif task_name == "vid_sort":
                    sft_item, text_item, mm_item, m = self.build_vid_sort_triplet(key, group)
                    if not sft_item or not text_item or not mm_item or not m:
                        continue
                    sft_item  = attach_meta(sft_item,  sample_id_next, task_name, key)
                    text_item = attach_meta(text_item, sample_id_next, task_name, key)
                    mm_item   = attach_meta(mm_item,   sample_id_next, task_name, key)

                    atc_dataset.append(sft_item)
                    baseline_text.append(text_item)
                    baseline_mm.append(mm_item)

                elif task_name == "vid_select":
                    sft_item, text_item, mm_item, m = self.build_vid_select_triplet(key, group)
                    if not sft_item or not text_item or not mm_item or not m:
                        continue
                    sft_item  = attach_meta(sft_item,  sample_id_next, task_name, key)
                    text_item = attach_meta(text_item, sample_id_next, task_name, key)
                    mm_item   = attach_meta(mm_item,   sample_id_next, task_name, key)

                    atc_dataset.append(sft_item)
                    baseline_text.append(text_item)
                    baseline_mm.append(mm_item)

                elif task_name == "vid2aud":
                    sft_item, m = self.build_vid2aud_item(key, group)
                    if not sft_item or not m:
                        continue
                    sft_item = attach_meta(sft_item, sample_id_next, task_name, key)
                    atc_dataset.append(sft_item)
                    # baseline 无此任务
                    text_idx = None
                    mm_idx = None

                else:
                    continue

                # 写入 manifest 的 sample_id 与索引映射（使用 sample_id_next）
                m["sample_id"] = sample_id_next
                m["indices"] = {
                    "atc": atc_idx if task_name in {"vid2text", "vid_sort", "vid_select", "vid2aud"} else None,
                    "baseline_text": (text_idx if task_name in {"vid2text", "vid_sort", "vid_select"} else None),
                    "baseline_mm": (mm_idx if task_name in {"vid2text", "vid_sort", "vid_select"} else None),
                }
                manifest.append(m)

        return atc_dataset, baseline_text, baseline_mm, manifest


# ========== CLI ==========

def parse_task_ratios(arg: str) -> Dict[str, float]:
    ratios: Dict[str, float] = {}
    if not arg:
        return ratios
    for p in arg.split(","):
        p = p.strip()
        if not p or "=" not in p:
            continue
        name, val = p.split("=", 1)
        name = name.strip()
        try:
            v = float(val.strip())
        except ValueError:
            continue
        ratios[name] = v
    return ratios


def main():
    ap = argparse.ArgumentParser(
        description="Build multi-task ATC SFT data + aligned baselines + manifest (vid2text/vid_sort/vid_select)."
    )
    ap.add_argument("--input",type=str,default="/data/phd/miltonzhou/sft/data_preprocess/filter_data/BLCT_TEST_1112.jsonl",help="输入的 merged clip_table jsonl 路径",)
    ap.add_argument("--output_atc",type=str,default="T3_atc_vid2aud.json",help="输出：ATC 模型训练/测试数据（含所有任务，含 v_tok）",)
    ap.add_argument("--output_baseline_text",type=str,default="T3_baseline_text_vid2aud.json",help="输出：文本 baseline 数据（含 vid2text / vidsort / vid_select）",)
    ap.add_argument("--output_baseline_mm",type=str,default="T3_baseline_mm_vid2aud.json",help="输出：多模态 baseline 数据（含 vid2text / vidsort / vid_select）",)
    ap.add_argument("--output_manifest",type=str,default="T3_manifest_vid2aud.json",help="输出：题目检索表（mapping manifest）",)
    ap.add_argument("--task_ratios",type=str,default="vid2aud=1",help="任务比例配置，如 'vid2text=0.3,vid_sort=0.3,vid_select=0.4,vid2aud=0.0'",)
    ap.add_argument("--seed",type=int,default=42,help="随机种子",)
    ap.add_argument("--neg_pos_ratio",type=float,default=1.0,help="vid_select 负样本数量 / 正样本数量",)

    args = ap.parse_args()
    task_ratios = parse_task_ratios(args.task_ratios)

    gen = SFTDataGenerator(
        input_path=args.input,
        task_ratios=task_ratios,
        seed=args.seed,
        neg_pos_ratio=args.neg_pos_ratio,
    )

    atc_dataset, baseline_text, baseline_mm, manifest = gen.build_datasets()

    write_json(args.output_atc, atc_dataset)
    write_json(args.output_baseline_text, baseline_text)
    write_json(args.output_baseline_mm, baseline_mm)
    write_json(args.output_manifest, manifest)

    print(f"[OK] ATC 样本数: {len(atc_dataset)} -> {args.output_atc}")
    print(f"[OK] 文本 baseline 样本数: {len(baseline_text)} -> {args.output_baseline_text}")
    print(f"[OK] 多模态 baseline 样本数: {len(baseline_mm)} -> {args.output_baseline_mm}")
    print(f"[OK] Manifest 记录数: {len(manifest)} -> {args.output_manifest}")
    print("[INFO] 相同输入与 seed 下，任务分配、乱序顺序、负样本采样与映射均可完全复现。")
    print("[INFO] 每条样本顶层已注入 sample_id / task / ad_key，可直接与 manifest 对齐。")


if __name__ == "__main__":
    main()
