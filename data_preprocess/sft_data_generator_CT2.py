#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SFT 多任务数据构造器（ShareGPT 格式，基于 chunk_id + ad_id 分组）

已实现任务：
- vid2text:
    给定同一广告下多个视频片段的 v_tok，生成对应多行中文台词。
- vid_sort:
    给定广告台词正确顺序 + 相同数量的乱序视频片段，输出正确片段编号序列。
- vid_select:
    给定商品信息 + 正确广告台词 + 包含正负样本的视频片段素材池，
    选择与该广告匹配的片段编号（不要求排序）。
- vid2aud:
    给定商品信息 + 全量广告台词 + 视频片段（按真实顺序，含 text+v_tok），
    输出对应广告的背景音频 token 序列（<|audio_start|>...<|audio_end|>）。

说明：
- 输入可能来自多个 chunk，每个 chunk 内 ad_id 从 1 开始；
- 使用 (chunk_id, ad_id) 作为广告唯一标识，避免跨 chunk 混合；
- 输出为一个 JSON 数组，每个元素是 ShareGPT 格式样本：
  {
    "conversations": [...],
    "system": "",
    "tools": ""
  }
- meta 信息（task, chunk_id, ad_id 等）目前不写入输出，如需可从注释恢复。

用法示例：
python sft_data_generator_CT_multi.py \
  --input merged_clip_table.jsonl \
  --output sft_1109_eval.json \
  --task_ratios vid2text=0.3,vid_sort=0.3,vid_select=0.2,vid2aud=0.2
"""

import argparse
import json
import sys
import random
from collections import defaultdict, Counter
from typing import Any, Dict, Iterable, List, Optional, Tuple


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
    """写出为一个 JSON 数组文件。"""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# ========== 通用工具函数 ==========

NULL_SENTINELS = {"", "null", "None", "无", "(null)", "(none)"}


def norm_str(x: Optional[str]) -> Optional[str]:
    """规范化字符串字段：去空白，过滤常见空值表示。"""
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


def majority_nonnull(values: List[Optional[str]]) -> str:
    """返回非空众数；若全为空则返回 '无'。"""
    cleaned = [norm_str(v) for v in values if norm_str(v) is not None]
    if not cleaned:
        return "无"
    cnt = Counter(cleaned)
    mode_val, _ = sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0]))[0]
    return mode_val


def ensure_clip_wrapper(v_tok: Optional[str]) -> str:
    """将单个片段的 v_tok 包裹为标准 clip+video 范式（无 text 的场景）。"""
    base = v_tok if isinstance(v_tok, str) and v_tok.strip() else "<|frame_start|><|frame_end|>"
    return f"<|clip_start|><|video_start|>{base}<|video_end|><|clip_end|>"


def ensure_audio_wrapper(aud_tok: Optional[str]) -> Optional[str]:
    """
    将 aud_tok 包裹为 <|audio_start|>...<|audio_end|>。
    - 若为空或为 '无' 等空值，返回 None；
    - 若已包含 audio_start/audio_end，则原样返回。
    """
    if aud_tok is None:
        return None
    if not isinstance(aud_tok, str):
        aud_tok = str(aud_tok)
    s = aud_tok.strip()
    if not s:
        return None
    # 过滤“无”等占位
    if s.lower() in {z.lower() for z in NULL_SENTINELS}:
        return None
    if "<|audio_start|>" in s and "<|audio_end|>" in s:
        return s
    return f"<|audio_start|>{s}<|audio_end|>"


def safe_line_text(t: Optional[str]) -> str:
    """将原始 text 标准化为监督信号的一行。"""
    if t is None:
        return "（无台词）"
    if t == "null":
        return "（无台词）"
    s = str(t).strip()
    return s if s else "（无台词）"


def sort_key_clip(rec: Dict[str, Any]) -> Tuple[int, int]:
    """片段排序键：先 clip_id（int），再按 frame_id 类型稳定排序。"""
    cid = rec.get("clip_id")
    try:
        cid = int(cid)
    except Exception:
        cid = 10**9
    fid = rec.get("frame_id") or ""
    return (cid, 0 if isinstance(fid, str) else 1)


# ========== 多任务 SFT 数据生成器 ==========

class SFTDataGenerator:
    """
    多任务 SFT 数据生成器：

    - 从 merged clip_table.jsonl 中按 (chunk_id, ad_id) 聚合，
      避免不同 chunk 下相同 ad_id 被合并。
    - 按 task_ratios 将各广告分配给不同任务（互斥分配，同一广告只用于一个主任务）。
    - 对于需要负样本（vid_select），可以从所有其他广告中采样片段作为负例。
    """

    def __init__(self,
                 input_path: str,
                 task_ratios: Dict[str, float],
                 seed: Optional[int] = 42) -> None:
        self.input_path = input_path
        self.task_ratios = task_ratios
        if seed is not None:
            random.seed(seed)

        # key: (chunk_id, ad_id) -> List[rec]
        self.by_ad: Dict[Tuple[Any, Any], List[Dict[str, Any]]] = defaultdict(list)
        # 全量片段池 (key, rec)，供负样本使用
        self.all_clips: List[Tuple[Tuple[Any, Any], Dict[str, Any]]] = []

        self._load()

    # ----- 加载与分组 -----

    def _load(self) -> None:
        for rec in read_jsonl(self.input_path):
            chunk_id = rec.get("chunk_id")
            ad_id = rec.get("ad_id")
            if chunk_id is None or ad_id is None:
                continue
            key = (chunk_id, ad_id)
            self.by_ad[key].append(rec)

        # 清理空组
        empty_keys = [k for k, g in self.by_ad.items() if not g]
        for k in empty_keys:
            self.by_ad.pop(k, None)

        # 建立全量片段池
        for key, group in self.by_ad.items():
            for rec in group:
                self.all_clips.append((key, rec))

    # ----- 任务分配策略 -----

    def _assign_tasks(self) -> Dict[str, List[Tuple[Tuple[Any, Any], List[Dict[str, Any]]]]]:
        """
        根据 task_ratios 把 (chunk_id, ad_id, group) 分配给各任务（互斥分配）。

        策略：
        - 将所有广告打乱；
        - 对每个任务，按比例 int(total * ratio) 切一段；
        - 一个广告只属于一个任务的数据主来源；
        - vid_select 的负样本可从全局其它广告中抽取，不影响互斥分配。
        """
        ad_items = list(self.by_ad.items())  # List[(key, group)]
        random.shuffle(ad_items)
        total = len(ad_items)

        task_to_pairs: Dict[str, List[Tuple[Tuple[Any, Any], List[Dict[str, Any]]]]] = {
            t: [] for t in self.task_ratios
        }
        cursor = 0

        for task, ratio in self.task_ratios.items():
            if ratio <= 0:
                continue
            n = int(total * ratio)
            if n <= 0:
                continue
            if cursor >= total:
                break
            end = min(cursor + n, total)
            task_to_pairs[task] = ad_items[cursor:end]
            cursor = end

        return task_to_pairs

    # ================== 任务：vid2text ==================

    @staticmethod
    def build_vid2text_item(key: Tuple[Any, Any],
                            group: List[Dict[str, Any]]) -> Dict[str, Any]:
        """vid2text: v_tok -> 多行中文台词"""
        chunk_id, ad_id = key
        clips = sorted(group, key=sort_key_clip)

        product = majority_nonnull([rec.get("product") for rec in clips])
        brand = majority_nonnull([rec.get("brand") for rec in clips])
        features = majority_nonnull([rec.get("features") for rec in clips])

        num_clips = len(clips)

        header = (
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

        numbered_tokens_lines: List[str] = []
        for idx, rec in enumerate(clips, 1):
            v_tok = rec.get("v_tok")
            wrapped = ensure_clip_wrapper(v_tok)
            numbered_tokens_lines.append(f"{idx}){wrapped}")

        human_value = (
            header
            + "\n"
            + "\n".join(numbered_tokens_lines)
            + "\n【现在开始输出台词】"
        )

        gpt_lines = [safe_line_text(rec.get("text")) for rec in clips]
        gpt_value = "\n".join(gpt_lines)

        return {
            "conversations": [
                {"from": "human", "value": human_value},
                {"from": "gpt", "value": gpt_value},
            ],
            "system": "",
            "tools": "",
        }

    def build_vid2text_dataset(
        self,
        pairs: List[Tuple[Tuple[Any, Any], List[Dict[str, Any]]]],
    ) -> List[Dict[str, Any]]:
        dataset: List[Dict[str, Any]] = []
        for key, group in pairs:
            if not group:
                continue
            item = self.build_vid2text_item(key, group)
            dataset.append(item)
        return dataset

    # ================== 任务：vid_sort ==================

    @staticmethod
    def build_vid_sort_item(key: Tuple[Any, Any],
                            group: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """vid_sort: 正确台词顺序 + 乱序视频片段 -> 输出排序编号序列"""
        chunk_id, ad_id = key
        clips = sorted(group, key=sort_key_clip)
        num_clips = len(clips)
        if num_clips < 2:
            return None

        script_lines = [safe_line_text(rec.get("text")) for rec in clips]
        if all(line == "（无台词）" for line in script_lines):
            return None

        product = majority_nonnull([rec.get("product") for rec in clips])
        brand = majority_nonnull([rec.get("brand") for rec in clips])
        features = majority_nonnull([rec.get("features") for rec in clips])

        orig_indices = list(range(num_clips))
        shuffled_indices = orig_indices[:]
        random.shuffle(shuffled_indices)

        shuffled_lines: List[str] = []
        for new_idx, orig_idx in enumerate(shuffled_indices, start=1):
            rec = clips[orig_idx]
            v_tok = rec.get("v_tok")
            wrapped = ensure_clip_wrapper(v_tok)
            shuffled_lines.append(f"{new_idx}){wrapped}")

        pos_of_orig = {orig_idx: (pos + 1) for pos, orig_idx in enumerate(shuffled_indices)}
        correct_order = [str(pos_of_orig[orig_idx]) for orig_idx in orig_indices]
        gpt_value = ",".join(correct_order)

        script_block = "\n".join(script_lines)

        human_value = (
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
            + "\n".join(shuffled_lines)
            + "\n【现在请输出按正确顺序排列的视频片段编号序列】"
        )

        return {
            "conversations": [
                {"from": "human", "value": human_value},
                {"from": "gpt", "value": gpt_value},
            ],
            "system": "",
            "tools": "",
        }

    def build_vid_sort_dataset(
        self,
        pairs: List[Tuple[Tuple[Any, Any], List[Dict[str, Any]]]],
    ) -> List[Dict[str, Any]]:
        dataset: List[Dict[str, Any]] = []
        for key, group in pairs:
            if not group:
                continue
            item = self.build_vid_sort_item(key, group)
            if item is not None:
                dataset.append(item)
        return dataset

    # ================== 任务：vid_select ==================

    def build_vid_select_item(
        self,
        key: Tuple[Any, Any],
        group: List[Dict[str, Any]],
        neg_pos_ratio: float = 1.0,
    ) -> Optional[Dict[str, Any]]:
        """vid_select: 在正负混合素材池中选择与当前广告匹配的片段编号"""
        chunk_id, ad_id = key
        clips = sorted(group, key=sort_key_clip)
        num_pos = len(clips)
        if num_pos < 1:
            return None

        script_lines = [safe_line_text(rec.get("text")) for rec in clips]
        if all(line == "（无台词）" for line in script_lines):
            return None

        product = majority_nonnull([rec.get("product") for rec in clips])
        brand = majority_nonnull([rec.get("brand") for rec in clips])
        features = majority_nonnull([rec.get("features") for rec in clips])

        # 负样本候选：其他广告的片段
        neg_candidates = [
            (k2, rec2)
            for (k2, rec2) in self.all_clips
            if k2 != key
        ]
        if not neg_candidates:
            return None

        neg_target = int(num_pos * neg_pos_ratio)
        neg_target = max(1, neg_target) if num_pos > 1 else 1
        neg_target = min(neg_target, len(neg_candidates))

        sampled_negs = random.sample(neg_candidates, neg_target)

        pool: List[Tuple[str, Dict[str, Any]]] = []
        for rec in clips:
            pool.append(("pos", rec))
        for _k, rec in sampled_negs:
            pool.append(("neg", rec))

        random.shuffle(pool)

        shuffled_lines: List[str] = []
        correct_indices: List[str] = []

        for idx, (tag, rec) in enumerate(pool, start=1):
            v_tok = rec.get("v_tok")
            wrapped = ensure_clip_wrapper(v_tok)
            shuffled_lines.append(f"{idx}){wrapped}")
            if tag == "pos":
                correct_indices.append(str(idx))

        if len(correct_indices) != num_pos:
            return None

        gpt_value = ",".join(correct_indices)
        script_block = "\n".join(script_lines)

        human_value = (
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
            + "\n".join(shuffled_lines)
            + "\n【现在请输出相关的视频片段编号】"
        )

        return {
            "conversations": [
                {"from": "human", "value": human_value},
                {"from": "gpt", "value": gpt_value},
            ],
            "system": "",
            "tools": "",
        }

    def build_vid_select_dataset(
        self,
        pairs: List[Tuple[Tuple[Any, Any], List[Dict[str, Any]]]],
    ) -> List[Dict[str, Any]]:
        dataset: List[Dict[str, Any]] = []
        for key, group in pairs:
            if not group:
                continue
            item = self.build_vid_select_item(key, group)
            if item is not None:
                dataset.append(item)
        return dataset

    # ================== 新任务：vid2aud ==================

    @staticmethod
    def build_vid2aud_item(
        key: Tuple[Any, Any],
        group: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """
        vid2aud:
        根据商品信息 + 全量台词 + 对应视频片段（按真实顺序），
        预测广告背景音频 tokens。

        输入（human）：
        - 指令 + 输出要求；
        - 【商品】【品牌】【卖点】；
        - 【台词和视频片段如下】：
            以多个 clip 串接，每个 clip:
            <|clip_start|><|text_start|>...<|text_end|><|video_start|>...<|video_end|><|clip_end|>
        输出（gpt）：
        - 仅一行：<|audio_start|>...<|audio_end|>
        - aud_tok 从该 (chunk_id, ad_id) 下的 aud_tok 众数提取，并标准化包裹。
        """
        chunk_id, ad_id = key
        clips = sorted(group, key=sort_key_clip)
        if not clips:
            return None

        # 聚合音频监督信号（同一广告通常共享 aud_tok）
        aud_raw_list = [rec.get("aud_tok") for rec in clips if rec.get("aud_tok")]
        if not aud_raw_list:
            return None
        aud_mode = majority_nonnull(aud_raw_list)
        aud_wrapped = ensure_audio_wrapper(aud_mode)
        if not aud_wrapped:
            return None

        product = majority_nonnull([rec.get("product") for rec in clips])
        brand = majority_nonnull([rec.get("brand") for rec in clips])
        features = majority_nonnull([rec.get("features") for rec in clips])

        # 构造 clip 串（含 text + video），结构贴近预训练格式
        clip_blocks: List[str] = []
        for rec in clips:
            text = safe_line_text(rec.get("text"))
            v_tok = rec.get("v_tok")
            v_inner = v_tok if isinstance(v_tok, str) and v_tok.strip() else "<|frame_start|><|frame_end|>"
            block = (
                "<|clip_start|>"
                f"<|text_start|>{text}<|text_end|>"
                f"<|video_start|>{v_inner}<|video_end|>"
                "<|clip_end|>"
            )
            clip_blocks.append(block)

        human_value = (
            "你是专业的广告背景音乐生成助手。请根据广告的商品信息、台词、以及视频片段，输出对应广告的背景音频。\n"
            "【输出要求】\n"
            "1) 只输出一行，且仅包含 <|audio_start|>…<|audio_end|>；\n"
            "2) 禁止输出任何解释、编号、引号或额外符号。\n\n"
            f"【商品】：{product}\n"
            f"【品牌】：{brand}\n"
            f"【卖点】：{features}\n"
            "【台词和视频片段如下】\n"
            + "\n".join(clip_blocks)
            + "\n【现在请输出相关的背景音乐】"
        )

        gpt_value = aud_wrapped

        return {
            "conversations": [
                {"from": "human", "value": human_value},
                {"from": "gpt", "value": gpt_value},
            ],
            "system": "",
            "tools": "",
        }

    def build_vid2aud_dataset(
        self,
        pairs: List[Tuple[Tuple[Any, Any], List[Dict[str, Any]]]],
    ) -> List[Dict[str, Any]]:
        dataset: List[Dict[str, Any]] = []
        for key, group in pairs:
            if not group:
                continue
            item = self.build_vid2aud_item(key, group)
            if item is not None:
                dataset.append(item)
        return dataset

    # ================== 汇总构造 ==================

    def build_dataset(self) -> List[Dict[str, Any]]:
        """
        主入口：根据 task_ratios 分配 (chunk_id, ad_id) 并为各任务构造样本。
        启用任务：
        - vid2text
        - vid_sort
        - vid_select
        - vid2aud
        """
        task_assignments = self._assign_tasks()
        all_items: List[Dict[str, Any]] = []

        if "vid2text" in task_assignments:
            all_items.extend(self.build_vid2text_dataset(task_assignments["vid2text"]))

        if "vid_sort" in task_assignments:
            all_items.extend(self.build_vid_sort_dataset(task_assignments["vid_sort"]))

        if "vid_select" in task_assignments:
            all_items.extend(self.build_vid_select_dataset(task_assignments["vid_select"]))

        if "vid2aud" in task_assignments:
            all_items.extend(self.build_vid2aud_dataset(task_assignments["vid2aud"]))

        return all_items


# ========== CLI ==========

def parse_task_ratios(arg: str) -> Dict[str, float]:
    """
    将形如 "vid2text=0.3,vid_sort=0.3,vid_select=0.2,vid2aud=0.2" 的字符串解析为 dict。
    """
    ratios: Dict[str, float] = {}
    if not arg:
        return ratios
    parts = arg.split(",")
    for p in parts:
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
    ap = argparse.ArgumentParser(description="Build multi-task ShareGPT SFT data from merged clip_table.jsonl")
    ap.add_argument(
        "--input",
        default="/data/phd/miltonzhou/sft/data_preprocess/filter_data/data_chunk_1102/Final_CT_train_1109.jsonl",
        help="path to merged clip_table.jsonl",
    )
    ap.add_argument(
        "--output",
        default="/data/phd/miltonzhou/sft/data_preprocess/sft_1109_train.json",
        help="path to save ShareGPT JSON (array)",
    )
    ap.add_argument(
        "--task_ratios",
        type=str,
        default="vid_sort=0.25,vid_select=0.25,vid2aud=0.25",
        help="任务比例配置",
    )
    ap.add_argument("--seed", type=int, default=42, help="随机种子（用于任务分配和乱序）")
    args = ap.parse_args()

    task_ratios = parse_task_ratios(args.task_ratios)
    # if not task_ratios:
    #     print("[WARN] 未提供有效的 task_ratios，默认使用 vid2text=1.0")
    #     task_ratios = {"vid2text": 1.0}

    gen = SFTDataGenerator(input_path=args.input, task_ratios=task_ratios, seed=args.seed)
    dataset = gen.build_dataset()
    write_json(args.output, dataset)
    print(f"[OK] 共生成 {len(dataset)} 条 ShareGPT 样本，已保存到：{args.output}")
    print(f"[INFO] 任务比例: {task_ratios}")


if __name__ == "__main__":
    main()
