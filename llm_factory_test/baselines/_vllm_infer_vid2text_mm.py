#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vLLM LLMEngine offline inference (Qwen2.5-VL) — vid2text 台词生成任务
(With meta passthrough in outputs, meta is NOT injected into prompts.)
"""

import argparse
import json
import os
import re
from typing import Dict, List, Tuple

import torch
import pandas as pd
from tqdm import tqdm
from PIL import Image

from transformers import AutoTokenizer
from vllm import EngineArgs, LLMEngine, RequestOutput, SamplingParams
from vllm.utils import FlexibleArgumentParser

# ---------------------------
# Helpers
# ---------------------------

def build_sampling_params(args) -> SamplingParams:
    return SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_new_tokens,
        n=args.n,
    )

def _load_images_safe(paths: List[str]) -> List[Image.Image]:
    imgs = []
    for p in paths:
        if not p or p.startswith("<") or (not os.path.exists(p)):
            continue
        try:
            imgs.append(Image.open(p).convert("RGB"))
        except Exception as e:
            print(f"[warn] fail to load image: {p} ({e})")
    return imgs

def _read_records_auto(path: str) -> List[Dict]:
    """Read JSON or JSONL automatically into list[dict]."""
    try:
        return pd.read_json(path, lines=False).to_dict("records")
    except ValueError:
        return pd.read_json(path, lines=True).to_dict("records")

def _extract_human_value(rec: Dict) -> str:
    """
    从 conversations 里取第一条 human/user 的 value；
    若没有，则退回第 0 条的 value；若不存在则返回空字符串。
    """
    conv = rec.get("conversations", [])
    if not conv:
        return ""
    for m in conv:
        if m.get("from") in ("human", "user"):
            return m.get("value", "") or ""
    return conv[0].get("value", "") or ""

def _extract_frame_ids(text: str) -> List[str]:
    """从文本中提取所有 [frame_id]（中括号内为纯数字），按出现顺序返回。"""
    return re.findall(r"\[(\d+)\]", text)

def _remove_frame_ids(text: str) -> str:
    """
    从文本中删除 [frame_id]，保留其它内容（包括前缀编号如 '1)' 等）。
    例如： '1)[1626601758120003]' -> '1)'
    """
    return re.sub(r"\[(\d+)\]", "", text)

def prepare_prompts_from_frames(
    records: List[Dict],
    frames_dir: str,
    tokenizer,
    add_vision_id: bool,
    model_type: str,
) -> List[Tuple[str, str, int, List[str]]]:
    """
    针对 vid2text 任务构造多模态输入，不注入任何 meta：
      1) 提取 human 文本 raw_text；
      2) 提取 frame_ids（用于加载图片），并从文本中删除 [frame_id]，得到 prompt_text；
      3) {frame_id}.jpg 顺序与出现顺序一致；
      4) messages = [{"role":"user","content":[images..., {"type":"text","text":prompt_text}]}]
      5) tokenizer.apply_chat_template(..., add_generation_prompt=True)
    返回: List[(request_id, prompt_text, record_index, image_paths)]
    """
    items: List[Tuple[str, str, int, List[str]]] = []

    for idx, rec in enumerate(records):
        try:
            raw_text = _extract_human_value(rec)
            if not raw_text:
                print(f"[warn] idx={idx}: 未找到 human 文本，跳过。")
                continue

            frame_ids = _extract_frame_ids(raw_text)
            if not frame_ids:
                print(f"[warn] idx={idx}: 未解析到任何 [frame_id]，跳过。")
                continue

            prompt_text = _remove_frame_ids(raw_text)
            image_paths = [os.path.join(frames_dir, f"{fid}.jpg") for fid in frame_ids]

            if "minicpm" in model_type.lower():
                vision_tokens = "\n".join(["(<image>./</image>)" for _ in image_paths])
                content = f"{vision_tokens}\n\n{prompt_text}"
            elif "llava" in model_type.lower():
                vision_tokens = "\n".join(["<image>" for _ in image_paths])
                content = f"{vision_tokens}\n\n{prompt_text}"
            else:
                # 多模态 content：先所有图片，再文本
                content = [{"type": "image"} for _ in image_paths]
                content.append({"type": "text", "text": prompt_text})

            messages = [{"role": "user", "content": content}]

            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
                add_vision_id=add_vision_id,
            )

            if len(items) == 0:
                print("----- PROMPT (first record, truncated) -----")
                print(text[:2000], "...\n")

            request_id = str(idx)
            items.append((request_id, text, idx, image_paths))

        except Exception as e:
            # 单条样本构造 prompt 出问题，不要影响整体
            print(f"[error] idx={idx}: 构造多模态 prompt 失败，跳过该样本: {e}")
            continue

    return items

def build_output_record(src: Dict, gen_text: str) -> Dict:
    """
    生成“精简但可溯源”的输出结构（meta 放在最前面）：
    {
      "sample_id": ...,
      "task": "...",
      "ad_key": {"chunk_id": ..., "ad_id": ...},
      "conversations": [...],
      "system": "",
      "tools": "",
      "model_generate": "..."
    }
    """
    out: Dict = {}
    # ---- meta 放前面 ----
    out["sample_id"] = src.get("sample_id", None)
    out["task"] = src.get("task", None)
    out["ad_key"] = src.get("ad_key", None)
    # ---- 原始必要字段（供溯源） ----
    out["conversations"] = src.get("conversations", [])
    out["system"] = src.get("system", "")
    out["tools"] = src.get("tools", "")
    # ---- 模型输出 ----
    out["model_generate"] = gen_text
    return out

# ---------------------------
# Engine loop
# ---------------------------

def process_with_engine(
    engine: LLMEngine,
    pending_prompts: List[Tuple[str, str, int, List[str]]],
    base_records: List[Dict],
    sampling_params: SamplingParams,
    out_fp,
):
    """
    逐条将请求送入 vLLM Engine，并把生成结果写回文件（包含 meta，不改 prompt）。
    pending_prompts: (request_id, prompt_text, record_index, image_paths)
    """
    request_id_to_index: Dict[str, int] = {}
    queue = pending_prompts.copy()

    total = len(pending_prompts)
    pbar = tqdm(total=total, desc="vLLM requests finished", unit="req")
    written = 0

    try:
        while queue or engine.has_unfinished_requests():
            if queue:
                rid, prompt_text, idx, image_paths = queue.pop(0)

                # 对“送入引擎”的这一步做保护
                try:
                    images = _load_images_safe(image_paths)
                    if image_paths and (not images):
                        print(f"[warn] idx={idx}, rid={rid}: 所有图片加载失败，跳过。")
                        pbar.update(1)
                        continue

                    inputs = {"prompt": prompt_text}
                    if images:
                        inputs["multi_modal_data"] = {"image": images}

                    # 这里之前会因为 prompt 过长抛 ValueError
                    engine.add_request(rid, inputs, sampling_params)
                    request_id_to_index[rid] = idx

                except Exception as e:
                    # 这条样本无法送入 vLLM，写一条空的 model_generate，继续下一条
                    print(f"[error] idx={idx}, rid={rid}: add_request 失败，跳过该样本: {e}")
                    out_record = build_output_record(base_records[idx], "")
                    out_fp.write(json.dumps(out_record, ensure_ascii=False) + "\n")
                    out_fp.flush()
                    written += 1
                    pbar.update(1)
                    continue

            # 这里也可以出问题，保险起见再包一层
            try:
                request_outputs: List[RequestOutput] = engine.step()
            except Exception as e:
                print(f"[error] engine.step() 发生异常: {e}")
                # 如果这里崩了，通常是全局问题，很难恢复；先 break
                break

            for ro in request_outputs:
                if not ro.finished:
                    continue

                rid = ro.request_id
                if rid not in request_id_to_index:
                    continue

                idx = request_id_to_index.pop(rid)

                if ro.outputs and len(ro.outputs) > 0:
                    gen_text = ro.outputs[0].text or ""
                    prompt_str = getattr(ro, "prompt", None)
                    if isinstance(prompt_str, str) and gen_text.startswith(prompt_str):
                        gen_text = gen_text[len(prompt_str):]
                    gen_text = gen_text.strip()
                else:
                    gen_text = ""

                # 构造“meta 放前”的精简输出
                out_record = build_output_record(base_records[idx], gen_text)
                out_fp.write(json.dumps(out_record, ensure_ascii=False) + "\n")
                out_fp.flush()

                written += 1
                pbar.update(1)
    finally:
        pbar.close()
        print(f"[info] 共写入结果 {written} 条。")

# ---------------------------
# CLI & main
# ---------------------------

def parse_args():
    parser = FlexibleArgumentParser(
        description="vLLM LLMEngine offline inference (Qwen2.5-VL, vid2text, hide frame_id, meta passthrough)."
    )
    parser = EngineArgs.add_cli_args(parser)

    parser.add_argument(
        "--model_path",
        type=str,
        # default="/data/phd/hf_models/InternVL3-8B",
        # default="/data/phd/hf_models/MiniCPM-V-4_5",
        # default="/data/phd/hf_models/Qwen3-VL-8B-Instruct",
        default="/data/phd/hf_models/llava-v1.6-mistral-7b-hf",
    )
    parser.add_argument(
        "--prompts_path",
        type=str,
        default="/data/phd/miltonzhou/sft/data_preprocess/T_baseline_mm_vid2text.json",
        help="JSON/JSONL; 每条记录包含 `conversations`，human 文本中含有 [frame_id] 列表。",
    )
    parser.add_argument(
        "--frames_dir",
        type=str,
        default="/data/phd/miltonzhou/sft/data_preprocess/filter_data/downloaded_frames",
        help="帧图目录，文件名为 {frame_id}.jpg。",
    )
    parser.add_argument(
        "--result_path",
        type=str,
        default="/data/phd/qinsizhong/llm_factory_test/baselines/results/F___vid2text_llava.jsonl",
        help="输出 JSONL 路径（逐行写入包含 meta 的精简结构）。",
    )
    parser.add_argument(
        "--add_vision_id",
        action="store_true",
        default=True,
        help="是否让 tokenizer 自动为图片加入 'Picture k:' 标记（仅影响内部模板，不改你写的 prompt）。",
    )

    # 生成参数
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)

    return parser.parse_args()

def main():
    args = parse_args()
    args.model = args.model_path

    # vLLM engine args
    engine_args = EngineArgs.from_cli_args(args)
    engine_args.gpu_memory_utilization = 0.90
    engine_args.enable_expert_parallel = True if "235B" in args.model else False
    engine_args.tensor_parallel_size = torch.cuda.device_count()
    engine_args.max_model_len = 30000
    engine_args.trust_remote_code = True
    engine_args.limit_mm_per_prompt = {"image": 100, "video": 0, "audio": 0}

    print("[info] initializing LLMEngine ...")
    engine = LLMEngine.from_engine_args(engine_args)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    records = _read_records_auto(args.prompts_path)
    print(f"[info] 加载输入样本 {len(records)} 条。")

    pending_prompts = prepare_prompts_from_frames(
        records=records,
        frames_dir=args.frames_dir,
        tokenizer=tokenizer,
        add_vision_id=args.add_vision_id,
        model_type=args.model,
    )

    print(f"[info] 将要提交的请求数：{len(pending_prompts)}")
    if not pending_prompts:
        print("[error] 没有可提交的请求。请检查：\n"
              "1) conversations 中是否存在 human 文本；\n"
              "2) human 文本中是否包含形如 [frame_id] 的占位符；\n"
              "3) 对应 frames_dir/{frame_id}.jpg 是否存在。")
        return

    print(f"[info] 开始推理，输出写入：{args.result_path}")
    with open(args.result_path, "a", encoding="utf-8") as out_fp:
        try:
            process_with_engine(
                engine=engine,
                pending_prompts=pending_prompts,
                base_records=records,
                sampling_params=build_sampling_params(args),
                out_fp=out_fp,
            )
        finally:
            try:
                engine.shutdown()
            except Exception:
                pass
            print("[info] LLMEngine 已关闭。")

if __name__ == "__main__":
    main()
