#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vLLM LLMEngine offline inference script (Qwen3) — vid_select / vid_sort

- 仅使用 conversations[0].value 构造 prompt；
- 不把 sample_id / task / ad_key 等 meta 注入 prompt；
- 输出为精简对象：meta 放前，便于与 manifest 对齐检索。
"""

import argparse
import json
import os
from typing import Dict, List, Tuple

import torch
import pandas as pd
from tqdm import tqdm

from transformers import AutoTokenizer
from vllm import EngineArgs, LLMEngine, RequestOutput, SamplingParams
from vllm.utils import FlexibleArgumentParser


# ---------------------------
# Helper / main logic
# ---------------------------

def build_sampling_params(args) -> SamplingParams:
    return SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_new_tokens,
        n=args.n,
    )


def prepare_prompts(all_data: List[Dict], tokenizer) -> List[Tuple[str, str, int]]:
    """
    将每条记录转成 (request_id, prompt_text, index) 的列表。
    仅使用 conversations[0].value 作为 user 文本；不注入任何 meta。
    """
    items: List[Tuple[str, str, int]] = []
    for i, data in enumerate(all_data):
        convs = data.get("conversations", [])
        if not convs:
            continue
        user_text = convs[0].get("value", "")
        if not isinstance(user_text, str) or not user_text.strip():
            continue

        messages = [{'role': 'user', 'content': user_text}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        items.append((str(i), text, i))
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
    out["sample_id"] = src.get("sample_id", None)
    out["task"] = src.get("task", None)
    out["ad_key"] = src.get("ad_key", None)
    out["conversations"] = src.get("conversations", [])
    out["system"] = src.get("system", "")
    out["tools"] = src.get("tools", "")
    out["model_generate"] = gen_text
    return out


def process_with_engine(engine: LLMEngine,
                        pending_prompts: List[Tuple[str, str, int]],
                        all_data: List[Dict],
                        sampling_params: SamplingParams,
                        out_fp):
    """
    主事件循环：每次添加一个 request，然后 step；当 finished=True 时写出结果。
    输出使用 build_output_record，确保 meta 透传且不进入 prompt。
    """
    request_id_to_index = {rid: idx for (rid, _, idx) in pending_prompts}
    queue = pending_prompts.copy()

    total = len(pending_prompts)
    pbar = tqdm(total=total, desc="vLLM requests finished", unit="req")
    written = 0

    try:
        while queue or engine.has_unfinished_requests():
            if queue:
                rid, text, idx = queue.pop(0)
                engine.add_request(rid, text, sampling_params)
                request_id_to_index[rid] = idx

            request_outputs: List[RequestOutput] = engine.step()

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
                    if prompt_str and isinstance(prompt_str, str) and gen_text.startswith(prompt_str):
                        gen_text = gen_text[len(prompt_str):]
                    gen_text = gen_text.strip("\n")
                else:
                    gen_text = ""

                # 写入“精简输出”（meta 放前）
                out_obj = build_output_record(all_data[idx], gen_text)
                out_fp.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
                out_fp.flush()

                written += 1
                pbar.update(1)
    finally:
        pbar.close()
        print(f"[info] 完成推理样本数: {written}/{total}")


# ---------------------------
# CLI & bootstrap
# ---------------------------

def parse_args():
    parser = FlexibleArgumentParser(description="vLLM LLMEngine offline inference wrapper (Qwen3, select/sort)")
    parser = EngineArgs.add_cli_args(parser)

    parser.add_argument("--model_path", type=str, default="/data/phd/qinsizhong/llm_factory_test/saves/qwen-8b-sft-0127-textBL/checkpoint-10000",
                        help="(optional) alias to set --model for EngineArgs")
    parser.add_argument("--src_path", type=str,
                        default="/data/phd/qinsizhong/miltonzhou/sft/data_preprocess/T_baseline_text_vidsort.json",
                        help="输入 JSON/JSONL 文件（每条记录含 conversations 等）")
    parser.add_argument("--result_path", type=str,
                        default="/data/phd/qinsizhong/llm_factory_test/baselines/results/F___vidsort_Qwen3_8B_sft.jsonl",
                        help="输出 JSONL 文件路径")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--n", type=int, default=1, help="每条 prompt 的生成数（n）")
    parser.add_argument("--max_new_tokens", type=int, default=9000)
    return parser.parse_args()


def main():
    args = parse_args()
    args.model = args.model_path

    engine_args = EngineArgs.from_cli_args(args)
    engine_args.gpu_memory_utilization = 0.90
    engine_args.enable_expert_parallel = True if '235B' in args.model else False
    engine_args.tensor_parallel_size = torch.cuda.device_count()
    engine_args.max_model_len = 32768

    engine = LLMEngine.from_engine_args(engine_args)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # 读入数据（自动兼容 json / jsonl）
    try:
        all_data = pd.read_json(args.src_path, lines=False).to_dict("records")
    except ValueError:
        all_data = pd.read_json(args.src_path, lines=True).to_dict("records")

    result_path = args.result_path
    sampling_params = build_sampling_params(args)

    pending_prompts = prepare_prompts(all_data, tokenizer)

    print(f"[info] Starting vLLM LLMEngine inference, saving to {result_path}")
    with open(result_path, "a", encoding="utf-8") as out_fp:
        try:
            process_with_engine(engine, pending_prompts, all_data, sampling_params, out_fp)
        finally:
            try:
                engine.shutdown()
            except Exception:
                pass


if __name__ == "__main__":
    main()
