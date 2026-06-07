#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vLLM LLMEngine offline inference script (with meta passthrough)

Usage example:
CUDA_VISIBLE_DEVICES=0 python vllm_infer_sort_atc.py

Notes:
- Keeps your tokenizer.apply_chat_template(...) to build the chat prompt.
- Does NOT inject meta (sample_id/task/ad_key) into the prompt.
- Writes a simple JSONL per sample with meta fields placed at the front:
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
import argparse
import json
import os
from typing import Dict, List, Tuple
import torch
import pandas as pd
from tqdm import tqdm

from transformers import AutoTokenizer

# vLLM imports
from vllm import EngineArgs, LLMEngine, RequestOutput, SamplingParams
from vllm.utils import FlexibleArgumentParser


# ---------------------------
# Helper / main logic
# ---------------------------

def build_sampling_params(args) -> SamplingParams:
    # 将 CLI 参数映射到 vLLM 的 SamplingParams
    return SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_new_tokens,
        n=args.n,
        # 可按需添加 repetition_penalty, presence_penalty, frequency_penalty 等
    )

def prepare_prompts(all_data: List[Dict], tokenizer) -> List[Tuple[str, str, int]]:
    """
    将每条记录转成 (request_id, prompt_text, index) 的列表。
    request_id 使用字符串形式的 index（方便 engine 回传）。
    仅使用 conversations[0].value 构造 prompt，不注入任何 meta。
    """
    items = []
    for i, data in enumerate(all_data):
        # 只取用户消息进入 prompt，保证不把 meta 传入模型
        messages = [{'role': 'user', 'content': data['conversations'][0]['value']}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        print(text)
        items.append((str(i), text, i))
    return items

def build_output_record(src: Dict, gen_text: str) -> Dict:
    """
    生成“简单且包含 meta 的”输出结构，meta 放在前面。
    - 不改变 prompt/模型输入；
    - 只写必要字段，避免冗余。
    """
    out = {}

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

def process_with_engine(engine: LLMEngine, pending_prompts: List[Tuple[str, str, int]],
                        all_data: List[Dict], sampling_params: SamplingParams,
                        out_fp):
    """
    离线推理：结果按“输入顺序”写出。
    做法：先把已完成样本放入 buffer；只要轮到 next_to_write，就连续写出。
    """
    request_id_to_index = {rid: idx for (rid, _, idx) in pending_prompts}
    queue = pending_prompts.copy()

    total = len(all_data)
    pbar = tqdm(total=total, desc="vLLM requests finished", unit="req")

    # 新增：按输入顺序写出的缓冲与指针
    buffer: Dict[int, Dict] = {}   # idx -> record to write
    next_to_write = 0              # 下一个应该写出的样本下标（0..N-1）

    try:
        while queue or engine.has_unfinished_requests():
            # 提交一个请求（维持你的“每轮只加一个”的策略）
            if queue:
                rid, text, idx = queue.pop(0)
                engine.add_request(rid, text, sampling_params)
                request_id_to_index[rid] = idx

            # 前进一步
            request_outputs: List[RequestOutput] = engine.step()

            for ro in request_outputs:
                if not ro.finished:
                    continue

                rid = ro.request_id
                if rid not in request_id_to_index:
                    continue
                idx = request_id_to_index.pop(rid)

                # 收集生成结果
                if ro.outputs and len(ro.outputs) > 0:
                    gen_text = ro.outputs[0].text or ""
                    prompt_str = getattr(ro, "prompt", None)
                    if prompt_str and gen_text.startswith(prompt_str):
                        gen_text = gen_text[len(prompt_str):]
                    gen_text = gen_text.strip("\n")
                else:
                    gen_text = ""

                # 仅透传 meta 到输出，不影响 prompt
                src = all_data[idx]
                out_record = build_output_record(src, gen_text)

                # 放入缓冲（不立即写）
                buffer[idx] = out_record

                # 只要“下一个应写”的样本已经在缓冲里，就按顺序连续写出
                while next_to_write in buffer:
                    out_fp.write(json.dumps(buffer[next_to_write], ensure_ascii=False) + "\n")
                    out_fp.flush()
                    del buffer[next_to_write]
                    next_to_write += 1
                    pbar.update(1)

    finally:
        pbar.close()


# ---------------------------
# CLI & bootstrap
# ---------------------------

def parse_args():
    # 使用 FlexibleArgumentParser + EngineArgs.add_cli_args 保留 vLLM engine 的所有 CLI 参数
    parser = FlexibleArgumentParser(description="vLLM LLMEngine offline inference wrapper (meta passthrough)")
    parser = EngineArgs.add_cli_args(parser)

    # 脚本自身参数（与原脚本保持兼容）
    # parser.add_argument("--model_path", type=str,
    #                     default="/data/phd/qinsizhong/llm_factory_test/saves/qwen-8b-sft-1109-full1/checkpoint-12500",
    #                     help="(optional) alias to set --model for EngineArgs")
    
    parser.add_argument("--model_path", type=str,
                        default="/data/phd/qinsizhong/llm_factory_test/saves/qwen-8b-sft-1111_ablation_embsft2/checkpoint-24000",
                        help="(optional) alias to set --model for EngineArgs")

    # parser.add_argument("--model_path", type=str,
    #                     default="../saves/qwen-8b-sft-1111_ablation_embsft2/checkpoint-24000",
    #                     help="(optional) alias to set --model for EngineArgs")

    # parser.add_argument("--model_path", type=str,
    #                     default="/data/phd/qinsizhong/llm_factory_test/saves/qwen-8b-sft-1112_ablation_basesft/checkpoint-14000",
    #                     help="(optional) alias to set --model for EngineArgs")

    # parser.add_argument("--src_path", type=str,
    #                     default="/data/phd/miltonzhou/sft/data_preprocess/T_atc_vidsort_test.json",
    #                     help="输入 json 文件（数组，每元素一条样本）")

    parser.add_argument("--src_path", type=str,
                        default="../data/T_atc_vidsort_test.json",
                        help="输入 json 文件（数组，每元素一条样本）")
    # parser.add_argument("--result_path", type=str,
    #                     default="/data/phd/qinsizhong/llm_factory_test/baselines/results/F3___vidsort_atc_embsft.jsonl",
    #                     help="输出 jsonl 文件（逐行写出）")

    parser.add_argument("--result_path", type=str,
                        default="./results/F_atc_vidsort_test.jsonl",
                        help="输出 jsonl 文件（逐行写出）")

    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--n", type=int, default=1, help="每条 prompt 的生成数（n）")
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    return parser.parse_args()

def main():
    args = parse_args()
    args.model = args.model_path

    # 构造 engine_args 并初始化 LLMEngine
    engine_args = EngineArgs.from_cli_args(args)
    engine_args.gpu_memory_utilization = 0.95
    engine_args.enable_expert_parallel = True if '235B' in args.model else False
    engine_args.tensor_parallel_size = torch.cuda.device_count()
    engine_args.max_model_len = 32768

    engine = LLMEngine.from_engine_args(engine_args)

    # load tokenizer（保留原有 chat template 的逻辑）
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # 读入数据（数组 JSON，而非 JSONL）
    all_data = pd.read_json(args.src_path, lines=False).to_dict("records")

    # 输出路径
    result_path = args.result_path

    sampling_params = build_sampling_params(args)

    # 预组装 prompts（request_id, text, original_index）
    pending_prompts = prepare_prompts(all_data, tokenizer)

    print(f"Starting vLLM LLMEngine inference, saving to {result_path}")
    # 以追加模式写文件，这样即便中断也能保留已写的结果
    with open(result_path, "a", encoding="utf-8") as out_fp:
        try:
            process_with_engine(engine, pending_prompts, all_data, sampling_params, out_fp)
        finally:
            # 清理 engine
            try:
                engine.shutdown()
            except Exception:
                pass


if __name__ == "__main__":
    main()
