#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vLLM LLMEngine offline inference script (Qwen3) — vid2text 台词生成任务
(With meta passthrough in outputs, meta is NOT injected into prompts.)

Usage example:
CUDA_VISIBLE_DEVICES=0 python vllm_infer_vid2text_Qwen3.py
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
# Helpers
# ---------------------------

def build_sampling_params(args) -> SamplingParams:
    """将 CLI 采样参数映射到 vLLM SamplingParams。"""
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

    - 只使用 conversations[0].value 作为 user 消息；
    - 不拼接参考答案；
    - 不修改原始 prompt 文本内容；
    - 不注入任何 meta（sample_id/task/ad_key）。
    """
    items: List[Tuple[str, str, int]] = []

    for i, data in enumerate(all_data):
        convs = data.get("conversations", [])
        if not convs:
            print(f"[warn] idx={i}: no conversations field, skip.")
            continue

        # 取第一条 human/user 的 value（与你现有数据格式对齐）
        first = convs[0]
        user_text = first.get("value", "") if first.get("from") in ("human", "user") else ""
        if not user_text:
            # 退而求其次：直接用 conversations[0].value
            user_text = convs[0].get("value", "")

        if not user_text:
            print(f"[warn] idx={i}: empty user_text, skip.")
            continue

        messages = [{"role": "user", "content": user_text}]

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
    生成“简单且包含 meta 的”输出结构（meta 放在最前面，不影响 prompt）：
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


def process_with_engine(
    engine: LLMEngine,
    pending_prompts: List[Tuple[str, str, int]],
    all_data: List[Dict],
    sampling_params: SamplingParams,
    out_fp,
):
    """
    主推理循环：
    - 按顺序 add_request；
    - 调用 engine.step() 拉取结果；
    - 在 finished 时写入包含 meta 的精简结构（不把 meta 注入 prompt）。
    """
    request_id_to_index: Dict[str, int] = {
        rid: idx for (rid, _, idx) in pending_prompts
    }
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
                    # 有些实现会把 prompt 包含在 text 开头，这里防御性裁切一次
                    if isinstance(prompt_str, str) and gen_text.startswith(prompt_str):
                        gen_text = gen_text[len(prompt_str):]
                    gen_text = gen_text.strip()
                else:
                    gen_text = ""

                # 构造精简输出（meta 放前，不改 prompt）
                out_record = build_output_record(all_data[idx], gen_text)
                out_fp.write(json.dumps(out_record, ensure_ascii=False) + "\n")
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
    parser = FlexibleArgumentParser(
        description="vLLM LLMEngine offline inference (Qwen3, vid2text tagline generation, meta passthrough)."
    )
    parser = EngineArgs.add_cli_args(parser)

    # 模型与数据路径，根据你的环境调整默认值
    parser.add_argument(
        "--model_path",
        type=str,
        default="/data/phd/qinsizhong/llm_factory_test/saves/qwen-8b-sft-0127-textBL/checkpoint-10000", # rebuttal
        help="(optional) alias to set --model for EngineArgs",
    )
    parser.add_argument(
        "--src_path",
        type=str,
        default="/data/phd/qinsizhong/miltonzhou/sft/data_preprocess/T_baseline_text_vid2text.json",
        help="输入 JSON（数组）或 JSONL（行式）；建议使用 JSON（数组）与上游保持一致",
    )
    parser.add_argument(
        "--result_path",
        type=str,
        default="/data/phd/qinsizhong/llm_factory_test/baselines/results/F___vid2text_Qwen3_8B_sft.jsonl",
        help="输出 JSONL 文件路径，将写入包含 meta 的精简结构",
    )

    # 生成参数
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--n", type=int, default=1, help="每条 prompt 生成条数 (n)")
    parser.add_argument("--max_new_tokens", type=int, default=512)

    return parser.parse_args()


def main():
    args = parse_args()
    args.model = args.model_path

    # 构造 engine_args & 初始化 LLMEngine
    engine_args = EngineArgs.from_cli_args(args)
    engine_args.gpu_memory_utilization = 0.90
    engine_args.enable_expert_parallel = True if "235B" in args.model else False
    engine_args.tensor_parallel_size = torch.cuda.device_count()
    engine_args.max_model_len = 32768

    print("[info] initializing LLMEngine ...")
    engine = LLMEngine.from_engine_args(engine_args)

    # 初始化 tokenizer（保留 chat_template）
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # 读取输入数据（默认按 JSON 数组读取；若你给的是 JSONL，可自行改为 read_json(..., lines=True)）
    all_data = pd.read_json(args.src_path, lines=False).to_dict("records")

    # 组装 prompts
    pending_prompts = prepare_prompts(all_data, tokenizer)
    print(f"[info] 将要提交的请求数: {len(pending_prompts)}")

    if not pending_prompts:
        print("[error] 没有可用样本，请检查 src_path / conversations 格式。")
        return

    # 推理并写结果（逐行 JSONL）
    print(f"[info] 开始推理，输出写入: {args.result_path}")
    with open(args.result_path, "a", encoding="utf-8") as out_fp:
        try:
            sampling_params = build_sampling_params(args)
            process_with_engine(engine, pending_prompts, all_data, sampling_params, out_fp)
        finally:
            try:
                engine.shutdown()
            except Exception:
                pass
            print("[info] LLMEngine 已关闭。")


if __name__ == "__main__":
    main()
