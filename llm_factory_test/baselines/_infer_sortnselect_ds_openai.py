#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ModelScope(OpenAI-compatible) offline inference script (DeepSeek-V3.2) — vid_select / vid_sort

- 仅使用 conversations[0].value 构造 messages；
- 不把 sample_id / task / ad_key 等 meta 注入 prompt；
- 输出为精简对象：meta 放前，便于与 manifest 对齐检索；
- 输入自动兼容 JSON / JSONL；
- 逐条请求，完成即写盘（JSONL），便于容错与断点续跑。

Usage example:
  python infer_modelscope_deepseek_v3_2.py \
    --src_path /path/to/input.jsonl \
    --result_path /path/to/output.jsonl \
    --api_key_env MODELSCOPE_TOKEN \
    --enable_thinking false
"""

import argparse
import json
import os
import time
from typing import Dict, List, Optional

import pandas as pd
from tqdm import tqdm
from openai import OpenAI


# ---------------------------
# Output format (keep same as your vLLM version)
# ---------------------------

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


# ---------------------------
# Data loading (keep same behavior: json or jsonl)
# ---------------------------

def load_input_data(src_path: str) -> List[Dict]:
    """
    读入数据（自动兼容 json / jsonl）
    """
    try:
        return pd.read_json(src_path, lines=False).to_dict("records")
    except ValueError:
        return pd.read_json(src_path, lines=True).to_dict("records")


def extract_user_prompt(sample: Dict) -> Optional[str]:
    """
    仅使用 conversations[0].value 作为 user 文本；不注入任何 meta。
    """
    convs = sample.get("conversations", [])
    if not convs:
        return None
    user_text = convs[0].get("value", "")
    if not isinstance(user_text, str) or not user_text.strip():
        return None
    return user_text


def normalize_generation_text(gen_text: str, remove_spaces: bool = False) -> str:
    """
    轻量清洗：去首尾空白，合并换行。
    对你这种“必须一行输出”的任务很重要。
    """
    if gen_text is None:
        return ""
    s = gen_text.strip()
    # 严格一行：把换行去掉
    s = s.replace("\r", "").replace("\n", "")
    if remove_spaces:
        s = s.replace(" ", "")
    return s


# ---------------------------
# Inference logic: OpenAI SDK -> ModelScope
# ---------------------------

def call_model_with_retry(
    client: OpenAI,
    model_id: str,
    user_text: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    n: int,
    enable_thinking: bool,
    retries: int,
    backoff_base: float,
) -> Dict:
    """
    单条请求，带简单重试（429/5xx/网络抖动时更稳）。
    返回原始 response 对象（dict-like from SDK）。
    """
    extra_body = {"enable_thinking": bool(enable_thinking)}

    last_err: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": user_text}],
                stream=False,
                n=n,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                extra_body=extra_body,
            )
            return {"ok": True, "resp": resp}
        except Exception as e:
            last_err = e
            if attempt >= retries:
                break
            # 指数退避
            sleep_s = backoff_base * (2 ** attempt)
            time.sleep(sleep_s)

    return {"ok": False, "error": repr(last_err)}


def process_file(
    client: OpenAI,
    all_data: List[Dict],
    result_path: str,
    model_id: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    n: int,
    enable_thinking: bool,
    remove_spaces: bool,
    retries: int,
    backoff_base: float,
    max_samples: int,
    print_prompt: bool,
):
    """
    逐条推理：完成即写盘（JSONL）。
    保留你的输出结构与“meta 不进 prompt”的约束。
    """
    total = len(all_data) if max_samples <= 0 else min(len(all_data), max_samples)

    os.makedirs(os.path.dirname(result_path), exist_ok=True)

    written = 0
    with open(result_path, "a", encoding="utf-8") as out_fp:
        pbar = tqdm(total=total, desc="ModelScope requests finished", unit="req")
        try:
            for i in range(total):
                sample = all_data[i]
                user_text = extract_user_prompt(sample)
                if user_text is None:
                    # 空样本也写一条，保证可溯源（也可选择跳过）
                    out_obj = build_output_record(sample, "")
                    out_fp.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
                    out_fp.flush()
                    written += 1
                    pbar.update(1)
                    continue

                if print_prompt:
                    print("\n" + "=" * 80)
                    print(f"[debug] sample_index={i} sample_id={sample.get('sample_id')}")
                    print(user_text)
                    print("=" * 80 + "\n")

                res = call_model_with_retry(
                    client=client,
                    model_id=model_id,
                    user_text=user_text,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    n=n,
                    enable_thinking=enable_thinking,
                    retries=retries,
                    backoff_base=backoff_base,
                )

                if res["ok"]:
                    resp = res["resp"]
                    # 取第一个候选（与你 vLLM 版本 ro.outputs[0] 对齐）
                    msg = resp.choices[0].message
                    gen_text = getattr(msg, "content", "") or ""
                    gen_text = normalize_generation_text(gen_text, remove_spaces=remove_spaces)

                    # 如果你开启 thinking，想一起打印可打开这里
                    if enable_thinking:
                        reasoning = getattr(msg, "reasoning_content", None)
                        if reasoning:
                            # 仅打印，不写入结构（避免污染 model_generate 的严格格式）
                            print("[reasoning]\n" + str(reasoning) + "\n")
                else:
                    # 请求失败：写空字符串（不改输出结构），同时在 stderr/console 打印错误
                    gen_text = ""
                    print(f"[warn] request failed at index={i}, error={res['error']}")

                out_obj = build_output_record(sample, gen_text)
                out_fp.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
                out_fp.flush()

                written += 1
                pbar.update(1)
        finally:
            pbar.close()
            print(f"[info] 完成推理样本数: {written}/{total}")


# ---------------------------
# CLI
# ---------------------------

def parse_args():
    p = argparse.ArgumentParser(description="ModelScope(OpenAI-compatible) offline inference (DeepSeek-V3.2)")
    p.add_argument("--src_path", type=str, default="/data/phd/qinsizhong/miltonzhou/sft/data_preprocess/T_baseline_text_vidsort.json",
                   help="输入 JSON/JSONL 文件（每条记录含 conversations 等）")
    p.add_argument("--result_path", type=str, default="/data/phd/qinsizhong/llm_factory_test/baselines/results/F___vidsort_DS.jsonl",
                   help="输出 JSONL 文件路径（append 模式）")

    # ModelScope OpenAI-compatible endpoint
    p.add_argument("--base_url", type=str, default="https://api-inference.modelscope.cn/v1")
    p.add_argument("--model_id", type=str, default="deepseek-ai/DeepSeek-V3.2")

    # Token
    p.add_argument("--api_key", type=str, default="")
    p.add_argument("--api_key_env", type=str, default="MODELSCOPE_TOKEN",
                   help="从环境变量读取 token 的变量名，默认 MODELSCOPE_TOKEN")

    # Generation params
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--n", type=int, default=1)

    # Thinking control
    p.add_argument("--enable_thinking", type=str, default="false",
                   choices=["true", "false"], help="是否开启 reasoning_content 输出（建议 false）")

    # Output normalization
    p.add_argument("--remove_spaces", type=str, default="false",
                   choices=["true", "false"], help="是否移除输出中的空格（严格一行编号任务可选 true）")

    # Reliability
    p.add_argument("--retries", type=int, default=0)
    p.add_argument("--backoff_base", type=float, default=1.0)

    # Debug / partial run
    p.add_argument("--max_samples", type=int, default=0,
                   help="只跑前 N 条；0 表示跑全部")
    p.add_argument("--print_prompt", action="store_true",
                   help="打印每条 prompt（用于检查输入拼接）")

    return p.parse_args()


def main():
    args = parse_args()

    api_key = args.api_key.strip()
    if not api_key:
        api_key = os.environ.get(args.api_key_env, "").strip()
    if not api_key:
        raise RuntimeError(
            f"Missing ModelScope Token. Provide --api_key or set env var {args.api_key_env}."
        )

    enable_thinking = (args.enable_thinking.lower() == "true")
    remove_spaces = (args.remove_spaces.lower() == "true")

    client = OpenAI(
        base_url=args.base_url,
        api_key=api_key,
    )

    all_data = load_input_data(args.src_path)

    print(f"[info] Starting ModelScope inference: model={args.model_id}")
    print(f"[info] Input: {args.src_path}")
    print(f"[info] Output: {args.result_path}")

    process_file(
        client=client,
        all_data=all_data,
        result_path=args.result_path,
        model_id=args.model_id,
        max_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        n=args.n,
        enable_thinking=enable_thinking,
        remove_spaces=remove_spaces,
        retries=args.retries,
        backoff_base=args.backoff_base,
        max_samples=args.max_samples,
        print_prompt=args.print_prompt,
    )


if __name__ == "__main__":
    main()
