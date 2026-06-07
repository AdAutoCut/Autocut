#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vLLM LLMEngine offline inference for Qwen2.5-VL (image captioning)

新设定：
- 输入: JSONL（每行: 含 chunk_id, ad_id, clip_id, frame_id, text, 等任意字段）
- 图片: {frames_dir}/{frame_id}.jpg
- 输出: 原始记录 + "caption"
- 要求: caption 仅基于图像内容生成，
       明确要求忽略画面中的字幕/文字/水印/UI 文本等，
       不依赖输入 JSONL 中的 text / product / features 等字段。
"""

import argparse
import json
import os
from typing import Dict, List, Tuple, Any
import torch
from PIL import Image
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


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    """严格读取 JSONL（每行一个 JSON 对象）。"""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def build_messages(question: str) -> List[Dict[str, Any]]:
    """
    多模态 chat：image + text，由 tokenizer.apply_chat_template 注入视觉占位。
    system 明确限定：仅看图像内容，忽略字幕/文字。
    """
    return [
        {
            "role": "system",
            "content": (
                "你是一名图像描述助手，只能基于图像的视觉内容生成一句中文描述。忽略画面中的所有文字、字幕、水印、界面元素或商标，不要提及你在忽略文字，也不要描述文字内容。请客观地用一句简洁的中文描述图像中可见的场景或主要主体。"
            ),
        },
        {
            "role": "user",
            "content": [
                {"type": "image"},                  # 占位；真实图像通过 multi_modal_data 传入
                {"type": "text", "text": question}, # 指令文本
            ],
        },
    ]


def make_question(default_q: str) -> str:
    """
    不再使用 JSONL 中的 text / product / features。
    始终使用统一指令，确保仅基于图像内容。
    """
    return default_q


def load_image_if_exists(frames_dir: str, frame_id: Any) -> Image.Image | None:
    img_path = os.path.join(frames_dir, f"{str(frame_id)}.jpg")
    if not os.path.isfile(img_path):
        return None
    try:
        with Image.open(img_path) as im:
            return im.convert("RGB")
    except Exception:
        return None


def prepare_requests(
    records: List[Dict[str, Any]],
    frames_dir: str,
    tokenizer,
    default_question: str,
) -> Tuple[List[Tuple[str, Dict[str, Any], int]], List[int], int, int]:
    """
    返回:
      pending: [(request_id, inputs_dict, original_index)]
      empty_indices: 无法推理但需要输出空 caption 的样本索引
      missing_images, bad_lines: 统计
    """
    items: List[Tuple[str, Dict[str, Any], int]] = []
    empty_indices: List[int] = []
    num_missing_images = 0
    num_bad_lines = 0

    for i, rec in enumerate(records):
        try:
            frame_id = rec.get("frame_id", None)

            # 1) frame_id 缺失：直接记为空 caption
            if frame_id is None:
                empty_indices.append(i)
                continue

            image = load_image_if_exists(frames_dir, frame_id)

            # 2) 图片缺失/读失败：直接记为空 caption
            if image is None:
                num_missing_images += 1
                empty_indices.append(i)
                continue

            # 3) 正常可推理：统一问题，不用任何文本字段
            question = make_question(default_question)
            messages = build_messages(question)

            prompt_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            prompt_obj: Dict[str, Any] = {
                "prompt": prompt_text,
                "multi_modal_data": {"image": image},
            }
            items.append((str(i), prompt_obj, i))

        except Exception:
            # 异常行：也落为空 caption，避免丢样本
            num_bad_lines += 1
            empty_indices.append(i)
            continue

    return items, empty_indices, num_missing_images, num_bad_lines


def process_with_engine(
    engine: LLMEngine,
    pending: List[Tuple[str, Dict[str, Any], int]],
    all_records: List[Dict[str, Any]],
    sampling_params: SamplingParams,
    out_fp,
):
    """主事件循环：逐个 add_request + step；完成即写出（原记录 + caption）。"""
    request_id_to_index: Dict[str, int] = {}
    queue = pending.copy()

    pbar = tqdm(total=len(pending), desc="vLLM requests finished", unit="req")

    try:
        while queue or engine.has_unfinished_requests():
            if queue:
                rid, inputs_dict, idx = queue.pop(0)
                engine.add_request(rid, inputs_dict, sampling_params)
                request_id_to_index[rid] = idx

            request_outputs: List[RequestOutput] = engine.step()

            for ro in request_outputs:
                if not ro.finished:
                    continue

                rid = ro.request_id
                if rid not in request_id_to_index:
                    continue
                idx = request_id_to_index.pop(rid)

                # 取第一个候选文本
                if ro.outputs and len(ro.outputs) > 0:
                    gen_text = ro.outputs[0].text or ""
                    prompt_str = getattr(ro, "prompt", None)
                    if isinstance(prompt_str, str) and gen_text.startswith(prompt_str):
                        gen_text = gen_text[len(prompt_str):]
                    gen_text = gen_text.strip()
                else:
                    gen_text = ""

                # 在原始记录基础上新增 caption 字段（保留 chunk_id/ad_id/... 等全部原字段）
                out_obj = all_records[idx].copy()
                out_obj["caption"] = gen_text

                out_fp.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
                out_fp.flush()

                pbar.update(1)
    finally:
        pbar.close()


# ---------------------------
# CLI & bootstrap
# ---------------------------

def parse_args():
    parser = FlexibleArgumentParser(
        description="vLLM LLMEngine offline inference for Qwen2.5-VL captioning (image-only, ignore subtitles)"
    )
    parser = EngineArgs.add_cli_args(parser)

    parser.add_argument(
        "--model_path",
        type=str,
        default="/data/phd/hf_models/Qwen2.5-VL-7B-Instruct",
        help="(optional) alias to set --model for EngineArgs"
    )
    parser.add_argument(
        "--src_path",
        type=str,
        default="/data/phd/miltonzhou/sft/data_preprocess/filter_data/data_chunk_1102/TEST_DATA_1109.jsonl",
        help="输入 JSONL（每行至少包含 frame_id，可包含 chunk_id/ad_id/clip_id/text 等其他字段）"
    )
    parser.add_argument(
        "--frames_dir",
        type=str,
        default="/data/phd/miltonzhou/sft/data_preprocess/filter_data/downloaded_frames",
        help="包含 {frame_id}.jpg 的目录"
    )
    parser.add_argument(
        "--result_path",
        type=str,
        default="/data/phd/miltonzhou/sft/data_preprocess/filter_data/BLCT_TEST_1109.jsonl",
        help="输出 JSONL 路径（逐行写入）"
    )

    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=64)

    # 不再使用 use_ad_text
    parser.add_argument(
        "--question",
        type=str,
        default=(
            "请只根据图像中的画面内容，用一句中文客观描述最主要的主体和关键动作或场景，不要提及“图片”“画面”“照片”等词，"
            "不要描述镜头或相机信息，不要依赖或复述图中的任何文字或字幕，只输出这一句话。"
        ),
        help="统一问题指令；模型仅基于图像本身生成 caption。"
    )

    return parser.parse_args()


def main():
    args = parse_args()
    args.model = args.model_path

    # vLLM EngineArgs
    engine_args = EngineArgs.from_cli_args(args)

    # 多模态与性能默认
    if engine_args.tensor_parallel_size is None or engine_args.tensor_parallel_size == 1:
        engine_args.tensor_parallel_size = torch.cuda.device_count() or 1

    if engine_args.max_model_len is None:
        engine_args.max_model_len = 4096

    engine_args.limit_mm_per_prompt = {"image": 1, "video": 0, "audio": 0}
    engine_args.mm_processor_kwargs = {
        "min_pixels": 28 * 28,
        "max_pixels": 1280 * 28 * 28,
    }

    if engine_args.gpu_memory_utilization is None:
        engine_args.gpu_memory_utilization = 0.95

    # 初始化
    engine = LLMEngine.from_engine_args(engine_args)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # 读取 JSONL
    records = read_jsonl(args.src_path)

    # 组装请求（不使用 text 等字段）
    pending, empty_indices, miss_img, bad_lines = prepare_requests(
        records,
        args.frames_dir,
        tokenizer,
        default_question=args.question,
    )

    if len(pending) == 0:
        print(f"[Error] No valid requests. missing_images={miss_img}, bad_lines={bad_lines}")
        try:
            engine.shutdown()
        except Exception:
            pass
        return

    print(
        f"Total lines: {len(records)} | valid: {len(pending)} "
        f"| missing_images: {miss_img} | bad_lines: {bad_lines}"
    )

    # 先把空 caption 的样本写出去
    with open(args.result_path, "a", encoding="utf-8") as out_fp:
        for idx in empty_indices:
            out_obj = records[idx].copy()
            out_obj["caption"] = ""  # 空字符串
            out_fp.write(json.dumps(out_obj, ensure_ascii=False) + "\n")

    sampling_params = build_sampling_params(args)

    # 对有效样本逐行写出（断点友好）
    with open(args.result_path, "a", encoding="utf-8") as out_fp:
        try:
            process_with_engine(engine, pending, records, sampling_params, out_fp)
        finally:
            try:
                engine.shutdown()
            except Exception:
                pass

    print(f"Saved to: {args.result_path}")


if __name__ == "__main__":
    main()
