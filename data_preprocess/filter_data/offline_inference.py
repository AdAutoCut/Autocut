import os
import json
import time
import argparse
from typing import List, Dict, Optional
from PIL import Image
from vllm import LLM, EngineArgs, SamplingParams
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock


BASE_DIR = "/data/phd/miltonzhou/sft/data_preprocess/filter_data/cp_results"
# BATCH_SIZE = 8
BATCH_SIZE = 16

def build_prompt(text: str) -> str:
    return (
        "<|im_start|>system\n"
        "你是广告图文匹配度评分员，请根据输入说明格式判断匹配度。\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        "<|vision_start|><|image_pad|><|vision_end|>\n"
        "请你担任广告画面与广告台词的匹配度评估员，任务是根据给出的广告台词与图片内容，判断它们的匹配程度，并严格按照以下格式输出匹配度评分。\n\n"
        "请严格遵守以下要求：\n"
        "1. 禁止改写、扩写、修改台词内容。必须使用我提供的原始台词，不得增删任何字词。\n"
        "2. 不得根据画面自由联想、虚构内容，只能依据台词与图片的实际内容判断是否匹配。\n"
        "3. 如果画面中出现文字、字幕、标语等，这些都不算作画面匹配的依据，请忽略这些文字内容，禁止因为字幕文字与台词一致而提高匹配度分数。\n"
        "4. 输出结果必须严格按照以下格式，一条一行，无需添加其他解释或说明。\n\n"
        "【输出格式】\n"
        "每条样本输出格式如下：\n"
        "[台词：（广告片段中的台词）],[画面概括：（你从图片中识别出的内容，简要概括）],[匹配度：1-5]\n\n"
        "【评分标准】\n"
        "1分：完全不相关，画面与台词毫无联系\n"
        "2分：弱相关，可能有场景/气氛相关，但没有实质物品或主题支持\n"
        "3分：中等匹配，画面中有相关物品，但与台词细节不符（如类别错误、表达方式不同）\n"
        "4分：高匹配，画面出现了台词所提物品，但未完全体现其语义或卖点信息\n"
        "5分：完美匹配，画面与台词高度一致，呈现了完整物品与关键信息\n\n"
        "【参考示例（以广告台词“这款防晒喷雾SPF50+”为例）】\n"
        "[台词：这款防晒喷雾SPF50+],[画面概括：一个办公桌上的电脑键盘特写。],[匹配度：1]\n"
        "[台词：这款防晒喷雾SPF50+],[画面概括：画面是一张键盘特写，但画面上叠加了‘SPF50+’的字幕。],[匹配度：1]"
        "[台词：这款防晒喷雾SPF50+],[画面概括：烈日下的沙滩，阳光强烈但没有任何防晒相关物品。],[匹配度：2]\n"
        "[台词：这款防晒喷雾SPF50+],[画面概括：人物在涂抹防晒霜，使用的是传统瓶装涂抹式而非喷雾。],[匹配度：3]\n"
        "[台词：这款防晒喷雾SPF50+],[画面概括：画面展示喷雾防晒产品，但包装模糊，未显著标示SPF。],[匹配度：4]\n"
        "[台词：这款防晒喷雾SPF50+],[画面概括：人物手持一瓶标有“SPF50+”字样的喷雾防晒喷雾，清晰展示包装细节。],[匹配度：5]\n\n"
        "请注意：你只能输出上述格式的内容，禁止添加多余说明、标点、前缀或换行标题等内容。格式错误将被判为无效输出。\n\n"
        "现在请根据下面提供的台词与图像内容，进行匹配度判断。台词如下：\n"
        f"{text}\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

def load_clip_pairs(jsonl_path: str, frame_dir: str) -> List[Dict]:
    data = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                item = json.loads(line.strip())
                frame_path = os.path.join(frame_dir, f"{item['frame_id']}.jpg")
                if not os.path.exists(frame_path):
                    continue
                item["frame_path"] = frame_path
                data.append(item)
            except json.JSONDecodeError as e:
                print(f"skip: {e}")
    return data

def load_model(model_path: str) -> LLM:
    engine_args = EngineArgs(
        model=model_path,
        tokenizer=model_path,
        max_model_len=4096,
        max_num_seqs=BATCH_SIZE,
        tensor_parallel_size=4,
        limit_mm_per_prompt={"image": 1},
    )
    return LLM(**vars(engine_args))

def build_batch_inputs(batch: List[Dict], num_workers: int = 16) -> List[Dict]:
    def process_one(item: Dict) -> Optional[Dict]:
        try:
            prompt = build_prompt(item["text"])
            image = Image.open(item["frame_path"]).convert("RGB")

            # ➤ 如果图像太小，直接跳过
            if image.width < 28 or image.height < 28:
                return {
                    "meta": {
                        "chunk_id": item["chunk_id"],
                        "ad_id": item["ad_id"],
                        "global_id": item["global_id"],
                        "clip_id": item["clip_id"],
                        "frame_id": item["frame_id"],
                    },
                    "raw_text": item["text"],
                    "model_output": None,
                    "error": f"Too small image: {image.width}x{image.height}"
                }

            return {
                "prompt": prompt,
                "multi_modal_data": {"image": image},
                "meta": {
                    "chunk_id": item["chunk_id"],
                    "ad_id": item["ad_id"],
                    "global_id": item["global_id"],
                    "clip_id": item["clip_id"],
                    "frame_id": item["frame_id"],
                }
            }
        except Exception as e:
            return {
                "meta": {
                    "chunk_id": item["chunk_id"],
                    "ad_id": item["ad_id"],
                    "global_id": item["global_id"],
                    "clip_id": item["clip_id"],
                    "frame_id": item["frame_id"],
                },
                "raw_text": item["text"],
                "model_output": None,
                "error": f"Image processing error: {str(e)}"
            }

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(process_one, batch))

    # 把成功的 inputs 和失败的 error 分开
    inputs = [r for r in results if r.get("prompt")]
    errors = [r for r in results if not r.get("prompt")]
    return inputs, errors



def save_results_append_jsonl(results: List[Dict], save_path: str):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "a", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

def load_existing_result_keys(save_path: str) -> set:
    if not os.path.exists(save_path):
        return set()
    keys = set()
    with open(save_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                item = json.loads(line.strip())
                keys.add((item["global_id"], item["clip_id"]))
            except json.JSONDecodeError:
                continue
    return keys

def run_batch_inference(llm: LLM, batch_inputs: List[Dict]) -> List[Dict]:
    sampling_params = SamplingParams(temperature=0.0, max_tokens=74)
    raw_outputs = llm.generate(
        [inp for inp in batch_inputs],
        sampling_params=sampling_params
    )
    results = []
    for meta_input, output in zip(batch_inputs, raw_outputs):
        result = {
            **meta_input["meta"],
            "raw_text": meta_input["prompt"].split("台词如下：")[-1].split("\n<|im_end|>")[0].strip(),  # 提取纯台词内容
            "model_output": output.outputs[0].text.strip()
        }
        results.append(result)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk", type=int, required=True, help="Chunk ID to process, e.g., 21")
    args = parser.parse_args()

    chunk_id = args.chunk
    chunk_dir = os.path.join(BASE_DIR, f"cp_chunk{chunk_id}")
    data_path = os.path.join(chunk_dir, "clips_pairs.jsonl") ### 图文对应表格！！！
    frame_dir = os.path.join(chunk_dir, "extracted_frames") ### 图的路径！！！
    save_path = os.path.join(chunk_dir, "infer_results_p1.jsonl") ### 保存

    print(f"Processing chunk: {chunk_id}")
    print("Loading data...")
    all_data = load_clip_pairs(data_path, frame_dir)

    print(f"Loaded {len(all_data)} samples")

    print("Checking existing results...")
    existing_keys = load_existing_result_keys(save_path)
    print(f"Skipping {len(existing_keys)} existing samples")

    # 筛掉已存在或缺失字段的样本
    invalid_data = []
    filtered_data = []
    for item in all_data:
        key = (item["global_id"], item["clip_id"])
        if key in existing_keys:
            continue
        if not item.get("text") or not item.get("frame_id"):
            invalid_data.append({
                "chunk_id": item.get("chunk_id"),
                "ad_id": item.get("ad_id"),
                "global_id": item.get("global_id"),
                "clip_id": item.get("clip_id"),
                "frame_id": item.get("frame_id"),
                "raw_text": item.get("text"),
                "model_output": None,
                "error": "Missing text or frame_id"
            })
        else:
            filtered_data.append(item)

    # 写入无效样本
    if invalid_data:
        save_results_append_jsonl(invalid_data, save_path)
        print(f"Found {len(invalid_data)} invalid samples (empty text or frame_id).")

    print(f"Total samples to run: {len(filtered_data)}")

    print("Initializing model...")
    llm = load_model("/data/phd/hf_models/Qwen2.5-VL-32B-Instruct")

    print(f"Start inference with Batch Size {BATCH_SIZE}")
    total_batches = (len(filtered_data) + BATCH_SIZE - 1) // BATCH_SIZE

    checkpoint_time = time.time()

    for i in range(total_batches):
        batch = filtered_data[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]

        # 图像构造阶段处理错误图像
        batch_inputs, error_results = build_batch_inputs(batch, num_workers=16)

        if error_results:
            save_results_append_jsonl(error_results, save_path)
            print(f"[Batch {i+1}] Skipped {len(error_results)} invalid images")

        # 如果全部样本都失败了，跳过这个 batch
        if not batch_inputs:
            continue

        results = run_batch_inference(llm, batch_inputs)
        save_results_append_jsonl(results, save_path)

        if (i + 1) % 10 == 0 or (i + 1) == total_batches:
            print(f"[Batch {i+1}/{total_batches}] ")

        if (i + 1) % 100 == 0:
            current_time = time.time()
            elapsed = current_time - checkpoint_time
            speed = 100 / (elapsed / 3600)
            print(f"⚡ Batches {i - 99 + 1}–{i + 1}: average speed = {speed:.2f} batches/hour")
            checkpoint_time = current_time

    print(f"All results saved to: {save_path}")


if __name__ == "__main__":
    main()
