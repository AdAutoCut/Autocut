# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gc
import json
from typing import Optional

# --- fire 是可选依赖：有就用；没有就走 argparse ---
try:
    import fire as _fire
    _HAS_FIRE = True
except Exception:
    _HAS_FIRE = False

from tqdm import tqdm
from transformers import Seq2SeqTrainingArguments

from llamafactory.data import get_dataset, get_template_and_fix_tokenizer
from llamafactory.extras.constants import IGNORE_INDEX
from llamafactory.extras.misc import get_device_count
from llamafactory.extras.packages import is_vllm_available
from llamafactory.hparams import get_infer_args
from llamafactory.model import load_tokenizer


if is_vllm_available():
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest


def vllm_infer(
    model_name_or_path: str,
    adapter_name_or_path: str = None,
    dataset: str = "alpaca_en_demo",
    dataset_dir: str = "data",
    template: str = "default",
    cutoff_len: int = 2048,
    max_samples: Optional[int] = None,
    vllm_config: str = "{}",
    save_name: str = "generated_predictions.jsonl",
    temperature: float = 0.95,
    top_p: float = 0.7,
    top_k: int = 50,
    max_new_tokens: int = 1024,
    repetition_penalty: float = 1.0,
    skip_special_tokens: bool = True,
    default_system: Optional[str] = None,
    enable_thinking: bool = True,
    seed: Optional[int] = None,
    pipeline_parallel_size: int = 1,
    image_max_pixels: int = 768 * 768,
    image_min_pixels: int = 32 * 32,
    video_fps: float = 2.0,
    video_maxlen: int = 128,
    batch_size: int = 1024,
):
    r"""Perform batch generation using vLLM engine, which supports tensor parallelism.

    Usage: python vllm_infer.py --model_name_or_path meta-llama/Llama-2-7b-hf --template llama --dataset alpaca_en_demo
    """
    if pipeline_parallel_size > get_device_count():
        raise ValueError("Pipeline parallel size should be smaller than the number of gpus.")

    model_args, data_args, _, generating_args = get_infer_args(
        dict(
            model_name_or_path=model_name_or_path,
            adapter_name_or_path=adapter_name_or_path,
            dataset=dataset,
            dataset_dir=dataset_dir,
            template=template,
            cutoff_len=cutoff_len,
            max_samples=max_samples,
            preprocessing_num_workers=16,
            default_system=default_system,
            enable_thinking=enable_thinking,
            vllm_config=vllm_config,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
        )
    )

    training_args = Seq2SeqTrainingArguments(output_dir="dummy_dir")
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template_obj = get_template_and_fix_tokenizer(tokenizer, data_args)
    template_obj.mm_plugin.expand_mm_tokens = False  # for vllm generate

    engine_args = {
        "model": model_args.model_name_or_path,
        "trust_remote_code": True,
        "dtype": model_args.infer_dtype,
        "max_model_len": cutoff_len + max_new_tokens,
        "tensor_parallel_size": (get_device_count() // pipeline_parallel_size) or 1,
        "pipeline_parallel_size": pipeline_parallel_size,
        "disable_log_stats": True,
        "enable_lora": model_args.adapter_name_or_path is not None,
    }
    if template_obj.mm_plugin.__class__.__name__ != "BasePlugin":
        engine_args["limit_mm_per_prompt"] = {"image": 4, "video": 2, "audio": 2}

    if isinstance(model_args.vllm_config, dict):
        engine_args.update(model_args.vllm_config)

    llm = LLM(**engine_args)

    # load datasets
    dataset_module = get_dataset(template_obj, model_args, data_args, training_args, "ppo", **tokenizer_module)
    train_dataset = dataset_module["train_dataset"]

    sampling_params = SamplingParams(
        repetition_penalty=generating_args.repetition_penalty or 1.0,  # repetition_penalty must > 0
        temperature=generating_args.temperature,
        top_p=generating_args.top_p or 1.0,  # top_p must > 0
        top_k=generating_args.top_k or -1,  # top_k must > 0
        stop_token_ids=template_obj.get_stop_token_ids(tokenizer),
        max_tokens=generating_args.max_new_tokens,
        skip_special_tokens=skip_special_tokens,
        seed=seed,
    )
    if model_args.adapter_name_or_path is not None:
        lora_request = LoRARequest("default", 1, model_args.adapter_name_or_path[0])
    else:
        lora_request = None

    # Store all results in these lists
    all_prompts, all_preds, all_labels = [], [], []

    # Add batch process to avoid the issue of too many files opened
    for i in tqdm(range(0, len(train_dataset), batch_size), desc="Processing batched inference"):
        vllm_inputs, prompts, labels = [], [], []
        batch = train_dataset[i : min(i + batch_size, len(train_dataset))]

        for j in range(len(batch["input_ids"])):
            if batch["images"][j] is not None:
                image = batch["images"][j]
                multi_modal_data = {
                    "image": template_obj.mm_plugin._regularize_images(
                        image, image_max_pixels=image_max_pixels, image_min_pixels=image_min_pixels
                    )["images"]
                }
            elif batch["videos"][j] is not None:
                video = batch["videos"][j]
                multi_modal_data = {
                    "video": template_obj.mm_plugin._regularize_videos(
                        video,
                        image_max_pixels=image_max_pixels,
                        image_min_pixels=image_min_pixels,
                        video_fps=video_fps,
                        video_maxlen=video_maxlen,
                    )["videos"]
                }
            elif batch["audios"][j] is not None:
                audio = batch["audios"][j]
                audio_data = template_obj.mm_plugin._regularize_audios(
                    audio,
                    sampling_rate=16000,
                )
                multi_modal_data = {"audio": zip(audio_data["audios"], audio_data["sampling_rates"])}
            else:
                multi_modal_data = None

            vllm_inputs.append({"prompt_token_ids": batch["input_ids"][j], "multi_modal_data": multi_modal_data})
            prompts.append(tokenizer.decode(batch["input_ids"][j], skip_special_tokens=skip_special_tokens))
            labels.append(
                tokenizer.decode(
                    list(filter(lambda x: x != IGNORE_INDEX, batch["labels"][j])),
                    skip_special_tokens=skip_special_tokens,
                )
            )

        results = llm.generate(vllm_inputs, sampling_params, lora_request=lora_request)
        preds = [result.outputs[0].text for result in results]

        # Accumulate results
        all_prompts.extend(prompts)
        all_preds.extend(preds)
        all_labels.extend(labels)
        gc.collect()

    # Write all results at once outside the loop
    with open(save_name, "w", encoding="utf-8") as f:
        for text, pred, label in zip(all_prompts, all_preds, all_labels):
            f.write(json.dumps({"prompt": text, "predict": pred, "label": label}, ensure_ascii=False) + "\n")

    print("*" * 70)
    print(f"{len(all_prompts)} total generated results have been saved at {save_name}.")
    print("*" * 70)


if __name__ == "__main__":
    if _HAS_FIRE:
        # 正常路径：使用 fire 提供的 CLI
        _fire.Fire(vllm_infer)
    else:
        # 无 fire 时的回退：使用 argparse 解析参数，然后调用 vllm_infer
        import argparse

        def str2bool(x):
            return str(x).lower() in {"1", "true", "t", "yes", "y"}

        p = argparse.ArgumentParser("vllm_infer (no-fire fallback)")
        p.add_argument("--model_name_or_path", type=str, required=True)
        p.add_argument("--adapter_name_or_path", type=str, default=None)
        p.add_argument("--dataset", type=str, default="alpaca_en_demo")
        p.add_argument("--dataset_dir", type=str, default="data")
        p.add_argument("--template", type=str, default="default")
        p.add_argument("--cutoff_len", type=int, default=2048)
        p.add_argument("--max_samples", type=int, default=None)
        p.add_argument("--vllm_config", type=str, default="{}")
        p.add_argument("--save_name", type=str, default="generated_predictions.jsonl")
        p.add_argument("--temperature", type=float, default=0.95)
        p.add_argument("--top_p", type=float, default=0.7)
        p.add_argument("--top_k", type=int, default=50)
        p.add_argument("--max_new_tokens", type=int, default=1024)
        p.add_argument("--repetition_penalty", type=float, default=1.0)

        # skip_special_tokens: 默认 True；提供显式反转开关
        g = p.add_mutually_exclusive_group()
        g.add_argument("--skip_special_tokens", dest="skip_special_tokens", action="store_true")
        g.add_argument("--no_skip_special_tokens", dest="skip_special_tokens", action="store_false")
        p.set_defaults(skip_special_tokens=True)

        p.add_argument("--default_system", type=str, default=None)
        p.add_argument("--enable_thinking", type=str2bool, default=True)
        p.add_argument("--seed", type=int, default=None)
        p.add_argument("--pipeline_parallel_size", type=int, default=1)
        p.add_argument("--image_max_pixels", type=int, default=768 * 768)
        p.add_argument("--image_min_pixels", type=int, default=32 * 32)
        p.add_argument("--video_fps", type=float, default=2.0)
        p.add_argument("--video_maxlen", type=int, default=128)
        p.add_argument("--batch_size", type=int, default=1024)

        args = p.parse_args()
        vllm_infer(**vars(args))
