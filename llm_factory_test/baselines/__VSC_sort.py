#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VSC Evaluation for vid_sort Task (图像-台词匹配度评估)
-------------------------------------------------------
流程说明：
1. 从模型结果文件（ShareGPT 格式）中解析 meta 信息：
   【商品】【品牌】【卖点】，以及模型输出序列 model_generate。
2. 用 meta 信息在 manifest_sort.json 中找到对应广告（task="vid_sort"）。
3. 从 manifest 中读取：
   - label.vid_sort.correct_order：真实广告台词顺序；
   - clips：每个 local_id 对应的 frame_id 与 text。
4. 构建 “图文对”：
   - 台词按正确顺序排列；
   - 图像按模型输出顺序匹配。
     即第 i 行台词 ↔ 第 i 个预测视频帧。
5. 调用 GPT-4o 对每个 (frame_id, text) 打分 {0,1,2}。
6. 输出广告级和模型级平均分，并保留所有配对详情。

输出格式（JSON）：
{
  "model_VSC": 0.73,
  "ads_evaluated": 5,
  "ads_skipped": 1,
  "ads": [
    {
      "meta": {...},
      "vsc_ad": 0.64,
      "pairs": [
        {"frame_id": 1626601758120003, "text": "大家好，", "score": 2},
        ...
      ]
    }
  ]
}
"""

import os
import re
import json
import base64
import argparse
from typing import Dict, Any, List, Tuple
from tqdm import tqdm
from openai import AzureOpenAI


# ========== Prompt ==========
VSC_PROMPT = """
你是一名广告创意评估专家。现在请你评估一张图片与一行广告台词的语义匹配程度。
图片代表广告视频中的一个画面，台词是该广告中的一句文案。

请判断这张图片与这行台词的内容是否相符、是否传达了相同的语义或情境。

评分标准（只输出数字）：
0 = 完全不匹配（画面与台词无关或矛盾）
1 = 部分相关（画面与台词有一定联系，但不完全对应）
2 = 高度匹配（画面清晰体现台词内容，语义一致）

台词如下：
{line}
"""

# ========== I/O ==========
def read_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def read_jsonl(path: str):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    data.append(json.loads(line))
                except Exception as e:
                    print(f"[WARN] JSON 解析失败: {e}")
    return data

def write_json(path: str, obj: Any):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

# ========== Meta 解析与匹配 ==========
def normalize_field(s: str) -> str:
    if not s:
        return ""
    s = str(s).replace("'", '"').replace("，", ",").replace(" ", "")
    return s.strip()

def extract_meta_from_prompt(text: str) -> Tuple[str, str, str]:
    p = re.search(r"【商品】：([^\n]+)", text)
    b = re.search(r"【品牌】：([^\n]+)", text)
    f = re.search(r"【卖点】：([^\n]+)", text)
    prod = normalize_field(p.group(1)) if p else ""
    brand = normalize_field(b.group(1)) if b else ""
    feats = normalize_field(f.group(1)) if f else ""
    return prod, brand, feats

def build_manifest_map(manifest: List[Dict[str, Any]]):
    mapping = {}
    for entry in manifest:
        meta = entry.get("meta", {})
        key = (
            normalize_field(meta.get("product")),
            normalize_field(meta.get("brand")),
            normalize_field(meta.get("features")),
        )
        mapping[key] = entry
    return mapping

# ========== 图像编码与 GPT 调用 ==========
def encode_image_to_data_url(img_path: str) -> str:
    if not os.path.exists(img_path):
        return ""
    try:
        with open(img_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
        return f"data:image/jpeg;base64,{b64}"
    except Exception:
        return ""

def call_vsc(client: AzureOpenAI, deployment: str, img_path: str, text: str) -> int:
    img_url = encode_image_to_data_url(img_path)
    if not img_url:
        return -1
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": VSC_PROMPT.format(line=text)},
                {"type": "image_url", "image_url": {"url": img_url}},
            ],
        }
    ]
    try:
        resp = client.chat.completions.create(
            model=deployment,
            messages=messages,
            temperature=0.0,
            max_tokens=10,
        )
        raw = (resp.choices[0].message.content or "").strip()
        m = re.search(r"[0-2]", raw)
        return int(m.group()) if m else -1
    except Exception as e:
        print(f"[ERROR] GPT 调用失败: {e}")
        return -1

# ========== 主流程 ==========
def main():
    parser = argparse.ArgumentParser(description="VSC 评估 (图像-台词匹配度 via GPT-4o)")
    parser.add_argument("--manifest", default="/data/phd/qinsizhong/miltonzhou/sft/data_preprocess/T_manifest_vidsort.json", help="manifest_select.json 文件路径")
    parser.add_argument("--results", default="/data/phd/qinsizhong/llm_factory_test/baselines/results/F___vidsort_Qwen3_8B_sft.jsonl", help="模型推理结果 JSONL")
    parser.add_argument("--frames_dir", default="/data/phd/qinsizhong/miltonzhou/sft/data_preprocess/filter_data/downloaded_frames", help="帧图像目录 {frame_id}.jpg")
    parser.add_argument("--endpoint", type=str, default=os.getenv("ENDPOINT_URL", ""))
    parser.add_argument("--deployment", type=str, default=os.getenv("DEPLOYMENT_NAME", ""))
    parser.add_argument("--api_key", type=str, default=os.getenv("AZURE_OPENAI_API_KEY", ""))
    parser.add_argument("--output", default="/data/phd/qinsizhong/llm_factory_test/baselines/results/VSC___vidsort_Qwen3_8B_sft.jsonl", help="输出 JSON 文件 (含广告级与模型级 VSC)")
    args = parser.parse_args()

    client = AzureOpenAI(
        azure_endpoint=args.endpoint,
        api_key=args.api_key,
        api_version="2025-01-01-preview",
    )

    manifest = read_json(args.manifest)
    results = read_jsonl(args.results)
    manifest_map = build_manifest_map(manifest)

    ads_output = []
    ads_eval, ads_skip = 0, 0

    for rec in tqdm(results, desc="Evaluating vid_sort VSC"):
        # 1. 解析 meta
        human_val = ""
        if isinstance(rec.get("conversations"), list) and len(rec["conversations"]) > 0:
            human_val = rec["conversations"][0].get("value", "")
        meta_key = extract_meta_from_prompt(human_val)
        entry = manifest_map.get(meta_key)
        if not entry:
            ads_skip += 1
            continue

        # 2. 解析模型预测序列
        model_out = rec.get("model_generate", "").strip()
        if not re.match(r"^\d+(,\d+)*$", model_out):
            ads_skip += 1
            continue
        pred_order = [int(x) for x in model_out.split(",") if x.strip().isdigit()]

        # 3. 获取真实台词顺序
        order = entry["label"]["vid_sort"]["correct_order"]
        clips = entry["clips"]
        local_map = {c["local_id"]: c for c in clips}

        if len(order) != len(pred_order):
            print(f"[warn] 样本长度不一致：ad={entry['ad_key']} pred={len(pred_order)} gt={len(order)}")
            ads_skip += 1
            continue

        ad_lines = [local_map[lid]["orig"]["text"] for lid in order]

        # 4. 逐行匹配 frame_id + 台词
        pairs = []
        for i, pred_local_id in enumerate(pred_order):
            if pred_local_id not in local_map:
                continue
            frame_id = local_map[pred_local_id]["orig"]["frame_id"]
            text_line = ad_lines[i]
            img_path = os.path.join(args.frames_dir, f"{frame_id}.jpg")
            score = call_vsc(client, args.deployment, img_path, text_line)
            if score >= 0:
                pairs.append({
                    "frame_id": frame_id,
                    "text": text_line,
                    "score": score
                })

        if not pairs:
            ads_skip += 1
            continue

        vsc_ad = sum(p["score"] for p in pairs) / len(pairs)
        ads_output.append({
            "meta": entry["meta"],
            "vsc_ad": vsc_ad,
            "pairs": pairs
        })
        ads_eval += 1

    model_vsc = sum(ad["vsc_ad"] for ad in ads_output) / len(ads_output) if ads_output else 0.0

    result = {
        "model_VSC": model_vsc,
        "ads_evaluated": ads_eval,
        "ads_skipped": ads_skip,
        "ads": ads_output
    }

    write_json(args.output, result)
    print(f"[完成] 模型平均 VSC={model_vsc:.3f}, 有效广告={ads_eval}, 跳过={ads_skip}")
    print(f"[保存] {args.output}")


if __name__ == "__main__":
    main()
