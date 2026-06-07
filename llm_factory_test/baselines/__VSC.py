#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VSC Evaluation via GPT-4o (图像-台词匹配度评估，含详细图文对结果)
"""

import os
import re
import json
import base64
import argparse
from typing import Dict, Any, List, Tuple
from tqdm import tqdm
from openai import AzureOpenAI


# ========== VSC Prompt ==========
VSC_PROMPT = """
你是一名广告创意评估专家。现在请你评估一张图片与一行广告台词的语义匹配程度。
图片代表广告视频中的一个画面，台词是与该画面对应的一句广告台词。

请判断这张图片与这行台词的内容是否相符、是否传达了相同的语义或情境。

评分标准（仅输出数字）：
0 = 完全不匹配（画面与台词毫无关系或矛盾）
1 = 部分相关（画面与台词有一定联系，但不完全对应或语义模糊）
2 = 高度匹配（画面清晰体现了台词内容，语义一致，画面可直接支撑该台词）

只输出一个数字，不要解释。
台词如下：
{line}
"""

# ========== I/O ==========
def read_jsonl(path: str) -> List[Dict[str, Any]]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except Exception as e:
                    print(f"[WARN] JSON parse error: {e}")
    return data

def read_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def write_json(path: str, obj: Any):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

# ========== Meta 匹配 ==========
def normalize_meta_field(s: str) -> str:
    if s is None:
        return ""
    s = str(s).replace("'", '"').replace("，", ",").replace(" ", "")
    return s.strip()

def extract_meta_from_prompt(text: str) -> Tuple[str, str, str]:
    prod = re.search(r"【商品】：([^\n]+)", text)
    brand = re.search(r"【品牌】：([^\n]+)", text)
    feats = re.search(r"【卖点】：([^\n]+)", text)
    p = normalize_meta_field(prod.group(1)) if prod else ""
    b = normalize_meta_field(brand.group(1)) if brand else ""
    f = normalize_meta_field(feats.group(1)) if feats else ""
    return p, b, f

def build_manifest_map(manifest: List[Dict[str, Any]]) -> Dict[Tuple[str, str, str], Dict[str, Any]]:
    m = {}
    for entry in manifest:
        meta = entry.get("meta", {})
        key = (
            normalize_meta_field(meta.get("product")),
            normalize_meta_field(meta.get("brand")),
            normalize_meta_field(meta.get("features")),
        )
        m[key] = entry
    return m

# ========== GPT-4o 调用 ==========
def encode_image_to_data_url(img_path: str) -> str:
    if not os.path.exists(img_path):
        return ""
    try:
        with open(img_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
        return f"data:image/jpeg;base64,{b64}"
    except Exception:
        return ""

def call_vsc_model(client: AzureOpenAI, deployment: str, image_path: str, line: str) -> int:
    data_url = encode_image_to_data_url(image_path)
    if not data_url:
        return -1
    messages = [
        {"role": "user", "content": [
            {"type": "text", "text": VSC_PROMPT.format(line=line)},
            {"type": "image_url", "image_url": {"url": data_url}}
        ]}
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
        print(f"[error] GPT-4o 调用失败: {e}")
        return -1


# ========== 主流程 ==========
def main():
    parser = argparse.ArgumentParser(description="VSC 评估 (图像-台词匹配度 via GPT-4o)")
    parser.add_argument("--manifest", default="/data/phd/miltonzhou/sft/data_preprocess/TEST_manifest_sort.json", help="manifest_select.json 文件路径")
    parser.add_argument("--results", default="/data/phd/qinsizhong/llm_factory_test/baselines/results/FINAL___sort_Qwen3_8B.jsonl", help="模型推理结果 JSONL")
    parser.add_argument("--frames_dir", default="/data/phd/miltonzhou/sft/data_preprocess/filter_data/downloaded_frames", help="帧图像目录 {frame_id}.jpg")
    parser.add_argument("--endpoint", type=str, default=os.getenv("ENDPOINT_URL", ""))
    parser.add_argument("--deployment", type=str, default=os.getenv("DEPLOYMENT_NAME", ""))
    parser.add_argument("--api_key", type=str, default=os.getenv("AZURE_OPENAI_API_KEY", ""))
    parser.add_argument("--output", default="/data/phd/qinsizhong/llm_factory_test/baselines/results/VSC___sort_Qwen3_8B.jsonl", help="输出 JSON 文件 (含广告级与模型级 VSC)")
    args = parser.parse_args()

    client = AzureOpenAI(
        azure_endpoint=args.endpoint,
        api_key=args.api_key,
        api_version="2025-01-01-preview",
    )

    manifest = read_json(args.manifest)
    results = read_jsonl(args.results)
    manifest_map = build_manifest_map(manifest)

    ads_detailed = []
    match_count, skip_count = 0, 0

    for rec in tqdm(results, desc="VSC Evaluating"):
        human = ""
        if isinstance(rec.get("conversations"), list) and len(rec["conversations"]) > 0:
            human = rec["conversations"][0].get("value", "")
        product, brand, features = extract_meta_from_prompt(human)
        key = (product, brand, features)
        manifest_entry = manifest_map.get(key)
        if not manifest_entry:
            skip_count += 1
            continue

        model_out = rec.get("model_generate", "")
        if not model_out or not re.match(r"^\d+(,\d+)*$", model_out.strip()):
            skip_count += 1
            continue

        selected_ids = [int(x) for x in model_out.split(",") if x.strip().isdigit()]
        clips = manifest_entry.get("clips", [])
        local_map = {c["local_id"]: c for c in clips}
        ad_lines = [c["orig"]["text"] for c in clips if c["orig"]["ad_id"] == manifest_entry["ad_key"]["ad_id"] and c["role"] == "pos"]
        ad_lines.sort()

        if len(selected_ids) != len(ad_lines):
            skip_count += 1
            continue

        pairs = []
        for idx, local_id in enumerate(selected_ids):
            clip = local_map.get(local_id)
            if not clip:
                continue
            frame_id = clip["orig"]["frame_id"]
            line = ad_lines[idx]
            img_path = os.path.join(args.frames_dir, f"{frame_id}.jpg")
            score = call_vsc_model(client, args.deployment, img_path, line)
            if score >= 0:
                pairs.append({
                    "frame_id": frame_id,
                    "text": line,
                    "score": score
                })

        if not pairs:
            skip_count += 1
            continue

        vsc_ad = sum(p["score"] for p in pairs) / len(pairs)
        ads_detailed.append({
            "meta": {
                "product": manifest_entry["meta"]["product"],
                "brand": manifest_entry["meta"]["brand"],
                "features": manifest_entry["meta"]["features"]
            },
            "vsc_ad": vsc_ad,
            "pairs": pairs
        })
        match_count += 1

    avg_vsc = sum(a["vsc_ad"] for a in ads_detailed) / len(ads_detailed) if ads_detailed else 0.0
    output_data = {
        "model_VSC": avg_vsc,
        "ads_evaluated": match_count,
        "ads_skipped": skip_count,
        "ads": ads_detailed
    }

    write_json(args.output, output_data)
    print(f"[完成] 模型平均 VSC = {avg_vsc:.3f}, 有效广告 = {match_count}, 跳过 = {skip_count}")
    print(f"[输出] 已保存详细结果到 {args.output}")


if __name__ == "__main__":
    main()
