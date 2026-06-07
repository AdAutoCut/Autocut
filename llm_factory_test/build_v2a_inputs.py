#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
从 atc 的 vid_sort 结果 + sort-manifest 生成 vid2aud 推理输入（JSONL）。

- 文本：来自 sort human 的「【广告台词如下】」块，顺序不改
- 视频：按 sort 的 model_generate（local_id 序列）从 manifest 取 orig.v_tok
- 最终按位置配对：第 i 条台词 + 第 i 个 v_tok（长度不等按较短对齐并告警）

输出每行一条 JSON，结构与示例一致（task=vid2aud）。
"""

import json
import re
import argparse
from typing import Dict, List, Tuple

# ------------------------
# 小工具
# ------------------------
def parse_indices(s: str) -> List[int]:
    s = (s or "").replace("，", ",")
    out: List[int] = []
    for p in s.split(","):
        p = p.strip()
        if not p:
            continue
        try:
            out.append(int(p))
        except:
            pass
    return out

def load_sort_manifest_by_sid(path: str) -> Dict[int, dict]:
    with open(path, "r", encoding="utf-8") as f:
        root = json.loads(f.read().lstrip("\ufeff"))
    if not isinstance(root, list):
        raise TypeError("sort_manifest 根应为 JSON 数组。")

    sid_index: Dict[int, dict] = {}
    for obj in root:
        if not isinstance(obj, dict):
            continue
        sid = obj.get("sample_id")
        if sid is None:
            continue

        by_local: Dict[int, dict] = {}
        for c in obj.get("clips", []) or []:
            if isinstance(c, dict) and isinstance(c.get("local_id"), int):
                by_local[c["local_id"]] = c

        sid_index[sid] = {
            "clips_by_local": by_local,
            "meta": obj.get("meta", {}) or {},
            "ad_key": obj.get("ad_key", {}) or {}
        }
    return sid_index

def dedup_and_filter(ids: List[int], clips_by_local: Dict[int, dict]) -> List[int]:
    seen = set()
    out: List[int] = []
    for x in ids:
        if x in seen:
            continue
        if x not in clips_by_local:
            continue
        seen.add(x)
        out.append(x)
    return out

def ensure_vtok_wrapped(vtok: str) -> str:
    vtok = vtok or ""
    if ("<|video_start|>" in vtok) and ("<|video_end|>" in vtok):
        return vtok
    return f"<|video_start|>{vtok}<|video_end|>"

def extract_fields_from_human(human_value: str) -> Tuple[str, str, str, List[str]]:
    """
    从 sort 的 human 里抽取 商品/品牌/卖点/脚本行。
    脚本行从【广告台词如下】后开始，直到遇到【视频片段/片段的代表帧/现在请输出等锚点。
    """
    if not isinstance(human_value, str):
        return "", "", "", []

    m_prod = re.search(r"【商品】：(.*)", human_value)
    m_brand = re.search(r"【品牌】：(.*)", human_value)
    m_feat = re.search(r"【卖点】：(.*)", human_value)
    product = (m_prod.group(1).strip() if m_prod else "")
    brand = (m_brand.group(1).strip() if m_brand else "")
    features = (m_feat.group(1).strip() if m_feat else "")

    script_lines: List[str] = []
    start = human_value.find("【广告台词如下】")
    if start != -1:
        rem = human_value[start + len("【广告台词如下】") :]
        cut_patterns = ["【视频片段", "【片段的代表帧", "【片段", "【视频素材", "【视频片段素材", "【现在请输出"]
        cut = len(rem)
        for pat in cut_patterns:
            p = rem.find(pat)
            if p != -1:
                cut = min(cut, p)
        block = rem[:cut].strip("\n")
        for line in block.splitlines():
            t = line.strip()
            if t:
                script_lines.append(t.rstrip("，,"))
    return product, brand, features, script_lines

def make_clip_segment(text: str, vtok_wrapped: str) -> str:
    return f"<|clip_start|><|text_start|>{text or ''}<|text_end|>{vtok_wrapped}<|clip_end|>"

def build_vid2aud_human(product: str, brand: str, features: str,
                        texts: List[str], vtoks: List[str]) -> str:
    """
    构造 vid2aud 的 human 文本。
    文本按 texts 顺序，视频按 vtoks 顺序；按位置一一配对，长度不等按较短对齐。
    """
    n = min(len(texts), len(vtoks))
    if n == 0:
        return ""

    segments = []
    for i in range(n):
        segments.append(make_clip_segment(texts[i], ensure_vtok_wrapped(vtoks[i])))

    head = (
        "你是专业的广告背景音乐生成助手。请根据广告的商品信息、台词、以及视频片段，输出对应广告的背景音频。\n"
        "【输出要求】\n"
        "1) 只输出一行，且仅包含 <|audio_start|>…<|audio_end|>；\n"
        "2) 禁止输出任何解释、编号、引号或额外符号。\n\n"
        f"【商品】：{product}\n"
        f"【品牌】：{brand}\n"
        f"【卖点】：{features}\n"
        "【台词和视频片段如下】\n"
    )
    tail = "\n【现在请输出相关的背景音乐】"
    return head + "\n".join(segments) + tail

# ------------------------
# 主流程
# ------------------------
def main():
    ap = argparse.ArgumentParser(description="atc: 从 vid_sort 结果 + sort-manifest 生成 vid2aud 推理输入（JSONL）")
    ap.add_argument("--sort_result", type=str, default="/data/phd/qinsizhong/llm_factory_test/baselines/results/F2___vidsort_atc_embsft.jsonl", help="atc 的 vid_sort 结果 JSONL（每行一条）")
    ap.add_argument("--sort_manifest", type=str, default="/data/phd/miltonzhou/sft/data_preprocess/T2atc_manifest_vidsort.json", help="对应的 sort-manifest（JSON 数组）")
    ap.add_argument("--out_vid2aud", type=str, default="/data/phd/miltonzhou/sft/data_preprocess/T2_atc_vid2aud.json", help="输出 vid2aud 推理输入 JSONL 路径")
    ap.add_argument("--max_samples", type=int, default=None, help="最多处理多少条（调试用）")
    args = ap.parse_args()

    sid_index = load_sort_manifest_by_sid(args.sort_manifest)
    print(f"[info] sort-manifest 样本数：{len(sid_index)}")

    vid2aud_items = [] 
    n_total = n_ok = n_skip = 0
    with open(args.sort_result, "r", encoding="utf-8") as fin:
        for line in fin:
            if args.max_samples is not None and n_ok >= args.max_samples:
                break
            s = line.strip()
            if not s:
                continue
            n_total += 1

            try:
                mr = json.loads(s)
            except Exception as e:
                print(f"[warn] JSON 解析失败，跳过一行：{e}")
                n_skip += 1
                continue

            if mr.get("task") != "vid_sort":
                n_skip += 1
                continue

            sample_id = mr.get("sample_id")
            if sample_id is None or sample_id not in sid_index:
                print(f"[warn] 跳过：sample_id 缺失或不在 sort-manifest 中（sample_id={sample_id}）")
                n_skip += 1
                continue

            payload = sid_index[sample_id]
            clips_by_local = payload["clips_by_local"]
            ad_key = payload.get("ad_key", {}) or mr.get("ad_key", {}) or {}

            # human 信息与台词
            conversations = mr.get("conversations") or []
            human_value = conversations[0].get("value", "") if (conversations and conversations[0].get("from")=="human") else ""
            product, brand, features, script_lines = extract_fields_from_human(human_value)
            if not script_lines:
                print(f"[warn] sid={sample_id}: 未解析到台词行，跳过。")
                n_skip += 1
                continue

            # 模型排序 → vtok
            pred_ids = dedup_and_filter(parse_indices(mr.get("model_generate", "")), clips_by_local)
            if not pred_ids:
                print(f"[warn] sid={sample_id}: model_generate 无有效编号，跳过。")
                n_skip += 1
                continue

            vtoks: List[str] = []
            for lid in pred_ids:
                clip = clips_by_local.get(lid)
                v = (clip.get("orig") or {}).get("v_tok") if clip else None
                if v:
                    vtoks.append(v)

            if not vtoks:
                print(f"[warn] sid={sample_id}: 未取到任何 v_tok，跳过。")
                n_skip += 1
                continue

            # 组装 vid2aud human
            human_vid2aud = build_vid2aud_human(product, brand, features, script_lines, vtoks)
            if not human_vid2aud:
                print(f"[warn] sid={sample_id}: human 组装失败（可能长度为 0），跳过。")
                n_skip += 1
                continue

            # 一条 vid2aud 输入
            item = {
                "conversations": [
                    {"from": "human", "value": human_vid2aud}
                    # 推理时由模型输出 <|audio_start|>...<|audio_end|>，此处不带 gpt 字段
                ],
                "system": "",
                "tools": "",
                "sample_id": sample_id,
                "task": "vid2aud",
                "ad_key": ad_key
            }
            vid2aud_items.append(item)
            n_ok += 1

    with open(args.out_vid2aud, "w", encoding="utf-8") as f:
        json.dump(vid2aud_items, f, ensure_ascii=False, indent=2)  # 不加 indent => 非“格式化”JSON

    print(f"\n[done] 读取 {n_total} 行；生成 {n_ok} 条；跳过 {n_skip} 条。")
    print(f"[save] vid2aud 输入：{args.out_vid2aud}")

if __name__ == "__main__":
    main()
