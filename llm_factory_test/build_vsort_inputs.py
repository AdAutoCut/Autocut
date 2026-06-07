#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build vid_sort inputs (vtok or frame mode) + sort-manifest from vid_select results.

- 输入:
  --select_result     : vid_select 结果 JSONL（每行一条）
  --manifest_select   : vid_select 版 manifest（JSON 数组）
- 输出:
  --out_sort_inputs   : vid_sort 推理输入 JSON（数组；human 按固定 sort 模版，随 --sort_mode 切换）
  --out_sort_manifest : vid_sort 版 manifest（JSON 数组），clips 已按 select 顺序重排并 local_id=1..K

num_clips 取值优先级：
1) select_result 的 label.vid_select.correct_indices 长度
2) manifest.meta.num_pos
3) 从 human 的【广告台词如下】解析出的脚本行数
4) 选中候选数量（len(pred_ids)）

--sort_mode:
- vtok  : 片段块为 “【视频片段如下】”，每行输出 v_tok（包裹 <|video_start|>...）
- frame : 片段块为 “【片段的代表帧如下】”，每行输出 [frame_id]
"""

import json
import re
import argparse
from typing import Dict, List, Tuple, Any

# ------------------------
# 基础工具
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

def load_manifest_vidselect_by_sid(path: str) -> Dict[int, dict]:
    with open(path, "r", encoding="utf-8") as f:
        root = json.loads(f.read().lstrip("\ufeff"))
    if not isinstance(root, list):
        raise TypeError("manifest 根应为 JSON 数组。")

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

# ------------------------
# 从 select 的 human 文本解析字段（商品/品牌/卖点/脚本）
# ------------------------
def extract_fields_from_human(human_value: str) -> Tuple[str, str, str, List[str]]:
    """
    返回: (product, brand, features, script_lines)
    - features 原样字符串（方括号列表）不做 JSON 解析
    - script_lines: 从 '【广告台词如下】' 后开始，到遇到空行+后续 '【' 模块或文本末尾为止
    """
    if not isinstance(human_value, str):
        return "", "", "", []

    # 商品/品牌/卖点
    m_prod = re.search(r"【商品】：(.*)", human_value)
    m_brand = re.search(r"【品牌】：(.*)", human_value)
    m_feat  = re.search(r"【卖点】：(.*)", human_value)
    product = (m_prod.group(1).strip() if m_prod else "")
    brand   = (m_brand.group(1).strip() if m_brand else "")
    features= (m_feat.group(1).strip() if m_feat else "")

    # 脚本块
    script_lines: List[str] = []
    start = human_value.find("【广告台词如下】")
    if start != -1:
        rem = human_value[start + len("【广告台词如下】") :]
        # 更全面的截断锚点（包含“片段的代表帧”）
        cut_patterns = ["【视频片段", "【片段", "【片段的代表帧", "【视频素材", "【视频片段素材", "【现在请输出"]
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

# ------------------------
# 生成 sort 的 human（两种模版）
# ------------------------
def build_sort_human(num_clips: int,
                     product: str,
                     brand: str,
                     features: str,
                     script_lines: List[str],
                     clip_lines: List[str],
                     sort_mode: str) -> str:
    """sort_mode in {'vtok','frame'}"""
    script_block = "\n".join(script_lines)

    if sort_mode == "frame":
        head = (
            "你是专业的视频剪辑师。请你根据提供的广告台词顺序，将下方乱序的片段的代表帧重新排序，并输出正确的编号序列。\n"
            "【输出要求（必须严格遵守）】\n"
            " 1）只输出一行，共 {N} 个编号，使用半角逗号分隔，例如：\"1,3,2,4\"；\n"
            " 2）每个编号只能出现一次，不允许多余或缺少编号；\n"
            " 3）禁止输出任何解释、分析、理由、自然语言句子或多余符号；\n"
            " 4）如果你的回答包含汉字、句子、换行解释等非编号内容，则视为错误回答。\n"
            f"【商品】：{product}\n"
            f"【品牌】：{brand}\n"
            f"【卖点】：{features}\n"
            "【广告台词如下】\n"
            f"{script_block}\n\n"
            "【片段的代表帧如下】\n"
        ).replace("{N}", str(num_clips))
        tail = "\n【现在请输出正确顺序的编号序列】"
    else:
        head = (
            "你是专业的视频剪辑师。请你根据提供的广告台词顺序，将视频片段进行排序。\n"
            "【输出要求】\n"
            " 1）只输出与下方台词句数相同数量的视频片段编号；\n"
            " 2）输出的视频片段编号不能有重复；\n"
            f"【本次台词总句数】：{num_clips} 段；\n"
            f"【商品】：{product}\n"
            f"【品牌】：{brand}\n"
            f"【卖点】：{features}\n"
            "【广告台词如下】\n"
            f"{script_block}\n\n"
            "【视频片段如下】\n"
        )
        tail = "\n【现在请输出按正确顺序排列的视频片段编号序列】"

    return head + "\n".join(clip_lines) + tail

# ------------------------
# sort-manifest 条目
# ------------------------
def to_sort_manifest_entry(sample_id: int,
                           ad_key: dict,
                           meta: dict,
                           reordered_clips: List[dict]) -> dict:
    meta2 = dict(meta) if isinstance(meta, dict) else {}
    meta2["num_clips"] = len(reordered_clips)

    clips_out = []
    for i, c in enumerate(reordered_clips, 1):
        clips_out.append({
            "local_id": i,
            "role": "na",
            "orig": c.get("orig", {})
        })

    return {
        "task": "vid_sort",
        "ad_key": ad_key or {},
        "meta": meta2,
        "clips": clips_out,
        "label": {
            "vid_sort": {
                "correct_order": []  # 如有真值可后续补
            }
        },
        "sample_id": sample_id,
        "indices": {}
    }

# ------------------------
# 主流程
# ------------------------
def main():
    ap = argparse.ArgumentParser(description="从 vid_select 结果 + manifest 生成 vid_sort 输入（vtok 或 frame 模式）与 sort-manifest")
    ap.add_argument("--select_result", type=str, default="/data/phd/qinsizhong/llm_factory_test/baselines/results/F___vidselect_gpt4o.jsonl", help="vid_select 任务的结果 JSONL")
    ap.add_argument("--manifest_select", type=str, default="/data/phd/miltonzhou/sft/data_preprocess/T_manifest_vidselect.json", help="vid_select 版 manifest（JSON 数组）")
    ap.add_argument("--out_sort_inputs", type=str, default="/data/phd/miltonzhou/sft/data_preprocess/T2_gpt_vidsort.json", help="输出：vid_sort 推理输入 JSONL")
    ap.add_argument("--out_sort_manifest", type=str, default="/data/phd/miltonzhou/sft/data_preprocess/T2gpt_manifest_vidsort.json", help="输出：vid_sort 版 manifest JSON")
    ap.add_argument("--sort_mode", type=str, default="frame", choices=["vtok", "frame"], help="生成 sort 输入的片段表达形式")
    ap.add_argument("--max_samples", type=int, default=None, help="最多处理多少条样本（调试用）")
    args = ap.parse_args()

    sid_index = load_manifest_vidselect_by_sid(args.manifest_select)
    print(f"[info] manifest(vid_select) 条目：{len(sid_index)}；sort_mode={args.sort_mode}")

    sort_input_items: List[dict] = []
    sort_manifest_items: List[dict] = []

    total, ok, skip = 0, 0, 0
    with open(args.select_result, "r", encoding="utf-8") as fin:
        for line in fin:
            if args.max_samples is not None and ok >= args.max_samples:
                break
            s = line.strip()
            if not s:
                continue
            total += 1
            mr = json.loads(s)

            if mr.get("task") != "vid_select":
                skip += 1
                continue

            sample_id = mr.get("sample_id")
            if sample_id is None or sample_id not in sid_index:
                print(f"[warn] 跳过：sample_id 缺失或不在 manifest 中（sample_id={sample_id}）")
                skip += 1
                continue

            payload = sid_index[sample_id]
            clips_by_local = payload["clips_by_local"]
            meta = payload.get("meta", {})
            ad_key = payload.get("ad_key", {}) or mr.get("ad_key", {}) or {}
            conversations = mr.get("conversations") or []

            # 解析 select 的选择集合
            pred_ids_raw = parse_indices(mr.get("model_generate", ""))
            pred_ids = dedup_and_filter(pred_ids_raw, clips_by_local)
            if not pred_ids:
                print(f"[warn] 跳过：select 没有有效选择（sample_id={sample_id}）")
                skip += 1
                continue

            # 取被选中的 clip（保持 select 顺序），并构造 sort 的候选列表行
            selected_clips: List[dict] = []
            clip_lines: List[str] = []
            for i, lid in enumerate(pred_ids, 1):
                clip = clips_by_local.get(lid)
                if not clip:
                    continue
                o = clip.get("orig") or {}
                if args.sort_mode == "frame":
                    # 代表帧模式：输出 i)[frame_id]
                    frame_id = o.get("frame_id")
                    clip_line = f"{i})[{frame_id}]"
                else:
                    # vtok 模式：输出 i)<|clip_start|>...<|clip_end|>
                    vtok = o.get("v_tok") or ""
                    vtok_wrapped = ensure_vtok_wrapped(vtok)
                    clip_line = f"{i})<|clip_start|>{vtok_wrapped}<|clip_end|>"
                selected_clips.append(clip)
                clip_lines.append(clip_line)

            if not selected_clips:
                print(f"[warn] 跳过：全部选择都找不到对应 clip（sample_id={sample_id}）")
                skip += 1
                continue

            # 从 human 解析商品/品牌/卖点/脚本
            human_value_src = conversations[0].get("value", "") if (conversations and conversations[0].get("from")=="human") else ""
            product, brand, features, script_lines = extract_fields_from_human(human_value_src)

            # 计算 num_clips 的优先级
            lbl = (mr.get("label") or {}).get("vid_select") or {}
            lbl_correct = lbl.get("correct_indices") or []
            num_from_label  = len(lbl_correct) if isinstance(lbl_correct, list) else 0
            num_from_meta   = int(meta.get("num_pos")) if str(meta.get("num_pos", "")).isdigit() else 0
            num_from_script = len(script_lines)
            num_from_pred   = len(selected_clips)
            num_clips = num_from_label or num_from_meta or num_from_script or num_from_pred
            if num_from_script == 0:
                print(f"[note] sample_id={sample_id}: 未从 human 中解析到脚本行，将按 num_clips={num_clips} 输出。")

            # 构建 sort human（随模式切换）
            human_value = build_sort_human(
                num_clips=num_clips,
                product=product,
                brand=brand,
                features=features,
                script_lines=script_lines,
                clip_lines=clip_lines,
                sort_mode=args.sort_mode
            )

            # 输出一条 sort 输入（数组元素）
            sort_input_obj = {
                "conversations": [
                    {"from": "human", "value": human_value}
                ],
                "system": "",
                "tools": "",
                "sample_id": sample_id,
                "task": "vid_sort",
                "ad_key": ad_key
            }
            sort_input_items.append(sort_input_obj)

            # 生成 sort-manifest 条目（与模式无关，仍然携带 orig 供后续反推 vtok）
            sort_manifest_item = to_sort_manifest_entry(
                sample_id=sample_id,
                ad_key=ad_key,
                meta=meta,
                reordered_clips=selected_clips
            )
            sort_manifest_items.append(sort_manifest_item)

            ok += 1

    # 写文件（JSON 数组）
    with open(args.out_sort_inputs, "w", encoding="utf-8") as f:
        json.dump(sort_input_items, f, ensure_ascii=False, indent=2)
    with open(args.out_sort_manifest, "w", encoding="utf-8") as f:
        json.dump(sort_manifest_items, f, ensure_ascii=False, indent=2)

    print(f"\n[done] 读取 {total} 行；成功生成 {ok} 条；跳过 {skip} 条。")
    print(f"[save] sort inputs  : {args.out_sort_inputs}")
    print(f"[save] sort manifest: {args.out_sort_manifest}")

if __name__ == "__main__":
    main()
