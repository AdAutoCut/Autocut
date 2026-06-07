
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
把含 [ID=xx] + <|clip_start|>...<|clip_end|> 的 ShareGPT 数据，
用 captions 替换 tokens，并生成“数量受约束”的纯文本排序提示词：
- 候选区每行格式：[ID=nn] caption
- 提示词显式包含脚本句数 N 与可选 ID 集合，强约束输出必须恰好 N 个 ID
"""

import json
import re
from typing import Dict, Tuple, List, Any

# ---------- 标记（按你的新格式） ----------
SCRIPT_HEAD = "正确顺序的广告台词："
CLIPS_HEAD  = "素材池如下（包含正确片段与随机干扰片段，顺序已打乱）："

# [ID=xx] 标签与 clip token 块
ID_TAG = re.compile(r"\[ID\s*=\s*([0-9]+)\]\s*", re.IGNORECASE)
CLIP_BLOCK = re.compile(r"<\|clip_start\|>.*?<\|clip_end\|>", re.DOTALL)
ID_PLUS_CLIP = re.compile(
    r"\[ID\s*=\s*([0-9]+)\]\s*(?:\r?\n|\s)*(<\|clip_start\|>.*?<\|clip_end\|>)",
    re.IGNORECASE | re.DOTALL
)

# ---------- 文本工具 ----------
def normalize_script_lines(script_text: str) -> List[str]:
    """
    以换行作为分句基准；去掉空行与首尾空白。
    如需更细粒度，可在此加入基于标点（。！？；）的切分。
    """
    lines = [ln.strip() for ln in script_text.splitlines()]
    return [ln for ln in lines if ln != ""]

def numbered_script_block(lines: List[str]) -> str:
    """将脚本行编号为 (1)..(N) 的块文本"""
    return "\n".join(f"({i+1}) {s}" for i, s in enumerate(lines))

# ---------- 纯文本排序 Prompt 模板（强化版） ----------
def build_new_prompt(
    script_lines: List[str],
    candidates_text: str,
    id_list: List[str],
) -> str:
    N = len(script_lines)
    id_set_str = ",".join(id_list)

    return (
    "【任务】\n"
    "每张图片代表一个视频片段。请将脚本的每一句与最匹配的 Picture k 做一一对应，并按脚本顺序给出答案。\n\n"
    "【必须严格满足的硬性约束】\n"
    f"1) 只输出【恰好 {N} 行】答案，分别对应脚本句子 (1..{N})；\n"
    "2) 每个 Picture k 最多使用一次，禁止重复；\n"
    "3) 只在区间 [1..M] 中选择 Picture k；不得使用不存在的编号；\n"
    "4) 严禁输出形如 1,2,3,4,... 的单调升序列表；该答案将被判定为错误（0 分）；\n"
    "5) 即使不完全确定，也必须基于图像内容做出最可能的匹配；不要输出占位符或“无法判断”。\n\n"
    f"【脚本（共 {N} 句；按正确顺序）】\n"
    f"{numbered_script_block(script_lines)}\n\n"
    "【输出格式】\n"
    " 'Picture x1,Picture x2,...,Picture x{N}'，逗号分隔、无空格、无其他字符。\n\n"
    "【现在开始作答】\n"
    f"请按格式 A给出恰好 {N} 个且互不重复的 Picture 编号映射。不要输出任何解释。\n"
    )
    
    # return (
    # "请你给我所提供的每一张图片，一句话的描述\n"
    # "【输出格式】\n"
    # "Picture 1: （一句描述）\n"
    # "Picture 2: (描述2)...\n"
    # "【现在开始作答】\n"
    # )

# ---------- I/O ----------
def load_dataset(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else [data]

def save_dataset(objs: List[Dict[str, Any]], path: str) -> None:
    out_obj: Any = objs[0] if len(objs) == 1 else objs
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out_obj, f, ensure_ascii=False, indent=2)
    print(f"[OK] Saved to {path}")

def load_captions(jsonl_path: str) -> Dict[Tuple[int, int], str]:
    """
    读取 captions jsonl：每行需包含 ad_id, clip_id, caption
    (ad_id, clip_id) 的 clip_id 从 0 开始；与素材池对齐方式由 align 决定。
    """
    mapping: Dict[Tuple[int, int], str] = {}
    total = 0
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            ad = int(obj["ad_id"])
            cid = int(obj["clip_id"])
            cap = str(obj["caption"])
            mapping[(ad, cid)] = cap
            total += 1
    print(f"[INFO] Captions loaded: {total}")
    return mapping

# ---------- 解析 Human 文本 ----------
def split_human_text(human_value: str) -> Tuple[str, str, str]:
    """
    拆成三段：
    1) prefix_before_script_head
    2) script_text（SCRIPT_HEAD 到 CLIPS_HEAD 之间）
    3) pool_tail（CLIPS_HEAD 之后，包含 [ID=..] + token 块）
    """
    i = human_value.find(SCRIPT_HEAD)
    if i < 0:
        return human_value, "", ""
    prefix = human_value[:i]
    rest   = human_value[i + len(SCRIPT_HEAD):]

    j = rest.find(CLIPS_HEAD)
    if j < 0:
        return prefix, rest, ""

    script_text = rest[:j]
    tail = rest[j + len(CLIPS_HEAD):]
    return prefix, script_text, tail

def iter_id_clip_pairs(tail_text: str) -> List[Tuple[str, str]]:
    """
    按顺序提取 (disp_id, clip_block)：
      [ID=01] <clip_block1>
      [ID=02] <clip_block2>
      ...
    返回顺序即“出现顺序”，可用于 align=by_position。
    """
    pairs: List[Tuple[str, str]] = []
    for m in ID_PLUS_CLIP.finditer(tail_text):
        disp_id = m.group(1)
        clipblk = m.group(2)
        pairs.append((disp_id, clipblk))

    # 容错：如没匹配组合，则独立取 ID 与 clip 块，按最小个数对齐
    if not pairs:
        ids = [m.group(1) for m in ID_TAG.finditer(tail_text)]
        clips = [m.group(0) for m in CLIP_BLOCK.finditer(tail_text)]
        n = min(len(ids), len(clips))
        pairs = [(ids[k], clips[k]) for k in range(n)]
    return pairs

def build_candidates_block_by_pairs(
    pairs: List[Tuple[str, str]],
    ad_id: int,
    capmap: Dict[Tuple[int, int], str],
    missing_policy: str = "keep",
    align: str = "by_position",  # 或 "by_id"
) -> Tuple[str, List[str]]:
    """
    把 (disp_id, clip_block) 转成若干行 "[ID=disp_id] caption"
    返回：(候选区文本, id_list)
    默认按出现顺序 clip_id=0..n 与 captions 对齐；需要时可改 align="by_id"（01->0）。
    """
    out_lines: List[str] = []
    id_list: List[str] = []

    for clip_idx, (disp_id, _clipblk) in enumerate(pairs):
        key_cid = (int(disp_id) - 1) if align == "by_id" else clip_idx
        id_list.append(f"{int(disp_id):02d}")  # 统一两位

        cap = capmap.get((ad_id, key_cid), None)
        if cap is None:
            if missing_policy == "placeholder":
                line = f"[ID={disp_id}] [MISSING CAPTION]"
            else:
                line = f"[ID={disp_id}] [ORIGINAL TOKENS]"
        else:
            line = f"[ID={disp_id}] {cap}"
        out_lines.append(line)

    return "\n".join(out_lines) + ("\n" if out_lines else ""), id_list

def transform_one_human(
    human_value: str,
    ad_id: int,
    capmap: Dict[Tuple[int, int], str],
    missing_policy: str = "keep",
    align: str = "by_position",
) -> str:
    _prefix, script_text, tail = split_human_text(human_value)

    # 1) 脚本分句 + 编号
    script_lines = normalize_script_lines(script_text)

    # 2) 候选区转 caption，并收集 ID 集合
    pairs = iter_id_clip_pairs(tail)  # [(disp_id, clip_block), ...]
    candidates_text, id_list = build_candidates_block_by_pairs(
        pairs, ad_id, capmap, missing_policy=missing_policy, align=align
    )

    # 3) 组装强化后的 Prompt（含 N 与 ID 集合）
    new_human = build_new_prompt(script_lines, candidates_text, id_list)
    return new_human

# ---------- 批处理 ----------
def transform_dataset(
    dataset_path: str,
    captions_jsonl: str,
    output_path: str,
    assume_ad_id_by_index: bool = True,
    missing_policy: str = "keep",
    align: str = "by_position",  # 或 "by_id"
) -> None:
    objs = load_dataset(dataset_path)
    capmap = load_captions(captions_jsonl)

    for i, obj in enumerate(objs):
        conv = obj.get("conversations", [])
        if not conv or len(conv) < 1:
            continue
        if conv[0].get("from") != "human":
            continue

        ad_id = i if assume_ad_id_by_index else int(obj.get("ad_id", i))

        old_text = conv[0]["value"]
        new_text = transform_one_human(
            old_text, ad_id, capmap, missing_policy=missing_policy, align=align
        )
        conv[0]["value"] = new_text

    save_dataset(objs, output_path)

# ---------- 示例运行 ----------
if __name__ == "__main__":
    transform_dataset(
        dataset_path="/data/phd/miltonzhou/sft/data_preprocess/sft_1006_eval.json",
        captions_jsonl="/data/phd/miltonzhou/sft/data_preprocess/filter_data/temp/caption_list_1006.jsonl",
        output_path="/data/phd/miltonzhou/sft/data_preprocess/filter_data/temp/selectnsort_mmllm_test.json",
        assume_ad_id_by_index=True,
        missing_policy="keep",      # 或 "placeholder"
        align="by_id",              # 若 captions 是按 [ID] 数值生成，选 by_id；否则 by_position
    )
