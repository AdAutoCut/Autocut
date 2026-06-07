'''
替换sft_901_eval.json 中的tokens，用captions_list.jsonl里面的标签代替。生成sharegpt格式，为后续纯文本baseline做推理数据准备。
'''

import json
import re
from typing import Dict, Tuple, List, Any

# ---------- Configurable markers (Chinese) ----------
SCRIPT_HEAD = "正确顺序的广告台词："
CLIPS_HEAD  = "视频片段素材如下："

# Regex: match each tokenized clip block followed by its display id (e.g., 01), optional comma
CLIP_WITH_ID = re.compile(r"<\|clip_start\|>.*?<\|clip_end\|>\s*([0-9]+)\s*,?", re.DOTALL)

# ---------- New pure-text LLM prompt skeleton (Chinese) ----------
def build_new_prompt(script_text: str, candidates_text: str) -> str:
    return (
        "你是一个严格遵循格式的排序器。给定一段广告台词（脚本）和若干候选片段的“文字描述”（caption），"
        "你的任务是：找出最符合脚本顺序的片段编号序列。\n\n"
        "【脚本（按正确顺序）】\n"
        f"{script_text.strip()}\n\n"
        "【候选片段（乱序；每行=caption+编号）】\n"
        f"{candidates_text.rstrip()}\n\n"
        "要求：\n"
        "1. 仅输出一行、只包含逗号分隔的两位编号序列.\n"
        "2. 不要输出除编号外的任何文字；不要解释；不要重复脚本或caption。\n"
        "3. 保持所有出现过的编号各出现且仅出现一次（不丢失、不重复）。\n\n"
        "【现在开始，输出一行编号序列】"
    )

# ---------- I/O helpers ----------
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

# ---------- Core parsing ----------
def split_human_text(human_value: str) -> Tuple[str, str, str]:
    """
    Split human.value into: prefix_before_script_head, script_text, clips_block_and_after
    Expect markers SCRIPT_HEAD then CLIPS_HEAD.
    """
    i = human_value.find(SCRIPT_HEAD)
    if i < 0:
        # If SCRIPT_HEAD not found, treat all as prefix, empty script.
        return human_value, "", ""
    prefix = human_value[:i]
    rest   = human_value[i + len(SCRIPT_HEAD):]

    j = rest.find(CLIPS_HEAD)
    if j < 0:
        # If CLIPS_HEAD not found, everything after SCRIPT_HEAD is script_text
        return prefix, rest, ""

    script_text = rest[:j]
    tail = rest[j + len(CLIPS_HEAD):]
    return prefix, script_text, tail

def build_candidates_block(tail_text: str, ad_id: int, capmap: Dict[Tuple[int, int], str],
                           missing_policy: str = "keep") -> str:
    """
    Convert tokenized clip blocks in tail_text into caption+ID lines:
      '<|clip...|>...<|clip_end|>01,' -> 'caption01\n'
    Appearance order defines clip_id=0,1,2,...
    """
    out_lines: List[str] = []
    clip_idx = 0
    last = 0
    parts: List[str] = []

    # We will reconstruct tail_text replacing each matched block with a placeholder line,
    # but since the new prompt will completely re-format, we only need the candidate lines.
    for m in CLIP_WITH_ID.finditer(tail_text):
        disp_id = m.group(1)  # display id like "01"
        # But our regex stored id in group(1); careful: we captured ([0-9]+) as group(1)
        disp_id = m.group(1)

        # fetch caption by (ad_id, clip_idx)
        cap = capmap.get((ad_id, clip_idx), None)
        if cap is None:
            if missing_policy == "placeholder":
                line = f"[MISSING CAPTION]{disp_id}"
            else:
                # fallback: keep the original tokens shrunk to a generic marker
                line = f"[ORIGINAL TOKENS]{disp_id}"
        else:
            # emit caption + display_id + newline
            line = f"{cap}{disp_id}"
        out_lines.append(line)
        clip_idx += 1

    # Join with newline
    return "\n".join(out_lines) + ("\n" if out_lines else "")

def transform_one_human(human_value: str, ad_id: int,
                        capmap: Dict[Tuple[int, int], str],
                        missing_policy: str = "keep") -> str:
    prefix, script_text, tail = split_human_text(human_value)

    # Clean script_text (keep original lines as-is)
    script_text = script_text.strip("\n")

    # Build candidates from tail by replacing token blocks to caption+ID lines
    candidates_text = build_candidates_block(tail, ad_id, capmap, missing_policy=missing_policy)

    # Build the NEW prompt (discard old header/prefix entirely)
    new_human = build_new_prompt(script_text, candidates_text)
    return new_human

# ---------- Batch transform ----------
def transform_dataset(
    dataset_path: str,
    captions_jsonl: str,
    output_path: str,
    assume_ad_id_by_index: bool = True,
    missing_policy: str = "keep",
) -> None:
    objs = load_dataset(dataset_path)
    capmap = load_captions(captions_jsonl)

    for i, obj in enumerate(objs):
        conv = obj.get("conversations", [])
        if not conv or len(conv) < 1:
            continue
        if conv[0].get("from") != "human":
            continue

        # Decide ad_id
        ad_id = i if assume_ad_id_by_index else int(obj.get("ad_id", i))

        old_text = conv[0]["value"]
        new_text = transform_one_human(old_text, ad_id, capmap, missing_policy=missing_policy)
        conv[0]["value"] = new_text

    save_dataset(objs, output_path)

# ---------- Example run ----------
if __name__ == "__main__":
    transform_dataset(
        dataset_path="/data/phd/qinsizhong/llm_factory_test/baselines/data/sft_901_eval.json",     # 原始数据（单对象或列表都可）
        captions_jsonl="/data/phd/miltonzhou/sft/data_preprocess/filter_data/temp/caption_list.jsonl",  # 你的 captions
        output_path="/data/phd/miltonzhou/sft/data_preprocess/filter_data/temp/sort_929_textllm.jsonl",  # 输出文件
        assume_ad_id_by_index=True,           # 若条目里没有显式 ad_id，则按索引匹配
        missing_policy="keep",                # 或 "placeholder"
    )
