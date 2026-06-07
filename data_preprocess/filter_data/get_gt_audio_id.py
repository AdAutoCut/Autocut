
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#### 音频检索库是User Study的小库

import os
import json
import re
import argparse
import torch
from tqdm import tqdm
from tools.audio import init_audio_model, atoken2aemb, aemb2audio_t5, astr2tok

# 通用读取器：优先整块 JSON（对象或数组），失败则按 JSONL 行解析
def iter_items(path):
    with open(path, "r", encoding="utf-8") as f:
        txt = f.read().strip()
    if not txt:
        return
    try:
        data = json.loads(txt)
        if isinstance(data, list):
            for obj in data:
                yield obj
        else:
            yield data
        return
    except json.JSONDecodeError:
        pass
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

def astr_to_audio(astr: str, t):
    atok = astr2tok(astr)
    print(atok)                    # List[int]
    audio_tensor = torch.tensor(atok)
    with torch.no_grad():
        aemb = atoken2aemb(audio_tensor)     # embeddings
        audio = aemb2audio_t5(aemb, t)          # 最终 audio 结构（ID/路径/对象）
    return audio

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="/data/phd/qinsizhong/llm_factory_test/baselines/results/F2___vid2aud_atc_embsft.jsonl", help="输入 JSON 或 JSONL（每行含 conversations）")
    parser.add_argument("--output", default="/data/phd/miltonzhou/sft/data_preprocess/F2___audrank_atc_embsft.jsonl", help="输出 JSONL（每行 {ad_id, aud_id}）")
    args = parser.parse_args()

    # 只初始化一次
    init_audio_model(load_faiss=True, model_name="audio_8_256_0729")

    with open(args.output, "w", encoding="utf-8") as fout:
        # 无 total 的进度条（兼容大文件且不需要预扫描）
        for ad_id, obj in enumerate(tqdm(iter_items(args.input), desc="Processing", unit="ad"), start=1):
            sid = obj.get("sample_id")
            convs = obj.get("conversations", [])
            # if not isinstance(convs, list) or len(convs) < 2:
            #     continue  # 极简：不符合就跳过

            # gt_astr = convs[1].get("value", "")
            # gt_audio = astr_to_audio(gt_astr, t = 5)

            pred_astr = obj.get("model_generate","")
            pred_audio = astr_to_audio(pred_astr, t = 350)

            # 逐条写出 + 立即落盘
            rec = {"sample_id": sid, "predict_aud_id": pred_audio}
            fout.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
            fout.flush()
            os.fsync(fout.fileno())

if __name__ == "__main__":
    main()
