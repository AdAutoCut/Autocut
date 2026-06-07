import json
import re
import time
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from loguru import logger

logger.remove()
logger.add(lambda msg: print(msg, end=""), format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>\n")

# ---- 你可以根据需要补充/修改 special 列表 ----
SPECIAL_TOKEN_LIST = [
    "<|ad_start|>", "<|ad_end|>",
    "<|clip_start|>", "<|clip_end|>",
    "<|video_start|>", "<|video_end|>",
    "<|frame_start|>", "<|frame_end|>",
    "<|text_start|>", "<|text_end|>",
    "<|audio_start|>", "<|audio_end|>",
]

def extract_all_tokens(text):
    video_audio_token_pattern = r"<[va]_\d+_\d+>"
    special_token_pattern = r"<\|[^>]+?\|>"
    all_tokens = re.findall(f"{video_audio_token_pattern}|{special_token_pattern}", text)
    return all_tokens

def process_line(line):
    try:
        data = json.loads(line)
        text = data.get("text", "")
        tokens = extract_all_tokens(text)
        return Counter(tokens)
    except Exception as e:
        logger.warning(f"Error processing line: {e}")
        return Counter()

def count_tokens_in_jsonl(jsonl_path, max_workers=16, log_interval=1000):
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    total = len(lines)
    logger.info(f"Loaded {total} lines from {jsonl_path}")

    token_counter = Counter()
    completed = 0
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_line, line): i for i, line in enumerate(lines)}
        for future in as_completed(futures):
            token_counter.update(future.result())
            completed += 1
            if completed % log_interval == 0 or completed == total:
                logger.info(f"Processed {completed}/{total} lines")

    elapsed = time.time() - start_time
    logger.info(f"Token counting complete in {elapsed:.2f} seconds.")
    return token_counter

def ensure_all_tokens(token_counter: Counter, special_tokens=None):
    """把所有可能的 token 都补齐到 counter 里（未出现的置 0）"""
    # video tokens
    for i in range(8):
        for j in range(256):
            token_counter.setdefault(f"<v_{i}_{j}>", 0)
    # audio tokens
    for i in range(8):
        for j in range(256):
            token_counter.setdefault(f"<a_{i}_{j}>", 0)
    # special tokens（用你声明的全集；并合并数据中已出现的）
    existing_specials = [t for t in token_counter.keys() if t.startswith("<|")]
    full_specials = set(existing_specials) | set(special_tokens or [])
    for tok in full_specials:
        token_counter.setdefault(tok, 0)
    return token_counter

def classify_tokens(counter: Counter):
    video = Counter()
    audio = Counter()
    special = Counter()
    for token, freq in counter.items():
        if token.startswith("<v_"):
            video[token] = freq
        elif token.startswith("<a_"):
            audio[token] = freq
        elif token.startswith("<|"):
            special[token] = freq
    return video, audio, special

def plot_token_heatmap(token_counter: Counter, prefix: str, title: str, output_path: str):
    """热力图：count=0 的格子显示为白色"""
    matrix = np.zeros((8, 256), dtype=int)
    for i in range(8):
        for j in range(256):
            token = f"<{prefix}_{i}_{j}>"
            matrix[i, j] = token_counter.get(token, 0)

    mask = (matrix == 0)  # True 的位置将被透明化，用轴背景色填充
    plt.figure(figsize=(20, 4))
    ax = sns.heatmap(
        matrix,
        cmap="YlGnBu",   # 你也可以换成 "viridis"/"inferno"/"rocket" 等
        mask=mask,
        cbar=True,
    )
    # 被 mask 的地方用白色
    ax.set_facecolor('white')

    plt.title(title)
    plt.xlabel("j (0–255)")
    plt.ylabel("i (0–7)")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()

def plot_special_barplot(special_counter: Counter, top_k=None):
    items = special_counter.most_common(top_k) if top_k else list(special_counter.items())
    # 确保顺序稳定（按 token 名排序），便于比对
    items.sort(key=lambda x: x[0])
    tokens, counts = zip(*items)

    plt.figure(figsize=(max(10, len(tokens) * 0.4), 5))
    sns.barplot(x=list(tokens), y=list(counts))
    plt.xticks(rotation=45, ha="right")
    plt.title("Special Tokens Frequency")
    plt.xlabel("Token")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig("special_tokens_barplot.png")
    plt.show()

def main():
    jsonl_path = "/data/phd/qinsizhong/llm_factory_test/data/0729_train.jsonl"
    # jsonl_path = "fuyu.jsonl"

    # 1) 统计
    token_counter = count_tokens_in_jsonl(jsonl_path, max_workers=16)

    # 2) 补全所有 token（包含 count=0）
    token_counter = ensure_all_tokens(token_counter, special_tokens=SPECIAL_TOKEN_LIST)

    # 3) 分类
    video_counter, audio_counter, special_counter = classify_tokens(token_counter)

    # 4) 保存为 JSON（全量 token，含 0）
    with open("token_frequency.json", "w", encoding="utf-8") as f_out:
        json.dump(dict(token_counter), f_out, indent=2, ensure_ascii=False, sort_keys=True)

    # 5) 可视化（0 次出现的格子是白色）
    plot_token_heatmap(video_counter, prefix="v", title="Video Token Frequency Heatmap", output_path="video_token_heatmap.png")
    plot_token_heatmap(audio_counter, prefix="a", title="Audio Token Frequency Heatmap", output_path="audio_token_heatmap.png")
    plot_special_barplot(special_counter)

    logger.info('Finished.')

if __name__ == "__main__":
    main()
