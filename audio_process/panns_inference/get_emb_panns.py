import os
import json
import random
import torch
import torchaudio
from tqdm import tqdm
from typing import Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from panns_inference import AudioTagging


def get_bgm_embedding(audio_path, model, threshold=0.5):
    """使用 panns 模型获取某一音频是否为音乐、得分和 embedding"""
    waveform, sr = torchaudio.load(audio_path)
    waveform = waveform.mean(dim=0, keepdim=True)  # 转 mono
    clipwise_output, embedding = model.inference(waveform)
    is_music = float(clipwise_output[0, 137]) > threshold  # 137 是 "music"
    score = float(clipwise_output[0, 137])
    return is_music, score, embedding


def process_file(fpath, model):
    """处理单个音频文件，返回其结果字典"""
    try:
        is_music, score, emb = get_bgm_embedding(fpath, model)
        if isinstance(emb, torch.Tensor):
            emb = emb.cpu().numpy()
        result = {
            "is_music": is_music,
            "score": score,
            "embedding": emb.tolist()
        }
        return os.path.basename(fpath), result
    except Exception as e:
        print(f"Error processing {fpath}: {e}")
        return os.path.basename(fpath), None


def main(audio_dir, output_path=None, max_workers=8):
    """主程序：并行提取所有音频文件的 embedding 并保存为 JSON"""
    device = 'cpu'  # 多线程时必须使用 CPU，避免 GPU 冲突
    model = AudioTagging(checkpoint_path='./panns_data/audiotagging.pth', device=device)

    audio_files = [
        os.path.join(audio_dir, f)
        for f in os.listdir(audio_dir)
        if f.endswith(".wav")
    ]

    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_file, fpath, model): fpath
            for fpath in audio_files
        }

        for future in tqdm(as_completed(futures), total=len(futures)):
            fname, result = future.result()
            if result is not None:
                results[fname] = result
                print(f"{fname}: score={result['score']:.3f} → {'MUSIC' if result['is_music'] else 'NON-MUSIC'}")

    if output_path:
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n✅ Saved JSON to: {output_path}")


def convert_json_to_tensor_pt(json_path: str, output_dir: str, train_ratio: float = 0.9):
    """从 JSON 文件中读取 embedding，保存为 train.pt / test.pt"""
    with open(json_path, "r") as f:
        data = json.load(f)

    all_entries: Dict[str, torch.Tensor] = {}
    for fname, content in data.items():
        photoid = os.path.splitext(fname)[0]
        emb = content["embedding"]
        tensor_emb = torch.tensor(emb, dtype=torch.float32)
        all_entries[photoid] = tensor_emb

    keys = list(all_entries.keys())
    random.shuffle(keys)

    split_idx = int(len(keys) * train_ratio)
    train_keys = keys[:split_idx]
    test_keys = keys[split_idx:]

    train_data = {k: all_entries[k] for k in train_keys}
    test_data = {k: all_entries[k] for k in test_keys}

    os.makedirs(output_dir, exist_ok=True)
    torch.save(train_data, os.path.join(output_dir, "train1.pt"))
    torch.save(test_data, os.path.join(output_dir, "test1.pt"))

    print(f"✅ Saved {len(train_data)} training samples and {len(test_data)} testing samples to {output_dir}")


if __name__ == "__main__":
    # 路径配置
    audio_dir = "/data/phd/miltonzhou/audio_process/bgm_audio"
    json_path = "/data/phd/miltonzhou/audio_process/panns_embeddings.json"
    pt_output_dir = "/data/phd/miltonzhou/audio_process/panns_pt"

    # 步骤一：提取 embeddings 并存入 JSON
    main(audio_dir=audio_dir, output_path=json_path, max_workers=8)

    # 步骤二：将 JSON 转换为 .pt 文件
    convert_json_to_tensor_pt(json_path=json_path, output_dir=pt_output_dir)
