from transformers import AudioFlamingo3ForConditionalGeneration, AutoProcessor
import torch
import re

###需要下载最新版transformer

MODEL_ID = "/data/phd/hf_models/music-flamingo-hf"  ###修改

def load_music_flamingo_model(device="cuda"):
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AudioFlamingo3ForConditionalGeneration.from_pretrained(
        MODEL_ID,
        device_map=device,
        torch_dtype=torch.bfloat16,
    )
    model.eval()
    return model, processor

# ---------------------------
# 描述单个音频
# ---------------------------
def describe_music(file, model, processor, max_new_tokens=256):

    conversation = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Analyze this music: genre, tempo, key, instruments, structure, mood.",
                },
                {"type": "audio", "path": file},
            ],
        }
    ]

    batch = processor.apply_chat_template(
        conversation,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
    ).to(model.device)

    # 修复 float32 -> bf16
    for k, v in batch.items():
        if isinstance(v, torch.Tensor) and v.dtype == torch.float32:
            batch[k] = v.to(torch.bfloat16)

    out = model.generate(**batch, max_new_tokens=max_new_tokens)

    text = processor.batch_decode(
        out[:, batch.input_ids.shape[1] :], skip_special_tokens=True
    )[0]

    return text.strip()

# ---------------------------
# 比较两个描述文本
# ---------------------------
def compare_descriptions(desc1, desc2, model, processor):
    prompt = f"""
Here are two music descriptions:

Music A:
{desc1}

Music B:
{desc2}

Based on genre, instrumentation, structure, and emotional mood,
rate their similarity from 0 to 1. Only output a number.
"""

    conversation = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]

    batch = processor.apply_chat_template(
        conversation,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
    ).to(model.device)

    out = model.generate(**batch, max_new_tokens=64)

    text = processor.batch_decode(
        out[:, batch.input_ids.shape[1] :], skip_special_tokens=True
    )[0]

    nums = re.findall(r"\d+\.?\d*", text)
    values = [float(n) for n in nums if 0 <= float(n) <= 1]
    return values[0] if values else 0.0, text

def music_similarity(file1, file2, model, processor):
    """
    Compute similarity between two music files using Music Flamingo.
    Returns (score, model_raw_output).
    """

    # Step 1: Analyze music A
    desc1 = describe_music(file1, model, processor)

    # Step 2: Analyze music B
    desc2 = describe_music(file2, model, processor)

    # Step 3: Compare descriptions to get similarity score
    score, raw = compare_descriptions(desc1, desc2, model, processor)

    # Return final result
    return score

if __name__ == "__main__":
    model, processor = load_music_flamingo_model()

    score = music_similarity("bgm1.mp3", "bgm2.mp3") # pred,gt

    print("Similarity:", score)
