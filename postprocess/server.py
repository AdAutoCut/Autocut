#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Minimal vLLM HTTP server.

- Loads your SFT model + chat_template.jinja
- Keeps a single vLLM LLM instance alive
- Exposes POST /generate with JSON: {"prompt": "your text"}
- Returns: {"text": "model reply"}
"""

from pathlib import Path
from typing import Optional

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# -----------------------------
# 1) Config: model & template
# -----------------------------
# TODO: change this to your actual SFT model path
MODEL_DIR = Path("/data/phd/qinsizhong/llm_factory_test/saves/qwen-8b-sft-1111_ablation_embsft2/checkpoint-24000")
CHAT_TEMPLATE_PATH = MODEL_DIR / "chat_template.jinja"

# -----------------------------
# 2) Tokenizer + chat template
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
print('CHAT_TEMPLATE:\n', tokenizer.chat_template)

# if CHAT_TEMPLATE_PATH.exists():
#     with CHAT_TEMPLATE_PATH.open("r", encoding="utf-8") as f:
#         tokenizer.chat_template = f.read()
# else: fall back to tokenizer_config.json's chat_template if defined

# -----------------------------
# 3) Lazy global LLM instance
# -----------------------------
_llm: Optional[LLM] = None

def get_llm() -> LLM:
    """Create the global LLM instance on first use (lazy init)."""
    global _llm
    if _llm is None:
        _llm = LLM(
            model=str(MODEL_DIR),
            tensor_parallel_size=torch.cuda.device_count() or 1,
            gpu_memory_utilization=0.95,
            max_model_len=32768,
            # dtype="auto",  # or "bfloat16"/"half" if you used that offline
        )
    return _llm

# -----------------------------
# 4) Default sampling params
#    (set these to match your offline script!)
# -----------------------------
default_sampling = SamplingParams(
    temperature=0.0,
    top_p=0.95,
    top_k=20,
    max_tokens=512,  # == max_new_tokens
    n=1,
)

# -----------------------------
# 5) Prompt builder
# -----------------------------
def build_prompt_from_text(user_text: str) -> str:
    """
    Wrap a simple user string into a one-turn chat and apply chat_template.

    This is the same pipeline as your offline code:
    user text -> messages[] -> tokenizer.apply_chat_template(...)
    """
    messages = [
        {"role": "user", "content": user_text},
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return prompt

# -----------------------------
# 6) FastAPI app & schemas
# -----------------------------
app = FastAPI()

class GenerateRequest(BaseModel):
    prompt: str

class GenerateResponse(BaseModel):
    text: str

@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest) -> GenerateResponse:
    llm = get_llm()  # ensure LLM is created (only once per process)
    prompt = build_prompt_from_text(req.prompt)

    outputs = llm.generate([prompt], sampling_params=default_sampling)
    out = outputs[0]

    if not out.outputs:
        return GenerateResponse(text="")

    gen_text = out.outputs[0].text or ""
    return GenerateResponse(text=gen_text.strip("\n"))

