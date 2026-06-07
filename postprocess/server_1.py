#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import List, Optional, Dict, Any

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# ----------------------------
# 一次性加载模型（重要：不要在接口里重复加载）
# ----------------------------
MODEL_NAME = "/data/phd/qinsizhong/llm_factory_test/saves/qwen-8b-sft-1111_ablation_embsft2/checkpoint-24000"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype="auto",
    device_map="auto",
)
model.eval()

# ----------------------------
# 定义请求 / 响应的结构（兼容 OpenAI Chat 形式）
# ----------------------------

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: float = 0.01
    top_p: float = 0.95
    max_tokens: int = 4096
    extra_body: Optional[Dict[str, Any]] = None

class ChatCompletionChoiceMessage(BaseModel):
    role: str
    content: str

class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatCompletionChoiceMessage
    finish_reason: str = "stop"

class ChatCompletionResponse(BaseModel):
    id: str = "chatcmpl-1"
    object: str = "chat.completion"
    choices: List[ChatCompletionChoice]


# ----------------------------
# FastAPI 应用
# ----------------------------

app = FastAPI()

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
def chat_completions(req: ChatCompletionRequest):
    # 1) 把 Pydantic 对象转回 dict，构建 messages
    messages = [{"role": m.role, "content": m.content} for m in req.messages]

    # 2) 用 chat_template 构造模型输入 text（和你本地 predict 一样）
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    print("=== ONLINE CHAT TEMPLATE TEXT ===")
    print(repr(text))
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # 3) 生成
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=req.max_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
        )

    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):]
    content = tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")

    # 4) 封装成 OpenAI 风格返回
    choice = ChatCompletionChoice(
        index=0,
        message=ChatCompletionChoiceMessage(
            role="assistant",
            content=content,
        ),
        finish_reason="stop",
    )

    resp = ChatCompletionResponse(
        choices=[choice],
    )
    return resp


