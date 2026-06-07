#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
vLLM HTTP client

- 提供一个函数 generate_from_prompt(prompt: str, server_url: str | None = None)
- 默认连接到全局 SERVER_URL，你也可以在调用时覆盖 server_url
"""

import os
import requests
from typing import Optional

# 默认的服务地址，可以按需修改，或者用环境变量覆盖
DEFAULT_SERVER_URL = os.getenv("VLLM_SERVER_URL", "") ##


def generate_from_prompt(
    prompt: str,
    server_url: Optional[str] = None,
    timeout: int = 600,
) -> str:
    """
    调用 vLLM HTTP server 进行一次推理。

    参数
    ----
    prompt : str
        输入给模型的完整文本 prompt（你之前那串台词 + 视频 tokens）。
    server_url : str, optional
        服务的完整 URL，默认使用 DEFAULT_SERVER_URL。
    timeout : int
        请求超时时间（秒）。

    返回
    ----
    str
        模型返回的文本结果（data["text"]，如果不存在则返回空字符串）。
    """
    url = server_url or DEFAULT_SERVER_URL

    resp = requests.post(
        url,
        json={"prompt": prompt},
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()
    return data.get("text", "")


# 下面是方便你单独运行 client 做 quick test（可选）
if __name__ == "__main__":
    test_prompt = (
        "你是专业的视频剪辑师。请你根据提供的广告台词顺序，将视频片段进行排序...【现在请输出按正确顺序排列的视频片段编号序列】"
    )

    print("User:", test_prompt)
    answer = generate_from_prompt(test_prompt)
    print("Model:", answer)
