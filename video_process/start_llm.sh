#!/usr/bin/env bash

# 启动服务端
python -m sglang.launch_server \
  --model-path=/data/phd/hf_models/Qwen3-30B-A3B/ \
  --host=0.0.0.0 \
  --port=8090 \
  --tp-size=4 \
  --mem-fraction-static=0.85 \
  --cuda-graph-max-bs=128 \
  --served-model-name=Qwen3-30B-A3B \
  --context-length=32768 \
  --enable-torch-compile \
  --torch-compile-max-bs 128