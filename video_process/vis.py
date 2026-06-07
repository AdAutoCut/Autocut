import json
import os
from get_embeddings import download_and_extract_frames

# data=[]
# with open("data/output_result.jsonl",'r',encoding='utf-8') as f:
#     for line in f:
#         line=line.strip()
#         if not line:
#             continue
#         data.append(json.loads(line))
# print("hello")

if __name__ == "__main__":
    download_and_extract_frames(153248662354)
    print("hello")
    a = 3
