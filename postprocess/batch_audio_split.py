import os
import json
import traceback
import argparse
import threading
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils.blobstore import BlobStoreClientManager

from Vocal_split_pb2 import VocalSplitRequest
from kess.framework import ClientOption, GrpcClient
from google.protobuf.json_format import MessageToDict
from Vocal_split_pb2_grpc import VocalSplitServingStub
from batch_client import run_vocal_split_batch

# ---------- 配置参数 ----------
LOCAL_AUDIO_DIR = "/data/phd/miltonzhou/audio_process/audio_wavs"
BUCKET = "ad-nieuwland-material"
ADBUCKET = "nieuwland-material"
SERVICE_NAME = "ad-Vocal-Split"
RESULT_JSON = "split_results_UserStudy.json"
BGM_OUTPUT_DIR = "./bgm_UserStudy"
NUM_WORKERS = 4
# ------------------------------

# Step 1: 并发上传音频到 Blobstore
def upload_one(local_path, key, bucket):
    client = BlobStoreClientManager().get_client(bucket)
    return key, client.upload_file_with_retry(local_path, key)

def upload_all_audio(local_dir, bucket, num_workers):
    print("Upload audio to Blobstore...")
    files = [f for f in os.listdir(local_dir) if f.endswith(".wav")]
    tasks = [(os.path.join(local_dir, f), f) for f in files]
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(upload_one, path, key, bucket): key
            for path, key in tasks
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Uploading"):
            key, success = future.result()
            if not success:
                print(f"upload failed: {key}")
    print("upload completed")

# Step 2: 并发调用 gRPC 语音分离服务（使用线程局部变量避免 signal 注册错误）
###



# Step 3: 并发下载 BGM
def download_one_bgm(key, local_path, bucket):
    client = BlobStoreClientManager().get_client(bucket)
    return key, client.download_file_with_retry(key, local_path)

def download_all_bgm(result_json, output_dir, bucket, num_workers):
    print("downloading...")
    os.makedirs(output_dir, exist_ok=True)
    with open(result_json, "r") as f:
        data = json.load(f)

    tasks = []
    for fname, result in data.items():
        if "error" in result: continue
        base = os.path.splitext(fname)[0]
        key = result["bgm_key"]
        local = os.path.join(output_dir, f"{base}.wav")
        if os.path.exists(local): continue
        tasks.append((key, local))

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(download_one_bgm, key, local, bucket): key
            for key, local in tasks
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading"):
            key, success = future.result()
            if not success:
                print(f"download failed: {key}")
    print("BGM download complete")

# 主流程
if __name__ == "__main__":
    upload_all_audio(LOCAL_AUDIO_DIR, BUCKET, NUM_WORKERS)

    run_vocal_split_batch(
        audio_dir=LOCAL_AUDIO_DIR,
        bucket=ADBUCKET,
        service_name=SERVICE_NAME,
        output_json=RESULT_JSON,
        num_workers=NUM_WORKERS
    )

    download_all_bgm(RESULT_JSON, BGM_OUTPUT_DIR, BUCKET, NUM_WORKERS)
