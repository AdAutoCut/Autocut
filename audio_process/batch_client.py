import os
import json
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from client import Client 
from Vocal_split_pb2 import VocalSplitRequest

def call_vocal_split(fname, client, bucket):
    try:
        req = VocalSplitRequest(vocal_blob_key={"db": "ad", "table": bucket, "key": fname})
        resp = client.sync_run(req)
        if resp is not None and resp.get("status") == "SUCCESS" and "res" in resp:
            return fname, {
                "vocal_key": resp["res"]["vocalPart"]["key"],
                "bgm_key": resp["res"]["noVocalPart"]["key"]
            }
        else:
            return fname, {"error": "invalid or empty response", "raw": resp}
    except Exception as e:
        return fname, {"error": str(e)}

def safe_json_write(path, data):
    """原子方式写入 JSON，避免中断清空"""
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, path)

def run_vocal_split_batch(
    audio_dir: str,
    bucket: str,
    service_name: str = "ad-Vocal-Split",
    output_json: str = "split_results.json",
    num_workers: int = 128
):
    all_files = [f for f in os.listdir(audio_dir) if f.endswith(".wav")]

    if os.path.exists(output_json) and os.path.getsize(output_json) > 0:
        try:
            with open(output_json, "r") as f:
                existing = json.load(f)
        except json.JSONDecodeError:
            print(f" WARNING: {output_json} file error")
            existing = {}
    else:
        existing = {}

    already_done = {
        f for f, v in existing.items()
        if isinstance(v, dict) and "vocal_key" in v and "bgm_key" in v
    }

    to_process = [f for f in all_files if f not in already_done]
    total = len(to_process)
    print(f"Total {total} / {len(all_files)} files to process")

    results = existing
    client = Client(grpc_service_name=service_name)
    error_count = 0

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(call_vocal_split, fname, client, bucket): fname
            for fname in to_process
        }

        with tqdm(total=total, desc="gRPC vocal-splitting") as pbar:
            for future in as_completed(futures):
                fname, result = future.result()
                results[fname] = result

                if "error" in result:
                    error_count += 1
                    print(f"File error: {fname} → {result['error']}")

                # 安全写入 JSON
                safe_json_write(output_json, results)

                pbar.update(1)

    completed_num = total - error_count
    print(f"\nSplit results saved: {output_json}")
    print(f"Total failed files: {error_count} / {total}")
    print(f"Completed files: {completed_num}")
