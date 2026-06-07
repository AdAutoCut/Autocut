import os
os.environ['TORCH_HOME']='/data/phd/qinsizhong/.cache/torch'

from ..blobstore.client import BlobStoreClient
from .client import VocalSplitClient
from .Vocal_split_pb2 import VocalSplitRequest
import uuid

# import demucs.separate



def split_vocal(audio_bytes, photo_id=None):
    """返回 背景部分bytes，在存储桶中的key"""
    blobstore_client = BlobStoreClient('ad-nieuwland-material')
    vocal_split_client = VocalSplitClient()
    if photo_id is None:
        photo_id = uuid.uuid4()
    blobstore_client.upload_bytes_to_s3(audio_bytes, f"{photo_id}.mp3")
    req = VocalSplitRequest(vocal_blob_key={"db": "ad", "table": "nieuwland-material", "key": f"{photo_id}.mp3"})
    resp = vocal_split_client.sync_run(req)
    if resp:
        no_vocal_part = resp.get('res', {}).get('noVocalPart', {}).get('key', {})
    else:
        raise Exception("[处理失败_vocal_split] no resp")
    downloaded, bgm_bytes = blobstore_client.download_bytes_from_s3(no_vocal_part)
    if downloaded is False or bgm_bytes is None:
        raise Exception("[处理失败_no_vocal_part download]")
    return bgm_bytes, no_vocal_part


def split_vocal_demucs(audio_path, photo_id=None):
    """返回 背景部分bytes，在存储桶中的key"""
    demucs.separate.main(["--mp3", "--two-stems", "vocals", f"{audio_path}"])
