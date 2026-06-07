from .client import BlobStoreClient

def download_video_bytes(photo_id):
    # photo_id = '167107388360'
    key = f"{photo_id}.mp4"
    client = BlobStoreClient('video-def')
    success, video_bytes = client.download_bytes_from_s3(key)
    if success:
        return video_bytes
    else:
        return None