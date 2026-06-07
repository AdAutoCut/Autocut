from io import BytesIO
from pydub import AudioSegment

def extract_audio_from_video(video_bytes):
    """返回 audio_bytes"""
    video_buf = BytesIO(video_bytes)
    audio_buf = BytesIO()
    AudioSegment.from_file(video_buf).export(audio_buf, format='mp3')
    audio_bytes = audio_buf.getvalue()
    return audio_bytes