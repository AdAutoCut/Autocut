#!/usr/bin/env python3
import os
import sys
import argparse
from tools.common.blobstore import download_video_bytes

def main():
    parser = argparse.ArgumentParser(description="Download a single video by photo_id")
    parser.add_argument(
        "photo_id",
        nargs="?",                         # 允许零个或一个值
        default="162503447198",       # 替换为你想要的默认 photo_id
        help="ID of the photo/video to download (default: %(default)s)"
    )
    parser.add_argument(
        "--output-dir", 
        default="downloaded_vids", 
        help="Directory to save the downloaded video"
    )
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Download video bytes
    video_bytes = download_video_bytes(args.photo_id)
    if not video_bytes:
        print(f"Failed to download video for photo_id {args.photo_id}", file=sys.stderr)
        sys.exit(1)

    # Write bytes to file
    output_path = os.path.join(args.output_dir, f"{args.photo_id}.mp4")
    with open(output_path, "wb") as f:
        f.write(video_bytes)

    print(f"Video saved to: {output_path}")

if __name__ == "__main__":
    main()
