#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv

def main():
    input_dir = "/data/phd/miltonzhou/audio_process/downloaded_vids"
    output_csv = "inex_us.csv"

    # 获取所有 .mp4 文件名（去掉扩展名）
    photo_ids = [os.path.splitext(f)[0] for f in os.listdir(input_dir) if f.endswith(".mp4")]

    # 保存到 CSV
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["photo_id"])
        for pid in sorted(photo_ids):
            writer.writerow([pid])

    print(f"共保存 {len(photo_ids)} 个 photo_id 到 {output_csv}")

if __name__ == "__main__":
    main()
