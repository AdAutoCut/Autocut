def count_jsonl_lines(file_path: str) -> int:
    count = 0
    with open(file_path, "r", encoding="utf-8") as f:
        for _ in f:
            count += 1
    return count

if __name__ == "__main__":
    file_path = "/data/phd/qinsizhong/ad_edit_inference/inex_f831/831_train.jsonl"  # 修改为你的文件路径
    total = count_jsonl_lines(file_path)
    print("Total lines:", total)
