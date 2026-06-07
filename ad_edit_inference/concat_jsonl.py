import os


INPUTS = [
    "/data/phd/qinsizhong/ad_edit_inference/0729_eval.jsonl",
    "/data/phd/qinsizhong/ad_edit_inference/inex_f831/831_eval.jsonl",
]
OUT_PATH = "/data/phd/qinsizhong/ad_edit_inference/pt_831_eval.jsonl"

def concat_jsonl(inputs, out_path):
    # 基本校验
    if not inputs:
        raise SystemExit("没有配置输入文件（INPUTS 为空）")
    abs_out = os.path.abspath(out_path)
    abs_inputs = [os.path.abspath(p) for p in inputs]
    if abs_out in abs_inputs:
        raise SystemExit("输出文件路径不能与任一输入文件相同，请修改 OUT_PATH。")

    # 确保输出目录存在
    out_dir = os.path.dirname(out_path) or "."
    os.makedirs(out_dir, exist_ok=True)

    total = 0
    per_file = []
    with open(out_path, "w", encoding="utf-8") as fout:
        for fp in inputs:
            if not os.path.exists(fp):
                raise SystemExit(f"找不到文件：{fp}")
            n = 0
            with open(fp, "r", encoding="utf-8") as fin:
                for line in fin:
                    if not line or line.strip() == "":
                        continue  # 跳过空行
                    fout.write(line)
                    n += 1
                    total += 1
            per_file.append((fp, n))

    # 打印统计
    for fp, n in per_file:
        print(f"[OK] {fp} -> {n} 行")
    print(f"[DONE] {out_path} 共 {total} 行")

def main():
    concat_jsonl(INPUTS, OUT_PATH)

if __name__ == "__main__":
    main()
