import json
import os

def split_jsonl(input_path, output_dir, chunk_size=10000):
    os.makedirs(output_dir, exist_ok=True)
    with open(input_path, "r", encoding="utf-8") as fin:
        chunk_id = 0
        current_lines = []
        for i, line in enumerate(fin):
            line = line.strip()
            if not line:
                continue
            current_lines.append(line)

            if len(current_lines) >= chunk_size:
                out_path = os.path.join(output_dir, f"raw_chunk_{chunk_id}.jsonl")
                with open(out_path, "w", encoding="utf-8") as fout:
                    fout.write("\n".join(current_lines) + "\n")
                print(f"Saved chunk {chunk_id} with {len(current_lines)} lines.")
                current_lines = []
                chunk_id += 1

        # Write last chunk if any remaining
        if current_lines:
            out_path = os.path.join(output_dir, f"raw_chunk_{chunk_id}.jsonl")
            with open(out_path, "w", encoding="utf-8") as fout:
                fout.write("\n".join(current_lines) + "\n")
            print(f"Saved final chunk {chunk_id} with {len(current_lines)} lines.")

if __name__ == "__main__":
    input_jsonl = "/data/phd/miltonzhou/sft/data_preprocess/filter_data/prefiltered_data_1102_train.jsonl"
    output_dir = "data_chunk_1102"
    split_jsonl(input_jsonl, output_dir, chunk_size=10000)
