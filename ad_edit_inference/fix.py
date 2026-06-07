import jsonlines

new = []

with jsonlines.open('excycle/excycle_train.jsonl') as reader:
    for obj in reader:
        # obj 是一个 dict，表示一条记录
        text = obj["text"]
        if text.startswith('<|video'):
            continue
        if text.startswith('<|text_start|>'):
            begin_letter = text[14]
            if 'A' <= begin_letter <= 'Z' or 'a' <= begin_letter <= 'z':
                continue
        obj["text"] = "<|ad_start|>"+ text +"<|ad_end|>"
        new.append(obj)
with jsonlines.open('excycle/excycle_eval_train.jsonl', mode='w') as writer:
    writer.write_all(new)
