### Make SFT Dataset from PT dataset

import re
import json
import random

# Utils
def extract_text_video_pairs_string_tokens(sample_text):
    """
    提取每段 <|text_start|>...<|text_end|> 与其后紧跟的 <|video_start|>...<|video_end|>，
    保留所有原始 token 标记，作为原始字符串返回。
    
    返回 List[Dict]，每个 dict 包含：
    {
        "text": "就是用了这套青春期护肤套装，",
        "video_frames": "<|video_start|><|frame_start|>...<|frame_end|>...<|video_end|>"
    }
    """
    results = []

    # 匹配所有 text 位置
    text_pattern = re.compile(r"<\|text_start\|>(.*?)<\|text_end\|>", re.DOTALL)
    text_matches = list(text_pattern.finditer(sample_text))

    # 匹配所有 video 位置
    video_pattern = re.compile(r"<\|video_start\|>(.*?)<\|video_end\|>", re.DOTALL)
    video_matches = list(video_pattern.finditer(sample_text))

    # 遍历 text 匹配项，并找到紧跟的 video 匹配项
    video_index = 0
    for text_match in text_matches:
        text = text_match.group(1).strip()

        # 寻找紧随其后的 video（start位置大于 text_end）
        text_end_pos = text_match.end()
        matched_video = None

        while video_index < len(video_matches):
            video_match = video_matches[video_index]
            if video_match.start() >= text_end_pos:
                matched_video = video_match
                video_index += 1
                break
            video_index += 1

        if matched_video:
            video_full = sample_text[matched_video.start():matched_video.end()].strip()
        else:
            video_full = ""  # 如果没有对应视频段，填空

        results.append({
            "text": text,
            "video_frames": video_full
        })

    return results

def extract_audio_string(sample_text):
    """
    提取 <|audio_start|> ... <|audio_end|> 段的完整字符串，包括边界 token。
    
    返回值：
        - str, 例如 "<|audio_start|><a_0_597><a_1_419>...<|audio_end|>"
        - 如果没有音频段，返回空字符串 ""
    """
    audio_pattern = re.compile(r"(<\|audio_start\|>.*?<\|audio_end\|>)", re.DOTALL)
    match = audio_pattern.search(sample_text)
    if match:
        return match.group(1).strip()
    return ""

def vid_seg2str(segments):
    """
    将一组视频片段segments按顺序合并成一个字符串。
    """
    return "".join([seg["video_frames"] for seg in segments if seg["video_frames"].strip()])

def pre_select_error_segments(text_list, err_list_size):
    """
    从 text_list 中的每个样本提取视频片段，并随机选择 err_list_size 个错误片段。

    输入：
        - text_list: List[str]，每项为原始数据中的一条文本
        - err_list_size: 需要选取的错误片段数量

    输出：
        - err_segments: List[str]，包含 err_list_size 个错误视频片段的字符串列表
    """
    
    all_segments = []  # 用于存储所有的错误片段

    # 遍历 text_list 中的每条数据
    for sample in text_list:
        # 获取该数据样本中的所有视频片段
        segments = extract_text_video_pairs_string_tokens(sample)
        
        # 提取每个样本中的所有视频片段
        for seg in segments:
            video_frames = seg["video_frames"].strip()
            
            # 判断片段是否为空，或者仅是 <|video_start|><|video_end|> 之类的无效片段
            if video_frames and video_frames != "<|video_start|><|video_end|>":
                all_segments.append(seg)

    # 随机打乱所有的视频片段
    random.shuffle(all_segments)

    if err_list_size == 'FULL':
        err_segments = all_segments
    else :
        # 选取前 err_list_size 个错误片段
        err_segments = all_segments[:err_list_size]
    
    return err_segments

def split_vid_tokens(segments, n, position):
    """
    根据给定的位置来分割视频片段并返回适当的输入输出。

    输入：
        - segments: List[Dict]，每项包含 "video_frames"
        - n: int, 取 n 段作为补充片段
        - position: str, 'front' 表示在前补充，'end' 表示在后补充，'mid' 表示在中间补充

    输出：
        - [input_str, all_str]：两个字符串，分别是 A 区段和完整的视频序列
    """
    
    # 获取所有的视频片段
    all_segments = segments

    # 处理边界情况：segments 不足 n 段，返回空
    if n <= 0 or n > len(segments):
        return ["", ""]

    if position == "front":
        # 前补充：提取前 n 个片段作为补充，剩余作为输入
        add_segments = all_segments[:n]
        input_segments = all_segments[n:]

        # 组合字符串
        input_str = vid_seg2str(input_segments)
        all_str = vid_seg2str(all_segments)

    elif position == "end":
        # 后补充：提取后 n 个片段作为补充，剩余作为输入
        add_segments = all_segments[-n:]
        input_segments = all_segments[:-n]

        # 组合字符串
        input_str = vid_seg2str(input_segments)
        all_str = vid_seg2str(all_segments)

    elif position == "mid":
        # 中间补充：随机抽取 n 个片段作为补充，剩余的部分作为输入
        if len(all_segments) <= (n + 2):
            return ["", ""]  # 边界情况：无法从中间提取

        # 排除第一个和最后一个片段，然后从中间部分随机抽取 n 个片段
        middle_segments = all_segments[1:-1]
        random.shuffle(middle_segments)
        add_segments = middle_segments[:n]

        # 剩余的部分为 input_segments
        input_segments = [seg for seg in all_segments if seg not in add_segments]

        # 组合字符串
        input_str = vid_seg2str(input_segments)
        all_str = vid_seg2str(all_segments)
        return [input_str, all_str]

    else:
        raise ValueError("position should be 'front', 'end', or 'mid'")

    return [input_str, all_str]

def expand_vid_tokens(segments, error_segments, n, position):
    """
    扩展视频 tokens，插入随机错误片段。

    输入：
        - segments: List[Dict]，当前样本的视频片段（List of dictionaries, each containing 'video_frames' key）
        - error_segments: List[str]，用于插入的错误片段（来自 pre_select_error_segments 函数）
        - n: int，插入错误片段的数量
        - position: str，插入位置，可以是 'front'、'mid' 或 'end'

    输出：
        - input_str: 插入错误片段后的字符串（扩展输入）
        - all_str: 原始片段的字符串
    """

    # 随机抽取 n 个错误片段
    del_segments = random.sample(error_segments, n)

    # 扩展的输入片段
    if position == "front":
        # 在前面添加错误片段
        input_segments = del_segments + segments
    elif position == "mid":
        # 随机选择一个非边缘位置（1 到 len(segments) - 2）插入错误片段
        if len(segments) > 2:
            mid_pos = random.randint(1, len(segments) - 2)  # 随机选择一个非边缘位置
            input_segments = segments[:mid_pos] + del_segments + segments[mid_pos:]
        else:
            # 如果片段数量小于等于2，无法进行中间插入，选择前或后插入
            input_segments = del_segments + segments
    elif position == "end":
        # 在后面添加错误片段
        input_segments = segments + del_segments
    else:
        raise ValueError("position should be 'front', 'mid', or 'end'")

    # 使用 vid_seg2str 函数将 segments 列表转为字符串
    input_str = vid_seg2str(input_segments)
    all_str = vid_seg2str(segments)  # 只包含原始片段

    return input_str, all_str

def read_file(jsonl_path, max_samples=None):
    """
    读取 jsonl 文件，返回每条数据中的 record["text"] 字段（通常为预训练内容）。
    
    输入：
        - jsonl_path: 文件路径
        - max_samples: 最多读取多少条记录(None 表示全量）
    输出：
        - List[str]，每项为 record["text"]
    """
    contents = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if max_samples is not None and idx >= max_samples:
                break
            record = json.loads(line)
            if "text" in record:
                contents.append(record["text"])

    return contents

### Data for Diffenrent Tasks
def text2vid(text_list):
    """
    构造 SFT 数据(任务1): 脚本 → 视频 tokens
    输入：
        - text_list: List[str]，每项是一条原始数据
    输出：
        - List[Dict]，每项是符合 Alpaca 格式的样本，包含 instruction / input / output 字段
    """
    from copy import deepcopy

    results = []

    for raw_text in text_list:
        segments = extract_text_video_pairs_string_tokens(raw_text)

        # 拼接文本段（每段之间加换行）
        input_text = "\n".join([seg["text"] for seg in segments if seg["text"].strip()])

        # 拼接所有 video_frames 段
        output_vtok = "".join([seg["video_frames"] for seg in segments if seg["video_frames"].strip()])

        # 构造 Alpaca 格式字典
        item = {
            "instruction": "下面是一个广告视频的脚本，请根据脚本内容，输出与之匹配的视频片段的 token 编号。",
            "input": input_text.strip(),
            "output": output_vtok.strip()
        }

        results.append(item)

    return results

def vid2text(text_list):
    """
    构造 SFT 数据(任务2):视频片段 → 广告台词
    输入：
        - text_list: List[str]，每项是一条原始数据
    输出：
        - List[Dict]，每项为符合 Alpaca 格式的样本
    """
    results = []

    for raw_text in text_list:
        segments = extract_text_video_pairs_string_tokens(raw_text)

        # 拼接所有 video_frames（段落拼成一段输入）
        input_vtok = "".join([seg["video_frames"] for seg in segments if seg["video_frames"].strip()])

        # 拼接所有文本段（换行连接）
        output_text = "\n".join([seg["text"] for seg in segments if seg["text"].strip()])

        item = {
            "instruction": "下面是一些广告视频的片段，请根据画面内容生成合适的广告台词。",
            "input": input_vtok.strip(),
            "output": output_text.strip()
        }

        results.append(item)

    return results

def vid2aud(text_list):
    """
    构造 SFT 数据(任务3): 视频片段 → 音乐 token
    输入：
        - text_list: List[str]，每项是一条原始样本
    输出：
        - List[Dict]，每项为符合 Alpaca 格式的样本
    """
    results = []

    for raw_text in text_list:
        segments = extract_text_video_pairs_string_tokens(raw_text)
        audio_string = extract_audio_string(raw_text)

        # 拼接所有 video_frames
        input_video = "".join([seg["video_frames"] for seg in segments if seg["video_frames"].strip()])

        # 如果没有音频或视频，跳过该样本
        if not input_video.strip() or not audio_string.strip():
            continue

        item = {
            "instruction": "下面是一些广告视频的片段，请根据画面内容输出合适的背景音乐。",
            "input": input_video.strip(),
            "output": audio_string.strip()
        }

        results.append(item)

    return results

def vid2vid_add(text_list):
    """
    构造补充视频 tokens 的 SFT 数据样本（前补充、中间补充、后补充）。

    输入：
        - text_list: List[str]，每项为原始 jsonl 文件中 record["text"]

    输出：
        - 打印每个样本的 input_str 和 all_str，返回符合 Alpaca 格式的字典
    """
    
    # 划分数据集：30% front_add, 30% mid_add, 40% end_add
    front_add = text_list[:len(text_list) * 3 // 10]
    mid_add = text_list[len(text_list) * 3 // 10 : len(text_list) * 6 // 10]
    end_add = text_list[len(text_list) * 6 // 10:]

    results = []  # 存储所有 SFT 数据样本的列表

    # 遍历 front_add 并调用 split_vid_tokens 函数
    for sample in front_add:
        segments = extract_text_video_pairs_string_tokens(sample)
        # 如果片段数量小于5，跳过此数据
        if len(segments) < 5:
            continue

        # 随机确定 n，限制 n 最大值为 sample 的长度的 1/5
        n = random.randint(1, min(5, len(segments) // 5))
        input_str, all_str = split_vid_tokens(segments, n, position="front")
        
        # 构造符合 Alpaca 格式的字典
        result = {
            "instruction": "下面是一些广告视频的片段，请根据画面内容输出合适补充的视频片段。",
            "input": input_str,
            "output": all_str
        }
        results.append(result)

    # 遍历 mid_add 并调用 split_vid_tokens 函数
    for sample in mid_add:
        segments = extract_text_video_pairs_string_tokens(sample)
        # 如果片段数量小于5，跳过此数据
        if len(segments) < 5:
            continue

        # 随机确定 n，限制 n 最大值为 sample 的长度的 1/5
        n = random.randint(1, min(5, len(segments) // 5))
        input_str, all_str = split_vid_tokens(segments, n, position="mid")
        
        # 构造符合 Alpaca 格式的字典
        result = {
            "instruction": "下面是一些广告视频的片段，请根据画面内容输出合适补充的视频片段。",
            "input": input_str,
            "output": all_str
        }
        results.append(result)

    # 遍历 end_add 并调用 split_vid_tokens 函数
    for sample in end_add:
        segments = extract_text_video_pairs_string_tokens(sample)
        # 如果片段数量小于5，跳过此数据
        if len(segments) < 5:
            continue

        # 随机确定 n，限制 n 最大值为 sample 的长度的 1/5
        n = random.randint(1, min(5, len(segments) // 5))
        input_str, all_str = split_vid_tokens(segments, n, position="end")
        
        # 构造符合 Alpaca 格式的字典
        result = {
            "instruction": "下面是一些广告视频的片段，请根据画面内容输出合适补充的视频片段。",
            "input": input_str,
            "output": all_str
        }
        results.append(result)

    # 返回构造的结果列表
    return results

def vid2vid_delete(text_list, error_segments):
    """
    构造“删除”视频 tokens 的 SFT 数据样本（前删除、中间删除、后删除）。

    输入：
        - text_list: List[str]，每项为原始 jsonl 文件中 record["text"]
        - error_segments: List[Dict]，用于插入的错误片段（来自 pre_select_error_segments 函数）

    输出：
        - 打印每个样本的 input_str 和 output_str
    """

    results = []  # 存储所有 SFT 数据样本的列表

    # 划分数据集：30% front_del, 30% mid_del, 40% end_del
    front_del = text_list[:len(text_list) * 3 // 10]
    mid_del = text_list[len(text_list) * 3 // 10 : len(text_list) * 6 // 10]
    end_del = text_list[len(text_list) * 6 // 10:]

    # 遍历 front_del 并调用 expand_vid_tokens 函数
    for sample in front_del:
        segments = extract_text_video_pairs_string_tokens(sample)
        # 如果片段数量小于5，跳过此数据
        if len(segments) < 5:
            continue

        # 随机确定 n，限制 n 最大值为 segment 的长度的 1/5
        n = random.randint(1, min(5, len(segments) // 5))
        # 调用 expand_vid_tokens 来扩展视频 tokens，错误片段位置在前
        input_str, all_str = expand_vid_tokens(segments, error_segments, n, position="front") #### BUG HERE !!!!!!!!

        # 构造符合 Alpaca 格式的字典
        result = {
            "instruction": "下面是一些广告视频的片段，请删除不相关的片段。",
            "input": input_str,
            "output": all_str
        }
        results.append(result)

    # 遍历 mid_del 并调用 expand_vid_tokens 函数
    for sample in mid_del:
        segments = extract_text_video_pairs_string_tokens(sample)
        # 如果片段数量小于5，跳过此数据
        if len(segments) < 5:
            continue

        # 随机确定 n，限制 n 最大值为 segment 的长度的 1/5
        n = random.randint(1, min(5, len(segments) // 5))
        
        # 调用 expand_vid_tokens 来扩展视频 tokens，错误片段位置在中间
        input_str, all_str = expand_vid_tokens(segments, error_segments, n, position="mid")

        # 构造符合 Alpaca 格式的字典
        result = {
            "instruction": "下面是一些广告视频的片段，请删除不相关的片段。",
            "input": input_str,
            "output": all_str
        }
        results.append(result)

    # 遍历 end_del 并调用 expand_vid_tokens 函数
    for sample in end_del:
        segments = extract_text_video_pairs_string_tokens(sample)
        # 如果片段数量小于5，跳过此数据
        if len(segments) < 5:
            continue

        # 随机确定 n，限制 n 最大值为 segment 的长度的 1/5
        n = random.randint(1, min(5, len(segments) // 5))
        
        # 调用 expand_vid_tokens 来扩展视频 tokens，错误片段位置在后
        input_str, all_str = expand_vid_tokens(segments, error_segments, n, position="end")

        # 构造符合 Alpaca 格式的字典
        result = {
            "instruction": "下面是一些广告视频的片段，请删除不相关的片段。",
            "input": input_str,
            "output": all_str
        }
        results.append(result)

    # 返回构造的结果列表
    return results

def vid2vid_sort(text_list):
    """
    输入乱序的广告视频片段，输出正确顺序的视频片段。

    输入：
        - text_list: List[str]，每项为原始数据中的一条文本
        - seed: 可选，设置随机种子以确保可复现性

    输出：
        - 返回符合 Alpaca 格式的字典样本
    """
    all_samples = []

    # 遍历 text_list 中的每一条数据样本
    for sample in text_list:
        # 获取每个样本的视频片段
        segments = extract_text_video_pairs_string_tokens(sample)
        
        # 保留原始顺序作为 output_segments
        output_segments = segments
        
        # 复制原始 segments 来创建 input_segments，确保不会修改原始数据
        input_segments = segments.copy()

        # 乱序 video frames，作为错误顺序（input_segment）
        random.shuffle(input_segments)

        # 组合 input_str 和 output_str
        input_str = vid_seg2str(input_segments)
        output_str = vid_seg2str(output_segments)

        # 构造符合 Alpaca 格式的字典
        result = {
            "instruction": "下面是一些乱序的广告视频的片段，请合理排序视频片段，构建广告。",
            "input": input_str,
            "output": output_str
        }
        all_samples.append(result)

    return all_samples


def generate_all_sft_data(text_list, error_segments, task_ratios, output_path="sft_data.jsonl"):
    """
    生成不同任务的指令监督微调数据，并保存到 jsonl 文件。

    输入：
        - text_list: 任务样本数据
        - error_segments: 错误片段
        - output_path: 输出文件路径，默认为 "sft_data.jsonl"
        - task_ratios: 字典，定义每个任务所占的比例，如 {"t2v": 0.2, "v2t": 0.2, ...}
    """
    if task_ratios is None:
        task_ratios = {
            "t2v": 0.2,  # 脚本 → 视频 tokens
            "v2t": 0.2,  # 视频 tokens → 脚本
            "v2a": 0.2,  # 视频 tokens → 音乐
            "v2v_add": 0.2,  # 视频 tokens → 视频 tokens 补充
            "v2v_del": 0.1,  # 视频 tokens → 视频 tokens 删除
            "v2v_sort": 0.1,  # 视频 tokens → 视频 tokens 排序
        }

    # 检查比例之和是否为 1
    if sum(task_ratios.values()) != 1:
        raise ValueError("The sum of task_ratios must be 1.")

    all_samples = []

    # 根据比例划分 text_list
    total_samples = len(text_list)
    task_samples = {
        task: int(total_samples * ratio)
        for task, ratio in task_ratios.items()
    }
    print("[Task Samples]:\n", task_samples)

    # 划分数据集
    random.shuffle(text_list)
    task_data = {
        task: text_list[sum(list(task_samples.values())[:i]):sum(list(task_samples.values())[:i+1])]
        for i, task in enumerate(task_samples.keys())
    }

    # 任务一：脚本 → 视频 tokens
    t2v_samples = text2vid(task_data["t2v"])
    all_samples.extend(t2v_samples)

    # 任务二：视频 tokens → 脚本
    v2t_samples = vid2text(task_data["v2t"])
    all_samples.extend(v2t_samples)

    # 任务三：视频 tokens → 音乐
    v2a_samples = vid2aud(task_data["v2a"])
    all_samples.extend(v2a_samples)

    # 任务四：视频 tokens → 视频 tokens 补充
    v2v_add_samples = vid2vid_add(task_data["v2v_add"])
    all_samples.extend(v2v_add_samples)

    # 任务五：视频 tokens → 视频 tokens 删除
    v2v_del_samples = vid2vid_delete(task_data["v2v_del"], error_segments)
    all_samples.extend(v2v_del_samples)

    # 任务六：视频 tokens → 视频 tokens 排序
    v2v_sort_samples = vid2vid_sort(task_data["v2v_sort"])
    all_samples.extend(v2v_sort_samples)

    # 假设 all_samples 是你已经准备好的数据列表
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_samples, f, ensure_ascii=False, indent=2)  # 将列表作为单个 JSON 数组写入文件

    print(f"Generated {len(all_samples)} SFT data samples, saved to {output_path}")



if __name__ == "__main__":
    ### 参数
    input_file = "train_0722.jsonl"
    max_samples = None # None是不限制样本数量
    random.seed(44)
    # 数据分配比例，不定义就会默认（0.2,0.2,0.2,0.2,0.1,0.1）
    task_ratios = {
            "t2v": 0.2,  # 脚本 → 视频 tokens
            "v2t": 0.2,  # 视频 tokens → 脚本
            "v2a": 0.2,  # 视频 tokens → 音乐
            "v2v_add": 0.2,  # 视频 tokens → 视频 tokens 补充
            "v2v_del": 0.1,  # 视频 tokens → 视频 tokens 删除
            "v2v_sort": 0.1,  # 视频 tokens → 视频 tokens 排序
        }



    data = read_file(input_file, max_samples)
    error_segments = pre_select_error_segments(data, 'FULL')
    generate_all_sft_data(data, error_segments, task_ratios, output_path="sft_data_0722.json")

    
    ### TO-DO
    ### - 修改不同风格指令instructions？
    ### - SFT 效果？
