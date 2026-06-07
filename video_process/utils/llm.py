from openai import OpenAI
import time
import concurrent
import traceback
from typing import Dict, List

MODEL_NAME = 'Qwen3-30B-A3B' 

# ip_mapping = {} #

url = ip_mapping[MODEL_NAME]
client = OpenAI(
    base_url=url,
    api_key='empty',
)

def call_one_req(messages=None, stream=False, print_process=False):
    try:
        start_time = time.time()
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "user", "content": "1+1=？ "},
            ] if messages is None else messages,
            temperature=0.6,
            stream=stream,
            max_tokens=4096
        )

        result = ''
        has_reasoning_content_flag=False
        if stream:
            for chunk in completion:
                # print(chunk)
                if len(chunk.choices)>0:
                    reasoning_content = chunk.choices[0].delta.reasoning_content if hasattr(chunk.choices[0].delta,"reasoning_content")  else ''
                    answer_content = chunk.choices[0].delta.content
                    # print(f"reasoning_content: {reasoning_content}, answer_content: {answer_content}") 
                    if result =='' and reasoning_content is not None and  reasoning_content != '':
                        result += '<think>\n'
                        print("<think>")
                        has_reasoning_content_flag = True
                    if '<think>' in result and '</think>' not in result and answer_content is not None and has_reasoning_content_flag:
                        print('\n</think>')
                        result += '\n</think>\n'

                    tmp = reasoning_content if reasoning_content!='' and reasoning_content is not None  else answer_content
                    if tmp is None:
                        tmp = ''
                    result += tmp
                    if print_process:
                        print(tmp, end='', flush=True)
        else:
            reasoning_content = completion.choices[0].message.reasoning_content if hasattr(completion.choices[0].message,"reasoning_content")  else ''
            result = completion.choices[0].message.content
            if reasoning_content != '' and '<think>' not in result:
                result = f'<think>\n{reasoning_content}\n</think>\n' + result
            if print_process:
                print(result)

        return result
    except:
        traceback.print_exc()
        print("error")
        return None


prompt = """
你是一个智能助手，帮我检查和修正广告视频提取的 ASR（Automatic Speech Recognition）内容中的识别错误和语法错误。

## 任务要求：
1. 修正 ASR 中的错别字、断句错误、口语词等常见语病。不要编造任何原文中不存在的内容，尽量忠实原始识别结果。
2. 判断ASR是否为歌词，如果确认识别到有歌词（一般是流行歌短视频歌曲当作背景音乐），请你删除歌词相关的部分。
3. 如果和其他部分语义不连贯，可能是背景杂音，也请删除。
4. 输出结果是 list，每个元素是一个包含修正文本的字典。

## 以下是原始 ASR 内容（包括 startTime、endTime、confidence、text）：
{formatted_asr}

## 请输出修正后的 ASR list，格式如下：
[
  {{
    "confidence": 0.98,
    "startTime": 0.0,
    "endTime": 2.5,
    "text": "修正后的语句"
  }},
  {{
    "confidence": 0.95,
    "startTime": 2.6,
    "endTime": 4.0,
    "text": "另一段修正后的语句"
  }}
]
"""

def process_one_asr(asr):
    """
    单个调用
    """
    content = prompt.format(formatted_asr=asr)
    messages = [{"role": "user", "content": content}]
    reply = call_one_req(messages=messages, stream=False, print_process=False)
    return reply


def batch_process_asr(
    id_content_dict: Dict[str, str],
    max_workers: int = 4
):
    """
    并发调用 call_one_req，输入 id -> content 的 dict，输出 id -> reply 的 dict。
    """
    result_dict = {}

    def wrapper(_id: str, _content: str) -> (str, str):
        content = prompt.format(formatted_asr=_content)
        messages = [{"role": "user", "content": content}]
        reply = call_one_req(messages=messages, stream=False, print_process=True)
        return _id, reply

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(wrapper, _id, _content): _id
            for _id, _content in id_content_dict.items()
        }

        for future in concurrent.futures.as_completed(futures):
            try:
                _id, reply = future.result()
                result_dict[_id] = reply
            except Exception as e:
                print(f"Error processing {_id}: {e}")
                result_dict[_id] = None  # 失败标记

    return result_dict