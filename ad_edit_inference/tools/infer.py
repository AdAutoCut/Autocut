from transformers import AutoModelForCausalLM, AutoTokenizer
from openai import OpenAI

def predict(prompt, model_name = "Qwen/Qwen3-8B"):
    # load the tokenizer and the model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # prepare the model input
    messages = [
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False, # Switches between thinking and non-thinking modes. Default is True.
    )
    # print(text)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    # conduct text completion
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=8096,
        temperature = 0.7,
        top_p = 0.8
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 
    content = tokenizer.decode(output_ids, skip_special_tokens=False).strip("\n")
    return content

def predict_online(prompt, t,  url="http://10.82.136.14:19136/v1"): #10.62.19.26 t5 / 10.62.18.220 0730 / 10.62.18.187 full0730
    # load the tokenizer and the model    
    client = OpenAI(api_key="0",base_url=url)
    messages = [{"role": "user", "content": prompt}]
    result = client.chat.completions.create(
        messages=messages,
        model="XXX",
        temperature=t,
        top_p=0.95,
        extra_body={
            "top_k": 20,
            "chat_template_kwargs": {"enable_thinking": False},
        }
    )
    return result.choices[0].message.content