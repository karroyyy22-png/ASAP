import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import json
from tqdm import tqdm
import sys

gpu_id = int(sys.argv[1])
total_gpus = int(sys.argv[2])

model_path = '/nas/data_2/wanhongz/Qwen2.5-1.5B-Instruct'
json_file = '/nas/data_2/wanhongz/datasets/DGM4/metadata/train.json'
output_file = f'temp/llm_gpu{gpu_id}.json'
BATCH_SIZE = 8

os.makedirs('temp', exist_ok=True)

device = 'cuda:0'
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
tokenizer.padding_side = 'left'
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map=device,
    trust_remote_code=True
).eval()

with open(json_file, 'r') as f:
    ann_list = json.load(f)

seen = set()
unique_list = []
for item in ann_list:
    if item['image'] not in seen:
        seen.add(item['image'])
        unique_list.append(item)

chunk_size = len(unique_list) // total_gpus
start = gpu_id * chunk_size
end = start + chunk_size if gpu_id < total_gpus - 1 else len(unique_list)
my_list = unique_list[start:end]

if os.path.exists(output_file):
    with open(output_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
    print(f'GPU{gpu_id}: 已有 {len(results)} 条，继续从断点跑')
else:
    results = {}

my_list = [item for item in my_list if item['image'] not in results]
print(f'GPU{gpu_id}: 需处理 {len(my_list)} 条')

prompt_prefix = 'Refer to the following text to describe the specific information of the corresponding image: '

for i in tqdm(range(0, len(my_list), BATCH_SIZE)):
    batch = my_list[i:i+BATCH_SIZE]
    prompts = [f'{prompt_prefix}"{item["text"]}"' for item in batch]
    
    try:
        messages_batch = [[{"role": "user", "content": p}] for p in prompts]
        texts = [tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False) for m in messages_batch]
        inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=60, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        
        for j, item in enumerate(batch):
            input_len = inputs['input_ids'].shape[1]
            answer = tokenizer.decode(outputs[j][input_len:], skip_special_tokens=True)
            results[item['image']] = answer
    except Exception as e:
        print(f'Error: {e}')
        for item in batch:
            results[item['image']] = ''

    if (i // BATCH_SIZE + 1) % 100 == 0:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=True)

with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=True)
print(f'GPU{gpu_id} 完成，共 {len(results)} 条')