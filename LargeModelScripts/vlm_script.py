
import os
import sys
sys.path.append('/nas/data_2/wanhongz/VisCPM/')
import json
from tqdm import tqdm
from VisCPM import VisCPMChat
from PIL import Image
import torch

# 命令行参数：gpu_id, total_gpus
gpu_id = int(sys.argv[1])
total_gpus = int(sys.argv[2])

model_path = '/nas/data_2/wanhongz/VisCPM/VisCPM/pytorch_model.bin'
split = 'train'
ann_file = f'/nas/data_2/wanhongz/datasets/DGM4/metadata/{split}.json'
root_dir = '/nas/data_2/wanhongz/datasets/'
output_file = f'/nas/data_2/wanhongz/ASAP/LargeModelScripts/temp/caption_viscpm_gpu{gpu_id}.json'

os.makedirs('/nas/data_2/wanhongz/ASAP/LargeModelScripts/temp', exist_ok=True)

# 加载模型
viscpm_chat = VisCPMChat(model_path, image_safety_checker=False)

# 加载数据并去重
ann_list = json.load(open(ann_file, 'r'))
seen = set()
unique_list = []
for item in ann_list:
    if item['image'] not in seen:
        seen.add(item['image'])
        unique_list.append(item)

# 按 GPU 分片
chunk_size = len(unique_list) // total_gpus
start = gpu_id * chunk_size
end = start + chunk_size if gpu_id < total_gpus - 1 else len(unique_list)
my_list = unique_list[start:end]

# 断点续跑
if os.path.exists(output_file):
    with open(output_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
    print(f'GPU{gpu_id}: 已有 {len(results)} 条，继续从断点跑')
else:
    results = {}

print(f'GPU{gpu_id}: 需处理 {len(my_list)} 条')

for i, data in enumerate(tqdm(my_list)):
    img_dir = data['image']
    if img_dir in results:
        continue
    try:
        image_dir_all = f'{root_dir}/{img_dir}'
        image = Image.open(image_dir_all).convert('RGB')
        prompt = 'Give the caption of this picture'
        answer, context, vis_hidden = viscpm_chat.chat(image, prompt)
        results[img_dir] = answer
    except Exception as e:
        print(f'Error: {e}')
        results[img_dir] = ''

    if (i + 1) % 500 == 0:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=True)

with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=True)
print(f'GPU{gpu_id} 完成，共 {len(results)} 条')