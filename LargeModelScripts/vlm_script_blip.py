import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
import os
import json
from PIL import Image
from tqdm import tqdm

ann_file = '/nas/data_2/wanhongz/datasets/DGM4/metadata/train.json'
root_dir = '/nas/data_2/wanhongz/datasets/'
output_file = 'caption_train.json'
model_name = 'Salesforce/blip-image-captioning-large'

device = torch.device('cuda:0')
processor = BlipProcessor.from_pretrained(model_name)
model = BlipForConditionalGeneration.from_pretrained(
    model_name, torch_dtype=torch.float16
).to(device)
model.eval()

ann_list = json.load(open(ann_file, 'r'))

# 断点续跑：已有结果就跳过
if os.path.exists(output_file):
    with open(output_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
    print(f'已有 {len(results)} 条，继续从断点跑')
else:
    results = {}

print(f'共 {len(ann_list)} 条数据')
for i, data in enumerate(tqdm(ann_list)):
    img_key = data['image']
    if img_key in results:  # 已处理过就跳过
        continue
    img_path = os.path.join(root_dir, img_key)
    try:
        image = Image.open(img_path).convert('RGB')
        inputs = processor(image, return_tensors='pt').to(device, torch.float16)
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=100)
        caption = processor.decode(output[0], skip_special_tokens=True)
        results[img_key] = caption
    except Exception as e:
        print(f'Error: {img_path}: {e}')
        results[img_key] = ''
    
    # 每1000条保存一次
    if (i + 1) % 1000 == 0:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=True)

# 最终保存
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=True)
print(f'完成，共 {len(results)} 条，结果保存到 {output_file}')