import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import os
import json
from PIL import Image
from tqdm import tqdm

# 配置
ann_file = '/nas/data_2/wanhongz/datasets/DGM4/metadata/train.json'
root_dir = '/nas/data_2/wanhongz/datasets/'
output_file = 'caption_train.json'
model_name = 'Salesforce/blip2-opt-2.7b'

# 加载模型
device = torch.device('cuda:0')
processor = Blip2Processor.from_pretrained(model_name)
model = Blip2ForConditionalGeneration.from_pretrained(
    model_name, torch_dtype=torch.float16
).to(device)
model.eval()

# 加载数据
ann_list = json.load(open(ann_file, 'r'))
results = {}

print(f'共 {len(ann_list)} 条数据')
for data in tqdm(ann_list):
    img_path = os.path.join(root_dir, data['image'])
    try:
        image = Image.open(img_path).convert('RGB')
        inputs = processor(images=image, text='Describe this image:', return_tensors='pt').to(device, torch.float16)
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=100)
        caption = processor.decode(output[0], skip_special_tokens=True)
        results[data['image']] = caption
    except Exception as e:
        print(f'Error: {img_path}: {e}')
        results[data['image']] = ''

json.dump(results, open(output_file, 'w'), ensure_ascii=False)
print(f'完成，结果保存到 {output_file}')
