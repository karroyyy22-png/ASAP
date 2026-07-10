#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
sys.path.append('.')  # 确保可以导入项目模块

import torch
import numpy as np
import yaml
import argparse
from transformers import BertTokenizerFast
from dataset import create_dataset, create_loader
from models.ASAP import ASAP

def main():
    # ========== 配置参数（请根据实际情况修改） ==========
    CHECKPOINT_PATH = "/nas/data_2/wanhongz/ASAP/results/log20260602_134300_train_4gpu_batch16/checkpoint_best.pth"   # 改成您的改进后模型路径
    CONFIG_PATH = "configs/train.yaml"
    TEXT_ENCODER = "bert-base-uncased"
    DEVICE = "cuda"
    # ===================================================

    # 加载配置
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.load(f, Loader=yaml.Loader)

    # 创建验证集
    print("Creating dataset...")
    _, val_dataset = create_dataset(config)          # 根据 train.py，create_dataset 返回两个 dataset
    # 创建 DataLoader（注意返回值是列表）
    val_loaders = create_loader([val_dataset], [None],
                                batch_size=[config['batch_size_val']],
                                num_workers=[4],
                                is_trains=[False],
                                collate_fns=[None])
    val_loader = val_loaders[0]   # 取出单个 DataLoader

    # 初始化 tokenizer
    tokenizer = BertTokenizerFast.from_pretrained(TEXT_ENCODER)

    # 创建模型（需要构造 args 对象）
    args = argparse.Namespace()
    args.token_momentum = False   # 根据训练脚本，这里设为 False
    args.checkpoint = CHECKPOINT_PATH
    args.resume = False
    # 如果模型 __init__ 还需要其他 args 属性，请根据实际情况补充
    model = ASAP(
        args=args,
        config=config,
        text_encoder=TEXT_ENCODER,
        tokenizer=tokenizer,
        init_deit=True
    )
    # 加载权重
    print(f"Loading checkpoint from {CHECKPOINT_PATH}...")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu')
    model.load_state_dict(checkpoint['model'], strict=False)
    model = model.to(DEVICE)
    model.eval()

    s_vals = []
    labels = []

    print("Running evaluation on validation set...")
    with torch.no_grad():
        for i, (image, label, text, fake_image_box, fake_word_pos, W, H, real_cap, real_prom, res_fake_pos, res_fake_pos_patch) in enumerate(val_loader):
            image = image.to(DEVICE)
            # 获取真伪标签：label 列表中 'orig' 表示真实，其余为伪造
            cls_label = torch.ones(len(label), dtype=torch.long).to(DEVICE)
            real_label_pos = np.where(np.array(label) == 'orig')[0].tolist()
            cls_label[real_label_pos] = 0

            # 计算 s
            s = model.tamper_detector(image)   # [B]
            s_vals.extend(s.cpu().numpy())
            labels.extend(cls_label.cpu().numpy())

            if (i+1) % 50 == 0:
                print(f"Processed {i+1} batches")

    # 统计分析
    s_vals = np.array(s_vals)
    labels = np.array(labels)
    real_mask = (labels == 0)
    fake_mask = (labels == 1)

    print("\n========== Results ==========")
    print(f"Total samples: {len(s_vals)}")
    print(f"Real images: {np.sum(real_mask)}")
    print(f"Fake images: {np.sum(fake_mask)}")
    print(f"\nReal -> mean: {np.mean(s_vals[real_mask]):.6f}, std: {np.std(s_vals[real_mask]):.6f}")
    print(f"Fake -> mean: {np.mean(s_vals[fake_mask]):.6f}, std: {np.std(s_vals[fake_mask]):.6f}")
    print(f"Difference (Fake - Real): {np.mean(s_vals[fake_mask]) - np.mean(s_vals[real_mask]):.6f}")

if __name__ == "__main__":
    main()