import warnings
warnings.filterwarnings("ignore")

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import ruamel.yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from models.vit import interpolate_pos_embed
from transformers import BertTokenizerFast

import utils
from dataset import create_dataset, create_sampler, create_loader
from scheduler import create_scheduler
from optim import create_optimizer

import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
import logging
from types import MethodType
from tools.env import init_dist
from tqdm import tqdm

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d

from models import box_ops
from tools.multilabel_metrics import AveragePrecisionMeter, get_multi_label

from models.ASAP import ASAP

# ========== OmniFD 分数级融合相关配置 ==========
OMNI_SCORE_PATH = '/nas/data_2/wanhongz/face_cropped_features/score_full.npy'
OMNI_META_PATH = '/nas/data_2/wanhongz/face_cropped_features/meta_full.json'
OMNI_ALPHA = 0.5  # 融合权重：final = alpha*ASAP + (1-alpha)*OmniFD

def load_omni_lookup():
    """加载 OmniFD 的 path->score 查找表"""
    if not (os.path.exists(OMNI_SCORE_PATH) and os.path.exists(OMNI_META_PATH)):
        print(f"[WARNING] 未找到OmniFD分数文件，跳过融合，仅使用原始ASAP结果")
        return {}
    scores = np.load(OMNI_SCORE_PATH)
    meta = json.load(open(OMNI_META_PATH))
    paths = meta['paths']
    lookup = dict(zip(paths, scores))
    print(f"[OmniFD] 加载了 {len(lookup)} 条人脸伪造分数")
    return lookup


OMNI_FEAT_512_DIR = '/nas/data_2/wanhongz/ASAP'

def load_omni_feats_512(split):
    """加载指定split的OmniFD 512维特征查找表 (img_dir -> 512维tensor)，用于特征级融合"""
    path = f'{OMNI_FEAT_512_DIR}/omnifd_{split}_feats.pt'
    feats = torch.load(path, map_location='cpu')
    print(f'[Omni Fusion] loaded {len(feats)} feature entries from {path}')
    return feats

def build_omni_batch(img_dirs, omni_feats, device):
    """按img_dir列表查表拼成batch tensor，查不到的用全零占位"""
    vecs = []
    for d in img_dirs:
        if d in omni_feats:
            vecs.append(omni_feats[d])
        else:
            vecs.append(torch.zeros(512, dtype=torch.float32))
    return torch.stack(vecs, dim=0).to(device, non_blocking=True)

OMNI_FEAT_LOCAL_DIR = '/nas/data_2/wanhongz/ASAP'
def load_omni_feats_local(split):
    """加载指定split的OmniFD局部token特征查找表 (img_dir -> [64,512] tensor)"""
    feat_path = f'{OMNI_FEAT_LOCAL_DIR}/omnifd_{split}_feats_local.pt'
    hasface_path = f'{OMNI_FEAT_LOCAL_DIR}/omnifd_{split}_hasface_local.pt'
    feats = torch.load(feat_path, map_location='cpu')
    hasface = torch.load(hasface_path, map_location='cpu')
    print(f'[Omni Fusion Local] loaded {len(feats)} local-token entries from {feat_path}')
    print(f'[Omni Fusion Local] loaded {len(hasface)} hasface flags from {hasface_path}')
    return feats, hasface

def build_omni_batch_local(img_dirs, omni_feats, hasface_flags, device):
    """按img_dir列表查表拼成batch tensor
    返回: tokens [B,64,512], valid_mask [B,64] (bool, True=有效可参与attention)
    """
    vecs, masks = [], []
    for d in img_dirs:
        if d in omni_feats:
            vecs.append(omni_feats[d])
            is_valid = bool(hasface_flags.get(d, False))
            masks.append(torch.full((64,), is_valid, dtype=torch.bool))
        else:
            vecs.append(torch.zeros(64, 512, dtype=torch.float32))
            masks.append(torch.zeros(64, dtype=torch.bool))
    tokens = torch.stack(vecs, dim=0).to(device, non_blocking=True)
    valid_mask = torch.stack(masks, dim=0).to(device, non_blocking=True)
    return tokens, valid_mask

def is_face_related(label):
    """判断该样本标签是否属于人脸相关类别（含混合类别），只对这些类别做融合"""
    return ('face_swap' in label) or ('face_attribute' in label)


def setlogger(log_file):
    filehandler = logging.FileHandler(log_file)
    streamhandler = logging.StreamHandler()

    logger = logging.getLogger('')
    logger.setLevel(logging.INFO)
    logger.addHandler(filehandler)
    logger.addHandler(streamhandler)

    def epochInfo(self, set, idx, loss, acc):
        self.info('{set}-{idx:d} epoch | loss:{loss:.8f} | auc:{acc:.4f}%'.format(
            set=set,
            idx=idx,
            loss=loss,
            acc=acc
        ))

    logger.epochInfo = MethodType(epochInfo, logger)

    return logger


def text_input_adjust(text_input, fake_word_pos, device, cap=False):
    # input_ids adaptation
    input_ids_remove_SEP = [x[:-1] for x in text_input.input_ids]
    maxlen = max([len(x) for x in text_input.input_ids])-1
    input_ids_remove_SEP_pad = [x + [0] * (maxlen - len(x)) for x in input_ids_remove_SEP] # only remove SEP as HAMMER is conducted with text with CLS
    text_input.input_ids = torch.LongTensor(input_ids_remove_SEP_pad).to(device) 

    # attention_mask adaptation
    attention_mask_remove_SEP = [x[:-1] for x in text_input.attention_mask]
    attention_mask_remove_SEP_pad = [x + [0] * (maxlen - len(x)) for x in attention_mask_remove_SEP]
    text_input.attention_mask = torch.LongTensor(attention_mask_remove_SEP_pad).to(device)

    if cap:
        return text_input

    # fake_token_pos adaptation
    fake_token_pos_batch = []
    subword_idx_rm_CLSSEP_batch = []
    for i in range(len(fake_word_pos)):
        fake_token_pos = []

        fake_word_pos_decimal = np.where(fake_word_pos[i].numpy() == 1)[0].tolist() # transfer fake_word_pos into numbers

        subword_idx = text_input.word_ids(i)
        subword_idx_rm_CLSSEP = subword_idx[1:-1]
        subword_idx_rm_CLSSEP_array = np.array(subword_idx_rm_CLSSEP) # get the sub-word position (token position)
        
        subword_idx_rm_CLSSEP_batch.append(subword_idx_rm_CLSSEP_array)
        
        # transfer the fake word position into fake token position
        for i in fake_word_pos_decimal: 
            fake_token_pos.extend(np.where(subword_idx_rm_CLSSEP_array == i)[0].tolist())
        fake_token_pos_batch.append(fake_token_pos)

    return text_input, fake_token_pos_batch, subword_idx_rm_CLSSEP_batch


def compute_per_category(label_all, y_true_np, y_pred_np, pred_acc_np, IOU_pred_all, categories):
    """给定 y_pred / pred_acc，计算每个类别的 ACC/F1/IoU（AUC按原逻辑保留，不做修复，仅供参考）"""
    per_cat_results = {}
    for cat in categories:
        idx = np.where(label_all == cat)[0]
        if len(idx) == 0:
            continue
        y_t = y_true_np[idx]
        y_p = y_pred_np[idx]
        p_acc = pred_acc_np[idx]
        ACC = np.mean(p_acc == y_t)
        if len(np.unique(y_t)) >= 2:
            AUC = roc_auc_score(y_t, y_p)
        else:
            AUC = float('nan')
        TP = np.sum((y_t == 1) & (p_acc == 1))
        FP = np.sum((y_t == 0) & (p_acc == 1))
        FN = np.sum((y_t == 1) & (p_acc == 0))
        P = TP / (TP + FP) if (TP + FP) > 0 else 0
        R = TP / (TP + FN) if (TP + FN) > 0 else 0
        F1 = 2 * P * R / (P + R) if (P + R) > 0 else 0

        IOU_cat = IOU_pred_all[idx]
        IOU_mean = np.mean(IOU_cat)
        IOU_50 = np.mean(IOU_cat > 0.5)
        IOU_75 = np.mean(IOU_cat > 0.75)

        per_cat_results[cat] = {
            'N': len(idx),
            'AUC': AUC,
            'ACC': ACC,
            'F1': F1,
            'IoU': IOU_mean,
            'IoU@50': IOU_50,
            'IoU@75': IOU_75
        }
    return per_cat_results


@torch.no_grad()
def evaluation(args, model, data_loader, tokenizer, device, config, omni_lookup, omni_test_feats_512, omni_test_feats_local=None, omni_test_hasface_local=None):
    # test
    model.eval() 
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Evaluation:'    
    
    print('Computing features for evaluation...')
    start_time = time.time()   
    print_freq = 200 

    y_true, y_pred = [], []
    IOU_pred_all = []
    label_all = []
    pred_acc_all = []
    img_path_all = []

    cls_nums_all = 0
    cls_acc_all = 0   

    multi_label_meter = AveragePrecisionMeter(difficult_examples=False)
    multi_label_meter.reset()

    for i, (image, label, text, fake_image_box, fake_word_pos, W, H, real_cap, real_prom, res_fake_pos, res_fake_pos_patch, img_path) in enumerate(metric_logger.log_every(args, data_loader, print_freq, header)):
        image = image.to(device, non_blocking=True) 
        
        text_input = tokenizer(text, max_length=128, truncation=True, add_special_tokens=True, return_attention_mask=True, return_token_type_ids=False) 
        cap_input = tokenizer(real_cap, max_length=128, truncation=True, add_special_tokens=True, return_attention_mask=True, return_token_type_ids=False) 
        prom_input = tokenizer(real_prom, max_length=128, truncation=True, add_special_tokens=True, return_attention_mask=True, return_token_type_ids=False) 

        text_input, fake_token_pos, _ = text_input_adjust(text_input, fake_word_pos, device)
        cap_input = text_input_adjust(cap_input, fake_word_pos, device, True)
        prom_input = text_input_adjust(prom_input, fake_word_pos, device, True)

        omni_valid_mask_batch = None
        if args.use_omni_fusion_local:
            omni_feat_batch, omni_valid_mask_batch = build_omni_batch_local(img_path, omni_test_feats_local, omni_test_hasface_local, device)
        elif args.use_omni_fusion:
            omni_feat_batch = build_omni_batch(img_path, omni_test_feats_512, device)
        else:
            omni_feat_batch = None
        logits_real_fake, logits_multicls, output_coord, logits_tok = model(image, label, text_input, fake_image_box, fake_token_pos, cap_input, prom_input, res_fake_pos, res_fake_pos_patch, is_train=False, omni_feat=omni_feat_batch, omni_valid_mask=omni_valid_mask_batch)

        ##================= real/fake cls ========================## 
        cls_label = torch.ones(len(label), dtype=torch.long).to(image.device) 
        real_label_pos = np.where(np.array(label) == 'orig')[0].tolist()
        cls_label[real_label_pos] = 0
        
        prob_fake = F.softmax(logits_real_fake, dim=1)[:, 1]
        y_pred.extend(prob_fake.cpu().flatten().tolist())
        y_true.extend(cls_label.cpu().flatten().tolist())

        pred_acc = logits_real_fake.argmax(1)
        cls_nums_all += cls_label.shape[0]
        cls_acc_all += torch.sum(pred_acc == cls_label).item()
        
        label_all.extend(label)
        pred_acc_all.extend(pred_acc.cpu().tolist())
        img_path_all.extend(img_path)
        
        # ----- multi metrics -----
        target, _ = get_multi_label(label, image)
        multi_label_meter.add(logits_multicls, target)
        
        ##================= bbox cls ========================## 
        boxes1 = box_ops.box_cxcywh_to_xyxy(output_coord)
        boxes2 = box_ops.box_cxcywh_to_xyxy(fake_image_box)

        IOU, _ = box_ops.box_iou(boxes1, boxes2.to(device), test=True)
        IOU_pred_all.extend(IOU.cpu().tolist())

        ##================= token cls ========================##  
        token_label = text_input.attention_mask[:,1:].clone()
        token_label[token_label==0] = -100
        token_label[token_label==1] = 0

        for batch_idx in range(len(fake_token_pos)):
            fake_pos_sample = fake_token_pos[batch_idx]
            if fake_pos_sample:
                for pos in fake_pos_sample:
                    token_label[batch_idx, pos] = 1

        logits_tok_reshape = logits_tok.view(-1, 2)
        logits_tok_pred = logits_tok_reshape.argmax(1)
        token_label_reshape = token_label.view(-1)

        if i == 0:
            TP_all = torch.sum((token_label_reshape == 1) * (logits_tok_pred == 1)).item()
            TN_all = torch.sum((token_label_reshape == 0) * (logits_tok_pred == 0)).item()
            FP_all = torch.sum((token_label_reshape == 0) * (logits_tok_pred == 1)).item()
            FN_all = torch.sum((token_label_reshape == 1) * (logits_tok_pred == 0)).item()
        else:
            TP_all += torch.sum((token_label_reshape == 1) * (logits_tok_pred == 1)).item()
            TN_all += torch.sum((token_label_reshape == 0) * (logits_tok_pred == 0)).item()
            FP_all += torch.sum((token_label_reshape == 0) * (logits_tok_pred == 1)).item()
            FN_all += torch.sum((token_label_reshape == 1) * (logits_tok_pred == 0)).item()

    ##================= 转成numpy，准备融合 ========================##
    y_true_np = np.array(y_true)
    y_pred_np = np.array(y_pred)
    pred_acc_np = np.array(pred_acc_all)
    label_all_np = np.array(label_all)
    IOU_pred_all_np = np.array(IOU_pred_all)

    ##================= 融合前指标（baseline） ========================##
    AUC_cls_orig = roc_auc_score(y_true_np, y_pred_np)
    ACC_cls_orig = cls_acc_all / cls_nums_all
    fpr, tpr, _ = roc_curve(y_true_np, y_pred_np, pos_label=1)
    EER_cls_orig = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)

    ##================= OmniFD 融合 ========================##
    n = len(img_path_all)
    omni_scores = np.zeros(n)
    omni_valid = np.zeros(n, dtype=bool)

    for idx in range(n):
        path = img_path_all[idx]
        lbl = label_all[idx]
        if path in omni_lookup and is_face_related(lbl):
            omni_scores[idx] = omni_lookup[path]
            omni_valid[idx] = True

    n_fused = omni_valid.sum()
    print(f"[OmniFD融合] 共 {n} 张测试图，其中 {n_fused} 张(人脸相关类别且查到分数)参与融合")

    # 保存原始逐样本数据，供后续诊断分析和alpha搜索使用（避免重复跑模型推理）
    np.savez(os.path.join(args.output_dir, args.log_num, 'evaluation', 'raw_eval_data.npz'),
             y_true=y_true_np, y_pred_asap=y_pred_np, omni_scores=omni_scores,
             omni_valid=omni_valid, label_all=label_all_np,
             img_path_all=np.array(img_path_all, dtype=object))
    print("原始逐样本数据已保存")

    # ========== 诊断：ASAP判断错误时，OmniFD的判断是否正确 ==========
    asap_pred_binary = (y_pred_np >= 0.5).astype(np.int64)
    asap_wrong = (asap_pred_binary != y_true_np) & omni_valid
    n_asap_wrong = asap_wrong.sum()
    if n_asap_wrong > 0:
        omni_pred_binary = (omni_scores >= 0.5).astype(np.int64)
        omni_correct_where_asap_wrong = (omni_pred_binary == y_true_np)[asap_wrong]
        ratio = omni_correct_where_asap_wrong.mean()
        print(f"\n[诊断] ASAP判断错误且有OmniFD分数的样本数: {n_asap_wrong}")
        print(f"[诊断] 其中OmniFD判断正确的比例: {ratio:.4f}")
    else:
        print("\n[诊断] 没有找到ASAP判断错误且有OmniFD分数的样本")

    y_pred_fused = y_pred_np.copy()
    y_pred_fused[omni_valid] = OMNI_ALPHA * y_pred_np[omni_valid] + (1 - OMNI_ALPHA) * omni_scores[omni_valid]

    pred_acc_fused = (y_pred_fused >= 0.5).astype(np.int64)

    ##================= 融合后指标 ========================##
    AUC_cls_fused = roc_auc_score(y_true_np, y_pred_fused)
    ACC_cls_fused = np.mean(pred_acc_fused == y_true_np)
    fpr_f, tpr_f, _ = roc_curve(y_true_np, y_pred_fused, pos_label=1)
    EER_cls_fused = brentq(lambda x: 1. - x - interp1d(fpr_f, tpr_f)(x), 0., 1.)

    ##================= multi-label cls (不受OmniFD融合影响，按原逻辑) ========================## 
    MAP = multi_label_meter.value().mean()
    OP, OR, OF1, CP, CR, CF1 = multi_label_meter.overall()
    OP_k, OR_k, OF1_k, CP_k, CR_k, CF1_k = multi_label_meter.overall_topk(3)
    
    ##================= bbox cls (不受OmniFD融合影响) ========================##
    IOU_score = np.mean(IOU_pred_all_np)
    IOU_ACC_50 = np.mean(IOU_pred_all_np > 0.5)
    IOU_ACC_75 = np.mean(IOU_pred_all_np > 0.75)
    IOU_ACC_95 = np.mean(IOU_pred_all_np > 0.95)

    ##================= token cls ========================##
    ACC_tok = (TP_all + TN_all) / (TP_all + TN_all + FP_all + FN_all)
    Precision_tok = TP_all / (TP_all + FP_all) if (TP_all + FP_all) > 0 else 0
    Recall_tok = TP_all / (TP_all + FN_all) if (TP_all + FN_all) > 0 else 0
    F1_tok = 2 * Precision_tok * Recall_tok / (Precision_tok + Recall_tok) if (Precision_tok + Recall_tok) > 0 else 0

    # ================== 按操纵类别统计（融合前 vs 融合后）==================
    save_dir = os.path.join(args.output_dir, args.log_num, 'evaluation')
    os.makedirs(save_dir, exist_ok=True)

    categories = ['orig', 'face_swap', 'face_attribute', 'text_swap', 'text_attribute',
                  'face_swap&text_swap', 'face_swap&text_attribute',
                  'face_attribute&text_swap', 'face_attribute&text_attribute']

    per_cat_results_orig = compute_per_category(label_all_np, y_true_np, y_pred_np, pred_acc_np, IOU_pred_all_np, categories)
    per_cat_results_fused = compute_per_category(label_all_np, y_true_np, y_pred_fused, pred_acc_fused, IOU_pred_all_np, categories)

    # ========== 新增：打印/保存"融合前"（即模型直接输出，已包含omni token序列融合效果，
    #            但未经过外层OmniFD分数级平均）的真实结果，避免被下面的分数级融合掩盖 ==========
    print("\n===== Per-Category Results (融合前 / 模型直接输出, Orig) =====")
    print("(注: 若使用 --use_omni_fusion_local，这里的结果已包含token序列级融合效果；")
    print(" 若未使用任何omni开关，这里就是纯ASAP基线)")
    for cat, res in per_cat_results_orig.items():
        print(f"\n[{cat}] N={res['N']}")
        print(f"  AUC={res['AUC']:.4f}, ACC={res['ACC']:.4f}, F1={res['F1']:.4f}")
        print(f"  IoU={res['IoU']:.4f}, IoU@50={res['IoU@50']:.4f}, IoU@75={res['IoU@75']:.4f}")

    with open(os.path.join(save_dir, 'results_per_category_orig.json'), 'w') as f:
        json.dump(per_cat_results_orig, f, indent=2)
    print(f"\nPer-category results (orig, 未经分数级融合) saved to {save_dir}/results_per_category_orig.json")

    print("\n===== 整体指标 (融合前 / 模型直接输出, Orig) =====")
    print(f"AUC_cls_orig = {AUC_cls_orig*100:.4f}")
    print(f"ACC_cls_orig = {ACC_cls_orig*100:.4f}")
    print(f"EER_cls_orig = {EER_cls_orig*100:.4f}")

    print("\n===== Per-Category Results (融合后, Fused) =====")
    for cat, res in per_cat_results_fused.items():
        print(f"\n[{cat}] N={res['N']}")
        print(f"  AUC={res['AUC']:.4f}, ACC={res['ACC']:.4f}, F1={res['F1']:.4f}")
        print(f"  IoU={res['IoU']:.4f}, IoU@50={res['IoU@50']:.4f}, IoU@75={res['IoU@75']:.4f}")

    # results_per_category.json 保存融合后的最终结果（这是主要交付物）
    with open(os.path.join(save_dir, 'results_per_category.json'), 'w') as f:
        json.dump(per_cat_results_fused, f, indent=2)
    print(f"\nPer-category results (fused) saved to {save_dir}/results_per_category.json")

    # 计算 F1_multicls（复用per_cat_results的F1，与原代码逻辑一致）
    cat_names = ['face_swap', 'face_attribute', 'text_swap', 'text_attribute']

    F1_multicls_orig = np.zeros(4)
    F1_multicls_fused = np.zeros(4)
    for cls_idx, cat in enumerate(cat_names):
        if cat in per_cat_results_orig:
            F1_multicls_orig[cls_idx] = per_cat_results_orig[cat]['F1']
        if cat in per_cat_results_fused:
            F1_multicls_fused[cls_idx] = per_cat_results_fused[cat]['F1']

    # ================== 写 results_all.txt：融合前 vs 融合后 对比 ==================
    lines = []
    lines.append("="*70)
    lines.append("OmniFD 融合前后指标对比 (alpha=%.2f)" % OMNI_ALPHA)
    lines.append(f"参与融合样本数: {n_fused} / {n} (仅face_swap/face_attribute及其混合类别)")
    lines.append("="*70)

    def fmt_line(name, before, after):
        delta = after - before
        sign = '+' if delta >= 0 else ''
        return f"{name:20s}  Before={before*100:8.4f}  After={after*100:8.4f}  Delta={sign}{delta*100:.4f}"

    lines.append(fmt_line("AUC_cls", AUC_cls_orig, AUC_cls_fused))
    lines.append(fmt_line("ACC_cls", ACC_cls_orig, ACC_cls_fused))
    lines.append(fmt_line("EER_cls", EER_cls_orig, EER_cls_fused))
    lines.append(fmt_line("F1_FS", F1_multicls_orig[0], F1_multicls_fused[0]))
    lines.append(fmt_line("F1_FA", F1_multicls_orig[1], F1_multicls_fused[1]))
    lines.append(fmt_line("F1_TS", F1_multicls_orig[2], F1_multicls_fused[2]))
    lines.append(fmt_line("F1_TA", F1_multicls_orig[3], F1_multicls_fused[3]))
    lines.append("-"*70)
    lines.append("以下指标不受OmniFD融合影响（记录作为参照）：")
    lines.append(f"IOU_score            = {IOU_score*100:.4f}")
    lines.append(f"IOU_ACC_50            = {IOU_ACC_50*100:.4f}")
    lines.append(f"IOU_ACC_75            = {IOU_ACC_75*100:.4f}")
    lines.append(f"IOU_ACC_95            = {IOU_ACC_95*100:.4f}")
    lines.append(f"MAP                   = {MAP.item()*100:.4f}")
    lines.append(f"OF1 / CF1             = {OF1*100:.4f} / {CF1*100:.4f}")
    lines.append(f"ACC_tok / F1_tok      = {ACC_tok*100:.4f} / {F1_tok*100:.4f}")
    lines.append("="*70)

    result_all_path = os.path.join(save_dir, 'results_all.txt')
    with open(result_all_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"\n融合前后对比已保存到 {result_all_path}")
    print('\n'.join(lines))

    return AUC_cls_fused, ACC_cls_fused, EER_cls_fused, \
           MAP.item(), OP, OR, OF1, CP, CR, CF1, OP_k, OR_k, OF1_k, CP_k, CR_k, CF1_k, F1_multicls_fused, \
           IOU_score, IOU_ACC_50, IOU_ACC_75, IOU_ACC_95, \
           ACC_tok, Precision_tok, Recall_tok, F1_tok
    
def main_worker(gpu, args, config):

    if gpu is not None:
        args.gpu = gpu

    init_dist(args)

    eval_type = os.path.basename(config['val_file'][0]).split('.')[0]
    if eval_type == 'test':
        eval_type = 'all'
    log_dir = os.path.join(args.output_dir, args.log_num, 'evaluation')
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'shell_{eval_type}.txt')
    logger = setlogger(log_file)
    
    if args.log:
        logger.info('******************************')
        logger.info(args)
        logger.info('******************************')
        logger.info(config)
        logger.info('******************************')

    
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    # 加载OmniFD分数查找表
    omni_lookup = load_omni_lookup()
    omni_test_feats_512 = load_omni_feats_512('test')
    omni_test_feats_local, omni_test_hasface_local = load_omni_feats_local('test')

    #### Model #### 
    tokenizer = BertTokenizerFast.from_pretrained('/nas/data_2/wanhongz/bert-base-uncased')
    if args.log:
        print(f"Creating Model")
    model = ASAP(args=args, config=config, text_encoder='/nas/data_2/wanhongz/bert-base-uncased', tokenizer=tokenizer, init_deit=True)
    
    model = model.to(device)   

    checkpoint_dir = f'{args.output_dir}/{args.log_num}/checkpoint_{args.test_epoch}.pth'
    print(checkpoint_dir)
    checkpoint = torch.load(checkpoint_dir, map_location='cpu') 
    state_dict = checkpoint['model']                       

    pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'],model.visual_encoder)   
    state_dict['visual_encoder.pos_embed'] = pos_embed_reshaped       
                   
    if args.log:
        print('load checkpoint from %s'%checkpoint_dir)  
    msg = model.load_state_dict(state_dict, strict=False)
    if args.log:
        print(msg)  

    #### Dataset #### 
    if args.log:
        print("Creating dataset")
    _, val_dataset = create_dataset(config)
    
    if args.distributed:  
        samplers = create_sampler([val_dataset], [True], args.world_size, args.rank) + [None]    
    else:
        samplers = [None]

    val_loader = create_loader([val_dataset],
                                samplers,
                                batch_size=[config['batch_size_val']], 
                                num_workers=[4], 
                                is_trains=[False], 
                                collate_fns=[None])[0]

    
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    if args.log:
        print("Start evaluation")

    AUC_cls, ACC_cls, EER_cls, \
    MAP, OP, OR, OF1, CP, CR, CF1, OP_k, OR_k, OF1_k, CP_k, CR_k, CF1_k, F1_multicls, \
    IOU_score, IOU_ACC_50, IOU_ACC_75, IOU_ACC_95, \
    ACC_tok, Precision_tok, Recall_tok, F1_tok  = evaluation(args, model_without_ddp, val_loader, tokenizer, device, config, omni_lookup, omni_test_feats_512, omni_test_feats_local, omni_test_hasface_local)
    #============ evaluation info ============#
    val_stats = {"AUC_cls": "{:.4f}".format(AUC_cls*100),
                    "ACC_cls": "{:.4f}".format(ACC_cls*100),
                    "EER_cls": "{:.4f}".format(EER_cls*100),
                    "MAP": "{:.4f}".format(MAP*100),
                    "OP": "{:.4f}".format(OP*100),
                    "OR": "{:.4f}".format(OR*100),
                    "OF1": "{:.4f}".format(OF1*100),
                    "CP": "{:.4f}".format(CP*100),
                    "CR": "{:.4f}".format(CR*100),
                    "CF1": "{:.4f}".format(CF1*100),
                    "F1_FS": "{:.4f}".format(F1_multicls[0]*100),
                    "F1_FA": "{:.4f}".format(F1_multicls[1]*100),
                    "F1_TS": "{:.4f}".format(F1_multicls[2]*100),
                    "F1_TA": "{:.4f}".format(F1_multicls[3]*100),
                    "IOU_score": "{:.4f}".format(IOU_score*100),
                    "IOU_ACC_50": "{:.4f}".format(IOU_ACC_50*100),
                    "IOU_ACC_75": "{:.4f}".format(IOU_ACC_75*100),
                    "IOU_ACC_95": "{:.4f}".format(IOU_ACC_95*100),
                    "ACC_tok": "{:.4f}".format(ACC_tok*100),
                    "Precision_tok": "{:.4f}".format(Precision_tok*100),
                    "Recall_tok": "{:.4f}".format(Recall_tok*100),
                    "F1_tok": "{:.4f}".format(F1_tok*100),
    }
    if utils.is_main_process() or True: 
        log_stats = {**{f'val_{k}': v for k, v in val_stats.items()},
                        'epoch': args.test_epoch,
                    }    
        print('save result to:', os.path.join(log_dir, f"results_{eval_type}.txt"))         
        with open(os.path.join(log_dir, f"results_{eval_type}.txt"),"a") as f:
            f.write(json.dumps(log_stats) + "\n")

 
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/Pretrain.yaml')
    parser.add_argument('--checkpoint', default='')
    parser.add_argument('--use_omni_fusion', action='store_true', help='是否启用OmniFD特征融合，不加此参数则跑纯净ASAP基线')
    parser.add_argument('--use_omni_fusion_local', action='store_true', help='是否启用OmniFD局部token序列融合（新方案，与--use_omni_fusion互斥，优先级更高）') 
    parser.add_argument('--resume', default=False, type=bool)
    parser.add_argument('--output_dir', default='/mnt/lustre/share/rshao/data/FakeNews/Ours/results')
    parser.add_argument('--text_encoder', default='bert-base-uncased')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=777, type=int)
    parser.add_argument('--distributed', default=False, type=bool)
    parser.add_argument('--rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--world_size', default=1, type=int,
                        help='world size for distributed training')
    parser.add_argument('--dist-url', default='tcp://127.0.0.1:23451', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], default='none',
                        help='job launcher')
    parser.add_argument('--log_num', '-l', type=str)
    parser.add_argument('--model_save_epoch', type=int, default=5)
    parser.add_argument('--token_momentum', default=False, action='store_true')
    parser.add_argument('--test_epoch', default='best', type=str)


    parser.add_argument('--max_violation', action='store_true',
                        help='Use max instead of sum in the rank loss.')
    parser.add_argument('--margin', default=0.1, type=float,
                        help='Rank loss margin.')
    parser.add_argument('--raw_feature_norm', default="softmax",
                        help='clipped_l2norm|l2norm|clipped_l1norm|l1norm|no_norm|softmax')
    parser.add_argument('--agg_func', default="Mean",
                        help='LogSumExp|Mean|Max|Sum')
    parser.add_argument('--cross_attn', default="i2t",
                        help='t2i|i2t')
    parser.add_argument('--precomp_enc_type', default="basic",
                        help='basic|weight_norm')
    parser.add_argument('--bi_gru', action='store_true',
                        help='Use bidirectional GRU.')
    parser.add_argument('--lambda_lse', default=6., type=float,
                        help='LogSumExp temp.')
    parser.add_argument('--lambda_softmax', default=9., type=float,
                        help='Attention softmax temperature.')
                        
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
 
    main_worker(0, args, config)
