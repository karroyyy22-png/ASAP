import warnings
warnings.filterwarnings("ignore")

import os
# os.environ['CUDA_VISIBLE_DEVICES']='0,1,2,3'
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import ruamel.yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path
import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
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

from sklearn.metrics import roc_auc_score, roc_curve, f1_score
from scipy.optimize import brentq

OMNI_FEAT_DIR = '/nas/data_2/wanhongz/ASAP'

def load_omni_feats(split):
    """加载指定split的OmniFD 512维特征查找表 (img_dir -> 512维tensor)"""
    path = f'{OMNI_FEAT_DIR}/omnifd_{split}_feats.pt'
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
from scipy.interpolate import interp1d

from models import box_ops
from tools.multilabel_metrics import AveragePrecisionMeter, get_multi_label
from models.ASAP import ASAP
from torch.cuda.amp import autocast, GradScaler

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

def load_omni_hasface(split):
    """只加载hasface标记（残差诊断实验用，不加载几十GB的局部token特征）"""
    path = f'{OMNI_FEAT_LOCAL_DIR}/omnifd_{split}_hasface_local.pt'
    hasface = torch.load(path, map_location='cpu')
    print(f'[Omni Residual] loaded {len(hasface)} hasface flags from {path}')
    return hasface

# ===== noise control experiment: fixed random vector per img_dir =====
_NOISE_CACHE = {}
def _get_noise_vec(d, std=1.0):
    if d not in _NOISE_CACHE:
        g = torch.Generator()
        g.manual_seed(abs(hash(d)) % (2**31))
        _NOISE_CACHE[d] = torch.randn(512, generator=g, dtype=torch.float32) * std
    return _NOISE_CACHE[d]

def build_omni_residual_batch(img_dirs, omni_feats_512, hasface_flags, device, use_noise=False, noise_std=1.0):
    if use_noise:
        _v, _m = [], []
        for d in img_dirs:
            _v.append(_get_noise_vec(d, noise_std))
            _m.append(bool(hasface_flags.get(d, False)) if d in omni_feats_512 else False)
        return torch.stack(_v, 0).to(device, non_blocking=True), torch.tensor(_m, dtype=torch.bool, device=device)
    """返回 (omni_feat_global [B,512], face_mask [B] bool)"""
    vecs, masks = [], []
    for d in img_dirs:
        if d in omni_feats_512:
            vecs.append(omni_feats_512[d])
            masks.append(bool(hasface_flags.get(d, False)))
        else:
            vecs.append(torch.zeros(512, dtype=torch.float32))
            masks.append(False)
    feats = torch.stack(vecs, dim=0).to(device, non_blocking=True)
    face_mask = torch.tensor(masks, dtype=torch.bool, device=device)
    return feats, face_mask


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

def text_input_adjust(text_input, fake_word_pos, device, real_cap=False):
    # input_ids adaptation
    input_ids_remove_SEP = [x[:-1] for x in text_input.input_ids]
    maxlen = max([len(x) for x in text_input.input_ids])-1
    input_ids_remove_SEP_pad = [x + [0] * (maxlen - len(x)) for x in input_ids_remove_SEP] # only remove SEP as HAMMER is conducted with text with CLS
    text_input.input_ids = torch.LongTensor(input_ids_remove_SEP_pad).to(device) 

    # attention_mask adaptation
    attention_mask_remove_SEP = [x[:-1] for x in text_input.attention_mask]
    attention_mask_remove_SEP_pad = [x + [0] * (maxlen - len(x)) for x in attention_mask_remove_SEP]
    text_input.attention_mask = torch.LongTensor(attention_mask_remove_SEP_pad).to(device)

    if real_cap:
        return text_input

    # fake_token_pos adaptation
    fake_token_pos_batch = []
    for i in range(len(fake_word_pos)):
        fake_token_pos = []

        fake_word_pos_decimal = np.where(fake_word_pos[i].numpy() == 1)[0].tolist() # transfer fake_word_pos into numbers

        subword_idx = text_input.word_ids(i)
        subword_idx_rm_CLSSEP = subword_idx[1:-1]
        subword_idx_rm_CLSSEP_array = np.array(subword_idx_rm_CLSSEP) # get the sub-word position (token position)

        # transfer the fake word position into fake token position
        for i in fake_word_pos_decimal: 
            fake_token_pos.extend(np.where(subword_idx_rm_CLSSEP_array == i)[0].tolist())
        fake_token_pos_batch.append(fake_token_pos)

    return text_input, fake_token_pos_batch


def train(args, model, data_loader, optimizer, tokenizer, epoch, warmup_steps, device, scheduler, config, summary_writer, omni_train_feats=None, omni_train_feats_local=None, omni_train_hasface_local=None):
    # train
    model.train()  
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_MAC', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_BIC', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_bbox', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_giou', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_TMG', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_MLC', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    # metric_logger.add_meter('loss_Patch', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('lossAnother', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    
    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 100   
    step_size = 100
    warmup_iterations = warmup_steps*step_size  

    global_step = epoch*len(data_loader)
    
    if args.distributed:
        data_loader.sampler.set_epoch(epoch)
    # scaler = GradScaler()

    for i, (image, label, text, fake_image_box, fake_word_pos, W, H, real_cap,real_prom,res_fake_pos,res_fake_pos_patch,img_dir) in enumerate(metric_logger.log_every(args, data_loader, print_freq, header)):
        if config['schedular']['sched'] == 'cosine_in_step':
            scheduler.adjust_learning_rate(optimizer, i / len(data_loader) + epoch, args, config)        

        optimizer.zero_grad()
  
        image = image.to(device,non_blocking=True) 
        
        text_input = tokenizer(text, max_length=128, truncation=True, add_special_tokens=True, return_attention_mask=True, return_token_type_ids=False) 
        cap_input = tokenizer(real_cap, max_length=128, truncation=True, add_special_tokens=True, return_attention_mask=True, return_token_type_ids=False) 

        text_input, fake_token_pos = text_input_adjust(text_input, fake_word_pos, device)
        cap_input = text_input_adjust(cap_input, fake_word_pos, device, True)
        
        prom_input = tokenizer(real_prom, max_length=128, truncation=True, add_special_tokens=True, return_attention_mask=True, return_token_type_ids=False) 

        
        prom_input = text_input_adjust(prom_input, fake_word_pos, device, True)

 
        if epoch>0:
            alpha = config['alpha']
        else:
            alpha = config['alpha']*min(1,i/len(data_loader)) 
        
        # loss_MAC, loss_BIC, loss_bbox, loss_giou, loss_TMG, loss_MLC,loss_Patch,loss_image_text_BIC = model(image, label, text_input, fake_image_box, fake_token_pos, cap_input,prom_input,res_fake_pos, alpha = alpha)  
        # with autocast():
        omni_valid_mask_batch = None
        omni_feat_global_batch = None
        omni_face_mask_batch = None
        if args.use_omni_fusion_local:
            omni_feat_batch, omni_valid_mask_batch = build_omni_batch_local(img_dir, omni_train_feats_local, omni_train_hasface_local, device)
        elif args.use_omni_fusion_residual:
            omni_feat_batch = None
            omni_feat_global_batch, omni_face_mask_batch = build_omni_residual_batch(img_dir, omni_train_feats, omni_train_hasface_local, device, args.omni_noise_control, args.omni_noise_std)
        else:
            omni_feat_batch = build_omni_batch(img_dir, omni_train_feats, device)
        loss_MAC, loss_BIC, loss_bbox, loss_giou, loss_TMG, loss_MLC, lossAnother = model(
            image, label, text_input, fake_image_box, fake_token_pos, cap_input, prom_input, res_fake_pos, res_fake_pos_patch,alpha=alpha, omni_feat=omni_feat_batch, omni_valid_mask=omni_valid_mask_batch, omni_feat_global=omni_feat_global_batch, omni_face_mask=omni_face_mask_batch
        ) 
        loss = config['loss_MAC_wgt']*loss_MAC \
             + config['loss_BIC_wgt']*loss_BIC \
             + config['loss_bbox_wgt']*loss_bbox \
             + config['loss_giou_wgt']*loss_giou \
             + config['loss_TMG_wgt']*loss_TMG \
             + config['loss_MLC_wgt']*loss_MLC \
             + 0.1 * lossAnother \
          
        loss.backward()
        # 残差诊断实验：scheduler 会覆盖所有 param group 的 lr，
        # 这里在 step 前强制把 alpha group 的 lr 拉回设定值
        if args.use_omni_fusion_residual:
            for _pg in optimizer.param_groups:
                if _pg.get('is_alpha_group', False):
                    _pg['lr'] = args.alpha_lr
        optimizer.step()    
        # scaler.scale(loss).backward()
        # scaler.step(optimizer)
        # scaler.update() 

        metric_logger.update(loss_MAC=loss_MAC.item())
        metric_logger.update(loss_BIC=loss_BIC.item())
        metric_logger.update(loss_bbox=loss_bbox.item())
        metric_logger.update(loss_giou=loss_giou.item())
        metric_logger.update(loss_TMG=loss_TMG.item())
        metric_logger.update(loss_MLC=loss_MLC.item())
        # metric_logger.update(loss_Patch=loss_Patch.item())
        metric_logger.update(lossAnother=lossAnother.item())
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])         
        if args.use_omni_fusion_residual:
            _m = model.module if hasattr(model, 'module') else model
            metric_logger.update(omni_alpha=_m.omni_residual_alpha.item())
            if i < 20 or i % 100 == 0:
                _a = _m.omni_residual_alpha
                _g = _a.grad
                _algrp = [g for g in optimizer.param_groups if g.get('is_alpha_group')]
                _pw = _m.omni_residual_proj[1].weight.grad
                print(f'[ALPHA-DBG] step={i} alpha={_a.item():.10f} '
                      f'grad={"None" if _g is None else f"{_g.item():.3e}"} '
                      f'group_lr={_algrp[0]["lr"] if _algrp else "NO_GROUP"} '
                      f'proj_w_grad={"None" if _pw is None else f"{_pw.abs().mean().item():.3e}"} '
                      f'face_ratio={float(omni_face_mask_batch.float().mean()) if omni_face_mask_batch is not None else "None"}',
                      flush=True)
        
        if epoch==0 and i%step_size==0 and i<=warmup_iterations and config['schedular']['sched'] != 'cosine_in_step': 
            scheduler.step(i//step_size)   

        global_step+=1
        

        #============ tensorboard train log info ============#
        if args.log:
            lossinfo = {
                'lr': optimizer.param_groups[0]["lr"],                                                                                                  
                'loss_MAC': loss_MAC.item(),                                                                                                  
                'loss_BIC': loss_BIC.item(),                                                                                                  
                'loss_bbox': loss_bbox.item(),                                                                                                  
                'loss_giou': loss_giou.item(),                                                                                                  
                'loss_TMG': loss_TMG.item(),                                                                                                  
                'loss_MLC': loss_MLC.item(),
                # 'loss_Patch': loss_Patch.item(),
                'lossAnother': lossAnother.item(),                                                                                                    
                'loss': loss.item(),                                                                                                  
                    } 
            for tag, value in lossinfo.items():
                summary_writer.add_scalar(tag, value, global_step)
        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    if args.log:
        print("Averaged stats:", metric_logger.global_avg(), flush=True)  
 
    return {k: "{:.6f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}    



@torch.no_grad()
def evaluation(args, model, data_loader, tokenizer, device, config, omni_val_feats=None, omni_val_feats_local=None, omni_val_hasface_local=None):
    # test
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Evaluation:'    
    
    print('Computing features for evaluation...')
    start_time = time.time()   
    print_freq = 200 

    y_true, y_pred, IOU_pred, IOU_50, IOU_75, IOU_95 = [], [], [], [], [], []
    cls_nums_all = 0
    cls_acc_all = 0   
    
    TP_all = 0
    TN_all = 0
    FP_all = 0
    FN_all = 0

    multi_label_meter = AveragePrecisionMeter(difficult_examples=False)
    multi_label_meter.reset()

    for i, (image, label, text, fake_image_box, fake_word_pos, W, H, real_cap,real_prom,res_fake_pos,res_fake_pos_patch,img_dir) in enumerate(metric_logger.log_every(args, data_loader, print_freq, header)):
        image = image.to(device,non_blocking=True) 
        
        text_input = tokenizer(text, max_length=128, truncation=True, add_special_tokens=True, return_attention_mask=True, return_token_type_ids=False) 
        cap_input = tokenizer(real_cap, max_length=128, truncation=True, add_special_tokens=True, return_attention_mask=True, return_token_type_ids=False) 
        prom_input = tokenizer(real_prom, max_length=128, truncation=True, add_special_tokens=True, return_attention_mask=True, return_token_type_ids=False) 

        text_input, fake_token_pos = text_input_adjust(text_input, fake_word_pos, device)
        cap_input = text_input_adjust(cap_input, fake_word_pos, device, True)
        prom_input = text_input_adjust(prom_input, fake_word_pos, device, True)

        omni_valid_mask_batch = None
        omni_feat_global_batch = None
        omni_face_mask_batch = None
        if args.use_omni_fusion_local:
            omni_feat_batch, omni_valid_mask_batch = build_omni_batch_local(img_dir, omni_val_feats_local, omni_val_hasface_local, device)
        elif args.use_omni_fusion_residual:
            omni_feat_batch = None
            omni_feat_global_batch, omni_face_mask_batch = build_omni_residual_batch(img_dir, omni_val_feats, omni_val_hasface_local, device, args.omni_noise_control, args.omni_noise_std)
        else:
            omni_feat_batch = build_omni_batch(img_dir, omni_val_feats, device)
        logits_real_fake, logits_multicls, output_coord, logits_tok = model(image, label, text_input, fake_image_box, fake_token_pos, cap_input,prom_input,res_fake_pos,res_fake_pos_patch,is_train=False, omni_feat=omni_feat_batch, omni_valid_mask=omni_valid_mask_batch, omni_feat_global=omni_feat_global_batch, omni_face_mask=omni_face_mask_batch)

        ##================= real/fake cls ========================## 
        cls_label = torch.ones(len(label), dtype=torch.long).to(image.device) 
        real_label_pos = np.where(np.array(label) == 'orig')[0].tolist()
        cls_label[real_label_pos] = 0
        
        y_pred.extend(F.softmax(logits_real_fake,dim=1)[:,1].cpu().flatten().tolist())
        y_true.extend(cls_label.cpu().flatten().tolist())

        pred_acc = logits_real_fake.argmax(1)
        cls_nums_all += cls_label.shape[0]
        cls_acc_all += torch.sum(pred_acc == cls_label).item()
        # ----- multi metrics -----
        target, _ = get_multi_label(label, image)
        multi_label_meter.add(logits_multicls, target)
        
        ##================= bbox cls ========================## 
        boxes1 = box_ops.box_cxcywh_to_xyxy(output_coord)
        boxes2 = box_ops.box_cxcywh_to_xyxy(fake_image_box)

        IOU, _ = box_ops.box_iou(boxes1, boxes2.to(device), test=True)

        IOU_pred.extend(IOU.cpu().tolist())

        IOU_50_bt = torch.zeros(IOU.shape, dtype=torch.long)
        IOU_75_bt = torch.zeros(IOU.shape, dtype=torch.long)
        IOU_95_bt = torch.zeros(IOU.shape, dtype=torch.long)

        IOU_50_bt[IOU>0.5] = 1
        IOU_75_bt[IOU>0.75] = 1
        IOU_95_bt[IOU>0.95] = 1

        IOU_50.extend(IOU_50_bt.cpu().tolist())
        IOU_75.extend(IOU_75_bt.cpu().tolist())
        IOU_95.extend(IOU_95_bt.cpu().tolist())

        ##================= token cls ========================##  
        token_label = text_input.attention_mask[:,1:].clone() # [:,1:] for ingoring class token
        token_label[token_label==0] = -100 # -100 index = padding token
        token_label[token_label==1] = 0

        for batch_idx in range(len(fake_token_pos)):
            fake_pos_sample = fake_token_pos[batch_idx]
            if fake_pos_sample:
                for pos in fake_pos_sample:
                    token_label[batch_idx, pos] = 1
                    
        logits_tok_reshape = logits_tok.view(-1, 2)
        logits_tok_pred = logits_tok_reshape.argmax(1)
        token_label_reshape = token_label.view(-1)
        
        # F1
        TP_all += torch.sum((token_label_reshape == 1) * (logits_tok_pred == 1)).item()
        TN_all += torch.sum((token_label_reshape == 0) * (logits_tok_pred == 0)).item()
        FP_all += torch.sum((token_label_reshape == 0) * (logits_tok_pred == 1)).item()
        FN_all += torch.sum((token_label_reshape == 1) * (logits_tok_pred == 0)).item()

    ##================= real/fake cls ========================## 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    AUC_cls = roc_auc_score(y_true, y_pred)
    ACC_cls = cls_acc_all / cls_nums_all
    fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label=1)
    EER_cls = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)

    ##================= multi-label cls ========================## 
    MAP = multi_label_meter.value().mean()
    OP, OR, OF1, CP, CR, CF1 = multi_label_meter.overall()
    OP_k, OR_k, OF1_k, CP_k, CR_k, CF1_k = multi_label_meter.overall_topk(3)
    
    ##================= bbox cls ========================##
    IOU_score = sum(IOU_pred)/len(IOU_pred)
    IOU_ACC_50 = sum(IOU_50)/len(IOU_50)
    IOU_ACC_75 = sum(IOU_75)/len(IOU_75)
    IOU_ACC_95 = sum(IOU_95)/len(IOU_95)

    # ##================= token cls========================##
    ACC_tok = (TP_all + TN_all) / (TP_all + TN_all + FP_all + FN_all)
    Precision_tok = TP_all / (TP_all + FP_all)
    Recall_tok = TP_all / (TP_all + FN_all)
    F1_tok = 2*Precision_tok*Recall_tok / (Precision_tok + Recall_tok)

    
    # print("\n========== s distribution ==========")
    # print(f"Real images: mean={np.mean(s_vals[real_mask]):.6f}, std={np.std(s_vals[real_mask]):.6f}")
    # print(f"Fake images: mean={np.mean(s_vals[fake_mask]):.6f}, std={np.std(s_vals[fake_mask]):.6f}")
    # print(f"Difference (Fake - Real): {np.mean(s_vals[fake_mask]) - np.mean(s_vals[real_mask]):.6f}")
    return AUC_cls, ACC_cls, EER_cls, \
           MAP.item(), OP, OR, OF1, CP, CR, CF1, OP_k, OR_k, OF1_k, CP_k, CR_k, CF1_k, \
           IOU_score, IOU_ACC_50, IOU_ACC_75, IOU_ACC_95, \
           ACC_tok, Precision_tok, Recall_tok, F1_tok
    
def main_worker(gpu, args, config):

    if gpu is not None:
        args.gpu = gpu

    init_dist(args)

    log_dir = os.path.join(args.output_dir, 'log'+ args.log_num)
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'shell.txt')
    logger = setlogger(log_file)
    yaml.dump(config, open(os.path.join(log_dir, 'config.yaml'), 'w')) 
    
    if args.log:
        summary_writer = SummaryWriter(log_dir)
    else:
        summary_writer = None

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
    
    start_epoch = 0
    max_epoch = config['schedular']['epochs']
    warmup_steps = config['schedular']['warmup_epochs']  
    best = 0
    best_epoch = 0  

    #### Dataset #### 
    if args.log:
        print("Creating dataset")
    train_dataset, val_dataset = create_dataset(config)
    
    if args.distributed:
        samplers = create_sampler([train_dataset], [True], args.world_size, args.rank) + [None]    
    else:
        samplers = [None, None]

    train_loader, val_loader = create_loader([train_dataset, val_dataset],
                                samplers,
                                batch_size=[config['batch_size_train']]+[config['batch_size_val']], 
                                num_workers=[4, 4], 
                                is_trains=[True, False], 
                                collate_fns=[None, None])

    tokenizer = BertTokenizerFast.from_pretrained(args.text_encoder)
    # tokenizer = BertTokenizerFast.from_pretrained(args.text_encoder)

    #### Model #### 
    if args.log:
        print(f"Creating Model")
    model = ASAP(args=args, config=config, text_encoder='/nas/data_2/wanhongz/bert-base-uncased', tokenizer=tokenizer, init_deit=True)
    model = model.to(device)   

    global omni_train_feats, omni_val_feats
    omni_train_feats = load_omni_feats('train')
    omni_val_feats = load_omni_feats('val')
    omni_train_feats_local, omni_train_hasface_local = ({}, {})
    omni_val_feats_local, omni_val_hasface_local = ({}, {})
    if args.use_omni_fusion_local:
        omni_train_feats_local, omni_train_hasface_local = load_omni_feats_local('train')
        omni_val_feats_local, omni_val_hasface_local = load_omni_feats_local('val')
    if args.use_omni_fusion_residual:
        omni_train_hasface_local = load_omni_hasface('train')
        omni_val_hasface_local = load_omni_hasface('val')

    # ========== Omni特征融合微调：先加载checkpoint，再冻结参数，最后才建optimizer ==========
    if args.checkpoint:    
        checkpoint = torch.load(args.checkpoint, map_location='cpu') 
        state_dict = checkpoint['model'] 
        # print(state_dict.keys())                      
        if args.resume:
            pass  # resume场景下optimizer/scheduler状态在下面optimizer创建后单独处理
        else:
            # print('model visual encoder:', model.visual_encoder)
            pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'],model.visual_encoder)   
            state_dict['visual_encoder.pos_embed'] = pos_embed_reshaped       
        # model.load_state_dict(state_dict)  
        if args.log:
            print('load checkpoint from %s'%args.checkpoint)  
        msg = model.load_state_dict(state_dict, strict=False)
        # if args.log:
        #     print(msg)  

    # ========== 冻结除 omni_proj/omni_fusion_fc/cls_head/itm_head 外的所有参数 ==========
    # 残差诊断实验：只解冻 OmniFD 通路 + 分类头，不解冻 aggregator，
    # 保证 AUC 变化的唯一新增来源是 OmniFD 特征，alpha 的因果解释才干净
    UNFROZEN_KEYWORDS = ('omni_residual', 'cls_head', 'itm_head')
    n_trainable = 0
    n_frozen = 0
    for name, param in model.named_parameters():
        if any(kw in name for kw in UNFROZEN_KEYWORDS):
            param.requires_grad = True
            n_trainable += param.numel()
        else:
            param.requires_grad = False
            n_frozen += param.numel()
    if args.log:
        print(f'[Omni Fusion] trainable params: {n_trainable:,} | frozen params: {n_frozen:,}')
        for _n, _p in model.named_parameters():
            if 'omni_residual' in _n:
                print(f'[Omni Residual] param {_n}: shape={tuple(_p.shape)}, requires_grad={_p.requires_grad}')

    arg_opt = utils.AttrDict(config['optimizer'])
    optimizer = create_optimizer(arg_opt, model)

    # ========== 残差诊断实验：alpha 单独 param group ==========
    # 理由1: alpha 是标量，用主干 lr(1.25e-6) 时 50epoch 也只能移动 ~0.27，
    #        观察不到"增大/被压缩到0"的趋势，诊断标准无法判定
    # 理由2: weight_decay 必须为 0——否则 alpha 会被 wd 持续拉向 0，
    #        与"模型主动关闭 OmniFD 通路"的诊断信号混淆
    if args.use_omni_fusion_residual:
        _alpha_p = model.omni_residual_alpha
        for _g in optimizer.param_groups:
            _g['params'] = [p for p in _g['params'] if p is not _alpha_p]
        optimizer.add_param_group({
            'params': [_alpha_p],
            'lr': args.alpha_lr,
            'initial_lr': args.alpha_lr,
            'weight_decay': 0.0,
            'is_alpha_group': True,
        })
        if args.log:
            print(f'[Omni Residual] alpha param group added: lr={args.alpha_lr}, weight_decay=0.0')
            if args.omni_noise_control:
                print(f'[NOISE CONTROL] *** OmniFD feats REPLACED by fixed random vectors (std={args.omni_noise_std}) ***')
            print(f'[Omni Residual] total param groups = {len(optimizer.param_groups)}')

    arg_sche = utils.AttrDict(config['schedular'])
    lr_scheduler, _ = create_scheduler(arg_sche, optimizer)

    # 注: 原本这里有一个 monkey patch 保护 alpha group 的 lr，
    # 但局部函数无法被 pickle，保存 checkpoint 时会报错。
    # 训练循环里 optimizer.step() 前已经强制把 alpha group 的 lr 拉回 args.alpha_lr，
    # 效果等价且更稳，故移除。

    if config['schedular']['sched'] == 'cosine_in_step':
        args.lr = config['optimizer']['lr']

    if args.checkpoint and args.resume:
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        start_epoch = checkpoint['epoch']+1

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    if args.log:
        print("Start training")
    start_time = time.time()

    for epoch in range(start_epoch, max_epoch):
        train_stats = train(args, model, train_loader, optimizer, tokenizer, epoch, warmup_steps, device, lr_scheduler, config, summary_writer, omni_train_feats, omni_train_feats_local, omni_train_hasface_local) 
        AUC_cls, ACC_cls, EER_cls, \
        MAP, OP, OR, OF1, CP, CR, CF1, OP_k, OR_k, OF1_k, CP_k, CR_k, CF1_k, \
        IOU_score, IOU_ACC_50, IOU_ACC_75, IOU_ACC_95, \
        ACC_tok, Precision_tok, Recall_tok, F1_tok \
        = evaluation(args, model_without_ddp, val_loader, tokenizer, device, config, omni_val_feats, omni_val_feats_local, omni_val_hasface_local)

        #============ tensorboard train log info ============#
        if args.log:
            lossinfo = {
                'AUC_cls': round(AUC_cls*100, 4),                                                                                                  
                'ACC_cls': round(ACC_cls*100, 4),                                                                                                  
                'EER_cls': round(EER_cls*100, 4),                                                                                                  
                'MAP': round(MAP*100, 4),                                                                                                  
                'OP': round(OP*100, 4),                                                                                                  
                'OR': round(OR*100, 4), 
                'OF1': round(OF1*100, 4), 
                'CP': round(CP*100, 4), 
                'CR': round(CR*100, 4), 
                'CF1': round(CF1*100, 4), 
                'OP_k': round(OP_k*100, 4), 
                'OR_k': round(OR_k*100, 4), 
                'OF1_k': round(OF1_k*100, 4), 
                'CP_k': round(CP_k*100, 4), 
                'CR_k': round(CR_k*100, 4), 
                'CF1_k': round(CF1_k*100, 4), 
                'IOU_score': round(IOU_score*100, 4),                                                                                                  
                'IOU_ACC_50': round(IOU_ACC_50*100, 4),                                                                                                  
                'IOU_ACC_75': round(IOU_ACC_75*100, 4),                                                                                                  
                'IOU_ACC_95': round(IOU_ACC_95*100, 4),                                                                                                  
                'ACC_tok': round(ACC_tok*100, 4),                                                                                                  
                'Precision_tok': round(Precision_tok*100, 4),                                                                                                  
                'Recall_tok': round(Recall_tok*100, 4),                                                                                                  
                'F1_tok': round(F1_tok*100, 4),                                                                                                  
                    } 
            for tag, value in lossinfo.items():
                summary_writer.add_scalar(tag, value, epoch)

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
                     "OP_k": "{:.4f}".format(OP_k*100),
                     "OR_k": "{:.4f}".format(OR_k*100),
                     "OF1_k": "{:.4f}".format(OF1_k*100),
                     "CP_k": "{:.4f}".format(CP_k*100),
                     "CR_k": "{:.4f}".format(CR_k*100),
                     "CF1_k": "{:.4f}".format(CF1_k*100),
                     "IOU_score": "{:.4f}".format(IOU_score*100),
                     "IOU_ACC_50": "{:.4f}".format(IOU_ACC_50*100),
                     "IOU_ACC_75": "{:.4f}".format(IOU_ACC_75*100),
                     "IOU_ACC_95": "{:.4f}".format(IOU_ACC_95*100),
                     "ACC_tok": "{:.4f}".format(ACC_tok*100),
                     "Precision_tok": "{:.4f}".format(Precision_tok*100),
                     "Recall_tok": "{:.4f}".format(Recall_tok*100),
                     "F1_tok": "{:.4f}".format(F1_tok*100),
        }
        
        if utils.is_main_process(): 
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                            **{f'val_{k}': v for k, v in val_stats.items()},
                            'epoch': epoch,
                            'omni_alpha': float(model_without_ddp.omni_residual_alpha.item()),
                        }             
            print(f"[Omni Residual] epoch {epoch} | alpha = {model_without_ddp.omni_residual_alpha.item():.6f} | val AUC_cls = {val_stats['AUC_cls']}")
            with open(os.path.join(log_dir, "log.txt"),"a") as f:
                f.write(json.dumps(log_stats) + "\n")

            if config['schedular']['sched'] != 'cosine_in_step':
                save_obj = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'config': config,
                    'epoch': epoch,
                }
            else:
                save_obj = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr': optimizer.param_groups[0]["lr"],
                    'config': config,
                    'epoch': epoch,
                }                    
            if (epoch % args.model_save_epoch == 0 and epoch!=0):
                torch.save(save_obj, os.path.join(log_dir, 'checkpoint_%02d.pth'%epoch)) 
            if float(val_stats['AUC_cls'])>best:
                torch.save(save_obj, os.path.join(log_dir, 'checkpoint_best.pth')) 
                best = float(val_stats['AUC_cls'])
                best_epoch = epoch 

        if config['schedular']['sched'] != 'cosine_in_step':
            lr_scheduler.step(epoch+warmup_steps+1)  
        dist.barrier() 

    if utils.is_main_process():
        torch.save(save_obj, os.path.join(log_dir, 'checkpoint_%02d.pth'%epoch))   
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    if args.log:
        print('Training time {}'.format(total_time_str)) 
        with open(os.path.join(log_dir, "log.txt"),"a") as f:
            f.write("best epoch: {}, Training time: {}".format(best_epoch, total_time_str))    
       

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/Pretrain.yaml')
    parser.add_argument('--checkpoint', default='') 
    parser.add_argument('--resume', default=False, type=bool)
    parser.add_argument('--output_dir', default='results')
    parser.add_argument('--text_encoder', default='bert-base-uncased')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=777, type=int)
    parser.add_argument('--distributed', default=True, type=bool)
    parser.add_argument('--rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--world_size', default=1, type=int,
                        help='world size for distributed training')
    parser.add_argument('--dist-url', default='tcp://127.0.0.1:23459', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], default='none',
                        help='job launcher')
    parser.add_argument('--log_num', '-l', type=str)
    parser.add_argument('--model_save_epoch', type=int, default=20)
    parser.add_argument('--token_momentum', default=False, action='store_true')
    parser.add_argument('--use_omni_fusion_local', action='store_true', help='是否启用OmniFD局部token序列融合（训练时解冻omni_proj+aggregator）')
    parser.add_argument('--use_omni_fusion_residual', action='store_true', help='残差加和诊断实验：vision_feat + mask*alpha*proj(omni_global)，与--use_omni_fusion_local互斥')
    parser.add_argument('--omni_noise_control', action='store_true', help='control exp: replace OmniFD feats with fixed random vectors')
    parser.add_argument('--omni_noise_std', type=float, default=0.255, help='noise std, matches real OmniFD global std')
    parser.add_argument('--alpha_lr', type=float, default=1e-3, help='omni_residual_alpha 专用学习率（独立param group，无weight_decay，不受cosine调度影响）')

    args = parser.parse_args()
    assert not (args.use_omni_fusion_local and args.use_omni_fusion_residual), \
        '--use_omni_fusion_local 与 --use_omni_fusion_residual 互斥，不能同时使用'

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    # main(args, config)
    if args.launcher == 'none':
        args.launcher = 'pytorch'
        main_worker(0, args, config)
    else:
        ngpus_per_node = torch.cuda.device_count()
        args.ngpus_per_node = ngpus_per_node
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(args, config))
