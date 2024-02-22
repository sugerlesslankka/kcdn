# Training this with multi-keyword from each sentence

import os
import random
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.cuda.amp as amp
from torch.nn import functional as nnf
from torch.utils.data import Dataset, DataLoader
from enum import Enum
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from utils.tf_adpt import GPT2LMHeadModel
from transformers import GPT2Tokenizer, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
import random
import pickle
import sys
import argparse
import json
from typing import Tuple, Optional, Union
from clip1 import clip
from clip1.clip import _transform
import math
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import skimage.io as io1
from PIL import Image
from PIL import ImageFile
from timm.models.layers import trunc_normal_
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
from project.scheduler.lr_scheduler import build_scheduler
from utils.misc import generate2_adpt_if, evaluate_on_coco_caption
import torch.nn.functional as F
import nltk
from project.dataset.dataset_div import DivDataset
from project.models.trans import *

class DivModel(nn.Module):

    def forward(self, image: torch.Tensor, tokens: torch.Tensor):
        # self.clip_model.eval()
        self.gpt.eval()
    
        with torch.no_grad():    
            embedding_text = self.gpt.transformer.wte(tokens)
            # prefix, len_cls = self.image_encode(image)
        # print(embedding_text.shape)
        # out = self.attn(prefix, embedding_text)
        out = self.bc(embedding_text)
        out = self.fl(out)
        out = self.bc2(out)
        out = self.act(out)
        return out

    def __init__(self):
        super(DivModel, self).__init__()
        configuration = GPT2Config.from_pretrained("gpt2")
        configuration = configuration.__dict__.copy()
        configuration.update({'scale_attn_by_inverse_layer_idx': False})
        configuration.update({'reorder_and_upcast_attn': False})
        configuration = GPT2Config(**configuration)
        self.gpt = GPT2LMHeadModel(configuration)
        self.gpt.requires_grad_(False)
        # self.clip_model, _ = clip.load("ViT-B/16", device='cpu', jit=False)
        # self.clip_model.requires_grad_(False)
        # self.image_encode = self.clip_model.encode_image

        # self.attn = Transformer(768, 1, 1, 768)
        self.bc = MLP((768, 10))
        self.fl = nn.Flatten()
        self.bc2 = MLP((10*20, 2))
        self.act = nn.Sigmoid()


def setup_for_distributed(is_master):
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

def train(model, epoch, train_dataloader, optimizer, lr_scheduler, scaler, test_dataloader):
    model.train()
    num_steps = len(train_dataloader)
    train_dataloader.sampler.set_epoch(epoch)
    print(f">>> Training epoch {epoch}")
    sys.stdout.flush()
    progress = tqdm(total=len(train_dataloader))
    for idx, (image, tokens, label) in enumerate(train_dataloader):
        # prefix is raw images
        tokens = tokens.cuda(non_blocking=True)
        image = image.cuda(non_blocking=True)
        label = label.cuda(non_blocking=True)
        output = model(image, tokens)
        loss = nnf.cross_entropy(output, label)

        optimizer.zero_grad()
        scaler.scale(loss).backward()  # loss.backward()
        scaler.step(optimizer)  # optimizer.step()
        scaler.update()
        lr_scheduler.step_update(epoch * num_steps + idx)
        torch.cuda.synchronize()
        progress.set_postfix({"loss": loss.item(), 'lr': optimizer.param_groups[0]['lr']})
        progress.update()
    progress.close()
    model.eval()
    total = 0
    correct = 0
    for idx, (image, tokens, label) in enumerate(test_dataloader):
        # prefix is raw images
        tokens = tokens.cuda(non_blocking=True)
        image = image.cuda(non_blocking=True)
        label = label.cuda(non_blocking=True)
        output = model(image, tokens)
        _, predict = torch.max(output.data, 1)
        total += label.size(0)
        correct += (predict == label).sum().item()
    
    print(correct/total)
    
    return correct/total

def check_keywords_in_name(name, keywords=()):
    isin = False
    for keyword in keywords:
        if keyword in name:
            isin = True
    return isin

def get_pretrain_param_groups(model, skip_list=(), skip_keywords=()):
    has_decay = []
    no_decay = []
    has_decay_name = []
    no_decay_name = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name.endswith(".bias") or (name in skip_list) or \
                check_keywords_in_name(name, skip_keywords):
            no_decay.append(param)
            no_decay_name.append(name)
        else:
            has_decay.append(param)
            has_decay_name.append(name)
    print(f'No decay params: {no_decay_name}')
    print(f'Has decay params: {has_decay_name}')
    return [{'params': has_decay},
            {'params': no_decay, 'weight_decay': 0.}
            ]

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--bs', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--wd', type=float, default=0.01)
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument("--local_rank", type=int, required=True, help='local rank for DistributedDataParallel')
    args = parser.parse_args()
    return args


def main(args):
    model = DivModel()
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_params:,} total parameters')
    local_rank = args.local_rank
    torch.cuda.set_device(local_rank)
    model = model.cuda()

    model = DDP(model, device_ids=[local_rank], broadcast_buffers=False)

    div_dataset = DivDataset('../dataset/MSCOCO_Caption/')
    train_size = int(len(div_dataset) * 0.7)
    train_dataset, test_dataset = torch.utils.data.random_split(div_dataset, [train_size, len(div_dataset)-train_size])

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, shuffle=True)
    train_dataloader = DataLoader(train_dataset, batch_size=args.bs, sampler=train_sampler, num_workers=8, pin_memory=True, drop_last=False)
    test_dataloader = DataLoader(test_dataset, batch_size=args.bs*20, sampler=test_sampler, num_workers=8, pin_memory=True, drop_last=False)
    parameters = get_pretrain_param_groups(model)
    optimizer = AdamW(parameters, lr=args.lr, weight_decay=args.wd)
    scaler = amp.GradScaler()
    lr_args = {"LR_SCHEDULER_NAME": "cosine", "EPOCHS": args.epochs, "WARMUP_EPOCHS": 5, "MIN_LR": 1e-6,
               "WARMUP_LR": 1e-7}
    lr_scheduler = build_scheduler(lr_args, optimizer, len(train_dataloader))

    for epoch in range(args.epochs):
        _ = train(model, epoch, train_dataloader, optimizer, lr_scheduler, scaler, test_dataloader)
        

if __name__ == '__main__':
    args = parse_args()
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1
    dist.init_process_group("nccl", init_method='env://', rank=args.local_rank, world_size=world_size)
    torch.distributed.barrier()
    setup_for_distributed(args.local_rank == 0)  ##### HERE

    seed = dist.get_rank() + args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    main(args)
