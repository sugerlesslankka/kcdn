# this is for all dataset used in caption

import sys
import os
from typing import Tuple, Optional, Union
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, GPT2Tokenizer
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
import torch
import torch.distributed as dist
from torch.utils.data import Dataset
from clip1 import clip
from clip1.clip import _transform
from .key_utils import keyword_extractor
import skimage.io as io1
from PIL import Image
from PIL import ImageFile
import numpy as np
import pickle
from enum import Enum
import json
from utils.misc import generate2_adpt_if_nodist, evaluate_on_coco_caption

class ClipCocoGPTDataset(Dataset):

    def __len__(self) -> int:
        return len(self.captions_tokens)
    # for keys, no endoftext and no need to return mask
    def pad_tokens(self, item: int, is_key):
        if is_key:
            tokens = torch.tensor(self.tokenizer.encode(keyword_extractor(self.captions[item], self.stage)), dtype=torch.int64)
        else:
            tokens = torch.cat((self.captions_tokens[item], torch.tensor(self.tokenizer.encode('<|endoftext|>'))), dim=0)
            gt = torch.cat((self.captions_tokens[item], torch.tensor(self.tokenizer.encode('<|endoftext|>'))), dim=0)
        padding = self.std_len - tokens.shape[0]
        if padding > 0:
            tokens = torch.cat((tokens, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            tokens = tokens[:self.std_len]
        mask = tokens.ge(0)  # mask is zero where we out of sequence
        tokens[~mask] = 0
        mask = mask.float()
        if is_key:
            return tokens
        pad_gt = self.std_len - gt.shape[0]
        if pad_gt > 0:
            gt = torch.cat((gt, torch.zeros(pad_gt, dtype=torch.int64) - 1))  # we set target == -1 as ignore target
        elif pad_gt < 0:
            gt = gt[:self.std_len]
        return tokens, mask, gt

    def set_stage(self, stage_num: int):
        self.stage = stage_num
        print("setting training stage into", self.stage)

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, ...]:
        tokens, mask, gt = self.pad_tokens(item, is_key=False)
        key_tokens = self.pad_tokens(item, is_key=True) 
        img_id = self.image_ids[item]
        # train+restvals
        filename = f"{self.data_root}/train2014/COCO_train2014_{int(img_id):012d}.jpg"
        try:
            image = io1.imread(filename)
        except:
            filename = f"{self.data_root}/val2014/COCO_val2014_{int(img_id):012d}.jpg"
            image = io1.imread(filename)
        image = Image.fromarray(image)
        image = self.preprocess(image)
        prefix = self.prefixes[self.caption2embedding[item]]
        if self.normalize_prefix:
            prefix = prefix.float()
            prefix = prefix / prefix.norm(2, -1)
        return tokens, mask, image, gt, key_tokens

    def __init__(self, data_root: str, data_path: str, tokenizer_type: str = "gpt2", len_seq=20, normalize_prefix=False):
        self.data_root = data_root
        self.tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_type)
        self.normalize_prefix = normalize_prefix
        self.preprocess = _transform(224)
        with open(data_path, 'rb') as f:
            all_data = pickle.load(f)
        print("Data size is %0d" % len(all_data["clip_embedding"]))
        sys.stdout.flush()
        self.prefixes = all_data["clip_embedding"]
        captions_raw = all_data["captions"]
        self.image_ids = [caption["image_id"] for caption in captions_raw]
        self.captions = [caption['caption'] for caption in captions_raw]
        self.stage = 0
        # self.set_stage(0)
        if os.path.isfile(f"{data_path[:-4]}_base.pkl"):
            with open(f"{data_path[:-4]}_base.pkl", 'rb') as f:
                self.captions_tokens, self.caption2embedding, self.max_seq_len = pickle.load(f)
        else:
            self.captions_tokens = []
            self.caption2embedding = []
            max_seq_len = 0
            for caption in captions_raw:
                self.captions_tokens.append(torch.tensor(self.tokenizer.encode(caption['caption']), dtype=torch.int64))
                self.caption2embedding.append(caption["clip_embedding"])  # just index
                max_seq_len = max(max_seq_len, self.captions_tokens[-1].shape[0])
        all_len = torch.tensor([len(self.captions_tokens[i]) for i in range(len(self))]).float()
        self.std_len = len_seq

class ClipCocoGPTValDataset(Dataset):

    def __len__(self) -> int:
        return len(self.files)

    def set_stage(self, stage_num: int):
        self.stage = stage_num
        print("setting valing stage into", self.stage)

    def __getitem__(self, item: int):
        _filename = self.files[item]
        filename = f"{self.data_root}/val2014/{_filename}"
        for x in self.annotation:
            if 'COCO_val2014_' + str(x['image_id']).zfill(12) + '.jpg' == _filename:
                caption = x['caption']
                break
        tokens = torch.tensor(self.tokenizer.encode(caption), dtype=torch.int64)
        tokens = torch.cat((tokens, torch.tensor(self.tokenizer.encode('<|endoftext|>'))), dim=0)
        gt = torch.clone(tokens)

        padding = self.std_len - tokens.shape[0]
        pad_gt = self.std_len - gt.shape[0]
        if padding > 0:
            tokens = torch.cat((tokens, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            tokens = tokens[:self.std_len]
        if pad_gt > 0:
            gt = torch.cat((gt, torch.zeros(pad_gt, dtype=torch.int64) - 1))  # we set target == -1 as ignore target
        elif pad_gt < 0:
            gt = gt[:self.std_len]
        mask = tokens.ge(0)  # mask is zero where we out of sequence
        tokens[~mask] = 0
        mask = mask.float()

        image = io1.imread(filename)
        image = Image.fromarray(image)
        image = self.preprocess(image)
        keys = ","
        key_tokens = torch.tensor(self.tokenizer.encode(keys), dtype=torch.int64)
        padding = self.std_len - key_tokens.shape[0]
        if padding > 0:
            key_tokens = torch.cat((key_tokens, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            key_tokens = key_tokens[:self.std_len]
        mask_key = key_tokens.ge(0)
        key_tokens[~mask_key] = 0
        # self.counts[item] += 1
        return image, _filename, tokens, gt, mask, key_tokens, keys
    
    def __init__(self, data_root: str):
        self.data_root = data_root
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.annotation = json.load(open("../dataset/MSCOCO_Caption/annotations/captions_val2014.json", "r"))["annotations"]
        with open('utils/captioneval/coco_test.txt') as f:
            self.files = f.read().splitlines()
        self.preprocess = _transform(224)
        self.std_len = 20
        self.stage = 0
        # self.set_stage(0)
        # self.counts = [0]*len(self.files)
