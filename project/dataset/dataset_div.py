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

# in diversity evaluation, double size and get label, no need for mask

class DivDataset(Dataset):

    def __len__(self) -> int:
        return len(self.files)*2

    def __getitem__(self, item: int):
        # read robot annos
        if item >= len(self.files):
            item -= len(self.files)
            _filename = self.files[item]
            filename = f"{self.data_root}/val2014/{_filename}"
            for x in self.robot_anno:
                if 'COCO_val2014_' + str(x['image_id']).zfill(12) + '.jpg' == _filename:
                    caption = x['caption']
                    break
            label = 1
        else:
            _filename = self.files[item]
            filename = f"{self.data_root}/val2014/{_filename}"
            for x in self.annotation:
                if 'COCO_val2014_' + str(x['image_id']).zfill(12) + '.jpg' == _filename:
                    caption = x['caption']
                    break
            label = 0
        tokens = torch.tensor(self.tokenizer.encode(caption), dtype=torch.int64)
        tokens = torch.cat((tokens, torch.tensor(self.tokenizer.encode('<|endoftext|>'))), dim=0)

        padding = self.std_len - tokens.shape[0]
        if padding > 0:
            tokens = torch.cat((tokens, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            tokens = tokens[:self.std_len]
        mask = tokens.ge(0)  # mask is zero where we out of sequence
        tokens[~mask] = 0
        mask = mask.float()

        image = io1.imread(filename)
        image = Image.fromarray(image)
        image = self.preprocess(image)

        return image, tokens, label
    
    def __init__(self, data_root: str):
        self.data_root = data_root
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.annotation = json.load(open("../dataset/MSCOCO_Caption/annotations/captions_val2014.json", "r"))["annotations"]
        self.robot_anno = json.load(open("path-to-result.json", 'r'))
        with open('utils/captioneval/coco_test.txt') as f:
            self.files = f.read().splitlines()
        self.preprocess = _transform(224)
        self.std_len = 20

