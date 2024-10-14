import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import torch.distributed as dist
from PIL import Image
from project.models.model import MappingType, CaptionModel
from transformers import GPT2Tokenizer
import torch.utils.data as data
from clip1.clip import _transform
import json
import skimage.io as io1
from utils.misc import generate2_adpt_if_nodist

dist.init_process_group("nccl", init_method='file:///tmp/somefile', rank=0, world_size=1)

# get image from coco dataset to current folder
id = 581929
base_path = '../dataset/MSCOCO_Caption/annotations/captions_val2014.json'
with open(base_path,'r') as f:
    dataset=json.load(f)
for data in dataset['annotations']:
    if data['image_id'] == id:
        cap = data['caption']
        break
raw_image = Image.open('../dataset/MSCOCO_Caption/val2014/COCO_val2014_000000'+str(id)+'.jpg').convert("RGB")
raw_image.save(cap+".png")

# read image
image = io1.imread('../dataset/MSCOCO_Caption/val2014/COCO_val2014_000000'+str(id)+'.jpg')
# image = io1.imread('./images/walk.jpg')
image = Image.fromarray(image)

# load model
model = CaptionModel(10, clip_length=10, prefix_size=512,
                                 num_layers=8, mapping_type=MappingType.MLP, Timestep=20,
                                 if_drop_rate=0.1)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
print('finished loading')
model.load_state_dict(torch.load('../keyword_diffusion/checkpoints/10/10-049.pt', map_location=torch.device('cpu'))["model"])
model = model.cuda()
model.eval()

# diverse function
def gen_many_caption(image, model, num_cap=1, keyword_set=[]):
    num_cap = int(num_cap)
    if num_cap < 1:
        return 'ERROR'
    if len(keyword_set) > num_cap:
        keyword_set = keyword_set[:num_cap]
    cur = ''
    while len(keyword_set) < num_cap:
        keyword_set.append(cur)
        cur += ', '
    image = _transform(224)(image)
    image = image.cuda(non_blocking=True)
    image = torch.cat([image.unsqueeze(0) for _ in range(num_cap)], dim=0)
    kt_set = []
    for kw in keyword_set:
        kt = torch.tensor(tokenizer.encode(kw))
        padding = 20 - kt.shape[0]
        if padding > 0:
            kt = torch.cat((kt, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            kt = kt[:20]
        mask_kt = kt.ge(0)
        kt[~mask_kt] = 0
        kt = kt.cuda(non_blocking=True)
        kt = kt.unsqueeze(0)
        kt = kt.long()
        kt_set.append(kt)
    kt = torch.cat(kt_set, dim=0)
    prefix, len_cls = model.image_encode(image)
    prefix_embed = model.clip_project(prefix)
    kt = model.gpt.transformer.wte(kt)
    len_pre = model.len_head(len_cls)
    prefix_embed = model.kw_att(prefix_embed, kt)
    generated_text_prefix = generate2_adpt_if_nodist(model, tokenizer, embed=prefix_embed, len_pre=len_pre.argmax(-1) + 1)
    print(generated_text_prefix)

gen_many_caption(image, model, 5)