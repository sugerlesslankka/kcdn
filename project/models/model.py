# this is the model with kw input

import torch
import torch.nn as nn
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from utils.tf_adpt import GPT2LMHeadModel
from clip1 import clip
from typing import Tuple, Optional, Union
from enum import Enum
from .trans import *
from .dif_utils import *


class MappingType(Enum):
    MLP = 'mlp'
    Transformer = 'transformer'


def setup_for_distributed(is_master):
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


class CaptionModel(nn.Module):

    def get_dummy_token(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.prefix_length, dtype=torch.int64, device=device)

    def forward(self, tokens: torch.Tensor, mask_tokens: torch.Tensor, prefix: torch.Tensor, keyword: Optional[torch.Tensor] = None, 
                mask: Optional[torch.Tensor] = None, t = None, 
                labels: Optional[torch.Tensor] = None):
        self.clip_model.eval()
        embedding_text = self.gpt.transformer.wte(tokens)
        batch_size = embedding_text.size()[0]
        seq_len = embedding_text.size()[1]
        bos_token_embedding = self.bos_embedding.unsqueeze(0).unsqueeze(0).repeat_interleave(repeats=batch_size, dim=0)
        bos_token_embedding = bos_token_embedding.repeat_interleave(repeats=seq_len, dim=1)
        mask_tokens = mask_tokens.unsqueeze(dim=2).repeat_interleave(repeats=self.gpt_embedding_size, dim=2)
        embedding_text = torch.where(mask_tokens == 50257, bos_token_embedding, embedding_text)
        with torch.no_grad():
            prefix, len_cls = self.image_encode(prefix)
        prefix_projections = self.clip_project(prefix)
        if self.training:
            empty_idx = int(self.if_drop_rate * batch_size)
            for i_image in range(empty_idx):
                prefix_projections[i_image,:,:] = self.pad_embedding
        if labels is not None:
            dummy_token = self.get_dummy_token(tokens.shape[0], tokens.device)
            labels = torch.cat((dummy_token, tokens), dim=1)
        # interact with kw
        if keyword is not None:
            keyword_text = self.gpt.transformer.wte(keyword)
            prefix_projections = self.kw_att(prefix_projections, keyword_text)
            out = self.gpt(t=t, inputs_embeds=embedding_text, labels=labels, attention_mask=mask,
                       encoder_hidden_states=prefix_projections)
        else:
            out = self.gpt(t=t, inputs_embeds=embedding_text, labels=labels, attention_mask=mask,
                       encoder_hidden_states=prefix_projections)
        return out, self.len_head(len_cls)
        # temp = self.len_head(len_cls)
        # print(temp.shape)
        
        # len_tokens = torch.sum(prefix_projections, dim=1, keepdim=False)
        # len_tokens = self.len_head(len_tokens)
        # print(len_tokens.shape)
        
        # return out, len_tokens

    def __init__(self, prefix_length: int, clip_length: Optional[int] = None, prefix_size: int = 512,
                 num_layers: int = 8, mapping_type: MappingType = MappingType.MLP, Timestep: int = 20, if_drop_rate=0.02):
        super(CaptionModel, self).__init__()
        self.prefix_length = prefix_length
        configuration = GPT2Config.from_pretrained("gpt2")
        configuration = configuration.__dict__.copy()
        configuration.update({'scale_attn_by_inverse_layer_idx': False})
        configuration.update({'reorder_and_upcast_attn': False})
        configuration = GPT2Config(**configuration)
        self.gpt = GPT2LMHeadModel(configuration)
        self.clip_model, _ = clip.load("ViT-B/16", device='cpu', jit=False)
        self.clip_model.requires_grad_(False)
        self.image_encode = self.clip_model.encode_image
        self.time_step = Timestep
        self.if_drop_rate = if_drop_rate
        self.num_classes = configuration.vocab_size + 1
        self.gpt_embedding_size = self.gpt.transformer.wte.weight.shape[1]
        self.bos_embedding = nn.Parameter(torch.randn(self.gpt_embedding_size)) # used as the vector of mask token
        self.pad_embedding = nn.Parameter(torch.randn(size=(196, 768), requires_grad=True, dtype=torch.float64)) # image free vector
        self.len_head = MLP((512, self.gpt_embedding_size // 2, 20))
        # self.len_head = MLP((self.gpt_embedding_size, self.gpt_embedding_size // 2, 20))
        
        self.kw_att = Transformer(self.gpt_embedding_size, 4, 2, self.gpt_embedding_size, enc_dec=True)

        if mapping_type == MappingType.MLP:
            self.clip_project = MLP((768, self.gpt_embedding_size // 2,
                                     self.gpt_embedding_size))
        else:
            self.clip_project = TransformerMapper(prefix_size, self.gpt_embedding_size, prefix_length,
                                                  clip_length, num_layers)
        at, bt, ct, att, btt, ctt = alpha_schedule(self.time_step, N=self.num_classes)  # alpha schedule
        at = torch.tensor(at.astype('float64'))
        bt = torch.tensor(bt.astype('float64'))
        ct = torch.tensor(ct.astype('float64'))
        log_at = torch.log(at)
        log_bt = torch.log(bt)
        log_ct = torch.log(ct)
        att = torch.tensor(att.astype('float64'))
        btt = torch.tensor(btt.astype('float64'))
        ctt = torch.tensor(ctt.astype('float64'))
        log_cumprod_at = torch.log(att)
        log_cumprod_bt = torch.log(btt)
        log_cumprod_ct = torch.log(ctt)

        log_1_min_ct = log_1_min_a(log_ct)
        log_1_min_cumprod_ct = log_1_min_a(log_cumprod_ct)

        assert log_add_exp(log_ct, log_1_min_ct).abs().sum().item() < 1.e-5
        assert log_add_exp(log_cumprod_ct, log_1_min_cumprod_ct).abs().sum().item() < 1.e-5

        self.diffusion_acc_list = [0] * self.time_step
        self.diffusion_keep_list = [0] * self.time_step
        # Convert to float32 and register buffers.
        self.register_buffer('log_at', log_at.float())
        self.register_buffer('log_bt', log_bt.float())
        self.register_buffer('log_ct', log_ct.float())
        self.register_buffer('log_cumprod_at', log_cumprod_at.float())
        self.register_buffer('log_cumprod_bt', log_cumprod_bt.float())
        self.register_buffer('log_cumprod_ct', log_cumprod_ct.float())
        self.register_buffer('log_1_min_ct', log_1_min_ct.float())
        self.register_buffer('log_1_min_cumprod_ct', log_1_min_cumprod_ct.float())

    def q_pred(self, log_x_start, t):  # q(xt|x0)
        # log_x_start can be onehot or not
        t = (t + (self.time_step + 1)) % (self.time_step + 1)
        log_cumprod_at = extract(self.log_cumprod_at, t, log_x_start.shape)  # at~
        log_cumprod_bt = extract(self.log_cumprod_bt, t, log_x_start.shape)  # bt~
        log_cumprod_ct = extract(self.log_cumprod_ct, t, log_x_start.shape)  # ct~
        log_1_min_cumprod_ct = extract(self.log_1_min_cumprod_ct, t, log_x_start.shape)  # 1-ct~

        # log_probs = torch.zeros(log_x_start.size()).type_as(log_x_start)
        p1 = log_add_exp(log_x_start[:, :-1, :] + log_cumprod_at, log_cumprod_bt)
        p2 = log_add_exp(log_x_start[:, -1:, :] + log_1_min_cumprod_ct, log_cumprod_ct)

        return torch.cat([p1, p2], dim=1)

    def log_sample_categorical(self, logits):  # use gumbel to sample onehot vector from log probability
        uniform = torch.rand_like(logits)
        gumbel_noise = -torch.log(-torch.log(uniform + 1e-30) + 1e-30)
        sample = (gumbel_noise + logits).argmax(dim=1)
        log_sample = index_to_log_onehot(sample, self.num_classes)
        return log_sample

    def q_sample(self, log_x_start, t):  # diffusion step, q(xt|x0) and sample xt
        log_EV_qxt_x0 = self.q_pred(log_x_start, t)

        log_sample = self.log_sample_categorical(log_EV_qxt_x0)

        return log_sample

    def q_posterior(self, log_x_start, log_x_t, t):  # p_theta(xt_1|xt) = sum(q(xt-1|xt,x0)*p(x0|xt))
        # notice that log_x_t is onehot
        # log(p_theta(xt_1|xt)) = log(q(xt-1|xt,x0)) + log(p(x0|xt))
        #                       = log(p(x0|xt)) + log(q(xt|xt_1,x0)) + log(q(xt_1|x0)) - log(q(xt|x0))  (*)
        assert t.min().item() >= 0 and t.max().item() < self.time_step
        batch_size = log_x_start.size()[0]
        onehot_x_t = log_onehot_to_index(log_x_t)
        mask = (onehot_x_t == self.num_classes - 1).unsqueeze(1)
        log_one_vector = torch.zeros(batch_size, 1, 1).type_as(log_x_t)
        log_zero_vector = torch.log(log_one_vector + 1.0e-30).expand(-1, -1, log_x_start.size(-1))

        # log(q(xt|x0))
        log_qt = self.q_pred(log_x_t, t)
        log_qt = torch.cat((log_qt[:, :-1, :], log_zero_vector), dim=1)
        log_cumprod_ct = extract(self.log_cumprod_ct, t, log_x_start.shape)  # ct~
        ct_cumprod_vector = log_cumprod_ct.expand(-1, self.num_classes - 1, -1)
        ct_cumprod_vector = torch.cat((ct_cumprod_vector, log_one_vector), dim=1)
        log_qt = (~mask) * log_qt + mask * ct_cumprod_vector

        # log(q(xt|xt_1,x0))
        log_qt_one_timestep = self.q_pred_one_timestep(log_x_t, t)  # q(xt|xt_1)
        log_qt_one_timestep = torch.cat((log_qt_one_timestep[:, :-1, :], log_zero_vector), dim=1)
        log_ct = extract(self.log_ct, t, log_x_start.shape)  # ct
        ct_vector = log_ct.expand(-1, self.num_classes - 1, -1)
        ct_vector = torch.cat((ct_vector, log_one_vector), dim=1)
        log_qt_one_timestep = (~mask) * log_qt_one_timestep + mask * ct_vector

        q = log_x_start - log_qt  # log(p(x0|xt)/q(xt|x0))
        q_log_sum_exp = torch.logsumexp(q, dim=1, keepdim=True)
        q = q - q_log_sum_exp  # norm(log(p(x0|xt)/q(xt|x0)))  to leverage self.q_pred
        log_EV_xtmin_given_xt_given_xstart = self.q_pred(q,
                                                         t - 1) + log_qt_one_timestep + q_log_sum_exp  # get (*), last term is re-norm
        return torch.clamp(log_EV_xtmin_given_xt_given_xstart, -70, 0)

    def q_pred_one_timestep(self, log_x_t, t):  # q(xt|xt_1)
        log_at = extract(self.log_at, t, log_x_t.shape)  # at
        log_bt = extract(self.log_bt, t, log_x_t.shape)  # bt
        log_ct = extract(self.log_ct, t, log_x_t.shape)  # ct
        log_1_min_ct = extract(self.log_1_min_ct, t, log_x_t.shape)  # 1-ct

        # log_probs = torch.zeros(log_x_t.size()).type_as(log_x_t)
        p1 = log_add_exp(log_x_t[:, :-1, :] + log_at, log_bt)
        p2 = log_add_exp(log_x_t[:, -1:, :] + log_1_min_ct, log_ct)

        return torch.cat([p1, p2], dim=1)


class CaptionPrefix(CaptionModel):

    def parameters(self, recurse: bool = True):
        return self.clip_project.parameters()

    def train(self, mode: bool = True):
        super(CaptionPrefix, self).train(mode)
        self.gpt.eval()
        return self
