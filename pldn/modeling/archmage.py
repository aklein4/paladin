import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from transformers import PreTrainedModel
from transformers.models.gpt2.configuration_gpt2 import GPT2Config

from modeling.mage import MAGEModel, MAGEBlock
from modeling.model_utils import get_timestep_embedding, add_noise
from utils.data_utils import DotDict


class ArchMAGEConfig(GPT2Config):
    def __init__(
        self,
        z_scale=1.0,
        t_dim=32,
        **kwargs
    ):
        self.z_scale = z_scale
        self.t_dim = t_dim

        super().__init__(**kwargs)
        self.model_type = "archmage"


class ArchMAGEBLock(MAGEBlock):
    def init_subclass_modules(self, config):
        self.z_proj = nn.Linear(config.hidden_size, self.cond_dim, bias=False)
        self.t_proj = nn.Linear(config.t_dim, self.cond_dim, bias=False)
        
        self.z_proj.weight.data.zero_()
        self.t_proj.weight.data.zero_()
    
    def get_cond(self, x, z, t):
        t = get_timestep_embedding(t, self.config.t_dim)
        
        z = self.z_proj(z)
        t = self.t_proj(t)

        return z + t

class ArchMAGETransformer(MAGEModel):
    _block_module = ArchMAGEBLock


class ArchMAGE(PreTrainedModel):
    """ Linear-flow language model based on MAGE.
    """
    config_class = ArchMAGEConfig

    def __init__(self, config):
        super().__init__(config)
        self.z_scale = config.z_scale

        self.transformer = ArchMAGETransformer(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)


    def encode(self, input_ids):
        z = self.transformer.wte(input_ids)
        z = z / z.norm(dim=-1, keepdim=True)
        z = z * self.z_scale * np.sqrt(z.shape[-1])

        # shift z to the left
        z_out = torch.zeros_like(z)
        z_out[:, :-1] = z[:, 1:]

        return z_out
    

    def forward(
        self,
        input_ids,
        z, t,
        memory=None,
    ):
        out = self.transformer(
            input_ids=input_ids,
            memory=memory,
            z=z, t=t
        )

        # use the same weights as the input embeddings
        logits = self.lm_head(out.output)
        logits = F.log_softmax(logits, dim=-1)

        return DotDict(
            logits=logits,
            memory=out.memory,
        )


    @torch.no_grad()
    def get_x0(self, logits):
        p = F.softmax(logits, dim=-1).unsqueeze(-1)

        codebook = self.transformer.wte.weight.detach()
        while codebook.dim() < p.dim():
            codebook = codebook.unsqueeze(0)

        return (p * codebook).sum(dim=-2)


    @torch.no_grad()
    def get_normal_logits(self, xt, t):
        if not isinstance(t, float):
            t = t.unsqueeze(-1)

        codebook = self.transformer.wte.weight.detach()
        while codebook.dim() < xt.dim()+1:
            codebook = codebook.unsqueeze(0)
        
        xt = xt.unsqueeze(-2).expand(*([-1]*(xt.dim()-1)), codebook.shape[-2], -1)

        dist = torch.distributions.Normal(
            codebook * (1-t),
            t
        )
        normal_logits = dist.log_prob(xt).sum(dim=-1)

        return normal_logits


    @torch.no_grad()
    def get_adj_logits(self, xt, logits, t):
        normal_logits = self.get_normal_logits(xt, t)

        adj_logits = logits - normal_logits
        adj_logits = F.log_softmax(adj_logits, dim=-1)

        return adj_logits


    def prepare_training(self, memory_grad):
        if memory_grad:
            self.transformer.enable_memory_grad()
        else:
            self.transformer.disable_memory_grad()

        self.requires_grad_(True)
        self.transformer.wte.requires_grad_(False)
        self.transformer.wpe.requires_grad_(False)
        self.lm_head.requires_grad_(False)
        if not memory_grad:
            self.transformer.h.requires_grad_(False)

        p = list(self.transformer.h_agent.parameters())
        p += list(self.transformer.ln_f.parameters())
        if memory_grad:
            p += list(self.transformer.h.parameters())

        return p
    

    @torch.no_grad()
    def load_gpt2(self, gpt2):
        self.transformer.load_gpt2(gpt2.transformer)
        self.lm_head.load_state_dict(gpt2.lm_head.state_dict())
        