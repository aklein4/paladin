import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import PreTrainedModel
from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from transformers.activations import ACT2FN

from model.mage import MAGEBlock, MAGEModel
from model.model_utils import get_timestep_embedding

from utils import DotDict


class MultiPassConfig(GPT2Config):
    def __init__(
        self,
        z_dim=8,
        t_dim=32,
        init_scale=0.02,
        **kwargs
    ):
        self.z_dim = z_dim
        self.t_dim = t_dim
        
        super().__init__(**kwargs)
        self.model_type = "multipass"


class MultiPassBlock(MAGEBlock):
    
    def init_subclass_modules(self, config):
        self.z_proj = nn.Linear(config.z_dim, self.cond_dim, bias=False)
        self.t_proj = nn.Linear(config.t_dim, self.cond_dim, bias=False)

        self.z_proj.weight.data.normal_(std=config.init_scale)
        self.t_proj.weight.data.normal_(std=config.init_scale)

    def get_cond(self, x, z, t):
        
        t = get_timestep_embedding(
            t, self.config.t_dim, max_period=1
        )

        z = self.z_proj(z)
        t = self.t_proj(t)

        return z + t


class MultiPassModel(MAGEModel):
    _block_module = MultiPassBlock


class MultiPassEncoder(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
    
        self.transformer = GPT2Model(config)
        self.z_head = nn.Linear(config.hidden_size, config.z_dim)


    def forward(self, input_ids):
        out = self.transformer(
            input_ids=input_ids,
        ).last_hidden_state

        z = self.z_head(out)

        # constant kl with unit gaussian for given t
        z = z / z.norm(dim=-1, keepdim=True)

        # shift z to the left
        z_out = torch.zeros_like(z)
        z_out[:, :-1] = z[:, 1:]

        return z_out
    

    def prepare_training(self):
        self.requires_grad_(True)
        self.transformer.wte.requires_grad_(False)
        self.transformer.wpe.requires_grad_(False)

        p = list(self.z_head.parameters())
        p += list(self.transformer.h.parameters())
        p += list(self.transformer.ln_f.parameters())

        return p


class MultiPassDecoder(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
    
        self.transformer = MultiPassModel(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)


    def forward(self, input_ids, z, t, memory=None):

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