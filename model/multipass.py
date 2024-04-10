import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import PreTrainedModel
from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from transformers.models.gpt2.configuration_gpt2 import GPT2Config

from model.mage import MAGEBlock, MAGEModel
from model.model_utils import get_timestep_embedding

from utils import DotDict


class MultiPassConfig(GPT2Config):
    def __init__(
        self,
        z_dim=8,
        t_dim=32,
        **kwargs
    ):
        self.z_dim = z_dim
        self.t_dim = t_dim
        
        super().__init__(**kwargs)
        self.model_type = "multipass"


class MultiPassBlock(MAGEBlock):
    
    def get_extra_dim(self, config):
        return config.z_dim + config.t_dim


    def subforward(self, x, proj_extras, emb_extras):
        
        # spectral embed t
        z, t = tuple(proj_extras)
        t = get_timestep_embedding(
            t, self.config.t_dim, max_period=1
        )
        proj_extras = [z, t]

        return x, proj_extras, emb_extras


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
        z = z - z.mean(dim=-1, keepdim=True)
        z = z / z.std(dim=-1, keepdim=True)

        # shift z to the left
        z_out = torch.zeros_like(z)
        z_out[:, :-1] = z[:, 1:]

        return z
    

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


    def forward(self, input_ids, z, t, memory=None):

        out = self.transformer(
            input_ids=input_ids,
            memory=memory,
            proj_extras=[z, t],
        )

        # use the same weights as the input embeddings
        logits = F.linear(out.output, self.transformer.wte.weight)
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
        if not memory_grad:
            self.transformer.h.requires_grad_(False)

        p = list(self.transformer.parameters())
        p += list(self.transformer.h_agent.parameters())
        p += list(self.transformer.ln_f.parameters())
        if memory_grad:
            p += list(self.transformer.h.parameters())

        return p