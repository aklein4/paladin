import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.models.embeddings import Timesteps

from model.base import BaseBlock, BaseModel
from model.layers import OneXAttention
from utils import DotDict


class DiffuserBlock(BaseBlock):
    def __init__(self, config, layer_idx=None):
        super().__init__(config, layer_idx=layer_idx)

        # attend to noise time
        self.attn_nt = OneXAttention(self.hidden_size, self.hidden_size, config)

        # attend to z
        self.attn_z = OneXAttention(self.hidden_size, config.latent_size, config)


    def subforward(
        self,
        x: torch.Tensor,
        nt: torch.Tensor,
        z: torch.Tensor
    ) -> torch.Tensor:
        
        # attend to z index
        x_nt = self.attn_nt(x, nt)
        x = x + x_nt

        # attend to z
        x_z = self.attn_z(x, z)
        x = x + x_z

        return x


class DiffuserModel(BaseModel):
    _block_module = DiffuserBlock

    def __init__(self, config):
        super().__init__(config)

        # embeddings for z index
        self.wnte = Timesteps(self.embed_dim)


    def forward(
        self,
        input_ids: torch.LongTensor,
        memory: torch.Tensor,
        nt: torch.LongTensor,
        z: torch.Tensor,
        token_type_ids=None,
        position_ids=None,
        **kwargs
    ) -> torch.Tensor:
        x = self.pre_forward(input_ids, token_type_ids, position_ids)

        nt_in = self.wnte(nt.view(-1))
        nt_in = nt_in.view(*(nt.shape + (-1,)))

        # run model
        for i, block in enumerate(self.h):
            x = block(x, memory=memory[i], nt=nt_in, z=z)

        # post process
        x = self.ln_f(x)

        return DotDict(
            output=x
        )
    
