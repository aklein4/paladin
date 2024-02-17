import torch
import torch.nn as nn
import torch.nn.functional as F

from model.base import BaseBlock, BaseModel
from model.layers import OneXAttention
from utils import DotDict


class DecoderBlock(BaseBlock):
    def __init__(self, config, layer_idx=None):
        super().__init__(config, layer_idx=layer_idx)

        # attend to z index
        self.attn_j = OneXAttention(self.hidden_size, self.hidden_size, config)

        # attend to z
        self.attn_z = OneXAttention(self.hidden_size, config.latent_size, config)


    def subforward(
        self,
        x: torch.Tensor,
        j: torch.Tensor,
        z: torch.Tensor
    ) -> torch.Tensor:
        
        # attend to z index
        x_j = self.attn_j(x, j)
        x += x_j

        # attend to z
        x_z = self.attn_z(x, z)
        x += x_z

        return x


class DecoderModel(BaseModel):
    _block_module = DecoderBlock

    def __init__(self, config):
        super().__init__(config)

        # embeddings for z index
        self.wje = nn.Embedding(config.z_window, self.embed_dim)


    def forward(
        self,
        input_ids: torch.LongTensor,
        memory: torch.Tensor,
        j: torch.LongTensor,
        z: torch.Tensor,
        token_type_ids=None,
        position_ids=None,
        **kwargs
    ) -> torch.Tensor:
        x = self.pre_forward(input_ids, token_type_ids, position_ids)

        assert torch.allclose(x, memory[0])

        j_in = self.wje(j)

        # run model
        for i, block in enumerate(self.h):
            x = block(x, memory=memory[i], j=j_in, z=z)

        # post process
        x = self.ln_f(x)

        return DotDict(
            output=x
        )
    
