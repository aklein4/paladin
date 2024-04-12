import torch
import torch.nn as nn
import torch.nn.functional as F

from model.base import BaseBlock, BaseModel
from model.layers import OneXAttention
from model.model_utils import get_timestep_embedding
from utils import DotDict


class DecoderBlock(BaseBlock):
    """ A decoder block based on BaseBlock.
     - Adds projection of t, the noise time.
     - Adds embedding of j, the position within the z window.
     - Cross attends to a z encoding vector.
    """

    def __init__(self, config, layer_idx=None):
        super().__init__(config, layer_idx=layer_idx)

        # projection of t
        self.proj_t = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        # j embedding
        self.wje = nn.Embedding(config.z_window, self.hidden_size)

        # attend to z
        self.attn_z = OneXAttention(self.hidden_size, config.z_size, config)


    def subforward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        j: torch.Tensor,
        z: torch.Tensor
    ) -> torch.Tensor:
        """ Extra operations for the decoder block.

        Args:
            x (torch.Tensor): Input tensor [B, S, D].
            t (torch.Tensor): Noise time [B, S], float in [0, 1].
            j (torch.Tensor): z position [B, S], long in [0, z_window).
            z (torch.Tensor): latent embedding [B, S, z_size].

        Returns:
            torch.Tensor: Output tensor [B, S, D].
        """
        
        # add t emb
        x_t = self.proj_t(
            get_timestep_embedding(
                t, self.hidden_size, max_period=1.0
            )
        )
        x = x + x_t

        # add j emb
        x_j = self.wje(j)
        x = x + x_j

        # attend to z
        x_z = self.attn_z(x, z)
        x = x + x_z

        return x


class DecoderModel(BaseModel):
    """ A decoder model based on BaseModel.
     - Uses DecoderBlock.
    """
    _block_module = DecoderBlock