from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.pytorch_utils import Conv1D
from transformers.activations import ACT2FN


class MAGEMLP(nn.Module):
    def __init__(
            self,
            intermediate_size: int,
            config
        ):
        """ A transformer MLP layer for MAGE blocks that can
        addd extra conditions to the intermediate layer.

        Args:
            intermediate_size (int): Size of transformer ff layer.
            config (GPT2Config-like): Config for the transformer.
        """
        super().__init__()

        self.intermediate_size = intermediate_size
        self.embed_dim = config.hidden_size

        self.c_fc = Conv1D(intermediate_size, self.embed_dim)
        self.c_proj = Conv1D(self.embed_dim, intermediate_size)
        
        self.act = ACT2FN[config.activation_function]
        self.dropout = nn.Dropout(config.resid_pdrop)


    def forward(
            self,
            x: torch.FloatTensor,
            cond: Optional[torch.FloatTensor] = None
        ) -> torch.FloatTensor:
        """ Forward pass of the MAGEMLP layer.

        Args:
            x (torch.FloatTensor): Input to the layer [B, S, D].
            cond (Optional[torch.FloatTensor]): Extra conditions [B, S, D_inter]. Defaults to None.

        Returns:
            torch.FloatTensor: Output of the layer [B, S, D].
        """

        h = self.c_fc(x)
        if cond is not None:
            h = h + cond

        h = self.act(h)
        h = self.c_proj(h)
        h = self.dropout(h)

        return h


class ConditionGate(nn.Module):

    def __init__(self, hidden_size, cond_size, inter_size, config):
        super().__init__()

        self.hidden_size = hidden_size
        self.cond_size = cond_size
        self.inter_size = inter_size

        self.q_proj = nn.Linear(hidden_size, inter_size)
        self.k_proj = nn.Linear(cond_size, inter_size)
        
        self.out_proj = nn.Linear(inter_size, hidden_size, bias=False)
        self.out_proj.weight.data.zero_()

        self.act = ACT2FN[config.activation_function]
        self.dropout = nn.Dropout(config.resid_pdrop)


    def forward(self, x, cond):
        q = self.act(self.q_proj(x))
        k = self.act(self.k_proj(cond))
        
        v = q * k

        out = self.out_proj(v)
        out = self.dropout(out)

        return out


class OneXAttention(nn.Module):
    
    def __init__(self, hidden_size, other_size, config):
        super().__init__()

        self.hidden_size = hidden_size
        self.other_size = other_size

        self.ln_x = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.ln_o = nn.LayerNorm(other_size, eps=config.layer_norm_epsilon)

        self.q = nn.Linear(hidden_size, other_size)
        self.k = nn.Linear(other_size, other_size)
        self.b = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.v = nn.Linear(other_size, other_size)

        self.proj = nn.Linear(other_size, hidden_size)
        self.proj.weight.data.zero_()
        self.proj.bias.data.zero_()

        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.proj_dropout = nn.Dropout(config.resid_pdrop)


    def forward(self, x, other):
        x = self.ln_x(x)
        other = self.ln_o(other)

        q = self.q(x)
        k = self.k(other)
        v = self.v(other)

        attn = torch.sigmoid(q * k + self.b)
        attn = self.attn_dropout(attn)

        out = self.proj(attn * v)
        out = self.proj_dropout(out)

        return out
