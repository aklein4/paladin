from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.pytorch_utils import Conv1D
from transformers.activations import ACT2FN


class MAGEMLP(nn.Module):
    def __init__(
            self,
            intermediate_size: int,
            extra_dim: int,
            config
        ):
        """ A transformer MLP layer for MAGE blocks that can
        take extra inputs.

        Args:
            intermediate_size (int): Size of transformer ff layer.
            extra_dim (int): Total dimension of extra inputs.
            config (GPT2Config-like): Config for the transformer.
        """
        super().__init__()

        self.intermediate_size = intermediate_size
        self.embed_dim = config.hidden_size
        self.extra_dim = extra_dim
        self.full_dim = self.embed_dim + self.extra_dim

        self.c_fc = Conv1D(intermediate_size, self.full_dim)
        self.c_proj = Conv1D(self.embed_dim, intermediate_size)
        
        self.act = ACT2FN[config.activation_function]
        self.dropout = nn.Dropout(config.resid_pdrop)


    def forward(
            self,
            x: torch.FloatTensor,
            proj_extras: List[torch.FloatTensor],
            emb_extras: List[torch.FloatTensor]
        ) -> torch.FloatTensor:
        """ Forward pass of the MAGEMLP layer.

        Args:
            x (torch.FloatTensor): Input to the layer [B, S, D].
            proj_extras (List[torch.FloatTensor]): List of extra inputs [B, S, <D>] projected into the ff layer.
            emb_extras (List[torch.FloatTensor]): List of extra inputs [B, S, D] embedded into the ff layer.

        Returns:
            torch.FloatTensor: Output of the layer [B, S, D].
        """
        for i in range(len(proj_extras)):
            s = proj_extras[i].shape
            assert s[:2] == x.shape[:2], f"Shape mismatch for proj_extra {i}: {s[:2]} != {x.shape[:2]}"
            
        targ_s = (x.shape[0], x.shape[1], self.intermediate_size)
        for i in range(len(emb_extras)):
            s = tuple(emb_extras[i].shape)
            assert s == targ_s, f"Shape mismatch for emb_extra {i}: {s} != {targ_s}"

        h = torch.cat([x] + proj_extras, dim=-1)

        h = self.c_fc(h)
        for emb in emb_extras:
            h = h + emb

        h = self.act(h)
        h = self.c_proj(h)
        h = self.dropout(h)

        return h


    @torch.no_grad()
    def load_gpt2(
        self,
        gpt2: nn.Module
    ):
        """ Load the weights of a GPT2MLP into this layer.

        Args:
            gpt2_mlp (nn.Module): GPT2MLP to load.
        """

        # easy loading
        self.act.load_state_dict(gpt2.act.state_dict())
        self.dropout.load_state_dict(gpt2.dropout.state_dict())
        self.c_proj.load_state_dict(gpt2.c_proj.state_dict())

        # initial conv needs zero padding
        self.c_fc.bias = gpt2.c_fc.bias
        self.c_fc.weight.data[:self.embed_dim] = gpt2.c_fc.weight.data.clone()


    @torch.no_grad()
    def init_control(
        self,
        scales: torch.FloatTensor
    ):
        """ Initialize the control parameters of the MAGEMLP.

        Args:
            scales (torch.FloatTensor): Scales for the control parameters [extra_dim]
        """
        n = torch.randn_like(self.c_fc.weight.data[self.embed_dim:])
        if isinstance(scales, (float, int)):
            n *= scales
        else:
            n *= scales.unsqueeze(-1).to(n.device).to(n.dtype)
        self.c_fc.weight.data[self.embed_dim:] = n


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
