import torch
import torch.nn as nn
import torch.nn.functional as F


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
    