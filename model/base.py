import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import PreTrainedModel
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention, GPT2MLP

from model.model_utils import gpt2_causal_mask
from utils import DotDict


class BaseBlock(nn.Module):

    def __init__(self, config, layer_idx=None):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.inner_dim = config.n_inner if config.n_inner is not None else 4 * self.hidden_size

        # self attn
        self.ln_1 = nn.LayerNorm(self.hidden_size, eps=config.layer_norm_epsilon)
        self.ln_mem = nn.LayerNorm(self.hidden_size, eps=config.layer_norm_epsilon)
        self.attn = GPT2Attention(
            config, layer_idx=layer_idx,
            is_cross_attention=True
        )

        # ff layer
        self.ln_2 = nn.LayerNorm(self.hidden_size, eps=config.layer_norm_epsilon)
        self.mlp = GPT2MLP(self.inner_dim, config)


    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor=None,
        **kwargs
    ) -> torch.Tensor:
        if memory is None:
            memory = x

        # (self) cross attention
        x_attn = self.attn(
            self.ln_1(x),
            encoder_hidden_states=self.ln_mem(memory),
            encoder_attention_mask=gpt2_causal_mask(memory, self.attn.num_heads)
        )[0] # output_attn: a, present, (attentions)
        x = x + x_attn

        # child extras
        x = self.subforward(x, **kwargs)

        # ff layer
        x_ff = self.mlp(
            self.ln_2(x)
        )
        x = x + x_ff

        return x


    # to be overwridden by subclasses
    def subforward(
        self,
        x: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        return x
    

    @torch.no_grad()
    def load_gpt2_extras(self, block):

        # copy the attention matrices
        self.attn.q_attn.weight.data = block.attn.c_attn.weight[:, :block.attn.embed_dim].data.clone()
        self.attn.q_attn.bias.data = block.attn.c_attn.bias[:block.attn.embed_dim].data.clone()
        
        self.attn.c_attn.weight.data = block.attn.c_attn.weight[:, block.attn.embed_dim:].data.clone()
        self.attn.c_attn.bias.data = block.attn.c_attn.bias[block.attn.embed_dim:].data.clone()

        # copy the normalizations
        self.ln_mem.weight.data = block.ln_1.weight.data.clone()
        self.ln_mem.bias.data = block.ln_1.bias.data.clone()


class BaseModel(PreTrainedModel):
    _block_module = BaseBlock

    def __init__(self, config):
        super().__init__(config)

        self.embed_dim = config.hidden_size

        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        self.wpe = nn.Embedding(config.max_position_embeddings, self.embed_dim)

        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([self._block_module(config, layer_idx=i) for i in range(config.num_hidden_layers)])
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

        # Initialize weights and apply final processing
        self.post_init()


    def pre_forward(
        self,
        input_ids: torch.LongTensor,
        token_type_ids=None,
        position_ids=None,
    ) -> torch.Tensor:
        batch_size, seq_size = input_ids.shape[0], input_ids.shape[1]
        device = input_ids.device

        # make sure we have positions
        if position_ids is None:
            position_ids = torch.arange(seq_size, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0)

        # get token + position embs
        inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        x = inputs_embeds + position_embeds

        # special id embs (?)
        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            x += token_type_embeds

        # apply input dropout
        x = self.drop(x)        

        return x


    def forward(
        self,
        input_ids: torch.LongTensor,
        token_type_ids=None,
        position_ids=None,
        **kwargs
    ) -> torch.Tensor:
        x = self.pre_forward(input_ids, token_type_ids, position_ids)

        # run model, save memory
        m = []
        for i, block in enumerate(self.h):
            m.append(x)
            x = block(x)

        # post process
        m = torch.stack(m)
        x = self.ln_f(x)

        return DotDict(
            output=x,
            memory=m,
        )


    @torch.no_grad()
    def load_gpt2(self, gpt2):

        # cross attention uses different size c_attn matrix
        state_to_load = gpt2.state_dict()
        state_to_load = {k: v.clone() for k, v in state_to_load.items() if k.split(".")[-2] != "c_attn"}
        
        # load all available weights, some don't match
        self.load_state_dict(state_to_load, strict=False)

        # load the extra attention block weights
        with torch.no_grad():
            for i in range(len(self.h)):
                self.h[i].load_gpt2_extras(gpt2.h[i])
