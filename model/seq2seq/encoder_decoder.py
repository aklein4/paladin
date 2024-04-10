import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import PreTrainedModel
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention, GPT2MLP

from model.model_utils import get_gpt2_padding_mask
from utils import DotDict


class EncoderBlock(nn.Module):
    """
    Encoder block based on gpt2.
     - Uses noncausal self-attention.
     - Handles paddding masks.
    """

    def __init__(self, config, layer_idx=None):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.inner_dim = config.n_inner if config.n_inner is not None else 4 * self.hidden_size

        # self attn layer
        self.ln_1 = nn.LayerNorm(self.hidden_size, eps=config.layer_norm_epsilon)
        self.attn = GPT2Attention(
            config, layer_idx=layer_idx,
            is_cross_attention=False
        )

        # ff layer
        self.ln_2 = nn.LayerNorm(self.hidden_size, eps=config.layer_norm_epsilon)
        self.mlp = GPT2MLP(self.inner_dim, config)


    def forward(
        self,
        x: torch.Tensor,
        padding_mask=None,
    ) -> torch.Tensor:
        """ Forward pass of the block with self-attention.
         - padding mask should be bool tensor with same shape as x (True where padding)
         - if padding mask is not passed, it defaults to no padding

        Args:
            x (torch.Tensor): Input tensor [B, S, D]
            paddding_mask (torch.Tensor, optional): Padding tensor [B, S, D]. Defaults to None.

        Returns:
            torch.Tensor: Output tensor [B, S, D]
        """
        if padding_mask is None:
            padding_mask = torch.zeros_like(x).bool()

        # self attn layer
        mask = get_gpt2_padding_mask(x, x, padding_mask, self.attn.num_heads)
        x_attn = self.attn(
            self.ln_1(x),
            encoder_attention_mask=mask,
        )[0]
        x = x + x_attn

        # ff layer
        x_ff = self.mlp(
            self.ln_2(x)
        )
        x = x + x_ff

        return x


class MAGEModel(PreTrainedModel):
    """ MAGE model based on gpt2.
     - Uses MAGEBlock instead of GPT2Block.
     - Can be easily modified to use different block types.
    """

    # change this to the desired block type
    _block_module = MAGEBlock

    def __init__(self, config):
        super().__init__(config)

        self.embed_dim = config.hidden_size

        # token embeddings
        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        # position embeddings
        self.wpe = nn.Embedding(config.max_position_embeddings, self.embed_dim)

        # transformer
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
        """ Preprocess the input ids for input to the transformer.

        Args:
            input_ids (torch.LongTensor): Token ids [B, S].
            token_type_ids (optional): Special types. Defaults to None.
            position_ids (optional): Token positions. Defaults to sequential.

        Returns:
            torch.Tensor: _description_
        """
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
        memory=None,
        **kwargs
    ) -> torch.Tensor:
        """ Forward pass of the model.
         - Accumulates memory at each layer.
         - Can use external memory for cross-attention.

        Args:
            input_ids (torch.LongTensor): Token ids [B, S].
            token_type_ids (optional): Special types. Defaults to None.
            position_ids (optional): Token positions. Defaults to sequential.
            memory (torch.Tensor, optional): External memory [B, S, D]. Defaults to None.
            kwargs: Extra arguments to be passed to subforward.

        Returns:
            torch.Tensor: _description_
        """
        x = self.pre_forward(input_ids, token_type_ids, position_ids)

        # run model, save memory
        m = []
        for i, block in enumerate(self.h):
            m.append(x)

            x = block(
                x,
                memory=(memory[i] if memory is not None else None),
                **kwargs
            )
        
        # post process
        m = torch.stack(m)
        x = self.ln_f(x)

        return DotDict(
            output=x,
            memory=m,
        )


    @torch.no_grad()
    def load_gpt2(self, gpt2):
        """ Load the weights of a gpt2 model into the current model.

        Args:
            gpt2 (GPT2Model): GPT2 model to load from.
        """

        # load everything except c_attn matrix (our blocks have different weights)
        state_to_load = gpt2.state_dict()
        state_to_load = {k: v.clone() for k, v in state_to_load.items() if k.split(".")[-2] != "c_attn"}
        
        # load all available weights
        self.load_state_dict(state_to_load, strict=False)

        # load the extra attention block weights
        with torch.no_grad():
            for i in range(len(self.h)):
                self.h[i].load_gpt2_extras(gpt2.h[i])
