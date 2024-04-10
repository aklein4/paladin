from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import PreTrainedModel
from transformers.models.gpt2.modeling_gpt2 import GPT2Block, GPT2Attention

from model.layers import MAGEMLP
from model.model_utils import get_gpt2_causal_mask
from utils import DotDict


class MAGEBlock(nn.Module):
    """
    MAGE block based on gpt2.
     - Uses cross-attention instead of self-attention.
        - KEY IDEA: if the tensor we cross-attend to is the same as the input, we get self-attention!
     - Can add extra sequence-independent inputs in subclasses.
     - attention is always causal.
    """

    def get_extra_dim(self, config):
        # used to determine the extra dim for the mlp
        return 0
    
    def init_subclass_modules(self, config):
        # used to initialize extra modules in subclasses
        pass


    def __init__(self, config, layer_idx=None):
        super().__init__()
        self.config = config

        self.hidden_size = config.hidden_size
        self.inner_dim = config.n_inner if config.n_inner is not None else 4 * self.hidden_size
        self.extra_dim = self.get_extra_dim(config)

        # attn layer
        self.ln_1 = nn.LayerNorm(self.hidden_size, eps=config.layer_norm_epsilon)
        self.ln_mem = nn.LayerNorm(self.hidden_size, eps=config.layer_norm_epsilon)
        self.attn = GPT2Attention(
            config, layer_idx=layer_idx,
            is_cross_attention=True
        )

        # ff layer
        self.ln_2 = nn.LayerNorm(self.hidden_size, eps=config.layer_norm_epsilon)
        self.mlp = MAGEMLP(self.inner_dim, self.extra_dim, config)

        # initialize extra modules
        self.init_subclass_modules(config)
        

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        proj_extras: Optional[List[torch.Tensor]]=[],
        emb_extras: Optional[List[torch.Tensor]]=[]
    ) -> torch.Tensor:
        """ Forward pass of the block with.
         - x and memory should be the same size.
         - extras can be used to pass extra inputs to the block.
         - subforward is called between attention and ff layers to modify x and extras.

        Args:
            x (torch.Tensor): Input tensor [B, S, D]
            memory (torch.Tensor): Memory tensor [B, S, D]
            proj_extras (List[torch.Tensor], optional): List of extra inputs [B, S, <D>] projected into the ff layer. Defaults to [].
            emb_extras (List[torch.Tensor], optional): List of extra inputs [B, S, D] embedded into the ff layer. Defaults to [].

        Returns:
            torch.Tensor: Output tensor [B, S, D]
        """

        # cross (self) attn layer
        x_attn = self.attn(
            self.ln_1(x),
            encoder_hidden_states=self.ln_mem(memory),
            encoder_attention_mask=get_gpt2_causal_mask(x, self.attn.num_heads),
        )[0]
        x = x + x_attn

        # extra handling
        x, proj_extras, emb_extras = self.subforward(x, proj_extras, emb_extras)

        # ff layer
        x_ff = self.mlp(
            self.ln_2(x),
            proj_extras,
            emb_extras
        )
        x = x + x_ff

        return x


    # to be overwridden by subclasses
    def subforward(
        self,
        x: torch.Tensor,
        proj_extras: List[torch.Tensor],
        emb_extras: List[torch.Tensor]
    ) -> torch.Tensor:
        """ Subforward pass of the block.
         - Can be used to modify x and extras before the ff layer.

        Args:
            x (torch.Tensor): Input tensor [B, S, D].
            proj_extras (List[torch.Tensor]): List of extra inputs [B, S, <D>] projected into the ff layer.
            emb_extras (List[torch.Tensor]): List of extra inputs [B, S, D] embedded into the ff layer.

        Returns:
            torch.Tensor: Modified x.
        """
        return x, proj_extras, emb_extras
    

    @torch.no_grad()
    def load_gpt2(self, block):
        """ Load the weights of a gpt2 block.

        Args:
            block: GPT2Block to load from.
        """

        # copy the ln layers
        self.ln_1.load_state_dict(block.ln_1.state_dict())
        self.ln_mem.load_state_dict({k: v.clone() for k, v in block.ln_1.state_dict().items()})
        self.ln_2.load_state_dict(block.ln_2.state_dict())

        # copy what we can from the attentions
        state = block.attn.state_dict()
        state = {k: v.clone() for k, v in state.items() if "c_attn" not in k}
        self.attn.load_state_dict(state, strict=False)

        # copy the attention matrices
        self.attn.q_attn.weight.data = block.attn.c_attn.weight[:, :block.attn.embed_dim].data.clone()
        self.attn.q_attn.bias.data = block.attn.c_attn.bias[:block.attn.embed_dim].data.clone()

        self.attn.c_attn.weight.data = block.attn.c_attn.weight[:, block.attn.embed_dim:].data.clone()
        self.attn.c_attn.bias.data = block.attn.c_attn.bias[block.attn.embed_dim:].data.clone()

        # copy the mlp
        self.mlp.load_gpt2(block.mlp)


class MAGEModel(PreTrainedModel):
    """ MAGE model based on gpt2.
     - Uses MAGEBlock instead of GPT2Block.
     - Can be easily modified to use different block types.
    """

    def _init_weights(*args, **kwargs):
        # need this for some versions for transformers?
        pass

    # change this to the desired block type
    _block_module = MAGEBlock

    def __init__(self, config):
        super().__init__(config)

        self.embed_dim = config.hidden_size

        # token embeddings
        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        # position embeddings
        self.wpe = nn.Embedding(config.max_position_embeddings, self.embed_dim)

        # io modules
        self.drop = nn.Dropout(config.embd_pdrop)
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

        # blocks
        self.h = nn.ModuleList([GPT2Block(config, layer_idx=i) for i in range(config.num_hidden_layers)])
        self.h_agent = nn.ModuleList([self._block_module(config, layer_idx=i) for i in range(config.num_hidden_layers)])

        # Initialize weights and apply final processing
        self.post_init()

        # whether to compute gradients for memory
        self.memory_grad = True


    def enable_memory_grad(self):
        self.memory_grad = True
    
    def disable_memory_grad(self):
        self.memory_grad = False


    def pre_forward(
        self,
        input_ids: torch.LongTensor,
    ) -> torch.Tensor:
        """ Preprocess the input ids for input to the transformer.

        Args:
            input_ids (torch.LongTensor): Token ids [B, S].

        Returns:
            torch.Tensor: Input tokens [B, S, D].
        """
        batch_size, seq_size = input_ids.shape[0], input_ids.shape[1]
        device = input_ids.device

        # make sure we have positions
        position_ids = torch.arange(seq_size, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0)

        # get token + position embs
        inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        x = inputs_embeds + position_embeds

        # apply input dropout
        x = self.drop(x)        

        return x


    def forward(
        self,
        input_ids: torch.LongTensor,
        memory: Optional[torch.Tensor]=None,
        proj_extras: Optional[List[torch.Tensor]]=[],
        emb_extras: Optional[List[torch.Tensor]]=[],
    ) -> torch.Tensor:
        """ Forward pass of the model.
         - Accumulates memory, then uses agent to get output.
         - Can pass extra inputs to the agent.
         - Can take precomputed memory.

        Args:
            input_ids (torch.LongTensor): Token ids [B, S].
            memory (torch.Tensor, optional): Pre-computed memory [B, S, D]. Defaults to None.
            proj_extras (List[torch.Tensor], optional): List of extra inputs [B, S, <D>] projected into the ff layer. Defaults to [].
            emb_extras (List[torch.Tensor], optional): List of extra inputs [B, S, D] embedded into the ff layer. Defaults to [].

        Returns:
            DotDict: Output tensor [B, S, D], memory tensor [L, B, S, D].
        """
        x = self.pre_forward(input_ids)

        # calculate memory
        if memory is None:
            print(self.memory_grad)
            with torch.set_grad_enabled(self.memory_grad):
                m = x
                memory = [m]

                for block in self.h:
                    m = block(m)[0]
                    memory.append(m)

                memory = torch.stack(memory)

        # calculate output
        for i, block in enumerate(self.h_agent):
            x = block(
                x,
                memory[i],
                proj_extras,
                emb_extras
            )

        # post process
        x = self.ln_f(x)

        return DotDict(
            output=x,
            memory=memory,
        )


    @torch.no_grad()
    def load_gpt2(self, gpt2):
        """ Load the weights of a gpt2 model into the current model.

        Args:
            gpt2 (GPT2Model): GPT2 model to load from.
        """

        # embeddings
        self.wte.load_state_dict(gpt2.wte.state_dict())
        self.wpe.load_state_dict(gpt2.wpe.state_dict())

        # io modules
        self.drop.load_state_dict(gpt2.drop.state_dict())
        self.ln_f.load_state_dict(gpt2.ln_f.state_dict())

        # copy blocks since these params are reused
        state = {k: v.clone() for k, v in gpt2.h.state_dict().items()}
        self.h.load_state_dict(state)

        # custom blocks
        for i in range(len(self.h_agent)):
            self.h_agent[i].load_gpt2(gpt2.h[i])
