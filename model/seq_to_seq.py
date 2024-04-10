import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from transformers import PreTrainedModel
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from transformers.activations import ACT2FN

from model.base import BaseModel, BaseBlock
from model.decoder import DecoderModel

from utils import DotDict
import constants as constants


class Seq2SeqDecoderBlock(nn.Module):
    _disable_ff = True

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)

        # modified ff to include z
        self.ln_2 = nn.LayerNorm(self.hidden_size, eps=config.layer_norm_epsilon)
        self.ln_z = nn.LayerNorm(config.z_size, eps=config.layer_norm_epsilon)
        
        self.proj_up = nn.Linear(self.hidden_size + config.z_size, self.inner_dim)
        self.act = ACT2FN[config.activation_function]
        self.ff_dropout = nn.Dropout(config.resid_pdrop)
        self.proj_down = nn.Linear(self.inner_dim, self.hidden_size)


    def subforward(
        self, x, z
    ):
        x_ln = self.ln_2(x)
        z_ln = self.ln_z(z)

        z_ln = z_ln.unsqueeze(1).expand(-1, x.shape[1], -1)
        xz = torch.cat([x_ln, z_ln], dim=-1)

        h = self.proj_up(xz)
        h = self.act(h)
        h = self.ff_dropout(h)
        h = self.proj_down(h)

        return x + h


class Seq2SeqDecoderModel(BaseModel):
    _disable_ff = True


class Seq2SeqConfig(GPT2Config):
    def __init__(
        self,
        z_size=128,
        **kwargs
    ):
        self.z_size = z_size

        super().__init__(**kwargs)
        self.model_type = "seq2seq"


class Seq2SeqModel(PreTrainedModel):
    config_class = Seq2SeqConfig

    def __init__(self, config):
        super().__init__(config)
        self.latent_size = config.latent_size

        # latent model + its head
        self.z_model = BaseModel(config, causal=False, use_cls=True)
        self.z_head = nn.Linear(config.hidden_size, config.latent_size, bias=False)

        # encoder
        self.encoder = BaseModel(config, causal=False)

        # decoder + language head
        self.decoder_m = Seq2SeqDecoderModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()


    def get_z(self, input_ids):
        pass


    def encode(self, input_ids):
        """ Encode the input ids into the latent space.
         - z is normalized to have zero mean and unit variance.
        
        Args:
            input_ids: token ids [B, S]

        Returns:
            z: latent representation [B, S, latent_size]
        """
        output = self.encoder(input_ids).output

        z = self.z_head(output)
        z = z - z.mean(dim=-1, keepdim=True)
        z = z / z.std(dim=-1, keepdim=True)

        return z


    def decode(self, input_ids, t, j, z):
        """ Decode the input ids and latents into lm_logits.

        Args:
            input_ids: token ids [B, S]
            t: noise times [B, S], float in [0, 1]
            j: z positions [B, S], long in [0, z_window)
            z: latents [B, S, latent_size]

        Returns:
            lm_logits: language model logits [B, S, vocab_size]
        """
        memory = self.decoder_m(input_ids).memory

        output = self.decoder_age(
            input_ids,
            memory=memory,
            t=t, j=j, z=z
        )

        lm_logits = self.lm_head(output)
        lm_logits = F.log_softmax(lm_logits, dim=-1)

        return lm_logits

