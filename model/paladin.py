import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from transformers import PreTrainedModel
from transformers.models.gpt2.configuration_gpt2 import GPT2Config

from model.base import BaseModel
from model.decoder import DecoderModel

from utils import DotDict
import constants as constants


class PaladinConfig(GPT2Config):
    model_type = "paladin"

    def __init__(
        self,
        latent_size=128,
        z_window=16,
        **kwargs
    ):
        self.latent_size = latent_size
        self.z_window = z_window

        super().__init__(**kwargs)


class PaladinModel(PreTrainedModel):
    config_class = PaladinConfig

    def __init__(self, config):
        super().__init__(config)
        self.latent_size = config.latent_size
        self.z_window = config.z_window

        # encoder + its head
        self.encoder = BaseModel(config)
        self.z_head = nn.Linear(config.hidden_size, config.latent_size, bias=False)

        # decoder + language head
        self.decoder_m = BaseModel(config)
        self.decoder_age = DecoderModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

        # cache vars
        self.cache_enabled = False


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


    def sample_z_j(
        self, z, padding_mask, generator=None
    ):
        bs, seq_len, _ = z.shape

        j = torch.randint(
            0, self.z_window, [bs, seq_len],
            generator=generator, dtype=torch.long,
        ).to(z.device)

        return DotDict(
            z=z_out,
            j=i_z,
            w_pos=w_pos
        )


    @staticmethod
    def from_gpt2lm(model, args, debug=False):

        # create config
        config = model.config
        for k, v in args.items():
            config.__setattr__(k, v)

        # create model from config
        out = PaladinModel(config)

        # split up base model
        gpt2 = model.transformer
        lm_head = model.lm_head

        # load transformer weights
        out.encoder.load_gpt2(gpt2)
        out.memory.load_gpt2(gpt2)
        out.decoder.load_gpt2(gpt2)

        # load language prediction weights
        out.lm_head.load_state_dict(lm_head.state_dict())

        # format
        out = out.to(constants.DEVICE).to(constants.DTYPE)

        # compile for speed
        if not debug:
            _ = torch.compile(out, mode="reduce-overhead", fullgraph=True)

        return out


    @staticmethod
    def from_pretrained_compile(*args, **kwargs):
        out = PaladinModel.from_pretrained(*args, **kwargs).to(constants.DEVICE).to(constants.DTYPE)

        _ = torch.compile(out, mode="reduce-overhead", fullgraph=True)

        return out
    

    def enable_cache(self):
        self.cache_enabled = True
        for m in [self.encoder, self.memory, self.decoder]:
            m.enable_cache()

    def disable_cache(self):
        self.cache_enabled = False
        for m in [self.encoder, self.memory, self.decoder]:
            m.disable_cache()

    def reset_cache(self):
        for m in [self.encoder, self.memory, self.decoder]:
            m.reset_cache()
