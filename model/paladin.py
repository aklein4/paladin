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
        context_length=128,
        prompt_length=8,
        **kwargs
    ):
        self.latent_size = latent_size
        self.z_window = z_window
        self.context_length=context_length
        self.prompt_length=prompt_length

        super().__init__(**kwargs)


class PaladinModel(PreTrainedModel):
    config_class = PaladinConfig

    def __init__(self, config):
        super().__init__(config)
        self.context_length = config.context_length
        self.z_window = config.z_window
        self.prompt_length = config.prompt_length

        # encoder + its heads
        self.encoder = BaseModel(config)
        self.mu_head = nn.Linear(config.hidden_size, config.latent_size)
        self.sigma_head = nn.Linear(config.hidden_size, config.latent_size)

        # encoder memory (no head)
        self.memory = BaseModel(config)

        # decoder + language head
        self.decoder = DecoderModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # useful for position calculations
        pos = torch.arange(0, config.context_length)[None].long()
        self.register_buffer("pos", pos, persistent=False)

        # remove the prompt from the context length
        out_length = config.context_length - config.prompt_length
        pos_out = torch.arange(0, out_length)[None].long()
        pos_fake = torch.arange(0, config.prompt_length)[None].long()

        # the z locations that the given input token could use
        l_locs = torch.clip(pos_out + 1, min=config.z_window, max=out_length-1)
        r_locs = torch.clip(pos_out + config.z_window + 1, max=out_length)

        l_locs = torch.cat([pos_fake+1, l_locs+config.prompt_length], dim=-1)
        r_locs = torch.cat([pos_fake+2, r_locs+config.prompt_length], dim=-1)
        
        self.register_buffer("l_locs", l_locs, persistent=False)
        self.register_buffer("r_locs", r_locs, persistent=False)

        # importance sampling weights for each position
        w_pos = (r_locs - l_locs).float() / config.z_window
        w_pos[:, :config.prompt_length] = 0
        self.register_buffer("w_pos", w_pos[:, :-1], persistent=False)

        # Initialize weights and apply final processing
        self.post_init()


    def sample_z_for_training(
        self, mus, sigmas, mode,
        generator=None, np_generator=None
    ):
        if np_generator is None:
            np_generator = np.random.default_rng()

        bs = mus.shape[0]
        z = self.sample_z(mus, sigmas, mode, generator=generator)

        pos = self.pos.expand(bs, -1)
        l_locs = self.l_locs.expand(bs, -1)
        r_locs = self.r_locs.expand(bs, -1)
        w_pos = self.w_pos.expand(bs, -1)

        # really stupid randint workaround
        i_z_abs = torch.from_numpy(
                np_generator.integers(
                    low=l_locs.detach().cpu().numpy(),
                    high=r_locs.detach().cpu().numpy()
                )
        ).to(constants.DEVICE).long()
        i_z = i_z_abs - pos - 1

        # fix the last throw-away
        i_z[:, -1] = 0

        # construct the shuffled z input
        z_out = torch.zeros_like(z)
        for i in range(bs):
            z_out[i] += z[i][i_z_abs[i]]

        return DotDict(
            z=z_out,
            j=i_z,
            w_pos=w_pos
        )


    def sample_z(self, mus, sigmas, mode, generator=None):
        noise = torch.randn(
                *mus.shape, device=mus.device, dtype=mus.dtype,
                generator=generator
        )
        if mode == "uncond":
            return noise
        elif mode == "sample":
            return mus + (sigmas * noise)
        elif mode == "mode":
            return mus
        else:
            raise ValueError("Invalid sampling mode!")


    def _post_encoder(self, x):
        # tuned to init with just under 1 kl per dim
        mus = self.mu_head(x) / 10
        sigmas = 1 + F.elu(self.sigma_head(x)/10 - 1) + 1e-4
        return mus, sigmas

    def encode(self, *args, **kwargs):
        x = self.encoder(*args, **kwargs).output
        mus, sigmas = self._post_encoder(x)
        return DotDict(
            mus=mus,
            sigmas=sigmas
        )


    def memorize(self, *args, **kwargs):
        return self.memory(*args, **kwargs).memory


    def decode(self, *args, **kwargs):
        x = self.decoder(*args, **kwargs).output
        return self.lm_head(x)


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
    def from_pretrained_wrapper(*args, **kwargs):
        debug = kwargs.pop("debug", False)

        out = PaladinModel.from_pretrained(*args, **kwargs).to(DEVICE).to(DTYPE)

        if not debug:
            _ = torch.compile(out, mode="reduce-overhead", fullgraph=True)

        return out
