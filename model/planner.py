import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from transformers import PreTrainedModel

from model.base import BaseModel
from model.diffuser import DiffuserModel
from model.paladin import PaladinConfig

from utils import DotDict
import constants as constants


class PaladinModel(PreTrainedModel):
    config_class = PaladinConfig

    def __init__(self, config):
        super().__init__(config)
        self.context_length = config.context_length
        self.z_window = config.z_window
        self.prompt_length = config.prompt_length

        # memory (no head)
        self.memory = BaseModel(config)

        # diffuser + latent head
        self.decoder = DiffuserModel(config)
        self.head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # useful for position calculations
        pos = torch.arange(0, config.context_length)[None].long()
        self.register_buffer("pos", pos, persistent=False)

        targ_locs = torch.clip(self.pos + config.z_window, max=config.context_length-1)
        self.register_buffer("targ_locs", targ_locs, persistent=False)

        # importance sampling weights for each position
        w_targ = torch.ones([config.context_length])[None].float()
        w_targ[:, :config.prompt_length] = 0
        w_targ[:, -config.z_window:] = 0
        self.register_buffer("w_pos", w_pos[:, :-1], persistent=False)

        # Initialize weights and apply final processing
        self.post_init()

        self.cache_enabled = False


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
