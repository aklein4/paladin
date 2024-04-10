import torch

import math


def get_gpt2_causal_mask(x, num_heads, causal=True):
    mask = torch.ones(
        x.shape[0],
        num_heads,
        x.shape[1],
        x.shape[1],
        device=x.device,
        dtype=x.dtype
    )

    mask = torch.triu(mask, diagonal=1) * torch.finfo(mask.dtype).min
    return mask


def get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: int = 10000,
):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models: Create sinusoidal timestep embeddings.

    :param timesteps: a tensor of any shape og_shape.
    :param embedding_dim: the dimension of the output. :param max_period: controls the minimum frequency of the
    embeddings. :return: an [og_shape, dim] Tensor of positional embeddings.
    """
    og_shape = timesteps.shape
    timesteps = timesteps.view(-1)

    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(
        start=0, end=half_dim, dtype=torch.float32, device=timesteps.device
    )
    exponent = exponent / (half_dim - downscale_freq_shift)

    emb = torch.exp(exponent)
    emb = timesteps[:, None].float() * emb[None, :]

    # scale embeddings
    emb = scale * emb

    # concat sine and cosine embeddings
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

    # flip sine and cosine embeddings
    if flip_sin_to_cos:
        emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)

    # zero pad
    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))

    emb = emb.view(*og_shape, -1)
    return emb


def add_noise(
    x: torch.Tensor,
    t: torch.Tensor,
    noise: torch.Tensor
):
    """ Add noise to the input tensor.
     - Uses simple linear noise mixing.

    Args:
        x (torch.Tensor): Input tensor [B, S, D].
        t (torch.Tensor): Noise time [B, S], float in [0, 1].
        noise (torch.Tensor): Noise tensor [B, S, D].

    Returns:
        torch.Tensor: Noisy tensor [B, S, D].
    """
    t = t[:, :, None]
    return (1 - t) * x + t * noise