import torch

import math


def get_gpt2_causal_mask(x, num_heads):
    mask = torch.ones(
        x.shape[0],
        num_heads,
        x.shape[1],
        x.shape[1],
        device=x.device,
        dtype=x.dtype
    )

    return torch.triu(mask, diagonal=1) * torch.finfo(mask.dtype).min


def get_gpt2_padding_mask(tgt, mem, padding_mask, num_heads):
    mask = torch.ones(
        tgt.shape[0],
        num_heads,
        tgt.shape[1],
        mem.shape[1],
        device=tgt.device,
        dtype=tgt.dtype
    )

    # one where there is padding, with extra dim for heads and tgt_len
    padding_mask = padding_mask[:, None, None, :].to(mask.dtype).to(mask.device)

    # -inf where there is padding
    mask = mask * padding_mask * torch.finfo(mask.dtype).min

    return mask


def get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: float = 1.0,
):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models: Create sinusoidal timestep embeddings.

    :param timesteps: a tensor of any shape og_shape.
    :param embedding_dim: the dimension of the output. :param max_period: controls the minimum frequency of the
    embeddings. :return: an [og_shape, dim] Tensor of positional embeddings.
    """
    og_shape = timesteps.shape
    timesteps = timesteps.reshape(-1)

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

    emb = emb.reshape(*og_shape, -1)
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
        t (torch.Tensor): Noise time [B, S], float in [0, 1]. Can also be float.
        noise (torch.Tensor): Noise tensor [B, S, D].

    Returns:
        torch.Tensor: Noisy tensor [B, S, D].
    """
    if not isinstance(t, float):
        t = t.unsqueeze(-1)
    return (1 - t) * x + t * noise


def flow_step(
    xt: torch.Tensor,
    t: torch.Tensor,
    pred_x0: torch.Tensor,
    dt: torch.Tensor
):
    """ Perform a single flow step.

    Args:
        xt (torch.Tensor): Input tensor [B, S, D].
        t (torch.Tensor): Time tensor [B, S], float in [0, 1]. Can also be float.
        pred_x0 (torch.Tensor): Predicted x0 tensor [B, S, D].
        dt (torch.Tensor): Time size [B, S], float in [0, 1]. Can also be float.
        
    Returns:
        torch.Tensor: Output tensor [B, S, D].
    """
    if isinstance(t, float):
        assert t > 0
    else:
        t = t.unsqueeze(-1)
        assert (t > 0).all()

    if isinstance(dt, float):
        assert dt > 0
    else:
        dt = dt.unsqueeze(-1)
        assert (dt > 0).all()

    x1 = pred_x0 + (xt - pred_x0) / t
    v = pred_x0 - x1

    return xt + v * dt
