import torch


def gpt2_causal_mask(x, num_heads):
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