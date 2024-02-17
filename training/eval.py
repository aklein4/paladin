import torch

import numpy as np


@torch.no_grad()
def get_eval_metrics(logits, logp, input_ids, prompt_length=0):
    logits = logits[:, prompt_length:]
    logp = logp[:, prompt_length:]
    input_ids = input_ids[:, prompt_length:]
    
    # calculate the npb and perplexity
    bpb = -logp.mean() / np.log(2)
    ppl = torch.exp(-logp.mean(-1)).mean()

    # calculate the accuracy
    acc = (
        torch.argmax(logits[:, :-1], dim=-1) ==
        input_ids[:, 1:]
    ).float().mean()

    return ppl.item(), bpb.item(), acc.item()