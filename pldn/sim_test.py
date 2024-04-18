import torch

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

N = 32
D = 8

NUM_TRIALS = 1000
NUM_STEPS = 50

def main():
    
    codes = torch.randn(N, D)
    codes /= torch.norm(codes, dim=-1, keepdim=True)
    codes *= np.sqrt(D)

    p = torch.randn(N)
    p = torch.softmax(p, dim=-1)

    order = torch.argsort(p, descending=True)
    codes = codes[order]
    p = p[order]

    counts = torch.zeros(N)
    for _ in tqdm(range(NUM_TRIALS)):

        x = torch.randn(D)
        for t in np.linspace(0.0, 1.0, NUM_STEPS)[::-1][:-1]:

            dist = torch.distributions.Normal(
                codes * (1-t),
                t
            )
            log_probs = dist.log_prob(x[None].repeat(N, 1))

            logits = log_probs.sum(dim=-1) + torch.log(p)
            probs = torch.softmax(logits, dim=-1)

            x0 = (codes * probs[:, None]).sum(dim=0)
            x1 = x0 + (x - x0)/t
            v = x0 - x1

            x = x + v/NUM_STEPS
        
        dists = torch.norm(codes - x[None], dim=-1)
        counts[torch.argmin(dists)] += 1
    
    counts /= counts.sum()

    plt.plot(p.numpy(), label="p")
    plt.plot(counts.numpy(), label="samples")

    plt.legend()
    plt.show()


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()