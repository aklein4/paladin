import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from transformers import GPT2TokenizerFast

from modeling.archmage import ArchMAGE
from modeling.model_utils import add_noise
import utils.constants as constants


TOKENIZER_URL = 'openai-community/gpt2'
MODEL_URL = 'aklein4/archmage-beta'

PROMPT = "This is a test"


@torch.no_grad()
def main():
    tokenizer = GPT2TokenizerFast.from_pretrained(TOKENIZER_URL)
    model = ArchMAGE.from_pretrained(MODEL_URL, subfolder="model")

    model = model.to(constants.DEVICE)
    model.eval()
    # _ = torch.compile(model, mode="reduce-overhead", fullgraph=True)

    inputs = tokenizer(PROMPT, return_tensors="pt").to(constants.DEVICE)

    z = model.encode(inputs.input_ids)
    noise = torch.randn_like(z)

    memory = model(inputs.input_ids, z, torch.zeros_like(z[:, :, 0])).memory

    p_store = []
    for t in np.linspace(0.0, 1.0, 10):

        t_tensor = torch.ones_like(z)[:, :, 0] * t
        z_noisy = add_noise(z, t_tensor, noise)

        logits = model(inputs.input_ids, z_noisy, t_tensor, memory=memory).logits
        logits = logits[0, -2]
        p = F.softmax(logits, dim=-1)

        p_store.append(p)
    p_store = torch.stack(p_store).cpu().numpy()

    top_inds = np.argsort(p_store[-1])[-10:]

    for i, t in enumerate(np.linspace(0.0, 1.0, 10)):
        plt.plot(p_store[i][top_inds], label=f"t={t:.2f}")

    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()