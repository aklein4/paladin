import torch

import numpy as np
import argparse
from tqdm import tqdm

from transformers import GPT2TokenizerFast

from model.paladin import PaladinModel
import constants as constants


MODEL_URL = r"C:\Users\adam3\.cache\huggingface\hub\models--aklein4--paladin-128-highkl\snapshots\40ff294fc0c8cd3cd10f4485a62581cb701af8eb"
TOKENIZER_URL = "openai-community/gpt2"


@torch.no_grad()
def sample_test(tokenizer, paladin, prompt, mode, t=0):
    paladin.eval()
    paladin.disable_cache()
    paladin.memory.enable_cache()

    x = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=paladin.context_length).to(constants.DEVICE).input_ids

    if t == 0:
        print("Context:")
        print(tokenizer.decode(x[0, :-paladin.z_window]))

        print("Original Output:")
        print(tokenizer.decode(x[0, -paladin.z_window:]))

    # calculate z
    encoder_out = paladin.encode(x)
    z = paladin.sample_z(encoder_out.mus, encoder_out.sigmas, mode=mode)
    z = z[:, -1:]

    z_other = torch.randn_like(z)
    z = z * (1-t) + z_other * t

    curr = x[:, :-paladin.z_window]
    memory = None

    for i in tqdm(range(paladin.z_window), "Sampling"):
        curr_in = curr if i == 0 else curr[:, -1:]

        mem_out = paladin.memorize(curr_in)
        if memory is None:
            memory = mem_out
        else:
            memory = torch.cat([memory, mem_out], dim=-2)

        decoder_out = paladin.decode(
            curr,
            memory=memory,
            z=z.expand(-1, curr.shape[1], -1),
            j=torch.full_like(curr, paladin.z_window-i-1)
        )

        curr = torch.cat(
            [curr, torch.argmax(decoder_out[:, -1:], dim=-1)],
            dim=-1
        )

    print("Generated Output:")
    print(tokenizer.decode(curr[0, -paladin.z_window:]))


def main(args):
    tokenizer = GPT2TokenizerFast.from_pretrained(TOKENIZER_URL)
    paladin = PaladinModel.from_pretrained(MODEL_URL)

    sample_test(tokenizer, paladin, args.text, args.mode)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--mode", type=str, default="mode")
    args = parser.parse_args()

    main(args)