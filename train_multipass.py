import torch

from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from loader.single_loader import SingleLoader
from loader.full_loader import FullLoader

from model.multipass import MultiPassEncoder, MultiPassDecoder
from training.mutipass_trainer import MultiPassTrainer

import constants as constants


ENCODER_URL = "distilbert/distilgpt2"
DECODER_URL = "openai-community/gpt2"

TRAIN_DATA_URL = 'JeanKaddour/minipile' # 'EleutherAI/the_pile_deduplicated'
VAL_DATA_URL = 'JeanKaddour/minipile'

NAME = "multipass-beta"

TRAIN_CONFIG = {
    "lr": 3e-5,
    "bs": 8,
    "num_steps": 20000,
    "warmup_steps": 1000,
    "eval_freq": 100,
    "checkpoint_freq": 5000,
    "dtype": torch.bfloat16,
    "max_length": 1024,
    "memory_grad": False,
    "max_eval_examples": 500
}


MODEL_CONFIG = {
    "z_dim": 8,
    "t_dim": 32,
    "init_scale": 0.0,
    "attn_pdrop": 0.0,
    "embd_pdrop": 0.0,
    "resid_pdrop": 0.0,
    "summary_first_dropout": 0.0
}


def main():
    
    print("Loading models...")
    tokenizer = GPT2TokenizerFast.from_pretrained(ENCODER_URL)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    gpt = GPT2LMHeadModel.from_pretrained(ENCODER_URL)
    config = gpt.config
    for k, v in MODEL_CONFIG.items():
        setattr(config, k, v)
    encoder = MultiPassEncoder(config)
    encoder.transformer.load_state_dict(gpt.transformer.state_dict())

    gpt = GPT2LMHeadModel.from_pretrained(DECODER_URL)
    config = gpt.config
    for k, v in MODEL_CONFIG.items():
        setattr(config, k, v)
    decoder = MultiPassDecoder(config)
    decoder.load_gpt2(gpt)

    # prevent tied weights
    encoder.load_state_dict({k: v.clone() for k, v in encoder.state_dict().items()})
    decoder.load_state_dict({k: v.clone() for k, v in decoder.state_dict().items()})

    encoder = encoder.to(constants.DEVICE)
    decoder = decoder.to(constants.DEVICE)

    _ = torch.compile(encoder, mode="reduce-overhead", fullgraph=True)
    _ = torch.compile(decoder, mode="reduce-overhead", fullgraph=True)

    print("Loading data...")
    train_loader = SingleLoader(TRAIN_DATA_URL, train=True, debug=False)
    val_loader = FullLoader(VAL_DATA_URL, train=False, debug=False)

    print("Train!")
    trainer = MultiPassTrainer(
        NAME,
        **TRAIN_CONFIG
    )
    trainer.train(
        tokenizer,
        encoder,
        decoder,
        train_loader,
        val_loader
    )


if __name__ == "__main__":
        main()
