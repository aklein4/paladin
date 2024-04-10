import torch

from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from loader.single_loader import SingleLoader
from loader.full_loader import FullLoader

from model.multipass import MultiPassEncoder, MultiPassDecoder
from training.mutipass_trainer import MultiPassTrainer

import constants as constants


ENCODER_URL = "distilbert/distilgpt2"
DECODER_URL = "distilbert/distilgpt2"

TRAIN_DATA_URL = 'JeanKaddour/minipile' # 'EleutherAI/the_pile_deduplicated'
VAL_DATA_URL = 'JeanKaddour/minipile'


TRAIN_CONFIG = {
    "lr": 1e-5,
    "bs": 3,
    "num_steps": 1000,
    "warmup_steps": 100,
    "eval_freq": 100,
    "checkpoint_freq": 500,
    "dtype": torch.float32,
    "max_length": 32,
    "memory_grad": False
}


MODEL_CONFIG = {
    "z_dim": 8,
    "t_dim": 32,
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

    encoder.load_state_dict({k: v.clone() for k, v in encoder.state_dict().items()})
    decoder.load_state_dict({k: v.clone() for k, v in decoder.state_dict().items()})

    encoder = encoder.to(constants.DEVICE)
    decoder = decoder.to(constants.DEVICE)

    print("Loading data...")
    train_loader = SingleLoader(TRAIN_DATA_URL, train=False, debug=True)
    val_loader = FullLoader(VAL_DATA_URL, train=False, debug=True)

    print("Training!")
    trainer = MultiPassTrainer(
        "multipass-test",
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
