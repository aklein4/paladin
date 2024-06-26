import torch

from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from loader.single_loader import SingleLoader
from loader.full_loader import FullLoader

from model.mage import MAGEModelLM
from training.mage_trainer import MAGETrainer

import constants as constants


MODEL_URL = "openai-community/gpt2"

TRAIN_DATA_URL = 'JeanKaddour/minipile' # 'EleutherAI/the_pile_deduplicated'
VAL_DATA_URL = 'JeanKaddour/minipile'

NAME = "mage-beta"

TRAIN_CONFIG = {
    "lr": 3e-5,
    "bs": 8,
    "num_steps": 10000,
    "warmup_steps": 1000,
    "eval_freq": 1000,
    "checkpoint_freq": 5000,
    "dtype": torch.bfloat16,
    "max_length": 1024,
    "memory_grad": False,
    "max_eval_examples": 500
}


MODEL_CONFIG = {
    "attn_pdrop": 0.0,
    "embd_pdrop": 0.0,
    "resid_pdrop": 0.0,
    "summary_first_dropout": 0.0
}


def main():
    
    print("Loading models...")
    tokenizer = GPT2TokenizerFast.from_pretrained(MODEL_URL)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    gpt = GPT2LMHeadModel.from_pretrained(MODEL_URL)
    config = gpt.config
    for k, v in MODEL_CONFIG.items():
        setattr(config, k, v)
    model = MAGEModelLM(config)
    model.load_gpt2(gpt)

    # prevent tied weights
    model.load_state_dict({k: v.clone() for k, v in model.state_dict().items()})

    model = model.to(constants.DEVICE)

    _ = torch.compile(model, mode="reduce-overhead", fullgraph=True)

    print("Loading data...")
    train_loader = SingleLoader(TRAIN_DATA_URL, train=True, debug=False)
    val_loader = FullLoader(VAL_DATA_URL, train=False, debug=False)

    print("Train!")
    trainer = MAGETrainer(
        NAME,
        **TRAIN_CONFIG
    )
    trainer.train(
        tokenizer,
        model,
        train_loader,
        val_loader
    )


if __name__ == "__main__":
        main()
