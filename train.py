
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from loader.single_loader import SingleLoader
from loader.full_loader import FullLoader
from model.paladin import PaladinModel
from training.paladin_trainer import PaladinTrainer


MODEL_URL = "openai-community/gpt2"
TRAIN_DATA_URL = 'EleutherAI/the_pile_deduplicated'
VAL_DATA_URL = 'JeanKaddour/minipile'

TRAIN_INIT_CONFIG = {
    "lr": 1e-5,
    "eval_freq": 100,
    "save_freq": 10000,
    "kl_limit": 256,
    "kl_penalty": 1.0
}

TRAIN_RUN_CONFIG = {
    "num_steps": 60000,
    "warmup_steps": 5000, # 5000
    "batch_size": 2,
    "seed": 0
}

MODEL_CONFIG = {
    "context_length": 128,
    "prompt_length": 8,
    "z_window": 16,

    "latent_size": 256,

    "attn_pdrop": 0.0,
    "embd_pdrop": 0.0,
    "resid_pdrop": 0.0,
    "summary_first_dropout": 0.0,
}


def main():
    
    tokenizer = GPT2TokenizerFast.from_pretrained(MODEL_URL)
    
    # paladin = PaladinModel.from_pretrained_wrapper(
    #         os.path.join(PATH, "epoch_1", "checkpoint", "paladin")
    # )
    paladin = PaladinModel.from_gpt2lm(
        GPT2LMHeadModel.from_pretrained(MODEL_URL),
        MODEL_CONFIG,
    )

    print(paladin.device)

    # optimizer_state = torch.load(
    #         os.path.join(PATH, "epoch_1", "checkpoint", "optimizer.pt")
    # )
    optimizer_state = None

    train_loader = SingleLoader(TRAIN_DATA_URL, train=True, debug=False)
    val_loader = FullLoader(VAL_DATA_URL, train=False, debug=False)

    trainer = PaladinTrainer(
        "el4_test",
        tokenizer,
        init_log=True,
        **TRAIN_INIT_CONFIG
    )

    trainer.train(
        paladin,
        train_loader,
        val_loader,
        optimizer_state=optimizer_state,
        **TRAIN_RUN_CONFIG
    )


if __name__ == "__main__":
        main()
