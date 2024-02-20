import torch

from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from model.paladin import PaladinModel
import constants as constants


TEST_MODEL = 'openai-community/gpt2'


@torch.no_grad()
def paladin_sanity_check(url):

    print("Creating models for sanity check...")
    tokenizer = GPT2TokenizerFast.from_pretrained(TEST_MODEL)
    model = GPT2LMHeadModel.from_pretrained(TEST_MODEL)

    paladin = PaladinModel.from_gpt2lm( 
        model,
        {"latent_size": 256, "z_window": 16, "context_length": 128, "prompt_length": 8},
        debug=True
    )

    model = model.to(constants.DEVICE).to(constants.DTYPE)

    model = model.eval()
    paladin = paladin.eval()

    inputs = tokenizer("This is a test", return_tensors="pt").to(constants.DEVICE)

    print("Running sanity check...")
    model_out = model.transformer(**inputs, output_attentions=True, output_hidden_states=True)
    memory_out = paladin.memory(inputs.input_ids)
    encoder_out = paladin.encoder(inputs.input_ids)
    decoder_out = paladin.decoder(
        **inputs, memory=memory_out.memory,
        z=paladin.mu_head(encoder_out.output),
        j=torch.zeros_like(inputs.input_ids)
    )

    try:
        diff = torch.max(torch.abs(model_out.last_hidden_state - memory_out.output))
        assert diff < 1e-4, f"Memory mismatch! ({diff.item()})"

        diff = torch.max(torch.abs(model_out.last_hidden_state - encoder_out.output))
        assert diff < 1e-4, f"Encoder mismatch! ({diff.item()})"

        diff = torch.max(torch.abs(model_out.last_hidden_state - decoder_out.output))
        assert diff < 1e-4, f"Decoder mismatch! ({diff.item()})"

        diff = torch.max(torch.abs(model.lm_head(model_out.last_hidden_state) - paladin.lm_head(decoder_out.output)))
        assert diff < 1e-4, f"Decoder logit mismatch! ({diff.item()})"

    except AssertionError as e:
        print(e)
        del model
        del paladin
        return

    print("Model sanity check successful!")

    del model
    del paladin


if __name__ == "__main__":
    paladin_sanity_check(TEST_MODEL)
