import torch

from transformers import GPT2Model, GPT2TokenizerFast

from model.mage import MAGEModel, MAGEBlock
import constants as constants


TEST_URL = 'openai-community/gpt2'


@torch.no_grad()
def test_mage_basic(tokenizer, model):

    mage = MAGEModel(model.config).to(constants.DEVICE)
    mage.load_gpt2(model)
    mage = mage.eval()

    inputs = tokenizer("This is a test", return_tensors="pt").to(constants.DEVICE)

    model_out = model(**inputs, output_attentions=True, output_hidden_states=True).last_hidden_state
    mage_out = mage(inputs.input_ids)
    mage_out_2 = mage(inputs.input_ids, memory=mage_out.memory)

    diff = torch.max(torch.abs(model_out - mage_out.output))
    assert diff < 1e-4, f"Output mismatch! ({diff.item()})"

    diff = torch.max(torch.abs(model_out - mage_out_2.output))
    assert diff < 1e-4, f"Cached output mismatch! ({diff.item()})"


class ExtraBlock(MAGEBlock):
    def get_extra_dim(self, config):
        return 100

class ExtraModel(MAGEModel):
    _block_module = ExtraBlock


@torch.no_grad()
def mage_test_extras(tokenizer, model):

    mage = ExtraModel(model.config).to(constants.DEVICE)
    mage.load_gpt2(model)
    mage = mage.eval()

    inputs = tokenizer("This is also a test", return_tensors="pt").to(constants.DEVICE)
    extras = [
        torch.randn(inputs.input_ids.shape[0], inputs.input_ids.shape[1], 33).to(constants.DEVICE),
        torch.randn(inputs.input_ids.shape[0], inputs.input_ids.shape[1], 67).to(constants.DEVICE),
    ]

    model_out = model(**inputs, output_attentions=True, output_hidden_states=True).last_hidden_state
    mage_out = mage(inputs.input_ids, proj_extras=extras)
    mage_out_2 = mage(inputs.input_ids, memory=mage_out.memory, proj_extras=extras)

    diff = torch.max(torch.abs(model_out - mage_out.output))
    assert diff < 1e-4, f"Output mismatch! ({diff.item()})"

    diff = torch.max(torch.abs(model_out - mage_out_2.output))
    assert diff < 1e-4, f"Cached output mismatch! ({diff.item()})"


if __name__ == "__main__":
    
    print("Loading reference model...")
    tokenizer = GPT2TokenizerFast.from_pretrained(TEST_URL)
    model = GPT2Model.from_pretrained(TEST_URL).to(constants.DEVICE)
    model.eval()

    print("\nRUNNING: mage_test_basic")
    test_mage_basic(tokenizer, model)
    print("PASSED: mage_test_basic")

    print("\nRUNNING: mage_test_extras")
    mage_test_extras(tokenizer, model)
    print("PASSED: mage_test_extras")

    print("\nAll tests passed!")
