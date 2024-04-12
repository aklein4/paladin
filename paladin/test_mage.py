import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import GPT2Model, GPT2TokenizerFast

from modeling.mage import MAGEModel, MAGEBlock
from modeling.layers import ConditionGate
import utils.constants as constants


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


class ConditionBlock(MAGEBlock):
    def init_subclass_modules(self, config):
        self.ln_cond = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.cond_gate = ConditionGate(config.hidden_size, 100, 250, config)

    def subforward(self, x, cond):
        x_cond = self.cond_gate(
            self.ln_cond(x),
            cond
        )
        x = x + x_cond

        return x

class ConditionModel(MAGEModel):
    _block_module = ConditionBlock


@torch.no_grad()
def mage_test_condition(tokenizer, model):

    mage = ConditionModel(model.config).to(constants.DEVICE)
    mage.load_gpt2(model)
    mage = mage.eval()

    inputs = tokenizer("This is also a test", return_tensors="pt").to(constants.DEVICE)
    cond = torch.randn(1, inputs.input_ids.shape[1], 100).to(constants.DEVICE)

    model_out = model(**inputs, output_attentions=True, output_hidden_states=True).last_hidden_state
    mage_out = mage(inputs.input_ids, cond=cond)
    mage_out_2 = mage(inputs.input_ids, memory=mage_out.memory, cond=cond)

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

    # print("\nRUNNING: mage_test_extras")
    # mage_test_condition(tokenizer, model)
    # print("PASSED: mage_test_extras")

    print("\nAll tests passed!")
