import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from modeling.mage import MAGEModel, MAGEModelLM
from modeling.archmage import ArchMAGE, ArchMAGEConfig
import utils.constants as constants


TEST_MODEL_URL = 'openai-community/gpt2'


@torch.no_grad()
def test_mage(tokenizer, model):

    mage = MAGEModel(model.config).to(constants.DEVICE)
    mage.load_gpt2(model)
    mage = mage.eval()

    inputs = tokenizer("This is a test", return_tensors="pt").to(constants.DEVICE)

    model_out = model(**inputs, output_hidden_states=True).last_hidden_state
    mage_out = mage(inputs.input_ids)
    mage_out_mem = mage(inputs.input_ids, memory=mage_out.memory)

    diff = torch.max(torch.abs(model_out - mage_out.output))
    assert diff < 1e-4, f"Output mismatch! ({diff.item()})"

    diff = torch.max(torch.abs(model_out - mage_out_mem.output))
    assert diff < 1e-4, f"Cached output mismatch! ({diff.item()})"


@torch.no_grad()
def test_mage_lm(tokenizer, model):

    magelm = MAGEModelLM(model.config).to(constants.DEVICE)
    magelm.load_gpt2(model)
    magelm = magelm.eval()

    inputs = tokenizer("This is a test", return_tensors="pt").to(constants.DEVICE)

    model_out = model(**inputs).logits
    magelm_out = magelm(inputs.input_ids)
    magelm_out_mem = magelm(inputs.input_ids, memory=magelm_out.memory)

    model_out = F.log_softmax(model_out, dim=-1)

    diff = torch.max(torch.abs(model_out - magelm_out.logits))
    assert diff < 1e-4, f"Output mismatch! ({diff.item()})"

    diff = torch.max(torch.abs(model_out - magelm_out_mem.logits))
    assert diff < 1e-4, f"Cached output mismatch! ({diff.item()})"


@torch.no_grad()
def test_archmage(tokenizer, model):
    conf = ArchMAGEConfig(**model.config.to_diff_dict())

    archmage = ArchMAGE(conf).to(constants.DEVICE)
    archmage.load_gpt2(model)
    archmage = archmage.eval()

    inputs = tokenizer("This is a test", return_tensors="pt").to(constants.DEVICE)
    z = torch.randn(*(list(inputs.input_ids.shape)+[conf.hidden_size])).to(constants.DEVICE)
    t = torch.rand_like(z[:, :, 0]).to(constants.DEVICE)

    model_out = model(**inputs).logits
    archmage_out = archmage(inputs.input_ids, z, t)
    archmage_out_mem = archmage(inputs.input_ids, z, t, memory=archmage_out.memory)

    model_out = F.log_softmax(model_out, dim=-1)

    diff = torch.max(torch.abs(model_out - archmage_out.logits))
    assert diff < 1e-4, f"Output mismatch! ({diff.item()})"

    diff = torch.max(torch.abs(model_out - archmage_out_mem.logits))
    assert diff < 1e-4, f"Cached output mismatch! ({diff.item()})"


if __name__ == "__main__":
    
    print("Loading reference model...")
    tokenizer = GPT2TokenizerFast.from_pretrained(TEST_MODEL_URL)
    model = GPT2LMHeadModel.from_pretrained(TEST_MODEL_URL).to(constants.DEVICE)
    model.eval()

    print("\nRUNNING: test_mage")
    test_mage(tokenizer, model.transformer)
    print("PASSED: test_mage")

    print("\nRUNNING: test_mage_lm")
    test_mage_lm(tokenizer, model)
    print("PASSED: test_mage_lm")

    print("\nRUNNING: test_archmage")
    test_archmage(tokenizer, model)
    print("PASSED: test_archmage")

    print("\nAll tests passed!")
