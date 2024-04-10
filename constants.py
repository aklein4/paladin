import torch

# best device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# for GPT2.generate()
PAD_TOKEN_ID = 50256

# local data path
LOCAL_DATA_PATH = "./local_data"

# huggingface login id
HF_ID = "aklein4"
