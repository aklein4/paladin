import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32

# for GPT2.generate()
PAD_TOKEN_ID = 50256


# local data path
LOCAL_DATA_PATH = "./local_data"


HF_ID = "aklein4"
