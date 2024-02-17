
import huggingface_hub as hf

import numpy as np
import pandas as pd
import os
from tqdm import tqdm

from loader.base_loader import BaseLoader


class FullLoader(BaseLoader):

    def __init__(self, url, train, debug=False):
        super().__init__(url, train, debug)
        
        files = hf.list_repo_files(url, repo_type="dataset")
        self.parquets = [f for f in files if f.endswith(".parquet")]
        if self.train:
            self.parquets = [f for f in self.parquets if f.count("train")]
        else:
            self.parquets = [f for f in self.parquets if f.count("validation")]

        data = []

        for file in (
            tqdm(self.parquets, desc="Loading") if not debug else
            os.listdir(self.parquets)
        ):

            df = pd.read_parquet(f"hf://datasets/{self.url}/{file}")
            data.append(np.array(df["text"]))

            if debug:
                break

        self.data = np.concatenate(data)

        self.curr_ind = 0
        self.done = False


    def reset(self):
        self.curr_ind = 0
        self.done = False


    def __len__(self):
        return len(self.data)


    def __call__(self, batchsize, length=None, tokenizer=None):
        assert (length is None) == (tokenizer is None)

        out = []
        while len(out) < batchsize:
            x = self.data[self.curr_ind]

            self.curr_ind += 1
            if self.curr_ind >= len(self.data):
                self.curr_ind = 0
                self.done = True

            if length is not None:
                check = tokenizer([x], return_tensors="pt").input_ids.shape[1]
                if check < length:
                    continue

            out.append(x)

        return out
    