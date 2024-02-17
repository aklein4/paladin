
import huggingface_hub as hf

import numpy as np
import pandas as pd
import os
from tqdm import tqdm

from loader.base_loader import BaseLoader


class SingleLoader(BaseLoader):

    def __init__(self, url, train, debug=False):
        super().__init__(url, train, debug)
        
        files = hf.list_repo_files(url, repo_type="dataset")
        self.parquets = [f for f in files if f.endswith(".parquet")]
        if self.train:
            self.parquets = [f for f in self.parquets if f.count("train")]
        else:
            self.parquets = [f for f in self.parquets if f.count("validation")]

        self.curr_ind = 0
        self.curr_file_ind = 0
        self.done = False

        self.data = None
        self.load_file(0)


    def load_file(self, file_ind):
        file = self.parquets[file_ind]
        df = pd.read_parquet(f"hf://datasets/{self.url}/{file}")
        self.data = np.array(df["text"])


    def reset(self):
        self.curr_ind = 0
        self.curr_file_ind = 0
        self.done = False

        self.load_file(0)


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
                self.curr_file_ind += 1

                if self.curr_file_ind >= len(self.parquets):
                    self.curr_file_ind = 0
                    self.done = True

                self.load_file(self.curr_file_ind)

            if length is not None:
                check = tokenizer([x], return_tensors="pt").input_ids.shape[1]
                if check < length:
                    continue

            out.append(x)

        return out
    