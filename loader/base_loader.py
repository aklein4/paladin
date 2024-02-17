
from abc import abstractclassmethod


class BaseLoader:
    _data_subfolder = "data"

    @abstractclassmethod
    def reset(self):
        raise NotImplementedError
    

    @abstractclassmethod
    def __len__(self):
        raise NotImplementedError
    

    @abstractclassmethod
    def __call__(self, batchsize, length=None, tokenizer=None):
        raise NotImplementedError