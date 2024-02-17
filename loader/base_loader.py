
from abc import abstractclassmethod


class BaseLoader:

    def __init__(self, url, train, debug=False):
        self.url = url
        self.name = f'{url.replace("/", "-")}-{"train" if train else "val"}'

        self.train = train
        self.debug = debug


    @abstractclassmethod
    def reset(self):
        raise NotImplementedError
    

    @abstractclassmethod
    def __len__(self):
        raise NotImplementedError
    

    @abstractclassmethod
    def __call__(self, batchsize, length=None, tokenizer=None):
        raise NotImplementedError