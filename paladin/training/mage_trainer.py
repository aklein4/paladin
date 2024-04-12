import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from tqdm.notebook import tqdm

from training.base_trainer import BaseTrainer
from training.training_utils import lm_metrics

from utils import DotDict
import constants as constants


class MAGETrainer(BaseTrainer):

    _log_file = os.path.join(constants.LOCAL_DATA_PATH, "log.csv")
    _progress_file = os.path.join(constants.LOCAL_DATA_PATH, "progress.png")

    _hyperparams = [
        "lr",
        "bs",
        "num_steps",
        "warmup_steps",
        "eval_freq",
        "checkpoint_freq",
        "dtype",
        "max_length",
        "memory_grad",
        "max_eval_examples"
    ]

    _metrics = ["loss", "bpb", "ppl", "acc"]

    def __init__(
        self,
        save_name,
        **kwargs
    ):
        super().__init__(save_name, **kwargs)

        self.log = DotDict()
        self.reset_log()


    def reset_log(self):
        self.log = DotDict(
            train=DotDict(),
            eval=DotDict()
        )

        for m in self._metrics:
            self.log.train[m] = []
            self.log.eval[m] = []


    def upload(self):

        # save log
        out_log = self.log.eval.to_dict()

        max_len = max([len(v) for v in out_log.values()])
        for k, v in out_log.items():
            out_log[k] = v + [None] * (max_len - len(v))
        df = pd.DataFrame(out_log)
        df.to_csv(self._log_file)

        # get rolling training metrics
        df = pd.DataFrame(self.log.train.to_dict())
        roll = df.rolling(window=self.eval_freq, center=False, min_periods=self.eval_freq//2)
        mask = ~np.isnan(roll["loss"].mean())
        x = np.arange(len(roll["loss"].mean()))[mask]

        # plot simple metrics
        fig, ax = plt.subplots(2, len(self._metrics), figsize=(5*len(self._metrics), 10))

        # plot training metrics
        for i, metric in enumerate(self._metrics):
            ax[0,i].plot(x, roll[self._metrics[i]].mean()[mask])
            ax[0,i].set_title(f"Training {metric.upper()}")

        # plot eval metrics
        for i, metric in enumerate(self._metrics):
            ax[1,i].plot(self.log.eval[metric])
            ax[1,i].set_title(f"Eval {metric.upper()}")

        # finish plot
        plt.suptitle(f"MAGE Training Progress ({len(self.log.train.loss)} steps)")
        plt.tight_layout()
        plt.savefig(self._progress_file)
        plt.close()

        super().upload(self._log_file, self._progress_file)


    def _get_tokens(self, loader, tokenizer):
        prompts = loader(self.bs)

        x = tokenizer(
            prompts,
            return_tensors="pt",
            padding='max_length',
            truncation='longest_first',
            max_length=self.max_length
        ).to(constants.DEVICE)
        x.padding_mask = x.attention_mask == 0

        # hack to fix the padding token
        x.input_ids = torch.where(
            x.padding_mask,
            torch.zeros_like(x.input_ids),
            x.input_ids
        )

        return x


    @torch.no_grad()
    def evaluate(
        self,
        tokenizer,
        model,
        val_loader
    ):
        model.eval()

        tmp_log = DotDict()
        for m in self._metrics:
            tmp_log[m] = []

        examples = 0
        val_loader.reset()
        with tqdm(desc="Evaluating", leave=False) as pbar:
            while not val_loader.done:

                enable_autocast = self.dtype != torch.float32
                with torch.autocast(
                    device_type=str(constants.DEVICE),
                    dtype=(torch.float16 if not enable_autocast else self.dtype),
                    enabled=enable_autocast
                ):

                    # handle inputs
                    x = self._get_tokens(val_loader, tokenizer)

                    # get reusable things
                    logits = model(x.input_ids).logits

                    # get metrics
                    metrics = lm_metrics(
                        x.input_ids, logits,
                        x.padding_mask
                    )

                    # save metrics
                    for m in self._metrics:
                        tmp_log[m].append(metrics[m].item())

                pbar.set_postfix({str(k): np.mean(v) for k, v in tmp_log.items()})
                pbar.update(self.bs)

                examples += self.bs
                if examples >= self.max_eval_examples:
                    break

        # save metrics
        for m in self._metrics:
            self.log.eval[m].append(np.mean(tmp_log[m]))


    def train(
        self,
        tokenizer,
        model,
        train_loader,
        val_loader
    ):

        params = model.prepare_training(self.memory_grad)

        optimizer = torch.optim.AdamW(params, lr=self.lr)
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1e-10,
            end_factor=1.0,
            total_iters=self.warmup_steps
        )
        scaler = torch.cuda.amp.GradScaler()

        self.evaluate(tokenizer, model, val_loader)

        with tqdm(range(self.num_steps), desc="Training") as pbar:
            for step in pbar:

                model.train()

                enable_autocast = self.dtype != torch.float32
                with torch.autocast(
                    device_type=str(constants.DEVICE),
                    dtype=(torch.float16 if not enable_autocast else self.dtype),
                    enabled=enable_autocast
                ):

                    # handle inputs
                    x = self._get_tokens(train_loader, tokenizer)

                    # get encoding
                    logits = model(x.input_ids).logits

                    # get metrics
                    metrics = lm_metrics(
                        x.input_ids, logits,
                        x.padding_mask
                    )
                
                if enable_autocast:
                    scaler.scale(metrics.loss).backward()
                else:
                    metrics.loss.backward()
                optimizer.step()
                optimizer.zero_grad(True)
                lr_scheduler.step()

                # save metrics
                for m in self._metrics:
                    self.log.train[m].append(metrics[m].item())
                pbar.set_postfix({k: v.item() for k, v in metrics.items()})

                if (step+1) % self.eval_freq == 0 or step == self.num_steps-1:
                    self.evaluate(tokenizer, model, val_loader)
                    self.upload()

                if (step+1) % self.checkpoint_freq == 0 or step == self.num_steps-1:
                    self.save_checkpoint(
                        {
                            "model": model
                        }
                    )
                
