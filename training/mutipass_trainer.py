import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

from training.base_trainer import BaseTrainer
from training.training_utils import lm_metrics
from model.model_utils import add_noise

from utils import DotDict
import constants as constants


class MultiPassTrainer(BaseTrainer):

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
        "memory_grad"
    ]

    _metrics = ["loss", "bpb", "ppl", "acc"]
    _eval_t = [0.0, 0.25, 0.5, 0.75, 1.0]

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
            self.log.eval[m] = DotDict()

            for t in self._eval_t:
                self.log.eval[m][t] = []


    def upload(self):

        # save log
        out_log = {}
        for k, v in self.log.train.items():
            out_log[f"train_{k}"] = v
        for k, v in self.log.eval.items():
            for t, w in v.items():
                out_log[f"eval_{k}_{t:.2f}"] = w
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
            for t in self._eval_t:
                ax[1,i].plot(self.log.eval[metric][t], label=f"{t:.2f}")
            
            ax[1,i].legend()
            ax[1,i].set_title(f"Eval {metric.upper()}")

        # finish plot
        plt.suptitle(f"MultiPass Training Progress ({len(self.log.train.loss)} steps)")
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

        # hack to fix the padding token
        x = torch.clip(x, max=len(tokenizer)-1)

        x.padding_mask = x.attention_mask == 0

        return x


    @torch.no_grad()
    def evaluate(
        self,
        tokenizer,
        encoder,
        decoder,
        val_loader
    ):
        encoder.eval()
        decoder.eval()

        tmp_log = DotDict()
        for m in self._metrics:
            tmp_log[m] = DotDict()
            for t in self._eval_t:
                tmp_log[m][t] = []

        generator = torch.Generator(device=constants.DEVICE)

        val_loader.reset()
        with tqdm(desc="Evaluating", leave=False) as pbar:
            while not val_loader.done:

                # handle inputs
                x = self._get_tokens(val_loader, tokenizer)

                # get reusable things
                z = encoder(x.input_ids)
                memory = decoder(
                    x.input_ids,
                    z, torch.zeros_like(z)[:, :, 0]
                ).memory
                noise = torch.randn(z.shape, generator=generator, device=constants.DEVICE)
                
                # try all different noise levels
                for t in self._eval_t:
                    t_tensor = torch.ones_like(z)[:, :, 0] * t

                    # get noised z
                    z_noisy = add_noise(
                        z, t_tensor, noise
                    )

                    # get logits
                    logits = decoder(
                        x.input_ids,
                        z_noisy, t_tensor,
                        memory=memory
                    ).logits

                    # get metrics
                    metrics = lm_metrics(
                        x.input_ids, logits,
                        x.padding_mask
                    )

                    # save metrics
                    for m in self._metrics:
                        tmp_log[m][t].append(metrics[m].item())

                pbar.set_postfix({str(k): np.mean(v) for k, v in tmp_log.acc.items()})
                pbar.update(self.bs)

        # save metrics
        for m in self._metrics:
            for t in self._eval_t:
                self.log.eval[m][t].append(np.mean(tmp_log[m][t]))


    def train(
        self,
        tokenizer,
        encoder,
        decoder,
        train_loader,
        val_loader
    ):

        enc_params = encoder.prepare_training()
        dec_params = decoder.prepare_training(self.memory_grad)

        optimizer = torch.optim.AdamW(enc_params + dec_params, lr=self.lr)
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1e-10,
            end_factor=1.0,
            total_iters=self.warmup_steps
        )
        scaler = torch.cuda.amp.GradScaler()

        self.evaluate(tokenizer, encoder, decoder, val_loader)

        generator = torch.Generator(device=constants.DEVICE)

        with tqdm(range(self.num_steps), desc="Training") as pbar:
            for step in pbar:

                optimizer.zero_grad(True)
                encoder.train()
                decoder.train()

                enable_autocast = self.dtype != torch.float32
                with torch.autocast(
                    device_type=str(constants.DEVICE),
                    dtype=(torch.float16 if not enable_autocast else self.dtype),
                    enabled=enable_autocast
                ):

                    # handle inputs
                    x = self._get_tokens(train_loader, tokenizer)

                    # get encoding
                    z = encoder(x.input_ids)

                    # get noised encoding
                    t = torch.rand(z.shape[:-1], device=constants.DEVICE, generator=generator)
                    noise = torch.randn(z.shape, device=constants.DEVICE, generator=generator)
                    z_noisy = add_noise(z, t, noise)

                    # get logits
                    logits = decoder(
                        x.input_ids,
                        z_noisy, t,
                    )

                    # get metrics
                    metrics = lm_metrics(
                        x.input_ids, logits.logits,
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
                pbar.set_postfix({k: v.item() for k, v in metrics})

                if (step+1) % self.eval_freq == 0 or step == self.num_steps-1:
                    self.evaluate(tokenizer, encoder, decoder, val_loader)
                    self.upload()

                if (step+1) % self.save_freq == 0 or step == self.num_steps-1:
                    self.save_checkpoint(
                        {
                            "encoder": encoder,
                            "decoder": decoder
                        }
                    )
                
