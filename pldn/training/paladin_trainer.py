import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from tqdm.notebook import tqdm

import huggingface_hub as hf
from huggingface_hub import HfApi

from training.eval import get_eval_metrics
import constants as constants


class PaladinTrainer:

    # minipile val set
    # context window 128, prompt length 8
    _ppl_s = 35.50
    _ppl_m = 25.12
    _ppl_l = 23.03
    _bpb_s = 4.832
    _bpb_m = 4.347
    _bpb_l = 4.178
    _acc_s = 0.414
    _acc_m = 0.448
    _acc_l = 0.467

    _metric_mins = {
        "ppl": -1,
        "bpb": -1,
        "acc": -1
    }
    _metric_maxs = {
        "ppl": 50,
        "bpb": 6,
        "acc": 1000
    }

    def __init__(
        self,
        save_name,
        tokenizer,
        lr,
        kl_limit,
        kl_penalty,
        eval_freq,
        save_freq,
        init_log=False
    ):
        self.save_name = save_name
        self.save_path = os.path.join(constants.LOCAL_DATA_PATH, save_name)
        os.makedirs(self.save_path, exist_ok=True)

        save_repo = f"paladin-{save_name}"
        hf.create_repo(
            save_repo, private=True, exist_ok=True
        )
        self.save_repo = f"{constants.HF_ID}/{save_repo}"

        self.tokenizer = tokenizer
        self.lr = lr
        self.kl_limit = kl_limit
        self.kl_penalty = kl_penalty
        self.eval_freq = eval_freq
        self.save_freq = save_freq

        self.log = {}
        self.reset_log(file=os.path.join(self.save_path, "log.csv") if init_log else None)


    def reset_log(self, file=None):
        if file is not None and os.path.exists(file):
            df = pd.read_csv(file)
            log_tmp = df.to_dict(orient="list")

            self.log = {}
            for k, v in log_tmp.items():
                self.log[k] = []
            for e in v:
                if not isinstance(e, float) or e != 0.0:
                    self.log[k].append(e)

        else:
            self.log = {
            "loss": [],
            "elbo": [],
            "kl_enc": [],
            "train_ppl": [],
            "train_bpb": [],
            "train_acc": [],
            }
            for k in ["ppl", "bpb", "acc"]:
                for mode in ["uncond", "sample", "mode"]:
                    self.log[f"val_{k}_{mode}"] = []


    def save_log(self):
        os.makedirs(self.save_path, exist_ok=True)

        csv_file = os.path.join(self.save_path, "log.csv")
        png_file = os.path.join(self.save_path, "progress.png")
        for file in [csv_file, png_file]:
            try:
                os.remove(file)
            except:
                pass

        # save log as csv
        l = len(self.log["elbo"])
        df = pd.DataFrame({k: np.array(v + [0]*(l-len(v))) for k, v in self.log.items()})
        df.to_csv(csv_file)

        # get rolling metrics
        roll = df.rolling(window=1000, center=False, min_periods=100)
        mask = ~np.isnan(roll["elbo"].mean())
        x = np.arange(len(roll["elbo"].mean()))[mask]

        # plot simple metrics
        fig, ax = plt.subplots(3, 3, figsize=(15, 15))

        ax[0,0].plot(x, roll["loss"].mean()[mask])
        ax[0,0].set_title("Loss")

        ax[0,1].plot(x, roll["elbo"].mean()[mask])
        ax[0,1].set_title("ELBO")

        ax[0,2].plot(x, roll["kl_enc"].mean()[mask])
        ax[0,2].set_title("Encoding KL")

        # plot train & val metrics
        for i, k in enumerate(["ppl", "bpb", "acc"]):
            ax[1,i].plot(x, roll[f"train_{k}"].mean()[mask])
            ax[1,i].set_title(f"Train {k.upper()}")

            ax[2,i].plot(
                np.clip(
                    self.log[f"val_{k}_uncond"],
                    a_min=self._metric_mins[k],
                    a_max=self._metric_maxs[k]
                ),
                label="uncond", color="red"
            )
            ax[2,i].plot(self.log[f"val_{k}_sample"], label="sample", color="cyan")
            ax[2,i].plot(self.log[f"val_{k}_mode"], label="mode", color="blue")
            ax[2,i].axhline(getattr(self, f"_{k}_s"), label="gpt2-s", ls="--", color="black")
            ax[2,i].axhline(getattr(self, f"_{k}_m"), label="gpt2-m", ls="--", color="darkgray")
            ax[2,i].axhline(getattr(self, f"_{k}_l"), label="gpt2-l", ls="--", color="lightgray")
            ax[2,i].legend()
            ax[2,i].set_title(f"Validation {k.upper()}")

        plt.suptitle(f"PALADIN Training Progress ({len(self.log['elbo'])} steps)")
        plt.tight_layout()
        plt.savefig(png_file)
        plt.close()

        api = HfApi()
        api.upload_file(
            path_or_fileobj=csv_file,
            path_in_repo="log.csv",
            repo_id=self.save_repo,
            repo_type="model"
        )
        api.upload_file(
            path_or_fileobj=png_file,
            path_in_repo="progress.png",
            repo_id=self.save_repo,
            repo_type="model"
        )


    def save_checkpoint(
        self,
        paladin,
        optimizer
    ):
        checkpoint_path = os.path.join(self.save_path, "checkpoint")
        os.makedirs(checkpoint_path, exist_ok=True)

        paladin.save_pretrained(os.path.join(checkpoint_path, "paladin"))
        torch.save(optimizer.state_dict(), os.path.join(checkpoint_path, "optimizer.pt"))

        paladin.push_to_hub(self.save_repo)


    @torch.no_grad()
    def evaluate(
        self,
        paladin,
        val_loader,
        batch_size,
    ):
        paladin.eval()

        ppls = {}
        bpbs = {}
        accs = {}
        for d in [ppls, bpbs, accs]:
            d["uncond"] = []
            d["sample"] = []
            d["mode"] = []

        val_loader.reset()
        with tqdm(desc="Evaluating", leave=False) as pbar:
            while not val_loader.done:

                # handle inputs
                prompt = val_loader(batch_size, paladin.context_length, self.tokenizer)
                x = self.tokenizer(
                    prompt, return_tensors="pt",
                    truncation=True, max_length=paladin.context_length
                ).to(constants.DEVICE)
                if x.input_ids.shape[1] != paladin.context_length:
                    continue

                # calculate memory
                memory = paladin.memorize(**x)

                # calculate z dist
                encoder_out = paladin.encode(**x)

                # try all different methods
                for mode in ["uncond", "sample", "mode"]:
                    noise_generator = torch.Generator(device=constants.DEVICE)
                    noise_generator.manual_seed(val_loader.curr_ind)
                    np_generator = np.random.default_rng(seed=val_loader.curr_ind)

                    # sample z
                    z_info = paladin.sample_z_for_training(
                        encoder_out.mus,
                        encoder_out.sigmas,
                        mode,
                        generator=noise_generator,
                        np_generator=np_generator
                    )

                    # calculate decoder logits
                    decoder_out = paladin.decode(
                        **x, memory=memory,
                        z=z_info.z, j=z_info.j
                    )

                    # get logp(x|z)
                    # cross entropy takes care of softmax for logits
                    logits = decoder_out[:, :-1]
                    logp = -F.cross_entropy(
                        logits.reshape(-1, logits.shape[-1]),
                        x.input_ids[:, 1:].reshape(-1),
                        reduction='none'
                    ).reshape(logits.shape[0], logits.shape[1])

                    # calculate metrics
                    ppl, bpb, acc = get_eval_metrics(decoder_out, logp, x.input_ids, prompt_length=paladin.prompt_length)

                    ppls[mode].append(ppl)
                    bpbs[mode].append(bpb)
                    accs[mode].append(acc)

                pbar.set_postfix({k: np.mean(v) for k, v in accs.items()})
                pbar.update(batch_size)

        for k, v in bpbs.items():
            self.log[f"val_bpb_{k}"].append(np.mean(v))
        for k, v in ppls.items():
            self.log[f"val_ppl_{k}"].append(np.mean(v))
        for k, v in accs.items():
            self.log[f"val_acc_{k}"].append(np.mean(v))


    def train(
        self,
        paladin,
        train_loader,
        val_loader,
        num_steps,
        warmup_steps,
        batch_size,
        seed=0,
        optimizer_state=None
    ):

        optimizer = torch.optim.AdamW(paladin.parameters(), lr=self.lr)
        if optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1e-10,
            end_factor=1.0,
            total_iters=warmup_steps
        )

        self.evaluate(paladin, val_loader, batch_size)

        noise_generator = torch.Generator(device=constants.DEVICE)
        noise_generator.manual_seed(seed)
        np_generator = np.random.default_rng(seed=seed)

        rolling_kl = None
        with tqdm(range(num_steps), desc="PALADIN Training") as pbar:
            for step in pbar:

                optimizer.zero_grad()
                if not paladin.training:
                    paladin.train()

                # handle inputs
                prompt = train_loader(batch_size, paladin.context_length, self.tokenizer)
                x = self.tokenizer(
                    prompt, return_tensors="pt",
                    truncation=True, max_length=paladin.context_length
                ).to(constants.DEVICE)
                if x.input_ids.shape[1] != paladin.context_length:
                    continue

                # calculate memory
                memory = paladin.memorize(**x)

                # calculate and sample z
                encoder_out = paladin.encode(**x)
                z_info = paladin.sample_z_for_training(
                    encoder_out.mus,
                    encoder_out.sigmas,
                    "sample",
                    generator=noise_generator,
                    np_generator=np_generator
                )

                # calculate decoder logits
                decoder_out = paladin.decode(
                    **x, memory=memory,
                    z=z_info.z, j=z_info.j
                )

                # get logp(x|z)
                # cross entropy takes care of softmax for logits
                logits = decoder_out[:, :-1]
                logp = -F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    x.input_ids[:, 1:].reshape(-1),
                    reduction='none'
                ).reshape(logits.shape[0], logits.shape[1])

                # calculate metrics
                ppl, bpb, acc = get_eval_metrics(decoder_out, logp, x.input_ids, prompt_length=paladin.prompt_length)

                # get the kl of the z distribution
                mus = encoder_out.mus[:, paladin.z_window:]
                sigmas = encoder_out.sigmas[:, paladin.z_window:]
                dist = torch.distributions.Normal(mus, sigmas)
                target_dist = torch.distributions.Normal(
                    torch.zeros_like(mus),
                    torch.ones_like(sigmas)
                )
                kl = torch.distributions.kl_divergence(dist, target_dist)

                # for the kl penalty
                if rolling_kl is None:
                    rolling_kl = kl.sum(-1).mean().item()
                else:
                    rolling_kl = rolling_kl * 0.99 + kl.sum(-1).mean().item() * 0.01

                # calculate the elbo
                elbo = logp.sum(-1).mean() - kl.sum(-1).mean()

                # backpropogate
                loss = -(z_info.w_pos * logp).mean()
                if rolling_kl > self.kl_limit:
                    loss += self.kl_penalty * kl.mean()
                loss.backward()
                optimizer.step()
                lr_scheduler.step()

                # save metrics
                self.log["loss"].append(loss.item())
                self.log["elbo"].append(elbo.item())
                self.log["kl_enc"].append(kl.sum(-1).mean().item())
                self.log["train_bpb"].append(bpb)
                self.log["train_ppl"].append(ppl)
                self.log["train_acc"].append(acc)

                pbar.set_postfix({k: v[-1] for k, v in self.log.items()})

                if (step+1) % self.eval_freq == 0 or step == num_steps-1:
                    self.evaluate(paladin, val_loader, batch_size)
                    self.save_log()

                if (step+1) % self.save_freq == 0 or step == num_steps-1:
                    self.save_checkpoint(paladin, optimizer)
                
