import glob
import logging
import os
from typing import *

import torch
import torch.nn as nn
import wandb
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

logger = logging.getLogger()


class BaseTrainer(object):
    def __init__(self, hparams, loaders, model):
        self.hparams = hparams
        self.distributed = self.hparams.gpu.distributed
        self.rank: int = self.hparams.gpu.rank
        self.main_process: bool = self.rank in [-1, 0]
        self.nprocs: int = torch.cuda.device_count()
        self.scaler = torch.cuda.amp.GradScaler() if self.hparams.amp else None
        if self.distributed:
            assert torch.cuda.is_available()
            self.device = f"cuda:{self.rank}"
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = model
        self.model = model.to(self.device, non_blocking=True)
        if self.distributed:
            self.model = DDP(self.model, device_ids=[self.rank])
        elif self.nprocs > 1:
            self.model = nn.DataParallel(self.model)

        self.trn_loader, self.dev_loader = loaders
        self.max_grad_norm = self.hparams.max_grad_norm
        self.gradient_accumulation_step = self.hparams.gradient_accumulation_step
        self.step_total = (
            len(self.trn_loader) // self.gradient_accumulation_step * self.hparams.epoch
        )

        # model saving options
        self.global_step = 0
        self.eval_step = (
            int(self.step_total * self.hparams.eval_ratio)
            if self.hparams.eval_ratio > 0
            else self.step_total // self.hparams.epoch
        )

        # early stopping options
        self.stop_train = False
        self.global_dev_loss = float("inf")
        if hparams.early_stop_tolerance > 0:
            self.early_stop = True
            self.early_stop_cnt = 0
        else:
            self.early_stop = False

        if self.main_process:
            self.hparams.ckpt_root = os.path.join(self.hparams.ckpt_root, wandb.run.id)
            self.log_step = hparams.log_step
            wandb.config.update(self.hparams)
            wandb.watch(self.model)
            wandb.run.summary["step_total"] = self.step_total

    def get_parameter_names(self, model, forbidden_layer_types):
        """
        Returns the names of the model parameters that are not inside a forbidden layer.
        """
        result = []
        for name, child in model.named_children():
            result += [
                f"{name}.{n}"
                for n in self.get_parameter_names(child, forbidden_layer_types)
                if not isinstance(child, tuple(forbidden_layer_types))
            ]
        # Add model specific parameters (defined with nn.Parameter) since they are not in any child.
        result += list(model._parameters.keys())
        return result

    def save_checkpoint(
        self, epoch: int, dev_loss: float, dev_acc: float, model: nn.Module, best=True
    ) -> None:
        latest_pth = os.path.join(self.hparams.ckpt_root, "latest")
        os.makedirs(latest_pth, exist_ok=True)

        if best:
            logger.info(
                f"Dev loss decreased ({self.global_dev_loss:.5f} → {dev_loss:.5f}). Saving best model ..."
            )
            best_pth = os.path.join(self.hparams.ckpt_root, "best")
            os.makedirs(best_pth, exist_ok=True)

            # save best model
            for filename in glob.glob(os.path.join(self.hparams.ckpt_root, "best", "ckpt_*.pt")):
                os.remove(filename)  # remove old checkpoint
            torch.save(
                model.state_dict(),
                os.path.join(best_pth, f"ckpt_step_{self.global_step}_loss_{dev_loss:.5f}.pt"),
            )

            self.global_dev_loss = dev_loss
            wandb.run.summary["best_step"] = self.global_step
            wandb.run.summary["best_epoch"] = epoch
            wandb.run.summary["best_dev_loss"] = dev_loss
            wandb.run.summary["best_dev_acc"] = dev_acc

        # save latest model
        logger.info(f"Saving latest model ...")
        for filename in glob.glob(os.path.join(self.hparams.ckpt_root, "latest", "ckpt_*.pt")):
            os.remove(filename)  # remove old checkpoint
        torch.save(
            model.state_dict(),
            os.path.join(latest_pth, f"ckpt_step_{self.global_step}_loss_{dev_loss:.5f}.pt"),
        )

    def configure_optimizers(self, *args, **kwargs):
        raise NotImplementedError

    def fit(self):
        raise NotImplementedError

    def _train_epoch(self, *args, **kwargs):
        raise NotImplementedError

    @torch.no_grad()
    def validate(self, *args, **kwargs):
        raise NotImplementedError

    @torch.no_grad()
    def test(self, *args, **kwargs):
        raise NotImplementedError
