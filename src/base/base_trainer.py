import glob
import os
from typing import *

import torch
import torch.nn as nn
import wandb
from tensorboardX import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP


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
        if self.main_process:
            self.global_dev_loss = float("inf")
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
        self, epoch: int, dev_loss: float, dev_acc: float, model: nn.Module, num_ckpt: int
    ) -> None:
        print(f"Dev loss decreased ({self.global_dev_loss:.5f} → {dev_loss:.5f}). Saving model ...")
        new_path = os.path.join(
            wandb.run.dir, f"best_model_step_{self.global_step}_loss_{dev_loss:.5f}.pt"
        )

        for filename in glob.glob(os.path.join(wandb.run.dir, "*.pt")):
            # TODO: save model up to num_ckpt
            os.remove(filename)  # remove old checkpoint
        torch.save(model.state_dict(), new_path)
        # model.save(new_path)

        self.global_dev_loss = dev_loss
        wandb.run.summary["best_step"] = self.global_step
        wandb.run.summary["best_epoch"] = epoch
        wandb.run.summary["best_dev_loss"] = dev_loss
        wandb.run.summary["best_dev_acc"] = dev_acc

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