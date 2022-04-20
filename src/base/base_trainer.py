import glob
import logging
import math
import os
from typing import *

import torch
import torch.nn as nn
import wandb
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup
from utils import EvalManager

if torch.distributed.is_available():
    from torch.distributed import ReduceOp

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
        activate_earlystop = True if hparams.early_stop_tolerance > 0 else False
        self.eval_mgr = EvalManager(
            patience=hparams.early_stop_tolerance, activate_early_stop=activate_earlystop
        )

        if self.main_process:
            self.hparams.ckpt_root = os.path.join(self.hparams.ckpt_root, wandb.run.id)
            self.log_step = hparams.log_step
            wandb.config.update(self.hparams)
            wandb.watch(self.model)
            wandb.run.summary["step_total"] = self.step_total

    def to_device(self, *tensors):
        bundle = []
        for tensor in tensors:
            bundle.append(tensor.to(self.device, non_blocking=True))
        return bundle if len(bundle) > 1 else bundle[0]

    def reduce_boolean_decision(
        self,
        decision: bool,
        reduce_op: Optional[Union[ReduceOp, str]] = ReduceOp.SUM,
        stop_option: str = "all",
    ) -> bool:
        """This function is partially modified from pytorch-lightning

        Args:
            decision (bool): Boolean value whether to early stop the process
            reduce_op (Optional[Union[ReduceOp, str]]): DDP reduce operator
            stop_option (str): Early stopping option according to each process decision

        Return: Reduced boolean value

        Ref:
            https://github.com/PyTorchLightning/pytorch-lightning/blob/939d56c6d69202318baf2fbf65ceda00c63363fd/pytorch_lightning/strategies/parallel.py#L113
        """
        assert stop_option in ["all", "half", "strict"]
        divide_by_world_size = False
        group = torch.distributed.group.WORLD
        decision = torch.tensor(int(decision), device=self.device)

        if isinstance(reduce_op, str):
            if reduce_op.lower() in ("avg", "mean"):
                op = ReduceOp.SUM
                divide_by_world_size = True
            else:
                op = getattr(ReduceOp, reduce_op.upper())
        else:
            op = reduce_op

        torch.distributed.barrier(group=group)
        torch.distributed.all_reduce(decision, op=op, group=group, async_op=False)
        if divide_by_world_size:
            decision = decision / torch.distributed.get_world_size(group)

        if stop_option == "all":  # stop if every process calls stopping
            decision = bool(decision == self.hparams.gpu.world_size)
        elif stop_option == "half":  # stop if more than half processes call stopping
            decision = bool(decision > int(self.hparams.gpu.world_size // 2))
        elif stop_option == "strict":  # stop if just one process calls stopping
            decision = bool(decision > 0)

        return decision

    def configure_optimizers(self):
        # optimizer
        decay_parameters = self.get_parameter_names(self.model, [torch.nn.LayerNorm])
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if n in decay_parameters],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [
                    p for n, p in self.model.named_parameters() if n not in decay_parameters
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.lr)

        # lr scheduler with warmup
        self.warmup_steps = math.ceil(self.step_total * self.hparams.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=self.step_total,
        )

        return optimizer, scheduler

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
        self,
        epoch: int,
        global_dev_loss: float,
        dev_loss: float,
        dev_acc: float,
        model: nn.Module,
        best=True,
    ) -> None:
        latest_pth = os.path.join(self.hparams.ckpt_root, "latest")
        os.makedirs(latest_pth, exist_ok=True)

        if best:
            logger.info(
                f"Dev loss decreased ({global_dev_loss:.5f} → {dev_loss:.5f}). Saving best model ..."
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
