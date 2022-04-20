import logging
import warnings
from typing import *

warnings.filterwarnings("ignore")

from typing import Tuple

import torch
import torch.nn.utils as torch_utils
import wandb
from datasets import load_metric
from torch import Tensor as T
from tqdm import tqdm

from base.base_trainer import BaseTrainer
from utils import AverageMeter

logger = logging.getLogger()


class Trainer(BaseTrainer):
    """
    This trainer inherits BaseTrainer. See base_trainer.py
    """

    def __init__(self, hparams, tokenizer, loaders, model):
        super(Trainer, self).__init__(hparams, loaders, model)
        self.tokenizer = tokenizer
        self.accuracy = load_metric("accuracy")

        # dataloader and distributed sampler
        if self.distributed:
            self.train_sampler = self.trn_loader.sampler

        # optimizer, scheduler
        self.optimizer, self.scheduler = self.configure_optimizers()
        if self.main_process:
            wandb.run.summary["step_warmup"] = self.warmup_steps

    def fit(self) -> dict:
        # this zero gradient update is needed to avoid a warning message in warmup setting
        self.optimizer.zero_grad()
        self.optimizer.step()
        for epoch in tqdm(range(self.hparams.epoch), desc="epoch", disable=not self.main_process):
            if self.distributed:
                self.train_sampler.set_epoch(epoch)
            self._train_epoch(epoch)
            if self.eval_mgr.early_stop:
                break

        if self.main_process:
            wandb.run.summary["early_stopped"] = True if self.eval_mgr.early_stop else False

    def _train_epoch(self, epoch: int) -> None:
        if self.main_process:
            train_loss = AverageMeter()
            train_acc = AverageMeter()

        for step, batch in tqdm(
            enumerate(self.trn_loader),
            desc="trn_steps",
            total=len(self.trn_loader),
            disable=not self.main_process,
        ):
            self.model.train()

            batch_input = self._set_batch_input(batch)

            loss, logit = self._compute_loss(batch_input)
            pred = torch.argmax(logit, dim=1)
            loss = self._aggregate_loss(loss)
            self._update_loss(loss, step)
            acc = self.accuracy.compute(
                references=batch_input["labels"].data, predictions=pred.data
            )

            if (step + 1) % self.gradient_accumulation_step != 0:
                continue

            # train logging
            if self.main_process:
                self._logging_train(epoch, train_loss, loss, train_acc, acc)

            # validate and logging
            if self.global_step != 0 and self.global_step % self.eval_step == 0:
                dev_loss, dev_acc = self.validate(epoch)
                is_best = self.eval_mgr(dev_loss, self.global_step, self.main_process)
                global_dev_loss = self.eval_mgr.global_dev_loss
                if self.main_process:
                    wandb.log({"dev": {"loss": dev_loss}}, step=self.global_step)
                    self.save_checkpoint(
                        epoch, global_dev_loss, dev_loss, dev_acc, self.model, best=is_best
                    )

                if self.eval_mgr.activate_early_stop:
                    if self.distributed:  # sync early stop with all processes in ddp
                        self.eval_mgr.early_stop = self.reduce_boolean_decision(
                            self.eval_mgr.early_stop, stop_option="all"
                        )
                    if self.eval_mgr.early_stop:
                        if self.main_process:
                            logger.info("### Every process called early stopping ###")
                        break

    @torch.no_grad()
    def validate(self, epoch: int) -> float:
        dev_loss = AverageMeter()
        dev_acc = AverageMeter()

        self.model.eval()
        for step, batch in tqdm(
            enumerate(self.dev_loader),
            desc="dev_steps",
            total=len(self.dev_loader),
            disable=not self.main_process,
        ):
            # load to machine
            input_ids = batch["input_ids"].squeeze(1)
            token_type_ids = batch["token_type_ids"].squeeze(1)
            attention_mask = batch["attention_mask"].squeeze(1)
            labels = batch["labels"]

            input_ids = input_ids.to(self.device, non_blocking=True)
            token_type_ids = token_type_ids.to(self.device, non_blocking=True)
            attention_mask = attention_mask.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            # compute loss
            output = self.model(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = output.loss.mean()
            dev_loss.update(loss.item())

            pred = torch.argmax(output.logits, dim=1)
            acc = self.accuracy.compute(references=labels.data, predictions=pred.data)
            dev_acc.update(acc["accuracy"])

        return dev_loss.avg, dev_acc.avg

    @torch.no_grad()
    def test(self, test_loader, state_dict) -> dict:
        test_loss = AverageMeter()
        test_acc = AverageMeter()

        self.model.load_state_dict(state_dict)
        self.model.eval()
        for step, batch in tqdm(enumerate(test_loader), desc="tst_steps", total=len(test_loader)):
            # load to machine
            input_ids = batch["input_ids"].squeeze(1)
            token_type_ids = batch["token_type_ids"].squeeze(1)
            attention_mask = batch["attention_mask"].squeeze(1)
            labels = batch["labels"]

            input_ids = input_ids.to(self.device, non_blocking=True)
            token_type_ids = token_type_ids.to(self.device, non_blocking=True)
            attention_mask = attention_mask.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            # compute loss
            output = self.model(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = output.loss.mean()
            test_loss.update(loss.item())

            pred = torch.argmax(output.logits, dim=1)
            acc = self.accuracy.compute(references=labels.data, predictions=pred.data)
            test_acc.update(acc["accuracy"])

        wandb.log({"tst": {"loss": test_loss.avg, "acc": test_acc.avg}})
        wandb.run.summary["tst_loss"] = test_loss.avg
        wandb.run.summary["tst_acc"] = test_acc.avg
        logger.info(f"[TST] tst loss: {test_loss.avg:.5f} | tst acc: {test_acc.avg:.5f}")

    def _set_batch_input(self, batch: T) -> dict:
        input_ids = batch["input_ids"].squeeze(1)
        token_type_ids = batch["token_type_ids"].squeeze(1)
        attention_mask = batch["attention_mask"].squeeze(1)
        labels = batch["labels"]
        input_ids, token_type_ids, attention_mask, labels = self.to_device(
            input_ids, token_type_ids, attention_mask, labels
        )

        return {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def _compute_loss(self, batch_input: dict) -> Tuple[T, T]:
        if self.hparams.amp:
            with torch.cuda.amp.autocast():
                output = self.model(
                    input_ids=batch_input["input_ids"],
                    token_type_ids=batch_input["token_type_ids"],
                    attention_mask=batch_input["attention_mask"],
                    labels=batch_input["labels"],
                )
        else:
            output = self.model(
                input_ids=batch_input["input_ids"],
                token_type_ids=batch_input["token_type_ids"],
                attention_mask=batch_input["attention_mask"],
                labels=batch_input["labels"],
            )

        return output.loss, output.logits

    def _aggregate_loss(self, loss: T) -> T:
        loss = loss / self.gradient_accumulation_step
        if not self.distributed:
            loss = loss.mean()
        return loss

    def _update_loss(self, loss: T, step: int) -> None:
        if self.hparams.amp:
            self.scaler.scale(loss).backward()
            if (step + 1) % self.gradient_accumulation_step == 0:
                self.scaler.unscale_(self.optimizer)
                torch_utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scheduler.step()
                self.scaler.update()
                self.optimizer.zero_grad()  # when accumulating, only after step()
                self.global_step += 1
        else:
            loss.backward()
            if (step + 1) % self.gradient_accumulation_step == 0:
                torch_utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1

    def _logging_train(
        self,
        epoch: int,
        train_loss: AverageMeter,
        step_loss: T,
        train_acc: AverageMeter,
        step_acc: float,
    ) -> None:
        train_loss.update(step_loss.item())
        train_acc.update(step_acc["accuracy"])
        if self.global_step != 0 and self.global_step % self.log_step == 0:
            wandb.log(
                {
                    "train": {"loss": train_loss.avg, "acc": train_acc.avg},
                    "lr": self.optimizer.param_groups[0]["lr"],
                    "epoch": epoch,
                },
                step=self.global_step,
            )
            logger.info(
                f"[TRN] Epoch: {epoch} | Global step: {self.global_step} | Train loss: {step_loss.item():.5f} | LR: {self.optimizer.param_groups[0]['lr']:.5f}"
            )
