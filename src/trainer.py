import logging
import math
from typing import *

import torch
import torch.nn.utils as torch_utils
from datasets import load_metric
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup

from base.base_trainer import BaseTrainer
from utils import AverageMeter


class Trainer(BaseTrainer):
    """
    This trainer inherits BaseTrainer. See base_trainer.py
    """

    def __init__(self, hparams, tokenizer, loaders, model):
        super(Trainer, self).__init__(hparams, loaders, model)
        self.tokenizer = tokenizer
        self.accuracy = load_metric("accuracy")

        # dataloader and distributed sampler
        if self.hparams.distributed:
            self.train_sampler = self.trn_loader.sampler

        # optimizer, scheduler
        self.optimizer, self.scheduler = self.configure_optimizers()
        logging.info(
            f"[SCHEDULER] Total_step: {self.step_total} | Warmup step: {self.warmup_steps} | Accumulation step: {self.gradient_accumulation_step}"
        )

    def configure_optimizers(self):
        # optimizer
        decay_parameters = self.get_parameter_names(self.model, [torch.nn.LayerNorm])
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in self.model.named_parameters() if n in decay_parameters
                ],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if n not in decay_parameters
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

    def fit(self, num_ckpt=1) -> dict:
        # this zero gradient update is needed to avoid a warning message in warmup setting
        self.optimizer.zero_grad()
        self.optimizer.step()
        for epoch in tqdm(
            range(self.hparams.epoch), desc="epoch", disable=not self.main_process
        ):
            if self.hparams.distributed:
                self.train_sampler.set_epoch(epoch)
            self._train_epoch(epoch, num_ckpt)

        if self.main_process:
            self.summarywriter.close()
        return self.best_result if self.main_process else None

    def _train_epoch(self, epoch: int, num_ckpt: int = 1) -> None:
        train_loss = AverageMeter()

        for step, batch in tqdm(
            enumerate(self.trn_loader),
            desc="trn_steps",
            total=len(self.trn_loader),
            disable=not self.main_process,
        ):
            self.model.train()

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
            if self.hparams.amp:
                with torch.cuda.amp.autocast():
                    output = self.model(
                        input_ids=input_ids,
                        token_type_ids=token_type_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                    loss = output.loss
            else:
                output = self.model(
                    input_ids=input_ids,
                    token_type_ids=token_type_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = output.loss

            # update
            loss = loss / self.gradient_accumulation_step
            if not self.hparams.distributed:
                loss = loss.mean()

            if self.hparams.amp:
                self.scaler.scale(loss).backward()
                if (step + 1) % self.gradient_accumulation_step == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch_utils.clip_grad_norm_(
                        self.model.parameters(), self.max_grad_norm
                    )
                    self.scaler.step(self.optimizer)
                    self.scheduler.step()
                    self.scaler.update()
                    self.optimizer.zero_grad()  # when accumulating, only after step()
                    self.global_step += 1
            else:
                loss.backward()
                if (step + 1) % self.gradient_accumulation_step == 0:
                    torch_utils.clip_grad_norm_(
                        self.model.parameters(), self.max_grad_norm
                    )
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1

            train_loss.update(loss.item())
            if (step + 1) % self.gradient_accumulation_step != 0:
                continue

            # validate and logging
            if self.global_step != 0 and self.global_step % self.eval_step == 0:
                dev_loss, dev_acc = self.validate(epoch)
                if self.main_process:
                    self.summarywriter.add_scalars(
                        "loss/step", {"dev": dev_loss}, self.global_step
                    )
                    self.summarywriter.add_scalars(
                        "acc/step", {"dev": dev_acc}, self.global_step
                    )
                    logging.info(
                        f"[DEV] global step: {self.global_step} | dev loss: {dev_loss:.5f} | dev acc: {dev_acc:.5f}"
                    )
                    if dev_loss < self.global_dev_loss:
                        self.save_checkpoint(epoch, dev_loss, self.model, num_ckpt)

            # train logging
            if self.main_process:
                if self.global_step != 0 and self.global_step % self.log_step == 0:
                    logging.info(
                        f"[TRN] Version: {self.version} | Epoch: {epoch} | Global step: {self.global_step} | Train loss: {loss.item():.5f} | LR: {self.optimizer.param_groups[0]['lr']:.5f}"
                    )
                    self.summarywriter.add_scalars(
                        "loss/step", {"train": train_loss.avg}, self.global_step
                    )
                    self.summarywriter.add_scalars(
                        "lr",
                        {"lr": self.optimizer.param_groups[0]["lr"]},
                        self.global_step,
                    )

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
        for step, batch in tqdm(
            enumerate(test_loader), desc="tst_steps", total=len(test_loader)
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
            test_loss.update(loss.item())

            pred = torch.argmax(output.logits, dim=1)
            acc = self.accuracy.compute(references=labels.data, predictions=pred.data)
            test_acc.update(acc["accuracy"])

        logging.info(
            f"[TST] Test Loss: {test_loss.avg:.5f} | Test Acc: {test_acc.avg:.5f}"
        )

        return {"test_loss": test_loss.avg, "test_acc": test_acc.avg}
