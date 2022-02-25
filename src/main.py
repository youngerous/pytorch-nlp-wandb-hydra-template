import glob
import os

import hydra
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import wandb
from datasets import load_dataset
from transformers import AutoTokenizer, BertForSequenceClassification

from config import *
from loader import get_trn_dev_loader, get_tst_loader
from trainer import Trainer
from utils import fix_seed


def worker(rank, hparams, ngpus_per_node: int):
    fix_seed(hparams.seed)
    batch_size = hparams.batch_size
    if hparams.gpu.distributed:
        hparams.gpu.rank = hparams.gpu.rank * ngpus_per_node + rank
        print(f"Use GPU {hparams.gpu.rank} for training")
        dist.init_process_group(
            backend=hparams.gpu.dist_backend,
            init_method=hparams.gpu.dist_url,
            world_size=hparams.gpu.world_size,
            rank=hparams.gpu.rank,
        )
        batch_size = hparams.batch_size // ngpus_per_node

    # get tokenizer
    if hparams.gpu.distributed:
        if rank != 0:
            dist.barrier()
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
        if rank == 0:
            dist.barrier()
    else:
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)

    # get dataloaders
    loaders = get_trn_dev_loader(
        dset=load_dataset("imdb", split="train"),
        tok=tokenizer,
        batch_size=batch_size,
        workers=hparams.workers,
        distributed=hparams.gpu.distributed,
    )

    # get model
    if hparams.gpu.distributed:
        if rank != 0:
            dist.barrier()
        model = BertForSequenceClassification.from_pretrained("bert-base-uncased")
        if rank == 0:
            dist.barrier()
    else:
        model = BertForSequenceClassification.from_pretrained("bert-base-uncased")

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if rank in [-1, 0]:
        wandb.init(
            project=hparams.wandb_project,
            entity=hparams.wandb_entity,
            config={"ngpus": ngpus_per_node, "num_params": num_params},
        )
        wandb.run.name = f"ep_{hparams.epoch}_bsz_{int(hparams.batch_size)*int(hparams.gradient_accumulation_step)}_lr_{hparams.lr}_wrmup_{hparams.warmup_ratio}_accum_{hparams.gradient_accumulation_step}_amp_{hparams.amp}_ddp_{hparams.gpu.distributed}"
        print(f"# Model Parameters: {num_params}")
        print(f"# WandB Run Name: {wandb.run.name}")
        print(f"# WandB Save Directory: {wandb.run.dir}")
        print(f"# Checkpoint Save Directory: {hparams.ckpt_root}")

    # training phase
    trainer = Trainer(hparams, tokenizer, loaders, model)
    trainer.fit()

    # testing phase
    if rank in [-1, 0]:
        if hparams.test:
            state_dict = torch.load(
                glob.glob(os.path.join(hparams.ckpt_root, "best", f"ckpt_*.pt"))[0]
            )
            test_loader = get_tst_loader(
                dset=load_dataset("imdb", split="test"),
                tok=tokenizer,
                batch_size=batch_size,
                workers=hparams.workers,
                distributed=False,
            )
            trainer.test(test_loader, state_dict)
        wandb.finish()


@hydra.main(config_path=None, config_name="train")
def main(cfg):
    ngpus_per_node = torch.cuda.device_count()

    if cfg.gpu.distributed:
        cfg.gpu.world_size = ngpus_per_node * cfg.gpu.world_size
        mp.spawn(worker, nprocs=ngpus_per_node, args=(cfg, ngpus_per_node))
    else:
        worker(cfg.gpu.rank, cfg, ngpus_per_node)
    return


if __name__ == "__main__":
    main()
