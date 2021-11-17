import glob
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from datasets import load_dataset
from transformers import BertForSequenceClassification, BertTokenizer

from config import load_config
from loader import get_trn_dev_loader, get_tst_loader
from trainer import Trainer
from utils import ResultWriter, fix_seed


def main(rank, hparams, ngpus_per_node: int):
    fix_seed(hparams.seed)
    resultwriter = ResultWriter(hparams.result_path)
    if hparams.distributed:
        hparams.rank = hparams.rank * ngpus_per_node + rank
        print(f"Use GPU {hparams.rank} for training")
        dist.init_process_group(
            backend=hparams.dist_backend,
            init_method=hparams.dist_url,
            world_size=hparams.world_size,
            rank=hparams.rank,
        )

    # get tokenizer
    if hparams.distributed:
        if rank != 0:
            dist.barrier()
        tokenizer = BertTokenizer.from_pretrained(
            "bert-base-uncased", do_lower_case=True
        )
        if rank == 0:
            dist.barrier()
    else:
        tokenizer = BertTokenizer.from_pretrained(
            "bert-base-uncased", do_lower_case=True
        )

    # get dataloaders
    loaders = get_trn_dev_loader(
        dset=load_dataset("imdb", split="train"),
        tok=tokenizer,
        batch_size=hparams.batch_size,
        workers=hparams.workers,
        distributed=hparams.distributed,
    )

    # get model
    if hparams.distributed:
        if rank != 0:
            dist.barrier()
        model = BertForSequenceClassification.from_pretrained("bert-base-uncased")
        if rank == 0:
            dist.barrier()
    else:
        model = BertForSequenceClassification.from_pretrained("bert-base-uncased")

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model Parameters] {num_params}")

    # training phase
    trainer = Trainer(hparams, tokenizer, loaders, model)
    best_result = trainer.fit(num_ckpt=1)

    # testing phase
    if rank in [-1, 0]:
        version = best_result["version"]
        state_dict = torch.load(
            glob.glob(
                os.path.join(hparams.ckpt_path, f"version-{version}/best_model_*.pt")
            )[0]
        )
        test_loader = get_tst_loader(
            dset=load_dataset("imdb", split="test"),
            tok=tokenizer,
            batch_size=hparams.batch_size,
            workers=hparams.workers,
            distributed=False,
        )
        test_result = trainer.test(test_loader, state_dict)

        # save result
        best_result.update(test_result)
        resultwriter.update(hparams, **best_result)


if __name__ == "__main__":
    hparams = load_config()
    ngpus_per_node = torch.cuda.device_count()

    if hparams.distributed:
        hparams.rank = 0
        hparams.world_size = ngpus_per_node * hparams.world_size
        mp.spawn(main, nprocs=ngpus_per_node, args=(hparams, ngpus_per_node))
    else:
        main(hparams.rank, hparams, ngpus_per_node)
