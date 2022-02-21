from dataclasses import MISSING, dataclass
from typing import Any
from hydra.core.config_store import ConfigStore


@dataclass
class Distributed:
    distributed: bool = MISSING
    rank: int = MISSING


@dataclass
class DDP(Distributed):
    distributed: bool = True
    dist_backend: str = "nccl"
    dist_url: str = "tcp://127.0.0.1:3456"
    world_size: int = 1
    rank: int = 0


@dataclass
class DP(Distributed):
    distributed: bool = False
    rank: int = -1


@dataclass
class TrainConf:
    gpu: Distributed
    test: bool = True
    amp: bool = True  # torch >= 1.6.x
    ckpt_root: str = "/repo/pytorch-nlp-wandb-hydra-template/src/checkpoints/"

    seed: int = 42
    workers: int = 1
    log_step: int = 200
    eval_ratio: float = 0.0  # evaluation will be done at the end of epoch if set to 0.0

    epoch: int = 10
    batch_size: int = 16  # it will be divided by num_gpu in DDP
    lr: float = 5e-5
    weight_decay: float = 0.1
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    gradient_accumulation_step: int = 1


cs = ConfigStore.instance()
cs.store(name="train", node=TrainConf)
cs.store(group="gpu", name="ddp", node=DDP)
cs.store(group="gpu", name="dp", node=DP)
