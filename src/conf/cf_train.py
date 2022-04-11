from typing import Any
import conf.defaults.cf_distributed as cf_distributed

from dataclasses import dataclass
from hydra.core.config_store import ConfigStore


@dataclass
class TrainConf:
    gpu: cf_distributed.Distributed
    wandb: Any

    test: bool = True
    amp: bool = True  # torch >= 1.6.x
    ckpt_root: str = "/repo/pytorch-nlp-wandb-hydra-template/src/checkpoints/"

    seed: int = 42
    workers: int = 1
    log_step: int = 200
    eval_ratio: float = 0.0  # evaluation will be done at the end of epoch if set to 0.0
    early_stop_tolerance: int = -1

    epoch: int = 10
    batch_size: int = 16  # it will be divided by num_gpu in DDP
    lr: float = 5e-5
    weight_decay: float = 0.1
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    gradient_accumulation_step: int = 1


cf_distributed.register_configs()

cs = ConfigStore.instance()
cs.store(name="train", node=TrainConf)
