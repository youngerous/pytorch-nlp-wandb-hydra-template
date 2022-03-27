from dataclasses import MISSING, dataclass

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


def register_configs() -> None:
    cs = ConfigStore.instance()
    cs.store(group="gpu", name="ddp", node=DDP)
    cs.store(group="gpu", name="dp", node=DP)
