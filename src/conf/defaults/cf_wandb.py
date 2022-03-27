from dataclasses import MISSING, dataclass

from hydra.core.config_store import ConfigStore


@dataclass
class DefaultWandB:
    project: str = MISSING
    entity: str = MISSING


def register_configs() -> None:
    cs = ConfigStore.instance()
    cs.store(group="wandb", name="default", node=DefaultWandB)
