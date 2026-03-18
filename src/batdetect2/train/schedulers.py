"""Scheduler configuration and factory utilities for training."""

from typing import Annotated, Literal

from pydantic import Field
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler

from batdetect2.core import (
    BaseConfig,
    ImportConfig,
    Registry,
    add_import_config,
)

__all__ = [
    "CosineAnnealingSchedulerConfig",
    "SchedulerConfig",
    "SchedulerImportConfig",
    "build_scheduler",
    "scheduler_registry",
]


class CosineAnnealingSchedulerConfig(BaseConfig):
    """Configuration for ``CosineAnnealingLR``.

    Attributes
    ----------
    name : Literal["cosine_annealing"]
        Discriminator field used by the scheduler registry.
    t_max : int
        Number of epochs to complete one cosine cycle.
    """

    name: Literal["cosine_annealing"] = "cosine_annealing"
    t_max: int = 100


scheduler_registry: Registry[LRScheduler, [Optimizer]] = Registry("scheduler")


@add_import_config(scheduler_registry, arg_names=["optimizer"])
class SchedulerImportConfig(ImportConfig):
    """Use any callable as a scheduler.

    Set ``name="import"`` and provide a ``target`` pointing to any callable
    that returns a scheduler. The optimizer instance is passed as the
    ``optimizer`` keyword argument.
    """

    name: Literal["import"] = "import"


@scheduler_registry.register(CosineAnnealingSchedulerConfig)
def build_cosine_scheduler(
    config: CosineAnnealingSchedulerConfig,
    optimizer: Optimizer,
) -> LRScheduler:
    """Build a cosine annealing scheduler.

    ``t_max`` is interpreted in epochs because Lightning steps the scheduler
    once per epoch when ``interval="epoch"`` is used.
    """
    return CosineAnnealingLR(optimizer, T_max=config.t_max)


SchedulerConfig = Annotated[
    CosineAnnealingSchedulerConfig | SchedulerImportConfig,
    Field(discriminator="name"),
]


def build_scheduler(
    optimizer: Optimizer,
    config: SchedulerConfig | None = None,
) -> LRScheduler:
    """Build a scheduler from configuration."""
    config = config or CosineAnnealingSchedulerConfig()

    return scheduler_registry.build(config, optimizer)
