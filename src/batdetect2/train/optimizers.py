"""Optimizer configuration and factory utilities for training."""

from collections.abc import Iterable
from typing import Annotated, Literal

from pydantic import Field
from torch import nn
from torch.optim import Adam, Optimizer

from batdetect2.core import (
    BaseConfig,
    ImportConfig,
    Registry,
    add_import_config,
)

__all__ = [
    "AdamOptimizerConfig",
    "OptimizerConfig",
    "OptimizerImportConfig",
    "build_optimizer",
    "optimizer_registry",
]


class AdamOptimizerConfig(BaseConfig):
    """Configuration for the Adam optimizer.

    Attributes
    ----------
    name : Literal["adam"]
        Discriminator field used by the optimizer registry.
    learning_rate : float
        Learning rate used by ``torch.optim.Adam``.
    """

    name: Literal["adam"] = "adam"
    learning_rate: float = 1e-3


optimizer_registry: Registry[Optimizer, [Iterable[nn.Parameter]]] = Registry(
    "optimizer"
)


@add_import_config(optimizer_registry, arg_names=["params"])
class OptimizerImportConfig(ImportConfig):
    """Use any callable as an optimizer.

    Set ``name="import"`` and provide a ``target`` pointing to any callable
    that returns an optimizer. The training parameters are passed as the
    ``params`` keyword argument.
    """

    name: Literal["import"] = "import"


@optimizer_registry.register(AdamOptimizerConfig)
def build_adam(
    config: AdamOptimizerConfig,
    params: Iterable[nn.Parameter],
) -> Optimizer:
    """Build an Adam optimizer from configuration."""
    return Adam(params, lr=config.learning_rate)


OptimizerConfig = Annotated[
    AdamOptimizerConfig | OptimizerImportConfig,
    Field(discriminator="name"),
]


def build_optimizer(
    parameters: Iterable[nn.Parameter],
    config: OptimizerConfig | None = None,
) -> Optimizer:
    """Build an optimizer from configuration.

    Parameters
    ----------
    parameters : Iterable[nn.Parameter]
        Model parameters to optimize.
    config : OptimizerConfig, optional
        Optimizer configuration. Defaults to ``AdamOptimizerConfig``.
    """
    config = config or AdamOptimizerConfig()
    return optimizer_registry.build(config, parameters)
