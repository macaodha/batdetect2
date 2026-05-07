from torch import nn
from torch.optim import SGD, Adam

from batdetect2.train.optimizers import OptimizerImportConfig, build_optimizer


def test_build_optimizer_defaults_to_adam():
    model = nn.Linear(4, 2)
    optimizer = build_optimizer(model.parameters())

    assert isinstance(optimizer, Adam)


def test_build_optimizer_supports_import_config():
    model = nn.Linear(4, 2)
    config = OptimizerImportConfig(
        target="torch.optim.SGD",
        arguments={"lr": 1e-3},
    )

    optimizer = build_optimizer(model.parameters(), config=config)
    assert isinstance(optimizer, SGD)
