from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR

from batdetect2.train.schedulers import (
    CosineAnnealingSchedulerConfig,
    SchedulerImportConfig,
    build_scheduler,
)


def test_build_scheduler_uses_epoch_t_max_directly():
    model = nn.Linear(4, 2)
    optimizer = SGD(model.parameters(), lr=1e-3)
    scheduler = build_scheduler(
        optimizer,
        config=CosineAnnealingSchedulerConfig(t_max=7),
    )

    assert isinstance(scheduler, CosineAnnealingLR)
    assert scheduler.T_max == 7


def test_build_scheduler_supports_import_config():
    model = nn.Linear(4, 2)
    optimizer = SGD(model.parameters(), lr=1e-3)
    scheduler = build_scheduler(
        optimizer,
        config=SchedulerImportConfig(
            target="torch.optim.lr_scheduler.StepLR",
            arguments={"step_size": 2},
        ),
    )

    assert isinstance(scheduler, StepLR)
