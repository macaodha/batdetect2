from typing import Optional, Union

from lightning import LightningModule
from lightning.pytorch import Trainer
from soundevent.data import PathLike
from torch.utils.data import DataLoader

from batdetect2.configs import BaseConfig, load_config
from batdetect2.train.dataset import LabeledDataset

__all__ = [
    "train",
    "TrainerConfig",
    "load_trainer_config",
]


class TrainerConfig(BaseConfig):
    accelerator: str = "auto"
    accumulate_grad_batches: int = 1
    deterministic: bool = True
    check_val_every_n_epoch: int = 1
    devices: Union[str, int] = "auto"
    enable_checkpointing: bool = True
    gradient_clip_val: Optional[float] = None
    limit_train_batches: Optional[Union[int, float]] = None
    limit_test_batches: Optional[Union[int, float]] = None
    limit_val_batches: Optional[Union[int, float]] = None
    log_every_n_steps: Optional[int] = None
    max_epochs: Optional[int] = None
    min_epochs: Optional[int] = 100
    max_steps: Optional[int] = None
    min_steps: Optional[int] = None
    max_time: Optional[str] = None
    precision: Optional[str] = None
    reload_dataloaders_every_n_epochs: Optional[int] = None
    val_check_interval: Optional[Union[int, float]] = None


def load_trainer_config(path: PathLike, field: Optional[str] = None):
    return load_config(path, schema=TrainerConfig, field=field)


def train(
    module: LightningModule,
    train_dataset: LabeledDataset,
    trainer_config: Optional[TrainerConfig] = None,
    dev_run: bool = False,
    overfit_batches: bool = False,
    profiler: Optional[str] = None,
):
    trainer_config = trainer_config or TrainerConfig()
    trainer = Trainer(
        **trainer_config.model_dump(
            exclude_unset=True,
            exclude_none=True,
        ),
        fast_dev_run=dev_run,
        overfit_batches=overfit_batches,
        profiler=profiler,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=module.config.train.batch_size,
        shuffle=True,
        num_workers=7,
    )
    trainer.fit(module, train_dataloaders=train_loader)
