from typing import TYPE_CHECKING

import lightning as L
import torch
from soundevent.data import PathLike
from torch.optim.adam import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from batdetect2.models import Model, ModelConfig, build_model
from batdetect2.train.losses import build_loss
from batdetect2.typing import ModelOutput, TrainExample

if TYPE_CHECKING:
    pass

__all__ = [
    "TrainingModule",
]


class TrainingModule(L.LightningModule):
    model: Model

    def __init__(
        self,
        model_config: dict | None = None,
        t_max: int = 100,
        learning_rate: float = 1e-3,
        loss: torch.nn.Module | None = None,
        model: Model | None = None,
    ):
        super().__init__()

        self.save_hyperparameters(ignore=["model", "loss"], logger=False)

        self.model_config = ModelConfig.model_validate(model_config or {})
        self.learning_rate = learning_rate
        self.t_max = t_max

        if loss is None:
            loss = build_loss()

        if model is None:
            model = build_model(config=self.model_config)

        self.loss = loss
        self.model = model

    def training_step(self, batch: TrainExample):
        outputs = self.model.detector(batch.spec)
        losses = self.loss(outputs, batch)
        self.log("total_loss/train", losses.total, prog_bar=True, logger=True)
        self.log("detection_loss/train", losses.total, logger=True)
        self.log("size_loss/train", losses.total, logger=True)
        self.log("classification_loss/train", losses.total, logger=True)
        return losses.total

    def validation_step(  # type: ignore
        self,
        batch: TrainExample,
        batch_idx: int,
    ) -> ModelOutput:
        outputs = self.model.detector(batch.spec)
        losses = self.loss(outputs, batch)
        self.log("total_loss/val", losses.total, prog_bar=True, logger=True)
        self.log("detection_loss/val", losses.total, logger=True)
        self.log("size_loss/val", losses.total, logger=True)
        self.log("classification_loss/val", losses.total, logger=True)
        return outputs

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.learning_rate)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.t_max)
        return [optimizer], [scheduler]


def load_model_from_checkpoint(
    path: PathLike,
) -> tuple[Model, ModelConfig]:
    """Load a model and its configuration from a Lightning checkpoint.

    Parameters
    ----------
    path : PathLike
        Path to a ``.ckpt`` file produced by the BatDetect2 training
        pipeline.

    Returns
    -------
    tuple[Model, ModelConfig]
        The restored ``Model`` instance and the ``ModelConfig`` that
        describes its architecture, preprocessing, postprocessing, and
        targets.
    """
    module = TrainingModule.load_from_checkpoint(path)  # type: ignore
    return module.model, module.model_config


def build_training_module(
    model_config: dict | None = None,
    t_max: int = 200,
    learning_rate: float = 1e-3,
    loss_config: dict | None = None,
) -> TrainingModule:
    from batdetect2.train.config import LossConfig
    from batdetect2.train.losses import build_loss

    loss = build_loss(LossConfig.model_validate(loss_config or {}))
    return TrainingModule(
        model_config=model_config,
        t_max=t_max,
        learning_rate=learning_rate,
        loss=loss,
    )
