from typing import Optional, Tuple

import lightning as L
import torch
from soundevent.data import PathLike
from torch.optim.adam import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from batdetect2.models import Model, build_model
from batdetect2.train.config import FullTrainingConfig
from batdetect2.train.losses import build_loss
from batdetect2.typing import ModelOutput, TrainExample

__all__ = [
    "TrainingModule",
]


class TrainingModule(L.LightningModule):
    model: Model

    def __init__(
        self,
        config: FullTrainingConfig,
        learning_rate: float = 0.001,
        t_max: int = 100,
        model: Optional[Model] = None,
        loss: Optional[torch.nn.Module] = None,
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.config = config
        self.learning_rate = learning_rate
        self.t_max = t_max

        if loss is None:
            loss = build_loss(self.config.train.loss)

        if model is None:
            model = build_model(self.config)

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
) -> Tuple[Model, FullTrainingConfig]:
    module = TrainingModule.load_from_checkpoint(path)  # type: ignore
    return module.model, module.config


def build_training_module(
    config: Optional[FullTrainingConfig] = None,
    t_max: int = 200,
) -> TrainingModule:
    config = config or FullTrainingConfig()
    return TrainingModule(
        config=config,
        learning_rate=config.train.optimizer.learning_rate,
        t_max=t_max,
    )
