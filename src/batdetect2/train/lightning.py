import lightning as L
import torch
from torch.optim.adam import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from batdetect2.models import Model
from batdetect2.typing import ModelOutput, TrainExample

__all__ = [
    "TrainingModule",
]


class TrainingModule(L.LightningModule):
    model: Model

    def __init__(
        self,
        model: Model,
        loss: torch.nn.Module,
        learning_rate: float = 0.001,
        t_max: int = 100,
    ):
        super().__init__()

        self.learning_rate = learning_rate
        self.t_max = t_max

        self.loss = loss
        self.model = model
        self.save_hyperparameters(logger=False)

    def forward(self, spec: torch.Tensor) -> ModelOutput:
        return self.model.detector(spec)

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
