import lightning as L
import torch
from torch.optim.adam import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from batdetect2.models import (
    DetectionModel,
    ModelOutput,
)
from batdetect2.postprocess.types import PostprocessorProtocol
from batdetect2.preprocess.types import PreprocessorProtocol
from batdetect2.targets.types import TargetProtocol
from batdetect2.train import TrainExample
from batdetect2.train.types import LossProtocol

__all__ = [
    "TrainingModule",
]


class TrainingModule(L.LightningModule):
    def __init__(
        self,
        detector: DetectionModel,
        loss: LossProtocol,
        targets: TargetProtocol,
        preprocessor: PreprocessorProtocol,
        postprocessor: PostprocessorProtocol,
        learning_rate: float = 0.001,
        t_max: int = 100,
    ):
        super().__init__()

        self.loss = loss
        self.detector = detector
        self.preprocessor = preprocessor
        self.targets = targets
        self.postprocessor = postprocessor

        self.learning_rate = learning_rate
        self.t_max = t_max

        # NOTE: Ignore detector and loss from hyperparameter saving
        # as they are nn.Module and should be saved regardless.
        self.save_hyperparameters(ignore=["detector", "loss"])

    def forward(self, spec: torch.Tensor) -> ModelOutput:
        return self.detector(spec)

    def training_step(self, batch: TrainExample):
        outputs = self.forward(batch.spec)
        losses = self.loss(outputs, batch)

        self.log("total_loss/train", losses.total, prog_bar=True, logger=True)
        self.log("detection_loss/train", losses.total, logger=True)
        self.log("size_loss/train", losses.total, logger=True)
        self.log("classification_loss/train", losses.total, logger=True)

        return losses.total

    def validation_step(  # type: ignore
        self, batch: TrainExample, batch_idx: int
    ) -> ModelOutput:
        outputs = self.forward(batch.spec)
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
