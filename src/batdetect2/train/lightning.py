import lightning as L
import torch
from pydantic import BaseModel
from torch.optim.adam import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from batdetect2.models import ModelOutput, build_model
from batdetect2.postprocess import build_postprocessor
from batdetect2.preprocess import build_preprocessor
from batdetect2.targets import build_targets
from batdetect2.train import TrainExample
from batdetect2.train.config import FullTrainingConfig
from batdetect2.train.losses import build_loss

__all__ = [
    "TrainingModule",
]


class TrainingModule(L.LightningModule):
    def __init__(self, config: FullTrainingConfig):
        super().__init__()

        # NOTE: Need to convert to vanilla python object so that DVCLive can
        # store it.
        self._config = (
            config.model_dump() if isinstance(config, BaseModel) else config
        )
        self.save_hyperparameters({"config": self._config})

        self.config = FullTrainingConfig.model_validate(self._config)
        self.loss = build_loss(self.config.train.loss)
        self.targets = build_targets(self.config.targets)
        self.detector = build_model(
            num_classes=len(self.targets.class_names),
            config=self.config.model,
        )
        self.preprocessor = build_preprocessor(self.config.preprocess)
        self.postprocessor = build_postprocessor(
            self.targets,
            min_freq=self.preprocessor.min_freq,
            max_freq=self.preprocessor.max_freq,
        )

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
        self,
        batch: TrainExample,
        batch_idx: int,
    ) -> ModelOutput:
        outputs = self.forward(batch.spec)
        losses = self.loss(outputs, batch)
        self.log("total_loss/val", losses.total, prog_bar=True, logger=True)
        self.log("detection_loss/val", losses.total, logger=True)
        self.log("size_loss/val", losses.total, logger=True)
        self.log("classification_loss/val", losses.total, logger=True)

        return outputs

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.config.train.learning_rate)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.config.train.t_max)
        return [optimizer], [scheduler]
