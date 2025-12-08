from typing import TYPE_CHECKING, Tuple

import lightning as L
import torch
from soundevent.data import PathLike
from torch.optim.adam import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from batdetect2.models import Model, build_model
from batdetect2.postprocess import build_postprocessor
from batdetect2.preprocess import build_preprocessor
from batdetect2.targets import build_targets
from batdetect2.train.losses import build_loss
from batdetect2.typing import ModelOutput, TrainExample

if TYPE_CHECKING:
    from batdetect2.config import BatDetect2Config

__all__ = [
    "TrainingModule",
]


class TrainingModule(L.LightningModule):
    model: Model

    def __init__(
        self,
        config: dict | None = None,
        t_max: int = 100,
        model: Model | None = None,
        loss: torch.nn.Module | None = None,
    ):
        from batdetect2.config import validate_config

        super().__init__()

        self.save_hyperparameters(logger=False)

        self.config = validate_config(config)
        self.input_samplerate = self.config.audio.samplerate
        self.learning_rate = self.config.train.optimizer.learning_rate
        self.t_max = t_max

        if loss is None:
            loss = build_loss(self.config.train.loss)

        if model is None:
            targets = build_targets(self.config.targets)

            preprocessor = build_preprocessor(
                config=self.config.preprocess,
                input_samplerate=self.input_samplerate,
            )

            postprocessor = build_postprocessor(
                preprocessor, config=self.config.postprocess
            )

            model = build_model(
                config=self.config.model,
                targets=targets,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
            )

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
) -> Tuple[Model, "BatDetect2Config"]:
    module = TrainingModule.load_from_checkpoint(path)  # type: ignore
    return module.model, module.config


def build_training_module(
    config: dict | None = None,
    t_max: int = 200,
) -> TrainingModule:
    return TrainingModule(config=config, t_max=t_max)
