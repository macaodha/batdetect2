import lightning as L
from soundevent.data import PathLike

from batdetect2.models import Model, ModelConfig, build_model
from batdetect2.models.types import ModelOutput
from batdetect2.train.config import TrainingConfig
from batdetect2.train.losses import build_loss
from batdetect2.train.optimizers import build_optimizer
from batdetect2.train.schedulers import build_scheduler
from batdetect2.train.types import LossProtocol, TrainExample

__all__ = [
    "TrainingModule",
]


class TrainingModule(L.LightningModule):
    model: Model
    loss: LossProtocol

    def __init__(
        self,
        model_config: dict | None = None,
        train_config: dict | None = None,
        loss: LossProtocol | None = None,
        model: Model | None = None,
    ):
        super().__init__()

        self.save_hyperparameters(ignore=["model", "loss"], logger=False)

        self.model_config = ModelConfig.model_validate(model_config or {})
        self.train_config = TrainingConfig.model_validate(train_config or {})

        if loss is None:
            loss = build_loss(config=self.train_config.loss)

        if model is None:
            model = build_model(config=self.model_config)

        self.loss = loss
        self.model = model

    def training_step(self, batch: TrainExample):
        outputs = self.model.detector(batch.spec)
        losses = self.loss(outputs, batch)
        self.log("total_loss/train", losses.total, prog_bar=True, logger=True)
        self.log("detection_loss/train", losses.detection, logger=True)
        self.log("size_loss/train", losses.size, logger=True)
        self.log(
            "classification_loss/train",
            losses.classification,
            logger=True,
        )
        return losses.total

    def validation_step(  # type: ignore
        self,
        batch: TrainExample,
        batch_idx: int,
    ) -> ModelOutput:
        outputs = self.model.detector(batch.spec)
        losses = self.loss(outputs, batch)
        self.log("total_loss/val", losses.total, prog_bar=True, logger=True)
        self.log("detection_loss/val", losses.detection, logger=True)
        self.log("size_loss/val", losses.size, logger=True)
        self.log("classification_loss/val", losses.classification, logger=True)
        return outputs

    def configure_optimizers(self):
        trainable_parameters = [
            parameter
            for parameter in self.parameters()
            if parameter.requires_grad
        ]

        if not trainable_parameters:
            raise ValueError("No trainable parameters available.")

        optimizer = build_optimizer(
            trainable_parameters,
            config=self.train_config.optimizer,
        )
        scheduler = build_scheduler(
            optimizer,
            config=self.train_config.scheduler,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }


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
    model_config: ModelConfig | None = None,
    train_config: TrainingConfig | None = None,
    model: Model | None = None,
) -> TrainingModule:
    if model_config is None:
        model_config = ModelConfig()

    if train_config is None:
        train_config = TrainingConfig()

    return TrainingModule(
        model_config=model_config.model_dump(mode="json"),
        train_config=train_config.model_dump(mode="json"),
        model=model,
    )
