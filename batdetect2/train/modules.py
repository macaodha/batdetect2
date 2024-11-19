from typing import Optional

import pytorch_lightning as L
import torch
from pydantic import Field
from torch import optim

from batdetect2.configs import BaseConfig
from batdetect2.models import (
    BBoxHead,
    ClassifierHead,
    ModelConfig,
    get_backbone,
)
from batdetect2.models.typing import ModelOutput
from batdetect2.preprocess import PreprocessingConfig
from batdetect2.train.dataset import TrainExample
from batdetect2.train.losses import LossConfig, compute_loss
from batdetect2.train.targets import TargetConfig


class OptimizerConfig(BaseConfig):
    learning_rate: float = 1e-3
    t_max: int = 100


class TrainingConfig(BaseConfig):
    loss: LossConfig = Field(default_factory=LossConfig)
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)


class ModuleConfig(BaseConfig):
    train: TrainingConfig = Field(default_factory=TrainingConfig)
    targets: TargetConfig = Field(default_factory=TargetConfig)
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    backbone: ModelConfig = Field(default_factory=ModelConfig)
    preprocessing: PreprocessingConfig = Field(
        default_factory=PreprocessingConfig
    )


class DetectorModel(L.LightningModule):
    config: ModuleConfig

    def __init__(
        self,
        config: Optional[ModuleConfig] = None,
    ):
        super().__init__()
        self.config = config or ModuleConfig()
        self.save_hyperparameters()

        self.backbone = get_backbone(
            input_height=self.config.preprocessing.spectrogram.size.height,
            config=self.config.backbone,
        )

        self.classifier = ClassifierHead(
            num_classes=len(self.config.targets.classes),
            in_channels=self.backbone.out_channels,
        )

        self.bbox = BBoxHead(in_channels=self.backbone.out_channels)

        conf = self.training_config.loss.classification
        self.class_weights = (
            torch.tensor(conf.class_weights) if conf.class_weights else None
        )

    def forward(self, spec: torch.Tensor) -> ModelOutput:  # type: ignore
        features = self.backbone(spec)
        detection_probs, classification_probs = self.classifier(features)
        size_preds = self.bbox(features)
        return ModelOutput(
            detection_probs=detection_probs,
            size_preds=size_preds,
            class_probs=classification_probs,
            features=features,
        )

    def training_step(self, batch: TrainExample):
        outputs = self.forward(batch.spec)
        losses = compute_loss(
            batch,
            outputs,
            conf=self.config.train.loss,
            class_weights=self.class_weights,
        )
        return losses.total

    def configure_optimizers(self):
        conf = self.config.train.optimizer
        optimizer = optim.Adam(self.parameters(), lr=conf.learning_rate)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=conf.t_max,
        )
        return [optimizer], [scheduler]
