from pathlib import Path
from typing import Optional

import lightning as L
import torch
from pydantic import Field
from soundevent import data
from torch.optim.adam import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from batdetect2.configs import BaseConfig
from batdetect2.evaluate.evaluate import match_predictions_and_annotations
from batdetect2.models import (
    BBoxHead,
    ClassifierHead,
    ModelConfig,
    build_architecture,
)
from batdetect2.models.typing import ModelOutput
from batdetect2.post_process import (
    PostprocessConfig,
    postprocess_model_outputs,
)
from batdetect2.preprocess import PreprocessingConfig, preprocess_audio_clip
from batdetect2.train.config import TrainingConfig
from batdetect2.train.dataset import LabeledDataset, TrainExample
from batdetect2.train.losses import compute_loss
from batdetect2.train.targets import (
    TargetConfig,
    build_decoder,
    build_encoder,
    get_class_names,
)

__all__ = [
    "DetectorModel",
]


class ModuleConfig(BaseConfig):
    train: TrainingConfig = Field(default_factory=TrainingConfig)
    targets: TargetConfig = Field(default_factory=TargetConfig)
    architecture: ModelConfig = Field(default_factory=ModelConfig)
    preprocessing: PreprocessingConfig = Field(
        default_factory=PreprocessingConfig
    )
    postprocessing: PostprocessConfig = Field(
        default_factory=PostprocessConfig
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

        self.backbone = build_architecture(self.config.architecture)

        self.classifier = ClassifierHead(
            num_classes=len(self.config.targets.classes),
            in_channels=self.backbone.out_channels,
        )

        self.bbox = BBoxHead(in_channels=self.backbone.out_channels)

        conf = self.config.train.loss.classification
        self.class_weights = (
            torch.tensor(conf.class_weights) if conf.class_weights else None
        )

        # Training targets
        self.class_names = get_class_names(self.config.targets.classes)
        self.encoder = build_encoder(
            self.config.targets.classes,
            replacement_rules=self.config.targets.replace,
        )
        self.decoder = build_decoder(self.config.targets.classes)

        self.validation_predictions = []

        self.example_input_array = torch.randn([1, 1, 128, 512])

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

        self.log("train/loss/total", losses.total, prog_bar=True, logger=True)
        self.log("train/loss/detection", losses.total, logger=True)
        self.log("train/loss/size", losses.total, logger=True)
        self.log("train/loss/classification", losses.total, logger=True)

        return losses.total

    def validation_step(self, batch: TrainExample, batch_idx: int) -> None:
        outputs = self.forward(batch.spec)

        losses = compute_loss(
            batch,
            outputs,
            conf=self.config.train.loss,
            class_weights=self.class_weights,
        )

        self.log("val/loss/total", losses.total, prog_bar=True, logger=True)
        self.log("val/loss/detection", losses.total, logger=True)
        self.log("val/loss/size", losses.total, logger=True)
        self.log("val/loss/classification", losses.total, logger=True)

        dataloaders = self.trainer.val_dataloaders
        assert isinstance(dataloaders, DataLoader)
        dataset = dataloaders.dataset
        assert isinstance(dataset, LabeledDataset)
        clip_annotation = dataset.get_clip_annotation(batch_idx)

        clip_prediction = postprocess_model_outputs(
            outputs,
            clips=[clip_annotation.clip],
            classes=self.class_names,
            decoder=self.decoder,
            config=self.config.postprocessing,
        )[0]

        matches = match_predictions_and_annotations(
            clip_annotation,
            clip_prediction,
        )

        self.validation_predictions.extend(matches)

    def on_validation_epoch_end(self) -> None:
        self.validation_predictions.clear()

    def configure_optimizers(self):
        conf = self.config.train.optimizer
        optimizer = Adam(self.parameters(), lr=conf.learning_rate)
        scheduler = CosineAnnealingLR(optimizer, T_max=conf.t_max)
        return [optimizer], [scheduler]

    def process_clip(
        self,
        clip: data.Clip,
        audio_dir: Optional[Path] = None,
    ) -> data.ClipPrediction:
        spec = preprocess_audio_clip(
            clip,
            config=self.config.preprocessing,
            audio_dir=audio_dir,
        )
        tensor = torch.from_numpy(spec.data).unsqueeze(0).unsqueeze(0)
        outputs = self.forward(tensor)
        return postprocess_model_outputs(
            outputs,
            clips=[clip],
            classes=self.class_names,
            decoder=self.decoder,
            config=self.config.postprocessing,
        )[0]
