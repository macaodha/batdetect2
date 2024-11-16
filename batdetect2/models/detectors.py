from typing import Optional, Type

import pytorch_lightning as L
import torch
import xarray as xr
from soundevent import data
from torch import nn, optim

from batdetect2.data.labels import ClassMapper
from batdetect2.data.preprocessing import (
    PreprocessingConfig,
    preprocess_audio_clip,
)
from batdetect2.models.feature_extractors import Net2DFast
from batdetect2.models.post_process import (
    PostprocessConfig,
    postprocess_model_outputs,
)
from batdetect2.models.typing import FeatureExtractorModel, ModelOutput
from batdetect2.train import losses
from batdetect2.train.dataset import TrainExample


class DetectorModel(L.LightningModule):
    def __init__(
        self,
        class_mapper: ClassMapper,
        feature_extractor_class: Type[FeatureExtractorModel] = Net2DFast,
        learning_rate: float = 1e-3,
        input_height: int = 128,
        num_features: int = 32,
        preprocessing_config: Optional[PreprocessingConfig] = None,
        postprocessing_config: Optional[PostprocessConfig] = None,
    ):
        super().__init__()

        preprocessing_config = preprocessing_config or PreprocessingConfig()
        postprocessing_config = postprocessing_config or PostprocessConfig()

        self.save_hyperparameters()

        self.preprocessing_config = preprocessing_config
        self.postprocessing_config = postprocessing_config
        self.class_mapper = class_mapper
        self.learning_rate = learning_rate
        self.input_height = input_height
        self.num_features = num_features
        self.num_classes = class_mapper.num_classes

        self.feature_extractor = feature_extractor_class(
            input_height=input_height,
            num_features=num_features,
        )

        self.classifier = nn.Conv2d(
            self.feature_extractor.num_features // 4,
            self.num_classes + 1,
            kernel_size=1,
            padding=0,
        )

        self.bbox = nn.Conv2d(
            self.feature_extractor.num_features // 4,
            2,
            kernel_size=1,
            padding=0,
        )

    def forward(self, spec: torch.Tensor) -> ModelOutput:  # type: ignore
        features = self.feature_extractor(spec)
        classification_logits = self.classifier(features)
        classification_probs = torch.softmax(classification_logits, dim=1)
        detection_probs = classification_probs[:, :-1].sum(dim=1, keepdim=True)
        return ModelOutput(
            detection_probs=detection_probs,
            size_preds=self.bbox(features),
            class_probs=classification_probs[:, :-1],
            features=features,
        )

    def compute_spectrogram(self, clip: data.Clip) -> xr.DataArray:
        return preprocess_audio_clip(
            clip,
            config=self.preprocessing_config,
        )

    def compute_clip_features(self, clip: data.Clip) -> torch.Tensor:
        spectrogram = self.compute_spectrogram(clip)
        return self.feature_extractor(
            torch.tensor(spectrogram.values).unsqueeze(0).unsqueeze(0)
        )

    def compute_clip_predictions(self, clip: data.Clip) -> data.ClipPrediction:
        spectrogram = self.compute_spectrogram(clip)
        spec_tensor = (
            torch.tensor(spectrogram.values).unsqueeze(0).unsqueeze(0)
        )
        outputs = self(spec_tensor)
        return postprocess_model_outputs(
            outputs,
            [clip],
            class_mapper=self.class_mapper,
            config=self.postprocessing_config,
        )[0]

    def compute_loss(
        self,
        outputs: ModelOutput,
        batch: TrainExample,
    ) -> torch.Tensor:
        detection_loss = losses.focal_loss(
            outputs.detection_probs,
            batch.detection_heatmap,
        )

        size_loss = losses.bbox_size_loss(
            outputs.size_preds,
            batch.size_heatmap,
        )

        valid_mask = batch.class_heatmap.any(dim=1, keepdim=True).float()
        classification_loss = losses.focal_loss(
            outputs.class_probs,
            batch.class_heatmap,
            valid_mask=valid_mask,
        )

        return detection_loss + size_loss + classification_loss

    def training_step(  # type: ignore
        self,
        batch: TrainExample,
    ):
        outputs = self.forward(batch.spec)
        loss = self.compute_loss(outputs, batch)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 100)
        return [optimizer], [scheduler]
