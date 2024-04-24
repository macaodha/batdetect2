import pytorch_lightning as L
import torch
import xarray as xr
from soundevent import data
from torch import nn, optim

from batdetect2.data.preprocessing import preprocess_audio_clip
from batdetect2.models.typing import EncoderModel, ModelOutput
from batdetect2.train import losses
from batdetect2.train.dataset import TrainExample
from batdetect2.models.post_process import (
    PostprocessConfig,
    postprocess_model_outputs,
)
from batdetect2.train.preprocess import PreprocessingConfig


class DetectorModel(L.LightningModule):
    def __init__(
        self,
        encoder: EncoderModel,
        num_classes: int,
        learning_rate: float = 1e-3,
        preprocessing_config: PreprocessingConfig = PreprocessingConfig(),
        postprocessing_config: PostprocessConfig = PostprocessConfig(),
    ):
        super().__init__()

        self.preprocessing_config = preprocessing_config
        self.postprocessing_config = postprocessing_config
        self.num_classes = num_classes
        self.learning_rate = learning_rate

        self.encoder = encoder

        self.classifier = nn.Conv2d(
            self.encoder.num_filts // 4,
            self.num_classes + 1,
            kernel_size=1,
            padding=0,
        )

        self.bbox = nn.Conv2d(
            self.encoder.num_filts // 4,
            2,
            kernel_size=1,
            padding=0,
        )

    def forward(self, spec: torch.Tensor) -> ModelOutput:  # type: ignore
        features = self.encoder(spec)

        classification_logits = self.classifier(features)
        classification_probs = torch.softmax(classification_logits, dim=1)
        detection_probs = classification_probs[:, :-1].sum(dim=1, keepdim=True)

        return ModelOutput(
            detection_probs=detection_probs,
            size_preds=self.bbox(features),
            class_probs=classification_probs,
            features=features,
        )

    def compute_spectrogram(self, clip: data.Clip) -> xr.DataArray:
        config = self.preprocessing_config

        return preprocess_audio_clip(
            clip,
            target_sampling_rate=config.target_samplerate,
            scale_audio=config.scale_audio,
            fft_win_length=config.fft_win_length,
            fft_overlap=config.fft_overlap,
            max_freq=config.max_freq,
            min_freq=config.min_freq,
            spec_scale=config.spec_scale,
            denoise_spec_avg=config.denoise_spec_avg,
            max_scale_spec=config.max_scale_spec,
        )

    def process_clip(self, clip: data.Clip):
        spectrogram = self.compute_spectrogram(clip)
        spec_tensor = (
            torch.tensor(spectrogram.values).unsqueeze(0).unsqueeze(0)
        )

        outputs = self(spec_tensor)

        config = self.postprocessing_config
        return postprocess_model_outputs(
            outputs,
            [clip],
            nms_kernel_size=config.nms_kernel_size,
            detection_threshold=config.detection_threshold,
            min_freq=config.min_freq,
            max_freq=config.max_freq,
            top_k_per_sec=config.top_k_per_sec,
        )

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
        features = self.encoder(batch.spec)

        classification_logits = self.classifier(features)
        classification_probs = torch.softmax(classification_logits, dim=1)
        detection_probs = classification_probs[:, :-1].sum(dim=1, keepdim=True)

        loss = self.compute_loss(
            ModelOutput(
                detection_probs=detection_probs,
                size_preds=self.bbox(features),
                class_probs=classification_probs,
                features=features,
            ),
            batch,
        )
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 100)
        return [optimizer], [scheduler]
