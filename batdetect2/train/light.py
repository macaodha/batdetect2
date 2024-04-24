import pytorch_lightning as L
from torch import Tensor, optim

from batdetect2.models.typing import DetectionModel, ModelOutput
from batdetect2.train import losses

from batdetect2.train.dataset import TrainExample


__all__ = [
    "LitDetectorModel",
]


class LitDetectorModel(L.LightningModule):
    model: DetectionModel

    def __init__(self, model: DetectionModel, learning_rate: float = 1e-3):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate

    def compute_loss(
        self,
        outputs: ModelOutput,
        batch: TrainExample,
    ) -> Tensor:
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

    def training_step(self, batch: TrainExample, batch_idx: int):  # type: ignore
        outputs: ModelOutput = self.model(batch.spec)
        loss = self.compute_loss(outputs, batch)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 100)
        return [optimizer], [scheduler]
