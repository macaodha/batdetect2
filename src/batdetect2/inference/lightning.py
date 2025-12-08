from typing import Sequence

from lightning import LightningModule
from torch.utils.data import DataLoader

from batdetect2.inference.dataset import DatasetItem, InferenceDataset
from batdetect2.models import Model
from batdetect2.postprocess import to_raw_predictions
from batdetect2.typing.postprocess import BatDetect2Prediction


class InferenceModule(LightningModule):
    def __init__(self, model: Model):
        super().__init__()
        self.model = model

    def predict_step(
        self,
        batch: DatasetItem,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> Sequence[BatDetect2Prediction]:
        dataset = self.get_dataset()

        clips = [dataset.clips[int(example_idx)] for example_idx in batch.idx]

        outputs = self.model.detector(batch.spec)

        clip_detections = self.model.postprocessor(
            outputs,
            start_times=[clip.start_time for clip in clips],
        )

        predictions = [
            BatDetect2Prediction(
                clip=clip,
                predictions=to_raw_predictions(
                    clip_dets.numpy(),
                    targets=self.model.targets,
                ),
            )
            for clip, clip_dets in zip(clips, clip_detections, strict=False)
        ]

        return predictions

    def get_dataset(self) -> InferenceDataset:
        dataloaders = self.trainer.predict_dataloaders
        assert isinstance(dataloaders, DataLoader)
        dataset = dataloaders.dataset
        assert isinstance(dataset, InferenceDataset)
        return dataset
