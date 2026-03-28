from typing import Sequence

from lightning import LightningModule
from torch.utils.data import DataLoader

from batdetect2.inference.dataset import DatasetItem, InferenceDataset
from batdetect2.models import Model
from batdetect2.outputs import OutputTransformProtocol, build_output_transform
from batdetect2.postprocess.types import ClipDetections


class InferenceModule(LightningModule):
    def __init__(
        self,
        model: Model,
        output_transform: OutputTransformProtocol | None = None,
        detection_threshold: float | None = None,
    ):
        super().__init__()
        self.model = model
        self.detection_threshold = detection_threshold
        self.output_transform = output_transform or build_output_transform(
            targets=model.targets
        )

    def predict_step(
        self,
        batch: DatasetItem,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> Sequence[ClipDetections]:
        dataset = self.get_dataset()

        clips = [dataset.clips[int(example_idx)] for example_idx in batch.idx]

        outputs = self.model.detector(batch.spec)

        clip_detections = self.model.postprocessor(
            outputs,
            detection_threshold=self.detection_threshold,
        )

        return [
            self.output_transform.to_clip_detections(
                detections=clip_dets,
                clip=clip,
            )
            for clip, clip_dets in zip(clips, clip_detections, strict=True)
        ]

    def get_dataset(self) -> InferenceDataset:
        dataloaders = self.trainer.predict_dataloaders
        assert isinstance(dataloaders, DataLoader)
        dataset = dataloaders.dataset
        assert isinstance(dataset, InferenceDataset)
        return dataset
