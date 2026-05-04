from typing import Sequence

from lightning import LightningModule
from torch.utils.data import DataLoader

from batdetect2.inference.dataset import DatasetItem, InferenceDataset
from batdetect2.models import Model
from batdetect2.outputs import OutputTransformProtocol, build_output_transform
from batdetect2.postprocess.types import ClipDetections
from batdetect2.targets.types import ROIMapperProtocol, TargetProtocol


class InferenceModule(LightningModule):
    def __init__(
        self,
        model: Model,
        targets: TargetProtocol | None = None,
        roi_mapper: ROIMapperProtocol | None = None,
        output_transform: OutputTransformProtocol | None = None,
        detection_threshold: float | None = None,
    ):
        super().__init__()
        self.model = model
        self.detection_threshold = detection_threshold

        if output_transform is None and targets is None:
            raise ValueError(
                "targets must be provided when building inference output "
                "transforms."
            )

        if output_transform is None and roi_mapper is None:
            raise ValueError(
                "roi_mapper must be provided when building inference output "
                "transforms."
            )

        self.output_transform = output_transform or build_output_transform(
            targets=targets,
            roi_mapper=roi_mapper,
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
