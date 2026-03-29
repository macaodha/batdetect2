from typing import Any, List

from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from soundevent import data
from torch.utils.data import DataLoader

from batdetect2.evaluate.types import EvaluatorProtocol
from batdetect2.logging import get_image_logger
from batdetect2.models.types import ModelOutput
from batdetect2.outputs import OutputTransformProtocol, build_output_transform
from batdetect2.postprocess.types import ClipDetections
from batdetect2.train.dataset import ValidationDataset
from batdetect2.train.lightning import TrainingModule
from batdetect2.train.types import TrainExample


class ValidationMetrics(Callback):
    def __init__(
        self,
        evaluator: EvaluatorProtocol,
        output_transform: OutputTransformProtocol | None = None,
    ):
        super().__init__()

        self.evaluator = evaluator
        self.output_transform = output_transform

        self._clip_annotations: List[data.ClipAnnotation] = []
        self._predictions: List[ClipDetections] = []

    def get_dataset(self, trainer: Trainer) -> ValidationDataset:
        dataloaders = trainer.val_dataloaders
        assert isinstance(dataloaders, DataLoader)
        dataset = dataloaders.dataset
        assert isinstance(dataset, ValidationDataset)
        return dataset

    def generate_plots(
        self,
        eval_outputs: Any,
        pl_module: LightningModule,
    ):
        plotter = get_image_logger(pl_module.logger)  # type: ignore

        if plotter is None:
            return

        for figure_name, fig in self.evaluator.generate_plots(eval_outputs):
            plotter(figure_name, fig, pl_module.global_step)

    def log_metrics(
        self,
        eval_outputs: Any,
        pl_module: LightningModule,
    ):
        metrics = self.evaluator.compute_metrics(eval_outputs)
        pl_module.log_dict(metrics)

    def on_validation_epoch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        eval_outputs = self.evaluator.evaluate(
            self._clip_annotations,
            self._predictions,
        )

        self.log_metrics(eval_outputs, pl_module)
        self.generate_plots(eval_outputs, pl_module)

        return super().on_validation_epoch_end(trainer, pl_module)

    def on_validation_epoch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        self._clip_annotations = []
        self._predictions = []
        return super().on_validation_epoch_start(trainer, pl_module)

    def on_validation_batch_end(  # type: ignore
        self,
        trainer: Trainer,
        pl_module: TrainingModule,
        outputs: ModelOutput,
        batch: TrainExample,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        model = pl_module.model
        if self.output_transform is None:
            self.output_transform = build_output_transform(
                targets=model.targets,
                roi_mapper=model.roi_mapper,
            )

        output_transform = self.output_transform
        assert output_transform is not None

        dataset = self.get_dataset(trainer)

        clip_annotations = [
            dataset.clip_annotations[int(example_idx)]
            for example_idx in batch.idx
        ]

        clip_detections = model.postprocessor(outputs)
        predictions = [
            output_transform.to_clip_detections(
                detections=clip_dets,
                clip=clip_annotation.clip,
            )
            for clip_annotation, clip_dets in zip(
                clip_annotations, clip_detections, strict=False
            )
        ]

        self._clip_annotations.extend(clip_annotations)
        self._predictions.extend(predictions)
