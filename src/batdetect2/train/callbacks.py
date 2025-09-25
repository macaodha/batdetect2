from typing import List

from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from soundevent import data
from torch.utils.data import DataLoader

from batdetect2.logging import get_image_logger
from batdetect2.postprocess import to_raw_predictions
from batdetect2.train.dataset import ValidationDataset
from batdetect2.train.lightning import TrainingModule
from batdetect2.typing import (
    EvaluatorProtocol,
    ModelOutput,
    RawPrediction,
    TrainExample,
)


class ValidationMetrics(Callback):
    def __init__(self, evaluator: EvaluatorProtocol):
        super().__init__()

        self.evaluator = evaluator

        self._clip_annotations: List[data.ClipAnnotation] = []
        self._predictions: List[List[RawPrediction]] = []

    def get_dataset(self, trainer: Trainer) -> ValidationDataset:
        dataloaders = trainer.val_dataloaders
        assert isinstance(dataloaders, DataLoader)
        dataset = dataloaders.dataset
        assert isinstance(dataset, ValidationDataset)
        return dataset

    def generate_plots(
        self,
        pl_module: LightningModule,
    ):
        plotter = get_image_logger(pl_module.logger)  # type: ignore

        if plotter is None:
            return

        for figure_name, fig in self.evaluator.generate_plots(
            self._clip_annotations,
            self._predictions,
        ):
            plotter(figure_name, fig, pl_module.global_step)

    def log_metrics(
        self,
        pl_module: LightningModule,
    ):
        metrics = self.evaluator.compute_metrics(
            self._clip_annotations,
            self._predictions,
        )
        pl_module.log_dict(metrics)

    def on_validation_epoch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        self.log_metrics(pl_module)
        self.generate_plots(pl_module)

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
        dataset = self.get_dataset(trainer)

        clip_annotations = [
            dataset.clip_annotations[int(example_idx)]
            for example_idx in batch.idx
        ]

        clip_detections = model.postprocessor(
            outputs,
            start_times=[ca.clip.start_time for ca in clip_annotations],
        )
        predictions = [
            to_raw_predictions(clip_dets.numpy(), targets=model.targets)
            for clip_dets in clip_detections
        ]

        self._clip_annotations.extend(clip_annotations)
        self._predictions.extend(predictions)
