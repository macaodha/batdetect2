from typing import List, Optional

from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from soundevent import data
from torch.utils.data import DataLoader

from batdetect2.evaluate.match import (
    MatchConfig,
    match_all_predictions,
)
from batdetect2.plotting.clips import PreprocessorProtocol
from batdetect2.plotting.evaluation import plot_example_gallery
from batdetect2.postprocess import get_raw_predictions
from batdetect2.train.dataset import ValidationDataset
from batdetect2.train.lightning import TrainingModule
from batdetect2.train.logging import get_image_plotter
from batdetect2.typing import (
    MatchEvaluation,
    MetricsProtocol,
)
from batdetect2.typing.models import ModelOutput
from batdetect2.typing.postprocess import RawPrediction
from batdetect2.typing.train import TrainExample


class ValidationMetrics(Callback):
    def __init__(
        self,
        metrics: List[MetricsProtocol],
        preprocessor: PreprocessorProtocol,
        plot: bool = True,
        match_config: Optional[MatchConfig] = None,
    ):
        super().__init__()

        if len(metrics) == 0:
            raise ValueError("At least one metric needs to be provided")

        self.match_config = match_config
        self.metrics = metrics
        self.preprocessor = preprocessor
        self.plot = plot

        self._clip_annotations: List[data.ClipAnnotation] = []
        self._predictions: List[List[RawPrediction]] = []

    def get_dataset(self, trainer: Trainer) -> ValidationDataset:
        dataloaders = trainer.val_dataloaders
        assert isinstance(dataloaders, DataLoader)
        dataset = dataloaders.dataset
        assert isinstance(dataset, ValidationDataset)
        return dataset

    def plot_examples(
        self,
        pl_module: LightningModule,
        matches: List[MatchEvaluation],
    ):
        plotter = get_image_plotter(pl_module.logger)  # type: ignore

        if plotter is None:
            return

        for class_name, fig in plot_example_gallery(
            matches,
            preprocessor=self.preprocessor,
            n_examples=4,
        ):
            plotter(
                f"examples/{class_name}",
                fig,
                pl_module.global_step,
            )

    def log_metrics(
        self,
        pl_module: LightningModule,
        matches: List[MatchEvaluation],
    ):
        metrics = {}
        for metric in self.metrics:
            metrics.update(metric(matches).items())

        pl_module.log_dict(metrics)

    def on_validation_epoch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        matches = match_all_predictions(
            self._clip_annotations,
            self._predictions,
            targets=pl_module.model.targets,
            config=self.match_config,
        )

        self.log_metrics(pl_module, matches)

        if self.plot:
            self.plot_examples(pl_module, matches)

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
        postprocessor = pl_module.model.postprocessor
        targets = pl_module.model.targets
        dataset = self.get_dataset(trainer)

        clip_annotations = [
            dataset.clip_annotations[int(example_idx)]
            for example_idx in batch.idx
        ]

        predictions = get_raw_predictions(
            outputs,
            start_times=[
                clip_annotation.clip.start_time
                for clip_annotation in clip_annotations
            ],
            targets=targets,
            postprocessor=postprocessor,
        )

        self._clip_annotations.extend(clip_annotations)
        self._predictions.extend(predictions)
