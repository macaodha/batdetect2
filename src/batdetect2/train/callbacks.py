from typing import List

from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import TensorBoardLogger
from soundevent import data
from torch.utils.data import DataLoader

from batdetect2.evaluate.match import match_sound_events_and_raw_predictions
from batdetect2.evaluate.types import MatchEvaluation, MetricsProtocol
from batdetect2.plotting.evaluation import plot_examples
from batdetect2.targets.types import TargetProtocol
from batdetect2.train.dataset import LabeledDataset, TrainExample
from batdetect2.train.lightning import TrainingModule
from batdetect2.train.types import ModelOutput


class ValidationMetrics(Callback):
    def __init__(self, metrics: List[MetricsProtocol], plot: bool = True):
        super().__init__()

        if len(metrics) == 0:
            raise ValueError("At least one metric needs to be provided")

        self.matches: List[MatchEvaluation] = []
        self.metrics = metrics
        self.plot = plot

    def get_dataset(self, trainer: Trainer) -> LabeledDataset:
        dataloaders = trainer.val_dataloaders
        assert isinstance(dataloaders, DataLoader)
        dataset = dataloaders.dataset
        assert isinstance(dataset, LabeledDataset)
        return dataset

    def plot_examples(self, pl_module: LightningModule):
        if not isinstance(pl_module.logger, TensorBoardLogger):
            return

        for class_name, fig in plot_examples(
            self.matches,
            preprocessor=pl_module.preprocessor,
            n_examples=5,
        ):
            pl_module.logger.experiment.add_figure(
                f"{class_name}/examples",
                fig,
                pl_module.global_step,
            )

    def log_metrics(self, pl_module: LightningModule):
        metrics = {}
        for metric in self.metrics:
            metrics.update(metric(self.matches).items())

        pl_module.log_dict(metrics)

    def on_validation_epoch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        self.log_metrics(pl_module)

        if self.plot:
            self.plot_examples(pl_module)

        return super().on_validation_epoch_end(trainer, pl_module)

    def on_validation_epoch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        self.matches = []
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
        dataset = self.get_dataset(trainer)

        clip_annotations = [
            _get_subclip(
                dataset.get_clip_annotation(example_id),
                start_time=start_time.item(),
                end_time=end_time.item(),
                targets=pl_module.targets,
            )
            for example_id, start_time, end_time in zip(
                batch.idx,
                batch.start_time,
                batch.end_time,
            )
        ]

        clips = [clip_annotation.clip for clip_annotation in clip_annotations]

        raw_predictions = pl_module.postprocessor.get_sound_event_predictions(
            outputs,
            clips,
        )

        for clip_annotation, clip_predictions in zip(
            clip_annotations, raw_predictions
        ):
            self.matches.extend(
                match_sound_events_and_raw_predictions(
                    clip_annotation=clip_annotation,
                    raw_predictions=clip_predictions,
                    targets=pl_module.targets,
                )
            )


def _is_in_subclip(
    sound_event_annotation: data.SoundEventAnnotation,
    targets: TargetProtocol,
    start_time: float,
    end_time: float,
) -> bool:
    (time, _), _ = targets.encode_roi(sound_event_annotation)
    return start_time <= time <= end_time


def _get_subclip(
    clip_annotation: data.ClipAnnotation,
    start_time: float,
    end_time: float,
    targets: TargetProtocol,
) -> data.ClipAnnotation:
    return data.ClipAnnotation(
        clip=data.Clip(
            recording=clip_annotation.clip.recording,
            start_time=start_time,
            end_time=end_time,
        ),
        sound_events=[
            sound_event_annotation
            for sound_event_annotation in clip_annotation.sound_events
            if _is_in_subclip(
                sound_event_annotation,
                targets,
                start_time=start_time,
                end_time=end_time,
            )
        ],
    )
