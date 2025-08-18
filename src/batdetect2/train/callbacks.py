import io
from typing import List, Optional, Tuple

import numpy as np
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import Logger, TensorBoardLogger
from lightning.pytorch.loggers.mlflow import MLFlowLogger
from loguru import logger
from soundevent import data
from torch.utils.data import DataLoader

from batdetect2.evaluate.match import (
    MatchConfig,
    match_sound_events_and_raw_predictions,
)
from batdetect2.evaluate.types import MatchEvaluation, MetricsProtocol
from batdetect2.plotting.evaluation import plot_example_gallery
from batdetect2.postprocess.types import (
    BatDetect2Prediction,
    PostprocessorProtocol,
)
from batdetect2.targets.types import TargetProtocol
from batdetect2.train.dataset import LabeledDataset, TrainExample
from batdetect2.train.lightning import TrainingModule
from batdetect2.train.types import ModelOutput


class ValidationMetrics(Callback):
    def __init__(
        self,
        metrics: List[MetricsProtocol],
        plot: bool = True,
        match_config: Optional[MatchConfig] = None,
    ):
        super().__init__()

        if len(metrics) == 0:
            raise ValueError("At least one metric needs to be provided")

        self.match_config = match_config
        self.metrics = metrics
        self.plot = plot

        self._matches: List[
            Tuple[data.ClipAnnotation, List[BatDetect2Prediction]]
        ] = []

    def get_dataset(self, trainer: Trainer) -> LabeledDataset:
        dataloaders = trainer.val_dataloaders
        assert isinstance(dataloaders, DataLoader)
        dataset = dataloaders.dataset
        assert isinstance(dataset, LabeledDataset)
        return dataset

    def plot_examples(
        self,
        pl_module: LightningModule,
        matches: List[MatchEvaluation],
    ):
        plotter = _get_image_plotter(pl_module.logger)  # type: ignore

        if plotter is None:
            return

        for class_name, fig in plot_example_gallery(
            matches,
            preprocessor=pl_module.preprocessor,
            n_examples=4,
        ):
            plotter(
                f"images/{class_name}_examples",
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
        matches = _match_all_collected_examples(
            self._matches,
            pl_module.targets,
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
        self._matches = []
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
        self._matches.extend(
            _get_batch_clips_and_predictions(
                batch,
                outputs,
                dataset=self.get_dataset(trainer),
                postprocessor=pl_module.postprocessor,
                targets=pl_module.targets,
            )
        )


def _get_batch_clips_and_predictions(
    batch: TrainExample,
    outputs: ModelOutput,
    dataset: LabeledDataset,
    postprocessor: PostprocessorProtocol,
    targets: TargetProtocol,
) -> List[Tuple[data.ClipAnnotation, List[BatDetect2Prediction]]]:
    clip_annotations = [
        _get_subclip(
            dataset.get_clip_annotation(example_id),
            start_time=start_time.item(),
            end_time=end_time.item(),
            targets=targets,
        )
        for example_id, start_time, end_time in zip(
            batch.idx,
            batch.start_time,
            batch.end_time,
        )
    ]

    clips = [clip_annotation.clip for clip_annotation in clip_annotations]

    raw_predictions = postprocessor.get_sound_event_predictions(
        outputs,
        clips,
    )

    return [
        (clip_annotation, clip_predictions)
        for clip_annotation, clip_predictions in zip(
            clip_annotations, raw_predictions
        )
    ]


def _match_all_collected_examples(
    pre_matches: List[Tuple[data.ClipAnnotation, List[BatDetect2Prediction]]],
    targets: TargetProtocol,
    config: Optional[MatchConfig] = None,
) -> List[MatchEvaluation]:
    logger.info("Matching all annotations and predictions...")
    return [
        match
        for clip_annotation, raw_predictions in pre_matches
        for match in match_sound_events_and_raw_predictions(
            clip_annotation,
            raw_predictions,
            targets=targets,
            config=config,
        )
    ]


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


def _get_image_plotter(logger: Logger):
    if isinstance(logger, TensorBoardLogger):

        def plot_figure(name, figure, step):
            return logger.experiment.add_figure(name, figure, step)

        return plot_figure

    if isinstance(logger, MLFlowLogger):

        def plot_figure(name, figure, step):
            image = _convert_figure_to_image(figure)
            return logger.experiment.log_image(
                run_id=logger.run_id,
                image=image,
                key=name,
                step=step,
            )

        return plot_figure


def _convert_figure_to_image(figure):
    with io.BytesIO() as buff:
        figure.savefig(buff, format="raw")
        buff.seek(0)
        data = np.frombuffer(buff.getvalue(), dtype=np.uint8)
    w, h = figure.canvas.get_width_height()
    im = data.reshape((int(h), int(w), -1))
    return im
