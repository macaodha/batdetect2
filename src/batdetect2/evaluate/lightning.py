from typing import Sequence

from lightning import LightningModule
from torch.utils.data import DataLoader

from batdetect2.evaluate.dataset import TestDataset, TestExample
from batdetect2.evaluate.tables import FullEvaluationTable
from batdetect2.logging import get_image_logger, get_table_logger
from batdetect2.models import Model
from batdetect2.postprocess import to_raw_predictions
from batdetect2.typing import ClipEvaluation, EvaluatorProtocol


class EvaluationModule(LightningModule):
    def __init__(
        self,
        model: Model,
        evaluator: EvaluatorProtocol,
    ):
        super().__init__()

        self.model = model
        self.evaluator = evaluator

        self.clip_evaluations = []

    def test_step(self, batch: TestExample):
        dataset = self.get_dataset()
        clip_annotations = [
            dataset.clip_annotations[int(example_idx)]
            for example_idx in batch.idx
        ]

        outputs = self.model.detector(batch.spec)
        clip_detections = self.model.postprocessor(
            outputs,
            start_times=[ca.clip.start_time for ca in clip_annotations],
        )
        predictions = [
            to_raw_predictions(
                clip_dets.numpy(),
                targets=self.evaluator.targets,
            )
            for clip_dets in clip_detections
        ]

        self.clip_evaluations.extend(
            self.evaluator.evaluate(clip_annotations, predictions)
        )

    def on_test_epoch_start(self):
        self.clip_evaluations = []

    def on_test_epoch_end(self):
        self.log_metrics(self.clip_evaluations)
        self.plot_examples(self.clip_evaluations)
        self.log_table(self.clip_evaluations)

    def log_table(self, evaluated_clips: Sequence[ClipEvaluation]):
        table_logger = get_table_logger(self.logger)  # type: ignore

        if table_logger is None:
            return

        df = FullEvaluationTable()(evaluated_clips)
        table_logger("full_evaluation", df, 0)

    def plot_examples(self, evaluated_clips: Sequence[ClipEvaluation]):
        plotter = get_image_logger(self.logger)  # type: ignore

        if plotter is None:
            return

        for figure_name, fig in self.evaluator.generate_plots(evaluated_clips):
            plotter(figure_name, fig, self.global_step)

    def log_metrics(self, evaluated_clips: Sequence[ClipEvaluation]):
        metrics = self.evaluator.compute_metrics(evaluated_clips)
        self.log_dict(metrics)

    def get_dataset(self) -> TestDataset:
        dataloaders = self.trainer.test_dataloaders
        assert isinstance(dataloaders, DataLoader)
        dataset = dataloaders.dataset
        assert isinstance(dataset, TestDataset)
        return dataset
