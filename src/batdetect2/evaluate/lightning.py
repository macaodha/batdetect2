from typing import Any, List

from lightning import LightningModule
from soundevent import data
from torch.utils.data import DataLoader

from batdetect2.evaluate.dataset import TestDataset, TestExample
from batdetect2.logging import get_image_logger
from batdetect2.models import Model
from batdetect2.outputs import OutputTransformProtocol, build_output_transform
from batdetect2.postprocess import to_raw_predictions
from batdetect2.typing import EvaluatorProtocol
from batdetect2.typing.postprocess import ClipDetections


class EvaluationModule(LightningModule):
    def __init__(
        self,
        model: Model,
        evaluator: EvaluatorProtocol,
        output_transform: OutputTransformProtocol | None = None,
    ):
        super().__init__()

        self.model = model
        self.evaluator = evaluator
        self.output_transform = output_transform or build_output_transform()

        self.clip_annotations: List[data.ClipAnnotation] = []
        self.predictions: List[ClipDetections] = []

    def test_step(self, batch: TestExample, batch_idx: int):
        dataset = self.get_dataset()
        clip_annotations = [
            dataset.clip_annotations[int(example_idx)]
            for example_idx in batch.idx
        ]

        outputs = self.model.detector(batch.spec)
        clip_detections = self.model.postprocessor(outputs)
        predictions = [
            ClipDetections(
                clip=clip_annotation.clip,
                detections=to_raw_predictions(
                    clip_dets.numpy(),
                    targets=self.evaluator.targets,
                ),
            )
            for clip_annotation, clip_dets in zip(
                clip_annotations, clip_detections, strict=False
            )
        ]
        predictions = self.output_transform(predictions)

        self.clip_annotations.extend(clip_annotations)
        self.predictions.extend(predictions)

    def on_test_epoch_start(self):
        self.clip_annotations = []
        self.predictions = []

    def on_test_epoch_end(self):
        clip_evals = self.evaluator.evaluate(
            self.clip_annotations,
            self.predictions,
        )
        self.log_metrics(clip_evals)
        self.generate_plots(clip_evals)

    def generate_plots(self, evaluated_clips: Any):
        plotter = get_image_logger(self.logger)  # type: ignore

        if plotter is None:
            return

        for figure_name, fig in self.evaluator.generate_plots(evaluated_clips):
            plotter(figure_name, fig, self.global_step)

    def log_metrics(self, evaluated_clips: Any):
        metrics = self.evaluator.compute_metrics(evaluated_clips)
        self.log_dict(metrics)

    def get_dataset(self) -> TestDataset:
        dataloaders = self.trainer.test_dataloaders
        assert isinstance(dataloaders, DataLoader)
        dataset = dataloaders.dataset
        assert isinstance(dataset, TestDataset)
        return dataset
