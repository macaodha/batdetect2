from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from torch.utils.data import DataLoader

from batdetect2.evaluate import match_predictions_and_annotations
from batdetect2.postprocess import PostprocessorProtocol
from batdetect2.train.dataset import LabeledDataset, TrainExample
from batdetect2.types import ModelOutput


class ValidationMetrics(Callback):
    def __init__(self, postprocessor: PostprocessorProtocol):
        super().__init__()
        self.postprocessor = postprocessor
        self.predictions = []

    def on_validation_epoch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        self.predictions = []
        return super().on_validation_epoch_start(trainer, pl_module)

    def on_validation_batch_end(  # type: ignore
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: ModelOutput,
        batch: TrainExample,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        dataloaders = trainer.val_dataloaders
        assert isinstance(dataloaders, DataLoader)
        dataset = dataloaders.dataset
        assert isinstance(dataset, LabeledDataset)
        clip_annotation = dataset.get_clip_annotation(batch_idx)

        # clip_prediction = postprocess_model_outputs(
        #     outputs,
        #     clips=[clip_annotation.clip],
        #     classes=self.class_names,
        #     decoder=self.decoder,
        #     config=self.config.postprocessing,
        # )[0]
        #
        # matches = match_predictions_and_annotations(
        #     clip_annotation,
        #     clip_prediction,
        # )
        #
        # self.validation_predictions.extend(matches)
        # return super().on_validation_batch_end(
        #     trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
        # )
