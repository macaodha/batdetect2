from pathlib import Path

import lightning as L
import torch
import xarray as xr
from soundevent import data

from batdetect2.models import build_model
from batdetect2.postprocess import build_postprocessor
from batdetect2.preprocess import build_preprocessor
from batdetect2.targets import build_targets
from batdetect2.train.lightning import TrainingModule
from batdetect2.train.losses import build_loss


def build_default_module():
    loss = build_loss()
    targets = build_targets()
    detector = build_model(num_classes=len(targets.class_names))
    preprocessor = build_preprocessor()
    postprocessor = build_postprocessor(targets)
    return TrainingModule(
        detector=detector,
        loss=loss,
        targets=targets,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
    )


def test_can_initialize_default_module():
    module = build_default_module()
    assert isinstance(module, L.LightningModule)


def test_can_save_checkpoint(tmp_path: Path, clip: data.Clip):
    module = build_default_module()
    trainer = L.Trainer()
    path = tmp_path / "example.ckpt"
    trainer.strategy.connect(module)
    trainer.save_checkpoint(path)

    recovered = TrainingModule.load_from_checkpoint(path)

    spec1 = module.preprocessor.preprocess_clip(clip)
    spec2 = recovered.preprocessor.preprocess_clip(clip)

    xr.testing.assert_equal(spec1, spec2)

    input1 = torch.tensor([spec1.values]).unsqueeze(0)
    input2 = torch.tensor([spec2.values]).unsqueeze(0)

    output1 = module(input1)
    output2 = recovered(input2)

    assert output1 == output2
