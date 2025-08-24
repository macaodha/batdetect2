from pathlib import Path

import lightning as L
import torch
import xarray as xr
from soundevent import data

from batdetect2.train import FullTrainingConfig, TrainingModule
from batdetect2.train.train import build_training_module


def build_default_module():
    config = FullTrainingConfig()
    return build_training_module(config)


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

    spec1 = module.model.preprocessor.preprocess_clip(clip)
    spec2 = recovered.model.preprocessor.preprocess_clip(clip)

    xr.testing.assert_equal(spec1, spec2)

    input1 = torch.tensor([spec1.values]).unsqueeze(0)
    input2 = torch.tensor([spec2.values]).unsqueeze(0)

    output1 = module(input1)
    output2 = recovered(input2)

    torch.testing.assert_close(output1, output2)
