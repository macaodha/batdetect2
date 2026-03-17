from pathlib import Path

import lightning as L
import torch
from soundevent import data

from batdetect2.config import BatDetect2Config
from batdetect2.train import TrainingModule
from batdetect2.train.train import build_training_module
from batdetect2.typing.preprocess import AudioLoader


def build_default_module():
    config = BatDetect2Config()
    return build_training_module(
        model_config=config.model.model_dump(mode="json"),
        train_config=config.train.model_dump(mode="json"),
    )


def test_can_initialize_default_module():
    module = build_default_module()
    assert isinstance(module, L.LightningModule)


def test_can_save_checkpoint(
    tmp_path: Path,
    clip: data.Clip,
    sample_audio_loader: AudioLoader,
):
    module = build_default_module()
    trainer = L.Trainer()
    path = tmp_path / "example.ckpt"
    trainer.strategy.connect(module)
    trainer.save_checkpoint(path)

    recovered = TrainingModule.load_from_checkpoint(path)

    wav = torch.tensor(sample_audio_loader.load_clip(clip)).unsqueeze(0)

    spec1 = module.model.preprocessor(wav)
    spec2 = recovered.model.preprocessor(wav)

    torch.testing.assert_close(spec1, spec2, rtol=0, atol=0)

    output1 = module.model(wav.unsqueeze(0))
    output2 = recovered.model(wav.unsqueeze(0))

    torch.testing.assert_close(output1, output2, rtol=0, atol=0)
