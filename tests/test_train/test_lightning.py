from pathlib import Path

import lightning as L
import torch
from soundevent import data

from batdetect2.api_v2 import BatDetect2API
from batdetect2.config import BatDetect2Config
from batdetect2.train import (
    TrainingModule,
    load_model_from_checkpoint,
    run_train,
)
from batdetect2.train.train import build_training_module
from batdetect2.typing.preprocess import AudioLoader


def build_default_module(config: BatDetect2Config | None = None):
    config = config or BatDetect2Config()
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


def test_load_model_from_checkpoint_returns_model_and_config(
    tmp_path: Path,
):
    module = build_default_module()
    trainer = L.Trainer()
    path = tmp_path / "example.ckpt"
    trainer.strategy.connect(module)
    trainer.save_checkpoint(path)

    model, model_config = load_model_from_checkpoint(path)

    assert model is not None
    assert model_config.model_dump(
        mode="json"
    ) == module.model_config.model_dump(mode="json")


def test_checkpoint_stores_train_config_hyperparameters(tmp_path: Path):
    config = BatDetect2Config()
    config.train.optimizer.learning_rate = 7e-4
    config.train.optimizer.t_max = 123

    module = build_default_module(config=config)
    trainer = L.Trainer()
    path = tmp_path / "example.ckpt"
    trainer.strategy.connect(module)
    trainer.save_checkpoint(path)

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    hyper_parameters = checkpoint["hyper_parameters"]

    assert (
        hyper_parameters["train_config"]["optimizer"]["learning_rate"] == 7e-4
    )
    assert hyper_parameters["train_config"]["optimizer"]["t_max"] == 123
    assert "learning_rate" not in hyper_parameters
    assert "t_max" not in hyper_parameters


def test_configure_optimizers_uses_train_config_values():
    config = BatDetect2Config()
    config.train.optimizer.learning_rate = 5e-4
    config.train.optimizer.t_max = 321

    module = build_default_module(config=config)

    optimizers, schedulers = module.configure_optimizers()

    assert optimizers[0].param_groups[0]["lr"] == 5e-4
    assert schedulers[0].T_max == 321


def test_api_from_checkpoint_reconstructs_model_config(tmp_path: Path):
    module = build_default_module()
    trainer = L.Trainer()
    path = tmp_path / "example.ckpt"
    trainer.strategy.connect(module)
    trainer.save_checkpoint(path)

    api = BatDetect2API.from_checkpoint(path)

    assert api.config.model.model_dump(
        mode="json"
    ) == module.model_config.model_dump(mode="json")
    assert api.config.audio.samplerate == module.model_config.samplerate


def test_train_smoke_produces_loadable_checkpoint(
    tmp_path: Path,
    example_annotations: list[data.ClipAnnotation],
    sample_audio_loader: AudioLoader,
):
    config = BatDetect2Config()
    config.train.trainer.limit_train_batches = 1
    config.train.trainer.limit_val_batches = 1
    config.train.trainer.log_every_n_steps = 1
    config.train.train_loader.batch_size = 1
    config.train.train_loader.augmentations.enabled = False

    run_train(
        train_annotations=example_annotations[:1],
        val_annotations=example_annotations[:1],
        config=config,
        num_epochs=1,
        train_workers=0,
        val_workers=0,
        checkpoint_dir=tmp_path,
        seed=0,
    )

    checkpoints = list(tmp_path.rglob("*.ckpt"))
    assert checkpoints

    model, model_config = load_model_from_checkpoint(checkpoints[0])
    assert model_config.samplerate == config.model.samplerate
    assert model_config.architecture.name == config.model.architecture.name
    assert model_config.preprocess.model_dump(
        mode="json"
    ) == config.model.preprocess.model_dump(mode="json")
    assert model_config.postprocess.model_dump(
        mode="json"
    ) == config.model.postprocess.model_dump(mode="json")

    wav = torch.tensor(
        sample_audio_loader.load_clip(example_annotations[0].clip)
    ).unsqueeze(0)
    outputs = model(wav.unsqueeze(0))
    assert outputs is not None
