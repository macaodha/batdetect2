from pathlib import Path

import lightning as L
import pytest
import torch
from deepdiff import DeepDiff
from soundevent import data
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from batdetect2.api_v2 import BatDetect2API
from batdetect2.audio.types import AudioLoader
from batdetect2.config import BatDetect2Config
from batdetect2.models import ModelConfig, build_model
from batdetect2.targets.classes import TargetClassConfig
from batdetect2.train import (
    TrainingConfig,
    TrainingModule,
    load_model_from_checkpoint,
    run_train,
)
from batdetect2.train.optimizers import AdamOptimizerConfig
from batdetect2.train.schedulers import CosineAnnealingSchedulerConfig
from batdetect2.train.train import build_training_module


def build_default_module(config: BatDetect2Config | None = None):
    config = config or BatDetect2Config()
    return build_training_module(
        model_config=config.model,
        train_config=config.train,
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
    input_model_config = ModelConfig(samplerate=192_000)
    expected_model_config = ModelConfig.model_validate(
        input_model_config.model_dump(mode="json")
    )
    train_config = TrainingConfig()
    module = build_training_module(
        model_config=input_model_config,
        train_config=train_config,
    )
    trainer = L.Trainer()
    path = tmp_path / "example.ckpt"
    trainer.strategy.connect(module)
    trainer.save_checkpoint(path)

    model, loaded_model_config = load_model_from_checkpoint(path)

    assert model is not None
    assert loaded_model_config.model_dump(
        mode="json"
    ) == expected_model_config.model_dump(mode="json")

    recovered = TrainingModule.load_from_checkpoint(path)
    assert recovered.train_config.model_dump(
        mode="json"
    ) == train_config.model_dump(mode="json")


def test_checkpoint_stores_train_config_hyperparameters(tmp_path: Path):
    model_config = ModelConfig(samplerate=384_000)
    expected_model_config = ModelConfig.model_validate(
        model_config.model_dump(mode="json")
    )
    train_config = TrainingConfig()
    train_config.optimizer = AdamOptimizerConfig(learning_rate=5e-4)
    train_config.scheduler = CosineAnnealingSchedulerConfig(t_max=123)
    train_config.trainer.max_epochs = 3
    train_config.train_loader.batch_size = 2

    module = build_training_module(
        model_config=model_config,
        train_config=train_config,
    )
    trainer = L.Trainer()
    path = tmp_path / "example.ckpt"
    trainer.strategy.connect(module)
    trainer.save_checkpoint(path)

    recovered = TrainingModule.load_from_checkpoint(path)
    assert not DeepDiff(
        recovered.model_config.model_dump(mode="json"),
        expected_model_config.model_dump(mode="json"),
    )
    assert not DeepDiff(
        recovered.train_config.model_dump(mode="json"),
        train_config.model_dump(mode="json"),
    )


def test_configure_optimizers_uses_train_config_values(tmp_path: Path):
    model_config = ModelConfig()
    expected_model_config = ModelConfig.model_validate(
        model_config.model_dump(mode="json")
    )
    train_config = TrainingConfig()
    train_config.optimizer = AdamOptimizerConfig(learning_rate=5e-4)
    train_config.scheduler = CosineAnnealingSchedulerConfig(t_max=321)

    module = build_training_module(
        model_config=model_config,
        train_config=train_config,
    )

    optimization_config = module.configure_optimizers()
    optimizer = optimization_config["optimizer"]
    scheduler = optimization_config["lr_scheduler"]["scheduler"]

    assert isinstance(optimizer, Adam)
    assert isinstance(scheduler, CosineAnnealingLR)
    assert optimizer.param_groups[0]["lr"] == 5e-4
    assert scheduler.T_max == 321

    trainer = L.Trainer()
    path = tmp_path / "example.ckpt"
    trainer.strategy.connect(module)
    trainer.save_checkpoint(path)

    recovered = TrainingModule.load_from_checkpoint(path)
    assert recovered.model_config.model_dump(
        mode="json"
    ) == expected_model_config.model_dump(mode="json")
    assert recovered.train_config.model_dump(
        mode="json"
    ) == train_config.model_dump(mode="json")

    loaded_optimization_config = recovered.configure_optimizers()
    loaded_optimizer = loaded_optimization_config["optimizer"]
    loaded_scheduler = loaded_optimization_config["lr_scheduler"]["scheduler"]
    assert loaded_optimizer.param_groups[0]["lr"] == 5e-4
    assert loaded_scheduler.T_max == 321


def test_api_from_checkpoint_reconstructs_model_config(tmp_path: Path):
    module = build_default_module()
    trainer = L.Trainer()
    path = tmp_path / "example.ckpt"
    trainer.strategy.connect(module)
    trainer.save_checkpoint(path)

    api = BatDetect2API.from_checkpoint(path)

    assert api.model_config.model_dump(
        mode="json"
    ) == module.model_config.model_dump(mode="json")
    assert api.audio_config.samplerate == module.model_config.samplerate


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
        train_config=config.train,
        model_config=config.model,
        audio_config=config.audio,
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


def test_build_training_module_uses_provided_model() -> None:
    model = build_model(ModelConfig())

    module = build_training_module(
        model_config=ModelConfig(),
        train_config=TrainingConfig(),
        model=model,
    )

    assert module.model is model


def test_run_train_rejects_incompatible_model_config(
    example_annotations: list[data.ClipAnnotation],
) -> None:
    model = build_model(ModelConfig())
    incompatible_config = ModelConfig()
    incompatible_config.targets.classification_targets.append(
        TargetClassConfig(
            name="dummy_class",
            tags=[data.Tag(key="class", value="Dummy class")],
        )
    )

    with pytest.raises(
        ValueError,
        match="Provided model is incompatible with model_config",
    ):
        run_train(
            train_annotations=example_annotations[:1],
            val_annotations=None,
            model=model,
            model_config=incompatible_config,
            train_config=TrainingConfig(),
        )
