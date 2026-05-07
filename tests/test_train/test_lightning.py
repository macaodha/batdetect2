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
from batdetect2.models import (
    ModelConfig,
    build_model,
    build_model_with_new_targets,
)
from batdetect2.targets import TargetConfig, build_roi_mapping, build_targets
from batdetect2.train import (
    TrainingConfig,
    TrainingModule,
    load_model_from_checkpoint,
    run_train,
)
from batdetect2.train.logging import (
    DatasetConfigArtifact,
    DatasetConfigArtifactLogging,
)
from batdetect2.train.optimizers import AdamOptimizerConfig
from batdetect2.train.schedulers import CosineAnnealingSchedulerConfig
from batdetect2.train.train import build_training_module


def build_default_module(
    target_config: TargetConfig | None = None,
    model_config: ModelConfig | None = None,
    train_config: TrainingConfig | None = None,
):
    target_config = target_config or TargetConfig()
    model_config = model_config or ModelConfig()
    train_config = train_config or TrainingConfig()
    targets = build_targets(target_config)
    roi_mapper = build_roi_mapping(target_config.roi)
    return build_training_module(
        model_config=model_config,
        class_names=targets.class_names,
        dimension_names=roi_mapper.dimension_names,
        train_config=train_config,
    )


def test_can_initialize_default_module():
    module = build_default_module()
    assert isinstance(module, L.LightningModule)


@pytest.mark.slow
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


def test_load_model_from_checkpoint_returns_model_and_configs(
    tmp_path: Path,
):
    input_model_config = ModelConfig(samplerate=192_000)
    expected_model_config = ModelConfig.model_validate(
        input_model_config.model_dump(mode="json")
    )
    train_config = TrainingConfig()
    targets_config = TargetConfig()
    targets = build_targets(targets_config)
    roi_mapper = build_roi_mapping(targets_config.roi)
    module = build_training_module(
        model_config=input_model_config,
        class_names=targets.class_names,
        dimension_names=roi_mapper.dimension_names,
        train_config=train_config,
    )
    trainer = L.Trainer()
    path = tmp_path / "example.ckpt"
    trainer.strategy.connect(module)
    trainer.save_checkpoint(path)

    model, loaded_configs = load_model_from_checkpoint(path)

    assert model is not None
    assert loaded_configs.model.model_dump(
        mode="json"
    ) == expected_model_config.model_dump(mode="json")
    assert loaded_configs.targets.model_dump(
        mode="json"
    ) == targets_config.model_dump(mode="json")
    assert loaded_configs.train.model_dump(
        mode="json"
    ) == train_config.model_dump(mode="json")
    assert model.class_names == targets.class_names
    assert model.dimension_names == roi_mapper.dimension_names

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
    targets_config = TargetConfig()
    targets = build_targets(targets_config)
    roi_mapper = build_roi_mapping(targets_config.roi)
    train_config.optimizer = AdamOptimizerConfig(learning_rate=5e-4)
    train_config.scheduler = CosineAnnealingSchedulerConfig(t_max=123)
    train_config.trainer.max_epochs = 3
    train_config.train_loader.batch_size = 2

    module = build_training_module(
        model_config=model_config,
        class_names=targets.class_names,
        dimension_names=roi_mapper.dimension_names,
        train_config=train_config,
    )
    trainer = L.Trainer()
    path = tmp_path / "example.ckpt"
    trainer.strategy.connect(module)
    trainer.save_checkpoint(path)

    _, recovered_configs = load_model_from_checkpoint(path)
    assert not DeepDiff(
        recovered_configs.model.model_dump(mode="json"),
        expected_model_config.model_dump(mode="json"),
    )
    assert not DeepDiff(
        recovered_configs.train.model_dump(mode="json"),
        train_config.model_dump(mode="json"),
    )


def test_load_model_from_checkpoint_includes_targets_config(tmp_path: Path):
    targets_config = TargetConfig()
    targets = build_targets(targets_config)
    roi_mapper = build_roi_mapping(targets_config.roi)
    module = build_training_module(
        model_config=ModelConfig(),
        targets_config=targets_config,
        class_names=targets.class_names,
        dimension_names=roi_mapper.dimension_names,
        train_config=TrainingConfig(),
    )
    trainer = L.Trainer()
    path = tmp_path / "example.ckpt"
    trainer.strategy.connect(module)
    trainer.save_checkpoint(path)

    _, loaded_configs = load_model_from_checkpoint(path)

    assert loaded_configs.targets.model_dump(
        mode="json"
    ) == targets_config.model_dump(mode="json")


def test_configure_optimizers_uses_train_config_values(tmp_path: Path):
    model_config = ModelConfig()
    expected_model_config = ModelConfig.model_validate(
        model_config.model_dump(mode="json")
    )
    train_config = TrainingConfig()
    targets_config = TargetConfig()
    targets = build_targets(targets_config)
    roi_mapper = build_roi_mapping(targets_config.roi)
    train_config.optimizer = AdamOptimizerConfig(learning_rate=5e-4)
    train_config.scheduler = CosineAnnealingSchedulerConfig(t_max=321)

    module = build_training_module(
        model_config=model_config,
        class_names=targets.class_names,
        dimension_names=roi_mapper.dimension_names,
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

    _, recovered_configs = load_model_from_checkpoint(path)
    assert recovered_configs.model.model_dump(
        mode="json"
    ) == expected_model_config.model_dump(mode="json")
    assert recovered_configs.train.model_dump(
        mode="json"
    ) == train_config.model_dump(mode="json")

    recovered = TrainingModule.load_from_checkpoint(path)

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

    _, stored_configs = load_model_from_checkpoint(path)
    api = BatDetect2API.from_checkpoint(path)

    assert api.model_config.model_dump(
        mode="json"
    ) == stored_configs.model.model_dump(mode="json")
    assert api.audio_config.samplerate == stored_configs.model.samplerate


def test_api_from_checkpoint_reconstructs_targets_from_checkpoint(
    tmp_path: Path,
) -> None:
    targets_config = TargetConfig()
    module = build_default_module(target_config=targets_config)
    trainer = L.Trainer()
    path = tmp_path / "example.ckpt"
    trainer.strategy.connect(module)
    trainer.save_checkpoint(path)

    api = BatDetect2API.from_checkpoint(path)

    assert api.targets.get_config() == targets_config.model_dump(mode="json")


@pytest.mark.slow
def test_train_smoke_produces_loadable_checkpoint(
    tmp_path: Path,
    example_annotations: list[data.ClipAnnotation],
    sample_audio_loader: AudioLoader,
):
    # Given
    train_config = TrainingConfig.model_validate(
        {
            "trainer": {
                "limit_train_batches": 1,
                "limit_val_batches": 1,
                "log_every_n_steps": 1,
            },
            "train_loader": {
                "batch_size": 1,
                "augmentations": {"enabled": False},
            },
        }
    )

    # When
    run_train(
        train_annotations=example_annotations[:1],
        val_annotations=example_annotations[:1],
        train_config=train_config,
        num_epochs=1,
        train_workers=0,
        val_workers=0,
        checkpoint_dir=tmp_path,
        seed=0,
    )

    # Then
    checkpoints = list(tmp_path.rglob("*.ckpt"))
    assert checkpoints

    model, model_config = load_model_from_checkpoint(checkpoints[0])

    wav = torch.tensor(
        sample_audio_loader.load_clip(example_annotations[0].clip)
    ).unsqueeze(0)
    outputs = model(wav.unsqueeze(0))
    assert outputs is not None


def test_build_training_module_uses_provided_model() -> None:
    targets = build_targets(TargetConfig())
    roi_mapper = build_roi_mapping(TargetConfig().roi)
    model = build_model(
        ModelConfig(),
        class_names=targets.class_names,
        dimension_names=roi_mapper.dimension_names,
    )

    module = build_training_module(
        model_config=ModelConfig(),
        class_names=targets.class_names,
        dimension_names=roi_mapper.dimension_names,
        train_config=TrainingConfig(),
        model=model,
    )

    assert module.model is model


def test_build_model_with_new_targets_reuses_backbone_and_rebuilds_heads() -> (
    None
):
    source_targets_config = TargetConfig()
    source_targets = build_targets(source_targets_config)
    source_roi_mapper = build_roi_mapping(source_targets_config.roi)
    source_model = build_model(
        ModelConfig(),
        class_names=source_targets.class_names,
        dimension_names=source_roi_mapper.dimension_names,
    )

    new_targets_config = TargetConfig.model_validate(
        {
            "classification_targets": [
                {
                    "name": "single_class",
                    "tags": [{"key": "class", "value": "single_class"}],
                }
            ]
        }
    )
    new_targets = build_targets(new_targets_config)
    new_roi_mapper = build_roi_mapping(new_targets_config.roi)

    rebuilt_model = build_model_with_new_targets(
        model=source_model,
        targets=new_targets,
        roi_mapper=new_roi_mapper,
    )

    source_detector = source_model.detector
    rebuilt_detector = rebuilt_model.detector

    assert rebuilt_detector.backbone is source_detector.backbone
    assert (
        rebuilt_detector.classifier_head is not source_detector.classifier_head
    )
    assert rebuilt_detector.size_head is not source_detector.size_head
    assert rebuilt_model.class_names == ["single_class"]
    assert rebuilt_model.dimension_names == ["width", "height"]


@pytest.mark.slow
def test_run_train_logs_training_artifacts(
    tmp_path: Path,
    example_annotations: list[data.ClipAnnotation],
    example_dataset,
) -> None:
    train_config = TrainingConfig.model_validate(
        {
            "trainer": {
                "limit_train_batches": 1,
                "limit_val_batches": 1,
                "log_every_n_steps": 1,
            },
            "train_loader": {
                "batch_size": 1,
                "augmentations": {"enabled": False},
            },
        }
    )

    run_train(
        train_annotations=example_annotations[:1],
        val_annotations=example_annotations[:1],
        train_config=train_config,
        num_epochs=1,
        train_workers=0,
        val_workers=0,
        checkpoint_dir=tmp_path / "checkpoints",
        log_dir=tmp_path / "logs",
        seed=0,
        logging_callbacks=[
            DatasetConfigArtifactLogging(
                train_dataset_config=DatasetConfigArtifact(
                    filename="train_dataset.yaml",
                    config=example_dataset,
                ),
                val_dataset_config=DatasetConfigArtifact(
                    filename="val_dataset.yaml",
                    config=example_dataset,
                ),
            )
        ],
    )

    artifact_root = next((tmp_path / "logs").rglob("training_artifacts"))

    assert (artifact_root / "targets.yaml").exists()
    assert (artifact_root / "train_dataset.yaml").exists()
    assert (artifact_root / "val_dataset.yaml").exists()
    assert (artifact_root / "train_class_summary.csv").exists()
    assert (artifact_root / "val_class_summary.csv").exists()


def test_run_train_rejects_incompatible_model_config(
    example_annotations: list[data.ClipAnnotation],
) -> None:
    # Given
    targets_config = TargetConfig()
    targets = build_targets(targets_config)
    roi_mapper = build_roi_mapping(targets_config.roi)
    incompatible_config = ModelConfig()
    incompatible_model = build_model(
        incompatible_config,
        class_names=targets.class_names,
        dimension_names=[*roi_mapper.dimension_names, "extra_dim"],
    )

    # When/Then
    with pytest.raises(
        ValueError,
        match="Provided model is incompatible with model_config",
    ):
        run_train(
            train_annotations=example_annotations[:1],
            val_annotations=None,
            model=incompatible_model,
            targets=targets,
            roi_mapper=roi_mapper,
            targets_config=targets_config,
            train_config=TrainingConfig(),
        )
