from pathlib import Path
from typing import cast

import pytest

from batdetect2.api_v2 import BatDetect2API
from batdetect2.models.detectors import Detector
from batdetect2.targets import TargetConfig
from batdetect2.train import load_model_from_checkpoint


@pytest.mark.slow
def test_user_can_finetune_only_heads(
    tmp_path: Path,
    example_annotations,
) -> None:
    """User story: fine-tune only prediction heads."""

    api = BatDetect2API.from_config()
    source_classifier_head = api.model.detector.classifier_head
    source_size_head = api.model.detector.size_head
    source_backbone = api.model.detector.backbone
    finetune_dir = tmp_path / "heads_only"

    finetuned_api = api.finetune(
        train_annotations=example_annotations[:1],
        val_annotations=example_annotations[:1],
        targets_config=TargetConfig(),
        trainable="heads",
        train_workers=0,
        val_workers=0,
        checkpoint_dir=finetune_dir,
        log_dir=tmp_path / "logs",
        num_epochs=1,
        seed=0,
    )

    detector = cast(Detector, finetuned_api.model.detector)

    backbone_params = list(detector.backbone.parameters())
    classifier_params = list(detector.classifier_head.parameters())
    bbox_params = list(detector.size_head.parameters())

    assert backbone_params
    assert classifier_params
    assert bbox_params
    assert all(not parameter.requires_grad for parameter in backbone_params)
    assert all(parameter.requires_grad for parameter in classifier_params)
    assert all(parameter.requires_grad for parameter in bbox_params)
    assert finetuned_api is not api
    assert detector.backbone is source_backbone
    assert detector.classifier_head is not source_classifier_head
    assert detector.size_head is not source_size_head
    assert list(finetune_dir.rglob("*.ckpt"))


@pytest.mark.slow
def test_finetune_replaces_targets_and_checkpoint_owns_new_targets(
    tmp_path: Path,
    example_annotations,
) -> None:
    """User story: fine-tuning writes checkpoints with the new targets."""

    source_api = BatDetect2API.from_config()
    source_evaluator = source_api.evaluator
    source_formatter = source_api.formatter
    source_output_transform = source_api.output_transform
    new_targets = TargetConfig.model_validate(
        {
            "classification_targets": [
                {
                    "name": "single_class",
                    "tags": [{"key": "class", "value": "single_class"}],
                }
            ],
            "roi": {"mapper": "top_left"},
        }
    )
    finetune_dir = tmp_path / "new_targets"

    finetuned_api = source_api.finetune(
        train_annotations=example_annotations[:1],
        val_annotations=example_annotations[:1],
        targets_config=new_targets,
        trainable="heads",
        train_workers=0,
        val_workers=0,
        checkpoint_dir=finetune_dir,
        log_dir=tmp_path / "logs",
        num_epochs=1,
        seed=0,
    )

    checkpoints = list(finetune_dir.rglob("*.ckpt"))

    assert source_api.targets.get_config() != new_targets.model_dump(
        mode="json"
    )
    assert finetuned_api.targets.get_config() == new_targets.model_dump(
        mode="json"
    )
    assert finetuned_api.evaluator is not source_evaluator
    assert finetuned_api.formatter is not source_formatter
    assert finetuned_api.output_transform is not source_output_transform
    assert finetuned_api.evaluator.targets is finetuned_api.targets
    assert finetuned_api.evaluator.transform is finetuned_api.output_transform
    assert finetuned_api.model.class_names == ["single_class"]
    assert finetuned_api.model.dimension_names == ["width", "height"]
    assert checkpoints

    _, configs = load_model_from_checkpoint(checkpoints[0])
    assert configs.targets.model_dump(mode="json") == new_targets.model_dump(
        mode="json"
    )
