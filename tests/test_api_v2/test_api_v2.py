from pathlib import Path
from typing import cast

import lightning as L
import numpy as np
import pytest
import torch
from soundevent.geometry import compute_bounds

from batdetect2.api_v2 import BatDetect2API
from batdetect2.inference import InferenceConfig
from batdetect2.models.detectors import Detector
from batdetect2.targets import TargetConfig
from batdetect2.train import TrainingConfig, load_model_from_checkpoint
from batdetect2.train.lightning import build_training_module


@pytest.fixture
def train_config() -> TrainingConfig:
    """Train config with a small batch size for testing."""
    return TrainingConfig.model_validate({"train_loader": {"batch_size": 2}})


@pytest.fixture
def inference_config() -> InferenceConfig:
    """Inference config with a small batch size for testing."""
    return InferenceConfig.model_validate({"loader": {"batch_size": 2}})


@pytest.fixture
def example_targets_config(example_data_dir: Path) -> TargetConfig:
    return TargetConfig.load(example_data_dir / "targets.yaml")


@pytest.fixture
def api_v2(
    train_config: TrainingConfig,
    inference_config: InferenceConfig,
) -> BatDetect2API:
    """User story: users can create a ready-to-use API from config."""

    api = BatDetect2API.from_config(
        train_config=train_config,
        inference_config=inference_config,
    )
    assert api.inference_config.loader.batch_size == 2
    return api


def test_process_file_returns_recording_level_predictions(
    api_v2: BatDetect2API,
    example_audio_files: list[Path],
) -> None:
    """User story: process a file and get detections in recording time."""

    # When
    prediction = api_v2.process_file(example_audio_files[0])

    # Then
    assert prediction.clip.recording.path == example_audio_files[0]
    assert prediction.clip.start_time == 0
    assert prediction.clip.end_time == prediction.clip.recording.duration

    for detection in prediction.detections:
        start, low, end, high = compute_bounds(detection.geometry)
        assert 0 <= start <= end <= prediction.clip.recording.duration
        assert prediction.clip.recording.samplerate > 2 * low
        assert prediction.clip.recording.samplerate > 2 * high
        assert detection.class_scores.shape[0] == len(
            api_v2.targets.class_names
        )


@pytest.mark.slow
def test_process_files_is_batch_size_invariant(
    api_v2: BatDetect2API,
    example_audio_files: list[Path],
) -> None:
    """User story: changing batch size should not change predictions."""

    # When
    preds_batch_1 = api_v2.process_files(example_audio_files, batch_size=1)
    preds_batch_3 = api_v2.process_files(example_audio_files, batch_size=3)

    # Then
    assert len(preds_batch_1) == len(preds_batch_3)

    by_key_1 = {
        (
            str(pred.clip.recording.path),
            pred.clip.start_time,
            pred.clip.end_time,
        ): pred
        for pred in preds_batch_1
    }
    by_key_3 = {
        (
            str(pred.clip.recording.path),
            pred.clip.start_time,
            pred.clip.end_time,
        ): pred
        for pred in preds_batch_3
    }

    assert set(by_key_1) == set(by_key_3)

    for key in by_key_1:
        pred_1 = by_key_1[key]
        pred_3 = by_key_3[key]
        assert pred_1.clip.start_time == pred_3.clip.start_time
        assert pred_1.clip.end_time == pred_3.clip.end_time
        assert len(pred_1.detections) == len(pred_3.detections)


def test_process_audio_matches_process_spectrogram(
    api_v2: BatDetect2API,
    example_audio_files: list[Path],
) -> None:
    """User story: users can call either audio or spectrogram entrypoint."""

    # When
    audio = api_v2.load_audio(example_audio_files[0])
    from_audio = api_v2.process_audio(audio)

    spec = api_v2.generate_spectrogram(audio)
    from_spec = api_v2.process_spectrogram(spec)

    # Then
    assert len(from_audio) == len(from_spec)

    for det_audio, det_spec in zip(from_audio, from_spec, strict=True):
        bounds_audio = np.array(compute_bounds(det_audio.geometry))
        bounds_spec = np.array(compute_bounds(det_spec.geometry))
        np.testing.assert_allclose(bounds_audio, bounds_spec, atol=1e-6)
        assert np.isclose(det_audio.detection_score, det_spec.detection_score)
        np.testing.assert_allclose(
            det_audio.class_scores,
            det_spec.class_scores,
            atol=1e-6,
        )


def test_process_spectrogram_rejects_batched_input(
    api_v2: BatDetect2API,
) -> None:
    """User story: invalid batched input gives a clear error."""

    # Given
    spec = torch.zeros((2, 1, 128, 64), dtype=torch.float32)

    # When/Then
    with pytest.raises(ValueError, match="Batched spectrograms not supported"):
        api_v2.process_spectrogram(spec)


def test_user_can_read_top_class_and_other_class_scores(
    api_v2: BatDetect2API,
    example_audio_files: list[Path],
) -> None:
    """User story: inspect top class and all class scores per detection."""

    prediction = api_v2.process_file(example_audio_files[0])

    assert len(prediction.detections) > 0

    top_classes = [
        api_v2.get_top_class_name(det) for det in prediction.detections
    ]
    other_class_scores = [
        api_v2.get_class_scores(det, include_top_class=False)
        for det in prediction.detections
    ]

    assert len(top_classes) == len(prediction.detections)
    assert all(isinstance(class_name, str) for class_name in top_classes)
    assert len(other_class_scores) == len(prediction.detections)
    assert all(len(scores) >= 1 for scores in other_class_scores)
    assert all(
        all(class_name != top_class for class_name, _ in scores)
        for top_class, scores in zip(
            top_classes,
            other_class_scores,
            strict=True,
        )
    )
    assert all(
        all(
            score_a >= score_b
            for (_, score_a), (_, score_b) in zip(
                scores, scores[1:], strict=False
            )
        )
        for scores in other_class_scores
    )


def test_user_can_read_extracted_features_per_detection(
    api_v2: BatDetect2API,
    example_audio_files: list[Path],
) -> None:
    """User story: inspect extracted feature vectors per detection."""

    prediction = api_v2.process_file(example_audio_files[0])

    assert len(prediction.detections) > 0

    feature_vectors = [
        api_v2.get_detection_features(det) for det in prediction.detections
    ]
    assert len(feature_vectors) == len(prediction.detections)
    assert all(vec.ndim == 1 for vec in feature_vectors)
    assert all(vec.size > 0 for vec in feature_vectors)


@pytest.mark.slow
def test_user_can_load_checkpoint_and_finetune(
    tmp_path: Path,
    example_targets_config: TargetConfig,
    example_annotations,
) -> None:
    """User story: load a checkpoint and continue training from it."""

    api = BatDetect2API.from_config(
        targets_config=example_targets_config,
    )
    module = build_training_module(
        model_config=api.model_config,
        targets_config=example_targets_config,
        class_names=api.targets.class_names,
        dimension_names=api.roi_mapper.dimension_names,
    )
    trainer = L.Trainer(enable_checkpointing=False, logger=False)
    checkpoint_path = tmp_path / "base.ckpt"
    trainer.strategy.connect(module)
    trainer.save_checkpoint(checkpoint_path)

    train_config = api.train_config.model_copy(deep=True)
    train_config.trainer.limit_train_batches = 1
    train_config.trainer.limit_val_batches = 1
    train_config.trainer.log_every_n_steps = 1
    train_config.train_loader.batch_size = 1
    train_config.train_loader.augmentations.enabled = False

    api = BatDetect2API.from_checkpoint(
        checkpoint_path,
        train_config=train_config,
    )
    finetune_dir = tmp_path / "finetuned"

    api.train(
        train_annotations=example_annotations[:1],
        val_annotations=example_annotations[:1],
        train_workers=0,
        val_workers=0,
        checkpoint_dir=finetune_dir,
        log_dir=tmp_path / "logs",
        num_epochs=1,
        seed=0,
    )

    checkpoints = list(finetune_dir.rglob("*.ckpt"))
    assert checkpoints


def test_checkpoint_with_same_targets_config_keeps_heads_unchanged(
    example_targets_config: TargetConfig,
    tmp_path: Path,
) -> None:
    """User story: same targets config does not rebuild prediction heads."""

    # Given
    source_api = BatDetect2API.from_config(
        targets_config=example_targets_config
    )
    module = build_training_module(
        model_config=source_api.model_config,
        targets_config=example_targets_config,
        class_names=source_api.targets.class_names,
        dimension_names=source_api.roi_mapper.dimension_names,
    )
    trainer = L.Trainer(enable_checkpointing=False, logger=False)
    checkpoint_path = tmp_path / "same_targets.ckpt"
    trainer.strategy.connect(module)
    trainer.save_checkpoint(checkpoint_path)

    source_model, _ = load_model_from_checkpoint(checkpoint_path)
    source_detector = cast(Detector, source_model.detector)

    # When
    api = BatDetect2API.from_checkpoint(
        checkpoint_path,
        targets_config=example_targets_config,
    )

    # Then
    detector = cast(Detector, api.model.detector)

    for key, value in source_detector.classifier_head.state_dict().items():
        assert key in detector.classifier_head.state_dict()
        torch.testing.assert_close(
            detector.classifier_head.state_dict()[key],
            value,
        )

    for key, value in source_detector.bbox_head.state_dict().items():
        assert key in detector.bbox_head.state_dict()
        torch.testing.assert_close(
            detector.bbox_head.state_dict()[key],
            value,
        )


@pytest.mark.slow
def test_user_can_finetune_only_heads(
    tmp_path: Path,
    example_annotations,
) -> None:
    """User story: fine-tune only prediction heads."""

    api = BatDetect2API.from_config()
    finetune_dir = tmp_path / "heads_only"

    api.finetune(
        train_annotations=example_annotations[:1],
        val_annotations=example_annotations[:1],
        trainable="heads",
        train_workers=0,
        val_workers=0,
        checkpoint_dir=finetune_dir,
        log_dir=tmp_path / "logs",
        num_epochs=1,
        seed=0,
    )
    detector = cast(Detector, api.model.detector)

    backbone_params = list(detector.backbone.parameters())
    classifier_params = list(detector.classifier_head.parameters())
    bbox_params = list(detector.bbox_head.parameters())

    assert backbone_params
    assert classifier_params
    assert bbox_params
    assert all(not parameter.requires_grad for parameter in backbone_params)
    assert all(parameter.requires_grad for parameter in classifier_params)
    assert all(parameter.requires_grad for parameter in bbox_params)
    assert list(finetune_dir.rglob("*.ckpt"))


@pytest.mark.slow
def test_user_can_evaluate_small_dataset_and_get_metrics(
    api_v2: BatDetect2API,
    example_annotations,
    tmp_path: Path,
) -> None:
    """User story: run evaluation and receive metrics."""

    metrics, predictions = api_v2.evaluate(
        test_annotations=example_annotations[:1],
        num_workers=0,
        output_dir=tmp_path / "eval",
        save_predictions=False,
    )

    assert isinstance(metrics, list)
    assert len(metrics) == 1
    assert isinstance(predictions, list)
    assert len(predictions) == 1


def test_user_can_save_evaluation_results_to_disk(
    api_v2: BatDetect2API,
    example_annotations,
    tmp_path: Path,
) -> None:
    """User story: evaluate saved predictions and persist results."""

    prediction = api_v2.process_file(
        example_annotations[0].clip.recording.path
    )
    metrics = api_v2.evaluate_predictions(
        annotations=[example_annotations[0]],
        predictions=[prediction],
        output_dir=tmp_path,
    )

    assert isinstance(metrics, dict)
    assert (tmp_path / "metrics.json").exists()


def test_process_file_uses_resolved_batch_size_by_default(
    api_v2: BatDetect2API,
    example_audio_files: list[Path],
    monkeypatch,
) -> None:
    """User story: process_file falls back to resolved inference config."""

    captured: dict[str, object] = {}

    def fake_process_files(
        audio_files,
        batch_size=None,
        **kwargs,
    ):
        captured["audio_files"] = audio_files
        captured["batch_size"] = batch_size
        captured["kwargs"] = kwargs
        return []

    monkeypatch.setattr(api_v2, "process_files", fake_process_files)

    api_v2.process_file(example_audio_files[0])

    assert captured["audio_files"] == [example_audio_files[0]]
    assert captured["batch_size"] == api_v2.inference_config.loader.batch_size


def test_detection_threshold_override_changes_process_file_results(
    api_v2: BatDetect2API,
    example_audio_files: list[Path],
) -> None:
    """User story: users can override threshold in process_file."""

    default_prediction = api_v2.process_file(example_audio_files[0])
    strict_prediction = api_v2.process_file(
        example_audio_files[0],
        detection_threshold=1.0,
    )

    assert len(strict_prediction.detections) <= len(
        default_prediction.detections
    )


@pytest.mark.slow
def test_detection_threshold_override_is_ephemeral_in_process_file(
    api_v2: BatDetect2API,
    example_audio_files: list[Path],
) -> None:
    """User story: per-call threshold override does not change defaults."""

    before = api_v2.process_file(example_audio_files[0])
    _ = api_v2.process_file(
        example_audio_files[0],
        detection_threshold=1.0,
    )
    after = api_v2.process_file(example_audio_files[0])

    assert len(before.detections) == len(after.detections)
    np.testing.assert_allclose(
        [det.detection_score for det in before.detections],
        [det.detection_score for det in after.detections],
        atol=1e-6,
    )


def test_detection_threshold_override_changes_spectrogram_results(
    api_v2: BatDetect2API,
    example_audio_files: list[Path],
) -> None:
    """User story: threshold override works in spectrogram path."""

    audio = api_v2.load_audio(example_audio_files[0])
    spec = api_v2.generate_spectrogram(audio)
    default_detections = api_v2.process_spectrogram(spec)
    strict_detections = api_v2.process_spectrogram(
        spec, detection_threshold=1.0
    )

    assert len(strict_detections) <= len(default_detections)


def test_user_can_create_api_with_custom_targets_and_model_metadata_matches(
    sample_targets,
) -> None:
    """User story: custom targets define model output names for a new API."""

    api = BatDetect2API.from_config(targets_config=sample_targets.config)

    assert api.model.class_names == sample_targets.class_names
