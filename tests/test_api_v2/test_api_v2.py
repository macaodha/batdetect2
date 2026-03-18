from pathlib import Path

import lightning as L
import numpy as np
import pytest
import torch
from soundevent.geometry import compute_bounds

from batdetect2.api_v2 import BatDetect2API
from batdetect2.config import BatDetect2Config
from batdetect2.train.lightning import build_training_module


@pytest.fixture
def api_v2() -> BatDetect2API:
    """User story: users can create a ready-to-use API from config."""

    config = BatDetect2Config()
    config.inference.loader.batch_size = 2
    return BatDetect2API.from_config(config)


def test_process_file_returns_recording_level_predictions(
    api_v2: BatDetect2API,
    example_audio_files: list[Path],
) -> None:
    """User story: process a file and get detections in recording time."""

    prediction = api_v2.process_file(example_audio_files[0])

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


def test_process_files_is_batch_size_invariant(
    api_v2: BatDetect2API,
    example_audio_files: list[Path],
) -> None:
    """User story: changing batch size should not change predictions."""

    preds_batch_1 = api_v2.process_files(example_audio_files, batch_size=1)
    preds_batch_3 = api_v2.process_files(example_audio_files, batch_size=3)

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

    audio = api_v2.load_audio(example_audio_files[0])
    from_audio = api_v2.process_audio(audio)

    spec = api_v2.generate_spectrogram(audio)
    from_spec = api_v2.process_spectrogram(spec)

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

    spec = torch.zeros((2, 1, 128, 64), dtype=torch.float32)

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


def test_user_can_load_checkpoint_and_finetune(
    tmp_path: Path,
    example_annotations,
) -> None:
    """User story: load a checkpoint and continue training from it."""

    module = build_training_module(model_config=BatDetect2Config().model)
    trainer = L.Trainer(enable_checkpointing=False, logger=False)
    checkpoint_path = tmp_path / "base.ckpt"
    trainer.strategy.connect(module)
    trainer.save_checkpoint(checkpoint_path)

    config = BatDetect2Config()
    config.train.trainer.limit_train_batches = 1
    config.train.trainer.limit_val_batches = 1
    config.train.trainer.log_every_n_steps = 1
    config.train.train_loader.batch_size = 1
    config.train.train_loader.augmentations.enabled = False

    api = BatDetect2API.from_checkpoint(checkpoint_path, config=config)
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
    assert isinstance(metrics[0], dict)
    assert len(metrics[0]) > 0
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
