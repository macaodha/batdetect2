from pathlib import Path
from typing import cast
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from soundevent import data as soundevent_data

from batdetect2.api_v2 import BatDetect2API
from batdetect2.outputs import build_output_formatter
from batdetect2.outputs.formats import (
    BatDetect2OutputConfig,
    SoundEventOutputConfig,
)
from batdetect2.outputs.formats.batdetect2 import BatDetect2Formatter
from batdetect2.postprocess.types import ClipDetections


@pytest.fixture
def api_v2() -> BatDetect2API:
    """User story: API object manages prediction IO formats."""

    return BatDetect2API.from_config()


@pytest.fixture
def file_prediction(api_v2: BatDetect2API, example_audio_files: list[Path]):
    """User story: users save/load predictions produced by API inference."""

    return api_v2.process_file(example_audio_files[0])


def test_save_and_load_predictions_roundtrip_default_raw(
    api_v2: BatDetect2API,
    file_prediction,
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "raw_preds"
    api_v2.save_predictions([file_prediction], path=output_dir)
    loaded = cast(list[ClipDetections], api_v2.load_predictions(output_dir))

    assert len(loaded) == 1
    loaded_prediction = loaded[0]
    assert loaded_prediction.clip == file_prediction.clip
    assert len(loaded_prediction.detections) == len(file_prediction.detections)

    for loaded_det, det in zip(
        loaded_prediction.detections,
        file_prediction.detections,
        strict=True,
    ):
        assert loaded_det.geometry == det.geometry
        assert np.isclose(loaded_det.detection_score, det.detection_score)
        np.testing.assert_allclose(
            loaded_det.class_scores,
            det.class_scores,
            atol=1e-6,
        )


def test_save_predictions_with_batdetect2_override(
    api_v2: BatDetect2API,
    file_prediction,
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "batdetect2_preds"
    api_v2.save_predictions(
        [file_prediction],
        path=output_dir,
        format="batdetect2",
    )

    formatter = build_output_formatter(
        targets=api_v2.targets,
        config=BatDetect2OutputConfig(),
    )
    loaded = formatter.load(output_dir)

    assert len(loaded) == 1
    assert "annotation" in loaded[0]
    assert len(loaded[0]["annotation"]) == len(file_prediction.detections)


def test_batdetect2_formatter_can_use_raw_class_names(
    api_v2: BatDetect2API,
    file_prediction,
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "batdetect2_raw_class_names"
    api_v2.save_predictions(
        [file_prediction],
        path=output_dir,
        config=BatDetect2OutputConfig(class_label_mode="class_name"),
    )

    loaded = cast(
        list[dict], api_v2.load_predictions(output_dir, format="batdetect2")
    )
    first_annotation = loaded[0]["annotation"][0]

    assert first_annotation["class"] in api_v2.targets.class_names


def test_batdetect2_formatter_can_use_decoded_species_tag() -> None:
    targets = Mock()
    targets.class_names = ["myodau"]
    targets.decode_class.return_value = [
        soundevent_data.Tag(
            key="dwc:scientificName",
            value="Myotis daubentonii",
        )
    ]

    formatter = BatDetect2Formatter(
        targets=targets,
        event_name="Echolocation",
        annotation_note="Automatically generated.",
    )

    assert formatter.get_class_name(0) == "Myotis daubentonii"


def test_batdetect2_formatter_can_fallback_to_class_name_when_key_missing() -> (
    None
):
    targets = Mock()
    targets.class_names = ["myodau"]
    targets.decode_class.return_value = []

    formatter = BatDetect2Formatter(
        targets=targets,
        event_name="Echolocation",
        annotation_note="Automatically generated.",
        decoded_label_key="dwc:scientificName",
        fallback_to_class_name=True,
    )

    assert formatter.get_class_name(0) == "myodau"


def test_batdetect2_formatter_rejects_missing_decoded_key_without_fallback() -> (
    None
):
    targets = Mock()
    targets.class_names = ["myodau"]
    targets.decode_class.return_value = []

    formatter = BatDetect2Formatter(
        targets=targets,
        event_name="Echolocation",
        annotation_note="Automatically generated.",
        decoded_label_key="dwc:scientificName",
        fallback_to_class_name=False,
    )

    with pytest.raises(ValueError, match="Could not decode class label"):
        formatter.get_class_name(0)


def test_load_predictions_with_format_override(
    api_v2: BatDetect2API,
    file_prediction,
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "batdetect2_preds_load"
    api_v2.save_predictions(
        [file_prediction],
        path=output_dir,
        format="batdetect2",
    )

    loaded = api_v2.load_predictions(output_dir, format="batdetect2")

    assert len(loaded) == 1
    loaded_item = loaded[0]
    assert isinstance(loaded_item, dict)
    assert "annotation" in loaded_item


def test_load_predictions_with_batdetect2_nested_layout(
    api_v2: BatDetect2API,
    example_audio_files: list[Path],
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "batdetect2_nested"
    predictions = [
        api_v2.process_file(audio_file) for audio_file in example_audio_files
    ]

    api_v2.save_predictions(
        predictions,
        path=output_dir,
        format="batdetect2",
        audio_dir=example_audio_files[0].parent,
    )

    loaded = api_v2.load_predictions(output_dir, format="batdetect2")

    assert len(loaded) == len(example_audio_files)


def test_save_predictions_with_batdetect2_writes_cnn_feature_csv(
    api_v2: BatDetect2API,
    file_prediction,
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "batdetect2_cnn"
    api_v2.save_predictions(
        [file_prediction],
        path=output_dir,
        config=BatDetect2OutputConfig(write_cnn_features_csv=True),
    )

    cnn_csvs = list(output_dir.rglob("*_cnn_features.csv"))
    assert len(cnn_csvs) == 1

    loaded_df = pd.read_csv(cnn_csvs[0])
    assert not loaded_df.empty


def test_save_predictions_with_soundevent_override(
    api_v2: BatDetect2API,
    file_prediction,
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "soundevent_preds"
    api_v2.save_predictions(
        [file_prediction],
        path=output_path,
        format="soundevent",
    )

    formatter = build_output_formatter(
        targets=api_v2.targets,
        config=SoundEventOutputConfig(),
    )
    load_path = output_path.with_suffix(".json")
    loaded = formatter.load(load_path)

    assert load_path.exists()
    assert len(loaded) == 1
    assert len(loaded[0].sound_events) == len(file_prediction.detections)
