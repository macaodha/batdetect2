import json
from pathlib import Path

from soundevent import data

from batdetect2.data import DatasetConfig, load_dataset
from batdetect2.data.conditions import (
    HasTagConfig,
    IdInListConfig,
    RecordingSatisfiesConfig,
)


def test_load_dataset_applies_clip_filter(
    example_dataset: DatasetConfig,
    tmp_path: Path,
) -> None:
    baseline = list(load_dataset(example_dataset))
    keep_recording_id = str(baseline[0].clip.recording.uuid)
    ids_path = tmp_path / "train_ids.json"
    ids_path.write_text(json.dumps([keep_recording_id]))

    config = example_dataset.model_copy(
        update={
            "clip_filter": RecordingSatisfiesConfig(
                condition=IdInListConfig(path=ids_path)
            )
        }
    )

    filtered = list(load_dataset(config))

    assert len(filtered) == 1
    assert str(filtered[0].clip.recording.uuid) == keep_recording_id


def test_load_dataset_clip_filter_is_skipped_when_filters_disabled(
    example_dataset: DatasetConfig,
    tmp_path: Path,
) -> None:
    baseline = list(load_dataset(example_dataset))
    keep_recording_id = str(baseline[0].clip.recording.uuid)
    ids_path = tmp_path / "train_ids.json"
    ids_path.write_text(json.dumps([keep_recording_id]))

    config = example_dataset.model_copy(
        update={
            "clip_filter": RecordingSatisfiesConfig(
                condition=IdInListConfig(path=ids_path)
            )
        }
    )

    filtered = list(load_dataset(config, apply_filters=False))

    assert len(filtered) == len(baseline)


def test_load_dataset_resolves_clip_filter_paths_from_base_dir(
    example_dataset: DatasetConfig,
    tmp_path: Path,
) -> None:
    baseline = list(load_dataset(example_dataset))
    keep_recording_id = str(baseline[0].clip.recording.uuid)
    split_dir = tmp_path / "splits"
    split_dir.mkdir()
    ids_path = split_dir / "train_ids.json"
    ids_path.write_text(json.dumps([keep_recording_id]))

    config = example_dataset.model_copy(
        update={
            "clip_filter": RecordingSatisfiesConfig(
                condition=IdInListConfig(path=Path("splits/train_ids.json"))
            )
        }
    )

    filtered = list(load_dataset(config, base_dir=tmp_path))

    assert len(filtered) == 1
    assert str(filtered[0].clip.recording.uuid) == keep_recording_id


def test_sound_event_filter_keeps_empty_clips(
    example_dataset: DatasetConfig,
) -> None:
    config = example_dataset.model_copy(
        update={
            "sound_event_filter": HasTagConfig(
                tag=data.Tag(key="species", value="__missing_species__")
            )
        }
    )

    filtered = list(load_dataset(config))

    assert len(filtered) == 3
    assert all(
        len(clip_annotation.sound_events) == 0 for clip_annotation in filtered
    )
