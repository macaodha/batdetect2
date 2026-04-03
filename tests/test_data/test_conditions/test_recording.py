import json
from pathlib import Path

import pytest
from soundevent import data

from batdetect2.data.conditions import (
    HasAllTagsConfig,
    HasAnyTagConfig,
    HasTagConfig,
    IdInListConfig,
    RecordingAllOfConfig,
    RecordingAnyOfConfig,
    RecordingNotConfig,
    build_recording_condition,
)


def test_id_in_list_condition(tmp_path: Path, create_recording) -> None:
    recording_a = create_recording(path=tmp_path / "a.wav")
    recording_b = create_recording(path=tmp_path / "b.wav")
    ids_path = tmp_path / "recording_ids.json"
    ids_path.write_text(json.dumps([str(recording_a.uuid)]))

    condition = build_recording_condition(IdInListConfig(path=ids_path))

    assert condition(recording_a)
    assert not condition(recording_b)


def test_id_in_list_condition_uses_base_dir(
    tmp_path: Path,
    create_recording,
) -> None:
    recording = create_recording(path=tmp_path / "a.wav")
    split_dir = tmp_path / "splits"
    split_dir.mkdir()
    ids_path = split_dir / "train_ids.json"
    ids_path.write_text(json.dumps([str(recording.uuid)]))

    condition = build_recording_condition(
        IdInListConfig(path=Path("splits/train_ids.json")),
        base_dir=tmp_path,
    )

    assert condition(recording)


def test_id_in_list_condition_raises_for_non_list_json(
    tmp_path: Path,
) -> None:
    ids_path = tmp_path / "recording_ids.json"
    ids_path.write_text(json.dumps({"id": "foo"}))

    with pytest.raises(TypeError, match="Expected JSON list"):
        build_recording_condition(IdInListConfig(path=ids_path))


def test_id_in_list_condition_raises_for_invalid_id(tmp_path: Path) -> None:
    ids_path = tmp_path / "recording_ids.json"
    ids_path.write_text(json.dumps(["not-a-uuid"]))

    with pytest.raises(ValueError, match="Invalid ID"):
        build_recording_condition(IdInListConfig(path=ids_path))


def test_id_in_list_condition_supports_txt_format(
    tmp_path: Path,
    create_recording,
) -> None:
    recording_a = create_recording(path=tmp_path / "a.wav")
    recording_b = create_recording(path=tmp_path / "b.wav")
    ids_path = tmp_path / "recording_ids.txt"
    ids_path.write_text(f"{recording_a.uuid}\n")

    condition = build_recording_condition(
        IdInListConfig(path=ids_path, list_format="txt")
    )

    assert condition(recording_a)
    assert not condition(recording_b)


def test_recording_has_tag_conditions(
    tmp_path: Path, create_recording
) -> None:
    train = data.Tag(key="split", value="train")
    uk = data.Tag(key="region", value="uk")
    eu = data.Tag(key="region", value="eu")

    recording = create_recording(
        path=tmp_path / "rec.wav",
        tags=[train, uk],
    )

    has_train = build_recording_condition(HasTagConfig(tag=train))
    has_all = build_recording_condition(HasAllTagsConfig(tags=[train, uk]))
    has_any = build_recording_condition(HasAnyTagConfig(tags=[eu, uk]))

    assert has_train(recording)
    assert has_all(recording)
    assert has_any(recording)


def test_recording_logical_conditions(
    tmp_path: Path, create_recording
) -> None:
    train = data.Tag(key="split", value="train")
    uk = data.Tag(key="region", value="uk")
    eu = data.Tag(key="region", value="eu")

    recording = create_recording(
        path=tmp_path / "rec.wav",
        tags=[train, uk],
    )

    all_condition = build_recording_condition(
        RecordingAllOfConfig(
            conditions=[
                HasTagConfig(tag=train),
                HasAnyTagConfig(tags=[eu, uk]),
            ]
        )
    )
    any_condition = build_recording_condition(
        RecordingAnyOfConfig(
            conditions=[
                HasTagConfig(tag=eu),
                HasTagConfig(tag=train),
            ]
        )
    )
    not_condition = build_recording_condition(
        RecordingNotConfig(condition=HasTagConfig(tag=eu))
    )

    assert all_condition(recording)
    assert any_condition(recording)
    assert not_condition(recording)
