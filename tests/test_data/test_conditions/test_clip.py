import json
from pathlib import Path

from soundevent import data

from batdetect2.data.conditions import (
    ClipAllOfConfig,
    ClipAnyOfConfig,
    ClipNotConfig,
    HasAllTagsConfig,
    HasAnyTagConfig,
    HasTagConfig,
    IdInListConfig,
    RecordingSatisfiesConfig,
    build_clip_annotation_condition,
)


def test_recording_satisfies_condition(
    tmp_path: Path,
    create_recording,
    create_clip,
    create_clip_annotation,
) -> None:
    recording_a = create_recording(path=tmp_path / "a.wav")
    recording_b = create_recording(path=tmp_path / "b.wav")
    clip_a = create_clip(recording_a)
    clip_b = create_clip(recording_b)
    clip_annotation_a = create_clip_annotation(clip_a)
    clip_annotation_b = create_clip_annotation(clip_b)
    ids_path = tmp_path / "recording_ids.json"
    ids_path.write_text(json.dumps([str(recording_a.uuid)]))

    condition = build_clip_annotation_condition(
        RecordingSatisfiesConfig(
            condition=IdInListConfig(path=ids_path),
        )
    )

    assert condition(clip_annotation_a)
    assert not condition(clip_annotation_b)


def test_clip_id_in_list_condition(
    tmp_path: Path,
    create_recording,
    create_clip,
    create_clip_annotation,
) -> None:
    recording_a = create_recording(path=tmp_path / "a.wav")
    recording_b = create_recording(path=tmp_path / "b.wav")
    clip_annotation_a = create_clip_annotation(create_clip(recording_a))
    clip_annotation_b = create_clip_annotation(create_clip(recording_b))
    ids_path = tmp_path / "clip_annotation_ids.json"
    ids_path.write_text(json.dumps([str(clip_annotation_a.uuid)]))

    condition = build_clip_annotation_condition(IdInListConfig(path=ids_path))

    assert condition(clip_annotation_a)
    assert not condition(clip_annotation_b)


def test_clip_has_tag_conditions(
    tmp_path: Path,
    create_recording,
    create_clip,
    create_clip_annotation,
) -> None:
    reviewed = data.Tag(key="status", value="reviewed")
    train = data.Tag(key="split", value="train")
    val = data.Tag(key="split", value="val")

    recording = create_recording(path=tmp_path / "rec.wav")
    clip = create_clip(recording)
    clip_annotation = create_clip_annotation(
        clip,
        clip_tags=[reviewed, train],
    )

    has_tag = build_clip_annotation_condition(HasTagConfig(tag=reviewed))
    has_all = build_clip_annotation_condition(
        HasAllTagsConfig(tags=[reviewed, train])
    )
    has_any = build_clip_annotation_condition(
        HasAnyTagConfig(tags=[val, train])
    )

    assert has_tag(clip_annotation)
    assert has_all(clip_annotation)
    assert has_any(clip_annotation)


def test_clip_logical_conditions(
    tmp_path: Path,
    create_recording,
    create_clip,
    create_clip_annotation,
) -> None:
    reviewed = data.Tag(key="status", value="reviewed")
    train = data.Tag(key="split", value="train")
    val = data.Tag(key="split", value="val")

    recording = create_recording(path=tmp_path / "rec.wav")
    clip = create_clip(recording)
    clip_annotation = create_clip_annotation(
        clip,
        clip_tags=[reviewed, train],
    )

    all_condition = build_clip_annotation_condition(
        ClipAllOfConfig(
            conditions=[
                HasTagConfig(tag=reviewed),
                HasAnyTagConfig(tags=[train, val]),
            ]
        )
    )
    any_condition = build_clip_annotation_condition(
        ClipAnyOfConfig(
            conditions=[
                HasTagConfig(tag=val),
                HasTagConfig(tag=reviewed),
            ]
        )
    )
    not_condition = build_clip_annotation_condition(
        ClipNotConfig(condition=HasTagConfig(tag=val))
    )

    assert all_condition(clip_annotation)
    assert any_condition(clip_annotation)
    assert not_condition(clip_annotation)
