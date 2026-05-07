import json
import textwrap
import uuid
from pathlib import Path

from pydantic import TypeAdapter
from soundevent import data

from batdetect2.core import load_config
from batdetect2.data.conditions import (
    ClipAnnotationConditionConfig,
    build_clip_annotation_condition,
)


def load_clip_condition_config(
    tmp_path: Path,
    yaml_string: str,
) -> ClipAnnotationConditionConfig:
    config_path = tmp_path / f"{uuid.uuid4().hex}.yaml"
    config_path.write_text(textwrap.dedent(yaml_string).strip())
    return load_config(
        config_path, schema=TypeAdapter(ClipAnnotationConditionConfig)
    )


def build_clip_condition_from_yaml(
    tmp_path: Path,
    yaml_string: str,
    base_dir: Path | None = None,
):
    config = load_clip_condition_config(tmp_path, yaml_string)
    return build_clip_annotation_condition(config, base_dir=base_dir)


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

    condition = build_clip_condition_from_yaml(
        tmp_path,
        f"""
        name: recording_satisfies
        condition:
            name: id_in_list
            path: {ids_path}
        """,
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

    condition = build_clip_condition_from_yaml(
        tmp_path,
        f"""
        name: id_in_list
        path: {ids_path}
        """,
    )

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

    recording = create_recording(path=tmp_path / "rec.wav")
    clip = create_clip(recording)
    clip_annotation = create_clip_annotation(
        clip,
        clip_tags=[reviewed, train],
    )
    clip_annotation_missing = create_clip_annotation(
        create_clip(recording),
        clip_tags=[train],
    )

    condition = build_clip_condition_from_yaml(
        tmp_path,
        """
        name: has_tag
        tag:
            key: status
            value: reviewed
        """,
    )

    assert condition(clip_annotation)
    assert not condition(clip_annotation_missing)


def test_clip_has_all_tags_condition(
    tmp_path: Path,
    create_recording,
    create_clip,
    create_clip_annotation,
) -> None:
    reviewed = data.Tag(key="status", value="reviewed")
    train = data.Tag(key="split", value="train")

    recording = create_recording(path=tmp_path / "rec.wav")
    clip_annotation = create_clip_annotation(
        create_clip(recording),
        clip_tags=[reviewed, train],
    )
    clip_annotation_missing = create_clip_annotation(
        create_clip(recording),
        clip_tags=[reviewed],
    )

    condition = build_clip_condition_from_yaml(
        tmp_path,
        """
        name: has_all_tags
        tags:
            - key: status
              value: reviewed
            - key: split
              value: train
        """,
    )

    assert condition(clip_annotation)
    assert not condition(clip_annotation_missing)


def test_clip_has_any_tag_condition(
    tmp_path: Path,
    create_recording,
    create_clip,
    create_clip_annotation,
) -> None:
    reviewed = data.Tag(key="status", value="reviewed")
    train = data.Tag(key="split", value="train")

    recording = create_recording(path=tmp_path / "rec.wav")
    clip_annotation = create_clip_annotation(
        create_clip(recording),
        clip_tags=[reviewed, train],
    )
    clip_annotation_missing = create_clip_annotation(
        create_clip(recording),
        clip_tags=[data.Tag(key="split", value="test")],
    )

    condition = build_clip_condition_from_yaml(
        tmp_path,
        """
        name: has_any_tag
        tags:
            - key: split
              value: val
            - key: split
              value: train
        """,
    )

    assert condition(clip_annotation)
    assert not condition(clip_annotation_missing)


def test_clip_all_of_condition(
    tmp_path: Path,
    create_recording,
    create_clip,
    create_clip_annotation,
) -> None:
    reviewed = data.Tag(key="status", value="reviewed")
    train = data.Tag(key="split", value="train")

    recording = create_recording(path=tmp_path / "rec.wav")
    clip = create_clip(recording)
    clip_annotation = create_clip_annotation(
        clip,
        clip_tags=[reviewed, train],
    )
    clip_annotation_missing = create_clip_annotation(
        create_clip(recording),
        clip_tags=[reviewed],
    )

    condition = build_clip_condition_from_yaml(
        tmp_path,
        """
        name: all_of
        conditions:
            - name: has_tag
              tag:
                key: status
                value: reviewed
            - name: has_any_tag
              tags:
                - key: split
                  value: train
                - key: split
                  value: val
        """,
    )

    assert condition(clip_annotation)
    assert not condition(clip_annotation_missing)


def test_clip_any_of_condition(
    tmp_path: Path,
    create_recording,
    create_clip,
    create_clip_annotation,
) -> None:
    reviewed = data.Tag(key="status", value="reviewed")

    recording = create_recording(path=tmp_path / "rec.wav")
    clip_annotation = create_clip_annotation(
        create_clip(recording),
        clip_tags=[reviewed],
    )
    clip_annotation_missing = create_clip_annotation(
        create_clip(recording),
        clip_tags=[data.Tag(key="status", value="unchecked")],
    )

    condition = build_clip_condition_from_yaml(
        tmp_path,
        """
        name: any_of
        conditions:
            - name: has_tag
              tag:
                key: split
                value: val
            - name: has_tag
              tag:
                key: status
                value: reviewed
        """,
    )

    assert condition(clip_annotation)
    assert not condition(clip_annotation_missing)


def test_clip_not_condition(
    tmp_path: Path,
    create_recording,
    create_clip,
    create_clip_annotation,
) -> None:
    recording = create_recording(path=tmp_path / "rec.wav")
    clip_annotation = create_clip_annotation(
        create_clip(recording),
        clip_tags=[data.Tag(key="split", value="train")],
    )
    clip_annotation_missing = create_clip_annotation(
        create_clip(recording),
        clip_tags=[data.Tag(key="split", value="val")],
    )

    condition = build_clip_condition_from_yaml(
        tmp_path,
        """
        name: not
        condition:
            name: has_tag
            tag:
                key: split
                value: val
        """,
    )

    assert condition(clip_annotation)
    assert not condition(clip_annotation_missing)
