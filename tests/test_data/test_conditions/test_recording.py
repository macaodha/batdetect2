import json
import textwrap
import uuid
from pathlib import Path

import pytest
from pydantic import TypeAdapter
from soundevent import data

from batdetect2.core import load_config
from batdetect2.data.conditions import (
    RecordingConditionConfig,
    build_recording_condition,
)


def load_recording_condition_config(
    tmp_path: Path,
    yaml_string: str,
) -> RecordingConditionConfig:
    config_path = tmp_path / f"{uuid.uuid4().hex}.yaml"
    config_path.write_text(textwrap.dedent(yaml_string).strip())
    return load_config(
        config_path,
        schema=TypeAdapter(RecordingConditionConfig),
    )


def build_recording_condition_from_yaml(
    tmp_path: Path,
    yaml_string: str,
    base_dir: Path | None = None,
):
    config = load_recording_condition_config(tmp_path, yaml_string)
    return build_recording_condition(config, base_dir=base_dir)


def test_id_in_list_condition(tmp_path: Path, create_recording) -> None:
    recording_a = create_recording(path=tmp_path / "a.wav")
    recording_b = create_recording(path=tmp_path / "b.wav")
    ids_path = tmp_path / "recording_ids.json"
    ids_path.write_text(json.dumps([str(recording_a.uuid)]))

    condition = build_recording_condition_from_yaml(
        tmp_path,
        f"""
        name: id_in_list
        path: {ids_path}
        """,
    )

    assert condition(recording_a)
    assert not condition(recording_b)


def test_id_in_list_condition_uses_base_dir(
    tmp_path: Path,
    create_recording,
) -> None:
    recording_a = create_recording(path=tmp_path / "a.wav")
    recording_b = create_recording(path=tmp_path / "b.wav")
    split_dir = tmp_path / "splits"
    split_dir.mkdir()
    ids_path = split_dir / "train_ids.json"
    ids_path.write_text(json.dumps([str(recording_a.uuid)]))

    condition = build_recording_condition_from_yaml(
        tmp_path,
        """
        name: id_in_list
        path: splits/train_ids.json
        """,
        base_dir=tmp_path,
    )

    assert condition(recording_a)
    assert not condition(recording_b)


def test_id_in_list_condition_raises_for_non_list_json(
    tmp_path: Path,
) -> None:
    ids_path = tmp_path / "recording_ids.json"
    ids_path.write_text(json.dumps({"id": "foo"}))

    with pytest.raises(TypeError, match="Expected JSON list"):
        build_recording_condition_from_yaml(
            tmp_path,
            f"""
            name: id_in_list
            path: {ids_path}
            """,
        )


def test_id_in_list_condition_raises_for_invalid_id(tmp_path: Path) -> None:
    ids_path = tmp_path / "recording_ids.json"
    ids_path.write_text(json.dumps(["not-a-uuid"]))

    with pytest.raises(ValueError, match="Invalid ID"):
        build_recording_condition_from_yaml(
            tmp_path,
            f"""
            name: id_in_list
            path: {ids_path}
            """,
        )


def test_id_in_list_condition_supports_txt_format(
    tmp_path: Path,
    create_recording,
) -> None:
    recording_a = create_recording(path=tmp_path / "a.wav")
    recording_b = create_recording(path=tmp_path / "b.wav")
    ids_path = tmp_path / "recording_ids.txt"
    ids_path.write_text(f"{recording_a.uuid}\n")

    condition = build_recording_condition_from_yaml(
        tmp_path,
        f"""
        name: id_in_list
        path: {ids_path}
        list_format: txt
        """,
    )

    assert condition(recording_a)
    assert not condition(recording_b)


def test_has_tag_condition(tmp_path: Path, create_recording) -> None:
    train = data.Tag(key="split", value="train")
    val = data.Tag(key="split", value="val")

    recording_a = create_recording(
        path=tmp_path / "a.wav",
        tags=[train],
    )
    recording_b = create_recording(
        path=tmp_path / "b.wav",
        tags=[val],
    )

    condition = build_recording_condition_from_yaml(
        tmp_path,
        """
        name: has_tag
        tag:
            key: split
            value: train
        """,
    )

    assert condition(recording_a)
    assert not condition(recording_b)


def test_has_all_tags_condition(tmp_path: Path, create_recording) -> None:
    train = data.Tag(key="split", value="train")
    uk = data.Tag(key="region", value="uk")

    recording_a = create_recording(
        path=tmp_path / "a.wav",
        tags=[train, uk],
    )
    recording_b = create_recording(
        path=tmp_path / "b.wav",
        tags=[train],
    )

    condition = build_recording_condition_from_yaml(
        tmp_path,
        """
        name: has_all_tags
        tags:
            - key: split
              value: train
            - key: region
              value: uk
        """,
    )

    assert condition(recording_a)
    assert not condition(recording_b)


def test_has_any_tag_condition(tmp_path: Path, create_recording) -> None:
    uk = data.Tag(key="region", value="uk")
    us = data.Tag(key="region", value="us")

    recording_a = create_recording(
        path=tmp_path / "a.wav",
        tags=[uk],
    )
    recording_b = create_recording(
        path=tmp_path / "b.wav",
        tags=[us],
    )

    condition = build_recording_condition_from_yaml(
        tmp_path,
        """
        name: has_any_tag
        tags:
            - key: region
              value: eu
            - key: region
              value: uk
        """,
    )

    assert condition(recording_a)
    assert not condition(recording_b)


def test_all_of_condition(tmp_path: Path, create_recording) -> None:
    train = data.Tag(key="split", value="train")
    uk = data.Tag(key="region", value="uk")
    us = data.Tag(key="region", value="us")

    recording_a = create_recording(
        path=tmp_path / "a.wav",
        tags=[train, uk],
    )
    recording_b = create_recording(
        path=tmp_path / "b.wav",
        tags=[train, us],
    )

    condition = build_recording_condition_from_yaml(
        tmp_path,
        """
        name: all_of
        conditions:
            - name: has_tag
              tag:
                key: split
                value: train
            - name: has_any_tag
              tags:
                - key: region
                  value: eu
                - key: region
                  value: uk
        """,
    )

    assert condition(recording_a)
    assert not condition(recording_b)


def test_any_of_condition(tmp_path: Path, create_recording) -> None:
    train = data.Tag(key="split", value="train")
    us = data.Tag(key="region", value="us")

    recording_a = create_recording(
        path=tmp_path / "a.wav",
        tags=[train],
    )
    recording_b = create_recording(
        path=tmp_path / "b.wav",
        tags=[us],
    )

    condition = build_recording_condition_from_yaml(
        tmp_path,
        """
        name: any_of
        conditions:
            - name: has_tag
              tag:
                key: region
                value: eu
            - name: has_tag
              tag:
                key: split
                value: train
        """,
    )

    assert condition(recording_a)
    assert not condition(recording_b)


def test_not_condition(tmp_path: Path, create_recording) -> None:
    uk = data.Tag(key="region", value="uk")
    us = data.Tag(key="region", value="us")

    recording_a = create_recording(
        path=tmp_path / "a.wav",
        tags=[uk],
    )
    recording_b = create_recording(
        path=tmp_path / "b.wav",
        tags=[us],
    )

    condition = build_recording_condition_from_yaml(
        tmp_path,
        """
        name: not
        condition:
            name: has_tag
            tag:
                key: region
                value: us
        """,
    )

    assert condition(recording_a)
    assert not condition(recording_b)
