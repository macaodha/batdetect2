import json
import textwrap
import uuid
from pathlib import Path

import pytest
from pydantic import TypeAdapter
from soundevent import data

from batdetect2.core import load_config
from batdetect2.data.conditions import (
    SoundEventConditionConfig,
    build_sound_event_condition,
)


def load_sound_event_condition_config(
    tmp_path: Path,
    yaml_string: str,
) -> SoundEventConditionConfig:
    config_path = tmp_path / f"{uuid.uuid4().hex}.yaml"
    config_path.write_text(textwrap.dedent(yaml_string).strip())
    return load_config(
        config_path,
        schema=TypeAdapter(SoundEventConditionConfig),
    )


def build_condition_from_str(
    tmp_path: Path,
    yaml_string: str,
    base_dir: Path | None = None,
):
    config = load_sound_event_condition_config(tmp_path, yaml_string)
    return build_sound_event_condition(config, base_dir=base_dir)


def create_sound_event_annotation(
    recording: data.Recording,
    geometry: data.Geometry,
    tags: list[data.Tag] | None = None,
) -> data.SoundEventAnnotation:
    return data.SoundEventAnnotation(
        sound_event=data.SoundEvent(
            recording=recording,
            geometry=geometry,
        ),
        tags=tags or [],
    )


def test_has_tag_condition(
    sound_event: data.SoundEvent, tmp_path: Path
) -> None:
    condition = build_condition_from_str(
        tmp_path,
        """
        name: has_tag
        tag:
            key: species
            value: Myotis myotis
        """,
    )

    passing = data.SoundEventAnnotation(
        sound_event=sound_event,
        tags=[data.Tag(key="species", value="Myotis myotis")],
    )
    failing = data.SoundEventAnnotation(
        sound_event=sound_event,
        tags=[data.Tag(key="species", value="Eptesicus fuscus")],
    )

    assert condition(passing)
    assert not condition(failing)


def test_has_all_tags_condition(
    sound_event: data.SoundEvent,
    tmp_path: Path,
) -> None:
    condition = build_condition_from_str(
        tmp_path,
        """
        name: has_all_tags
        tags:
            - key: species
              value: Myotis myotis
            - key: event
              value: Echolocation
        """,
    )

    passing = data.SoundEventAnnotation(
        sound_event=sound_event,
        tags=[
            data.Tag(key="species", value="Myotis myotis"),
            data.Tag(key="event", value="Echolocation"),
        ],
    )
    failing = data.SoundEventAnnotation(
        sound_event=sound_event,
        tags=[data.Tag(key="species", value="Myotis myotis")],
    )

    assert condition(passing)
    assert not condition(failing)


def test_has_any_tag_condition(
    sound_event: data.SoundEvent,
    tmp_path: Path,
) -> None:
    condition = build_condition_from_str(
        tmp_path,
        """
        name: has_any_tag
        tags:
            - key: species
              value: Myotis myotis
            - key: event
              value: Echolocation
        """,
    )

    passing = data.SoundEventAnnotation(
        sound_event=sound_event,
        tags=[data.Tag(key="event", value="Echolocation")],
    )
    failing = data.SoundEventAnnotation(
        sound_event=sound_event,
        tags=[
            data.Tag(key="species", value="Eptesicus fuscus"),
            data.Tag(key="event", value="Social"),
        ],
    )

    assert condition(passing)
    assert not condition(failing)


def test_not_condition(sound_event: data.SoundEvent, tmp_path: Path) -> None:
    condition = build_condition_from_str(
        tmp_path,
        """
        name: not
        condition:
            name: has_tag
            tag:
                key: species
                value: Myotis myotis
        """,
    )

    passing = data.SoundEventAnnotation(
        sound_event=sound_event,
        tags=[data.Tag(key="species", value="Eptesicus fuscus")],
    )
    failing = data.SoundEventAnnotation(
        sound_event=sound_event,
        tags=[data.Tag(key="species", value="Myotis myotis")],
    )

    assert condition(passing)
    assert not condition(failing)


def test_id_in_list_condition(
    sound_event: data.SoundEvent, tmp_path: Path
) -> None:
    passing = data.SoundEventAnnotation(sound_event=sound_event)
    failing = data.SoundEventAnnotation(sound_event=sound_event)
    ids_path = tmp_path / "sound_event_ids.json"
    ids_path.write_text(json.dumps([str(passing.uuid)]))

    condition = build_condition_from_str(
        tmp_path,
        f"""
        name: id_in_list
        path: {ids_path}
        """,
    )

    assert condition(passing)
    assert not condition(failing)


def test_id_in_list_condition_uses_base_dir(
    sound_event: data.SoundEvent,
    tmp_path: Path,
) -> None:
    passing = data.SoundEventAnnotation(sound_event=sound_event)
    failing = data.SoundEventAnnotation(sound_event=sound_event)
    split_dir = tmp_path / "splits"
    split_dir.mkdir()
    ids_path = split_dir / "sound_event_ids.json"
    ids_path.write_text(json.dumps([str(passing.uuid)]))

    condition = build_condition_from_str(
        tmp_path,
        """
        name: id_in_list
        path: splits/sound_event_ids.json
        """,
        base_dir=tmp_path,
    )

    assert condition(passing)
    assert not condition(failing)


@pytest.mark.parametrize(
    "operator,seconds,passing_duration,failing_duration",
    [
        ("lt", 2, 1, 2),
        ("lte", 2, 2, 3),
        ("gt", 2, 3, 2),
        ("gte", 2, 2, 1),
        ("eq", 2, 2, 3),
    ],
)
def test_duration_condition(
    tmp_path: Path,
    recording: data.Recording,
    operator: str,
    seconds: int,
    passing_duration: int,
    failing_duration: int,
) -> None:
    condition = build_condition_from_str(
        tmp_path,
        f"""
        name: duration
        operator: {operator}
        seconds: {seconds}
        """,
    )

    passing = create_sound_event_annotation(
        recording=recording,
        geometry=data.TimeInterval(coordinates=[0, passing_duration]),
    )
    failing = create_sound_event_annotation(
        recording=recording,
        geometry=data.TimeInterval(coordinates=[0, failing_duration]),
    )

    assert condition(passing)
    assert not condition(failing)


@pytest.mark.parametrize(
    "boundary,operator,hertz,passing_bbox,failing_bbox",
    [
        ("high", "lt", 300, [0, 100, 1, 200], [0, 100, 1, 300]),
        ("high", "lte", 300, [0, 100, 1, 300], [0, 100, 1, 400]),
        ("high", "gt", 300, [0, 100, 1, 400], [0, 100, 1, 300]),
        ("high", "gte", 300, [0, 100, 1, 300], [0, 100, 1, 200]),
        ("high", "eq", 300, [0, 100, 1, 300], [0, 100, 1, 400]),
        ("low", "lt", 200, [0, 100, 1, 400], [0, 200, 1, 400]),
        ("low", "lte", 200, [0, 200, 1, 400], [0, 300, 1, 400]),
        ("low", "gt", 200, [0, 300, 1, 400], [0, 200, 1, 400]),
        ("low", "gte", 200, [0, 200, 1, 400], [0, 100, 1, 400]),
        ("low", "eq", 200, [0, 200, 1, 400], [0, 300, 1, 400]),
    ],
)
def test_frequency_condition(
    tmp_path: Path,
    recording: data.Recording,
    boundary: str,
    operator: str,
    hertz: int,
    passing_bbox: list[int],
    failing_bbox: list[int],
) -> None:
    condition = build_condition_from_str(
        tmp_path,
        f"""
        name: frequency
        boundary: {boundary}
        operator: {operator}
        hertz: {hertz}
        """,
    )

    passing = create_sound_event_annotation(
        recording=recording,
        geometry=data.BoundingBox(
            coordinates=[float(value) for value in passing_bbox]
        ),
    )
    failing = create_sound_event_annotation(
        recording=recording,
        geometry=data.BoundingBox(
            coordinates=[float(value) for value in failing_bbox]
        ),
    )

    assert condition(passing)
    assert not condition(failing)


def test_frequency_condition_is_false_for_temporal_geometries(
    tmp_path: Path,
    recording: data.Recording,
) -> None:
    condition = build_condition_from_str(
        tmp_path,
        """
        name: frequency
        boundary: low
        operator: eq
        hertz: 200
        """,
    )

    passing = create_sound_event_annotation(
        recording=recording,
        geometry=data.BoundingBox(coordinates=[0, 200, 1, 400]),
    )
    failing = create_sound_event_annotation(
        recording=recording,
        geometry=data.TimeInterval(coordinates=[0, 3]),
    )

    assert condition(passing)
    assert not condition(failing)


def test_has_all_tags_fails_if_empty(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least one tag"):
        build_condition_from_str(
            tmp_path,
            """
            name: has_all_tags
            tags: []
            """,
        )


def test_all_of_condition(tmp_path: Path, recording: data.Recording) -> None:
    condition = build_condition_from_str(
        tmp_path,
        """
        name: all_of
        conditions:
            - name: has_tag
              tag:
                key: species
                value: Myotis myotis
            - name: duration
              operator: lt
              seconds: 1
        """,
    )

    passing = create_sound_event_annotation(
        recording=recording,
        geometry=data.TimeInterval(coordinates=[0, 0.5]),
        tags=[data.Tag(key="species", value="Myotis myotis")],
    )
    failing = create_sound_event_annotation(
        recording=recording,
        geometry=data.TimeInterval(coordinates=[0, 2]),
        tags=[data.Tag(key="species", value="Myotis myotis")],
    )

    assert condition(passing)
    assert not condition(failing)


def test_any_of_condition(tmp_path: Path, recording: data.Recording) -> None:
    condition = build_condition_from_str(
        tmp_path,
        """
        name: any_of
        conditions:
            - name: has_tag
              tag:
                key: species
                value: Myotis myotis
            - name: duration
              operator: lt
              seconds: 1
        """,
    )

    passing = create_sound_event_annotation(
        recording=recording,
        geometry=data.TimeInterval(coordinates=[0, 2]),
        tags=[data.Tag(key="species", value="Myotis myotis")],
    )
    failing = create_sound_event_annotation(
        recording=recording,
        geometry=data.TimeInterval(coordinates=[0, 2]),
        tags=[data.Tag(key="species", value="Eptesicus fuscus")],
    )

    assert condition(passing)
    assert not condition(failing)
