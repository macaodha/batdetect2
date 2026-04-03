import json
import textwrap
from pathlib import Path

import pytest
import yaml
from pydantic import TypeAdapter
from soundevent import data

from batdetect2.data.conditions import (
    IdInListConfig,
    SoundEventConditionConfig,
    build_sound_event_condition,
)


def build_condition_from_str(content, base_dir: Path | None = None):
    content = textwrap.dedent(content)
    content = yaml.safe_load(content)
    config = TypeAdapter(SoundEventConditionConfig).validate_python(content)
    return build_sound_event_condition(config, base_dir=base_dir)


def test_has_tag(sound_event: data.SoundEvent):
    condition = build_condition_from_str("""
    name: has_tag
    tag:
        key: species
        value: Myotis myotis
    """)

    sound_event_annotation = data.SoundEventAnnotation(
        sound_event=sound_event,
        tags=[data.Tag(key="species", value="Myotis myotis")],
    )
    assert condition(sound_event_annotation)

    sound_event_annotation = data.SoundEventAnnotation(
        sound_event=sound_event,
        tags=[data.Tag(key="species", value="Eptesicus fuscus")],
    )
    assert not condition(sound_event_annotation)


def test_has_all_tags(sound_event: data.SoundEvent):
    condition = build_condition_from_str("""
    name: has_all_tags
    tags:
        - key: species
          value: Myotis myotis
        - key: event
          value: Echolocation
    """)

    sound_event_annotation = data.SoundEventAnnotation(
        sound_event=sound_event,
        tags=[data.Tag(key="species", value="Myotis myotis")],
    )
    assert not condition(sound_event_annotation)

    sound_event_annotation = data.SoundEventAnnotation(
        sound_event=sound_event,
        tags=[
            data.Tag(key="species", value="Eptesicus fuscus"),
            data.Tag(key="event", value="Echolocation"),
        ],
    )
    assert not condition(sound_event_annotation)

    sound_event_annotation = data.SoundEventAnnotation(
        sound_event=sound_event,
        tags=[
            data.Tag(key="species", value="Myotis myotis"),
            data.Tag(key="event", value="Echolocation"),
        ],
    )
    assert condition(sound_event_annotation)

    sound_event_annotation = data.SoundEventAnnotation(
        sound_event=sound_event,
        tags=[
            data.Tag(key="species", value="Myotis myotis"),
            data.Tag(key="event", value="Echolocation"),
            data.Tag(key="sex", value="Female"),
        ],
    )
    assert condition(sound_event_annotation)


def test_has_any_tags(sound_event: data.SoundEvent):
    condition = build_condition_from_str("""
    name: has_any_tag
    tags:
        - key: species
          value: Myotis myotis
        - key: event
          value: Echolocation
    """)

    sound_event_annotation = data.SoundEventAnnotation(
        sound_event=sound_event,
        tags=[data.Tag(key="species", value="Myotis myotis")],
    )
    assert condition(sound_event_annotation)

    sound_event_annotation = data.SoundEventAnnotation(
        sound_event=sound_event,
        tags=[
            data.Tag(key="species", value="Eptesicus fuscus"),
            data.Tag(key="event", value="Echolocation"),
        ],
    )
    assert condition(sound_event_annotation)

    sound_event_annotation = data.SoundEventAnnotation(
        sound_event=sound_event,
        tags=[
            data.Tag(key="species", value="Myotis myotis"),
            data.Tag(key="event", value="Echolocation"),
        ],
    )
    assert condition(sound_event_annotation)

    sound_event_annotation = data.SoundEventAnnotation(
        sound_event=sound_event,
        tags=[
            data.Tag(key="species", value="Eptesicus fuscus"),
            data.Tag(key="event", value="Social"),
        ],
    )
    assert not condition(sound_event_annotation)


def test_not(sound_event: data.SoundEvent):
    condition = build_condition_from_str("""
    name: not
    condition:
        name: has_tag
        tag:
            key: species
            value: Myotis myotis
    """)

    sound_event_annotation = data.SoundEventAnnotation(
        sound_event=sound_event,
        tags=[data.Tag(key="species", value="Myotis myotis")],
    )
    assert not condition(sound_event_annotation)

    sound_event_annotation = data.SoundEventAnnotation(
        sound_event=sound_event,
        tags=[data.Tag(key="species", value="Eptesicus fuscus")],
    )
    assert condition(sound_event_annotation)

    sound_event_annotation = data.SoundEventAnnotation(
        sound_event=sound_event,
        tags=[
            data.Tag(key="species", value="Myotis myotis"),
            data.Tag(key="event", value="Echolocation"),
        ],
    )
    assert not condition(sound_event_annotation)


def test_id_in_list(sound_event: data.SoundEvent, tmp_path: Path):
    se1 = data.SoundEventAnnotation(sound_event=sound_event)
    se2 = data.SoundEventAnnotation(sound_event=sound_event)
    ids_path = tmp_path / "sound_event_ids.json"
    ids_path.write_text(json.dumps([str(se1.uuid)]))

    condition = build_sound_event_condition(IdInListConfig(path=ids_path))

    assert condition(se1)
    assert not condition(se2)


def test_id_in_list_uses_base_dir(
    sound_event: data.SoundEvent,
    tmp_path: Path,
) -> None:
    se = data.SoundEventAnnotation(sound_event=sound_event)
    split_dir = tmp_path / "splits"
    split_dir.mkdir()
    ids_path = split_dir / "sound_event_ids.json"
    ids_path.write_text(json.dumps([str(se.uuid)]))

    condition = build_sound_event_condition(
        IdInListConfig(path=Path("splits/sound_event_ids.json")),
        base_dir=tmp_path,
    )

    assert condition(se)


def test_duration(recording: data.Recording):
    se1 = data.SoundEventAnnotation(
        sound_event=data.SoundEvent(
            recording=recording, geometry=data.TimeInterval(coordinates=[0, 1])
        ),
    )
    se2 = data.SoundEventAnnotation(
        sound_event=data.SoundEvent(
            recording=recording, geometry=data.TimeInterval(coordinates=[0, 2])
        ),
    )
    se3 = data.SoundEventAnnotation(
        sound_event=data.SoundEvent(
            recording=recording, geometry=data.TimeInterval(coordinates=[0, 3])
        ),
    )

    condition = build_condition_from_str("""
    name: duration
    operator: lt
    seconds: 2
    """)
    assert condition(se1)
    assert not condition(se2)
    assert not condition(se3)

    condition = build_condition_from_str("""
    name: duration
    operator: lte
    seconds: 2
    """)

    assert condition(se1)
    assert condition(se2)
    assert not condition(se3)

    condition = build_condition_from_str("""
    name: duration
    operator: gt
    seconds: 2
    """)

    assert not condition(se1)
    assert not condition(se2)
    assert condition(se3)

    condition = build_condition_from_str("""
    name: duration
    operator: gte
    seconds: 2
    """)

    assert not condition(se1)
    assert condition(se2)
    assert condition(se3)

    condition = build_condition_from_str("""
    name: duration
    operator: eq
    seconds: 2
    """)

    assert not condition(se1)
    assert condition(se2)
    assert not condition(se3)


def test_frequency(recording: data.Recording):
    se12 = data.SoundEventAnnotation(
        sound_event=data.SoundEvent(
            recording=recording,
            geometry=data.BoundingBox(coordinates=[0, 100, 1, 200]),
        ),
    )
    se13 = data.SoundEventAnnotation(
        sound_event=data.SoundEvent(
            recording=recording,
            geometry=data.BoundingBox(coordinates=[0, 100, 2, 300]),
        ),
    )
    se14 = data.SoundEventAnnotation(
        sound_event=data.SoundEvent(
            recording=recording,
            geometry=data.BoundingBox(coordinates=[0, 100, 3, 400]),
        ),
    )
    se24 = data.SoundEventAnnotation(
        sound_event=data.SoundEvent(
            recording=recording,
            geometry=data.BoundingBox(coordinates=[0, 200, 3, 400]),
        ),
    )
    se34 = data.SoundEventAnnotation(
        sound_event=data.SoundEvent(
            recording=recording,
            geometry=data.BoundingBox(coordinates=[0, 300, 3, 400]),
        ),
    )

    condition = build_condition_from_str("""
    name: frequency
    boundary: high
    operator: lt
    hertz: 300
    """)
    assert condition(se12)
    assert not condition(se13)
    assert not condition(se14)

    condition = build_condition_from_str("""
    name: frequency
    boundary: high
    operator: lte
    hertz: 300
    """)

    assert condition(se12)
    assert condition(se13)
    assert not condition(se14)

    condition = build_condition_from_str("""
    name: frequency
    boundary: high
    operator: gt
    hertz: 300
    """)

    assert not condition(se12)
    assert not condition(se13)
    assert condition(se14)

    condition = build_condition_from_str("""
    name: frequency
    boundary: high
    operator: gte
    hertz: 300
    """)

    assert not condition(se12)
    assert condition(se13)
    assert condition(se14)

    condition = build_condition_from_str("""
    name: frequency
    boundary: high
    operator: eq
    hertz: 300
    """)

    assert not condition(se12)
    assert condition(se13)
    assert not condition(se14)

    # LOW

    condition = build_condition_from_str("""
    name: frequency
    boundary: low
    operator: lt
    hertz: 200
    """)
    assert condition(se14)
    assert not condition(se24)
    assert not condition(se34)

    condition = build_condition_from_str("""
    name: frequency
    boundary: low
    operator: lte
    hertz: 200
    """)

    assert condition(se14)
    assert condition(se24)
    assert not condition(se34)

    condition = build_condition_from_str("""
    name: frequency
    boundary: low
    operator: gt
    hertz: 200
    """)

    assert not condition(se14)
    assert not condition(se24)
    assert condition(se34)

    condition = build_condition_from_str("""
    name: frequency
    boundary: low
    operator: gte
    hertz: 200
    """)

    assert not condition(se14)
    assert condition(se24)
    assert condition(se34)

    condition = build_condition_from_str("""
    name: frequency
    boundary: low
    operator: eq
    hertz: 200
    """)

    assert not condition(se14)
    assert condition(se24)
    assert not condition(se34)


def test_frequency_is_false_for_temporal_geometries(recording: data.Recording):
    condition = build_condition_from_str("""
    name: frequency
    boundary: low
    operator: eq
    hertz: 200
    """)
    se = data.SoundEventAnnotation(
        sound_event=data.SoundEvent(
            geometry=data.TimeInterval(coordinates=[0, 3]),
            recording=recording,
        )
    )
    assert not condition(se)

    se = data.SoundEventAnnotation(
        sound_event=data.SoundEvent(
            geometry=data.TimeStamp(coordinates=3),
            recording=recording,
        )
    )
    assert not condition(se)


def test_has_tags_fails_if_empty():
    with pytest.raises(ValueError):
        build_condition_from_str("""
        name: has_tags
        tags: []
        """)


def test_all_of(recording: data.Recording):
    condition = build_condition_from_str("""
    name: all_of
    conditions:
        - name: has_tag
          tag:
            key: species
            value: Myotis myotis
        - name: duration
          operator: lt
          seconds: 1
    """)
    se = data.SoundEventAnnotation(
        sound_event=data.SoundEvent(
            geometry=data.TimeInterval(coordinates=[0, 0.5]),
            recording=recording,
        ),
        tags=[data.Tag(key="species", value="Myotis myotis")],
    )
    assert condition(se)

    se = data.SoundEventAnnotation(
        sound_event=data.SoundEvent(
            geometry=data.TimeInterval(coordinates=[0, 2]),
            recording=recording,
        ),
        tags=[data.Tag(key="species", value="Myotis myotis")],
    )
    assert not condition(se)

    se = data.SoundEventAnnotation(
        sound_event=data.SoundEvent(
            geometry=data.TimeInterval(coordinates=[0, 0.5]),
            recording=recording,
        ),
        tags=[data.Tag(key="species", value="Eptesicus fuscus")],
    )
    assert not condition(se)


def test_any_of(recording: data.Recording):
    condition = build_condition_from_str("""
    name: any_of
    conditions:
        - name: has_tag
          tag:
            key: species
            value: Myotis myotis
        - name: duration
          operator: lt
          seconds: 1
    """)
    se = data.SoundEventAnnotation(
        sound_event=data.SoundEvent(
            geometry=data.TimeInterval(coordinates=[0, 2]),
            recording=recording,
        ),
        tags=[data.Tag(key="species", value="Eptesicus fuscus")],
    )
    assert not condition(se)

    se = data.SoundEventAnnotation(
        sound_event=data.SoundEvent(
            geometry=data.TimeInterval(coordinates=[0, 0.5]),
            recording=recording,
        ),
        tags=[data.Tag(key="species", value="Myotis myotis")],
    )
    assert condition(se)

    se = data.SoundEventAnnotation(
        sound_event=data.SoundEvent(
            geometry=data.TimeInterval(coordinates=[0, 2]),
            recording=recording,
        ),
        tags=[data.Tag(key="species", value="Myotis myotis")],
    )
    assert condition(se)

    se = data.SoundEventAnnotation(
        sound_event=data.SoundEvent(
            geometry=data.TimeInterval(coordinates=[0, 0.5]),
            recording=recording,
        ),
        tags=[data.Tag(key="species", value="Eptesicus fuscus")],
    )
    assert condition(se)
