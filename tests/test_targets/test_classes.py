from pathlib import Path
from typing import Callable
from uuid import uuid4

import pytest
from soundevent import data
from soundevent.terms import get_term

from batdetect2.targets.classes import (
    TargetClassConfig,
    build_sound_event_decoder,
    build_sound_event_encoder,
    get_class_names_from_config,
)


@pytest.fixture
def sample_annotation(
    sound_event: data.SoundEvent,
) -> data.SoundEventAnnotation:
    """Fixture for a sample SoundEventAnnotation."""
    return data.SoundEventAnnotation(
        sound_event=sound_event,
        tags=[
            data.Tag(key="species", value="Pipistrellus pipistrellus"),
            data.Tag(key="quality", value="Good"),
        ],
    )


@pytest.fixture
def create_temp_yaml(tmp_path: Path) -> Callable[[str], Path]:
    """Create a temporary YAML file with the given content."""

    def factory(content: str) -> Path:
        temp_file = tmp_path / f"{uuid4()}.yaml"
        temp_file.write_text(content)
        return temp_file

    return factory


def test_get_class_names_from_config():
    target_class1 = TargetClassConfig(
        name="pippip",
        tags=[data.Tag(key="species", value="Pipistrellus pipistrellus")],
    )
    target_class2 = TargetClassConfig(
        name="myodau",
        tags=[data.Tag(key="species", value="Myotis daubentonii")],
    )
    names = get_class_names_from_config([target_class1, target_class2])
    assert names == ["pippip", "myodau"]


def test_build_encoder_from_config(
    sample_annotation: data.SoundEventAnnotation,
):
    classes = [
        TargetClassConfig(
            name="pippip",
            tags=[data.Tag(key="species", value="Pipistrellus pipistrellus")],
        )
    ]
    encoder = build_sound_event_encoder(classes)
    result = encoder(sample_annotation)
    assert result == "pippip"

    classes = []
    encoder = build_sound_event_encoder(classes)
    result = encoder(sample_annotation)
    assert result is None


def test_build_decoder_from_config():
    classes = [
        TargetClassConfig(
            name="pippip",
            tags=[data.Tag(key="species", value="Pipistrellus pipistrellus")],
            assign_tags=[data.Tag(key="call_type", value="Echolocation")],
        )
    ]
    decoder = build_sound_event_decoder(classes)
    tags = decoder("pippip")
    assert len(tags) == 1
    assert tags[0].term == get_term("event")
    assert tags[0].value == "Echolocation"

    # Test when output_tags is None, should fall back to tags
    classes = [
        TargetClassConfig(
            name="pippip",
            tags=[data.Tag(key="species", value="Pipistrellus pipistrellus")],
        )
    ]
    decoder = build_sound_event_decoder(classes)
    tags = decoder("pippip")
    assert len(tags) == 1
    assert tags[0].term == get_term("species")
    assert tags[0].value == "Pipistrellus pipistrellus"

    # Test raise_on_unmapped=True
    decoder = build_sound_event_decoder(classes, raise_on_unmapped=True)
    with pytest.raises(ValueError):
        decoder("unknown_class")

    # Test raise_on_unmapped=False
    decoder = build_sound_event_decoder(classes, raise_on_unmapped=False)
    tags = decoder("unknown_class")
    assert len(tags) == 0
