from collections.abc import Callable
from pathlib import Path

from soundevent import data

from batdetect2.targets import build_targets, load_target_config
from batdetect2.targets.terms import get_term_from_key


def test_can_override_default_roi_mapper_per_class(
    create_temp_yaml: Callable[..., Path],
    recording: data.Recording,
    sample_term_registry,
):
    yaml_content = """
    roi:
        name: anchor_bbox
        anchor: bottom-left
    classes:
        classes:
          - name: pippip
            tags:
              - key: species
                value: Pipistrellus pipistrellus
          - name: myomyo
            tags:
              - key: species
                value: Myotis myotis
            roi:
                name: anchor_bbox
                anchor: top-left
        generic_class:
          - key: order
            value: Chiroptera
    """
    config_path = create_temp_yaml(yaml_content)

    config = load_target_config(config_path)
    targets = build_targets(config, term_registry=sample_term_registry)

    geometry = data.BoundingBox(coordinates=[0.1, 12_000, 0.2, 18_000])

    species = get_term_from_key("species", term_registry=sample_term_registry)
    se1 = data.SoundEventAnnotation(
        sound_event=data.SoundEvent(recording=recording, geometry=geometry),
        tags=[data.Tag(term=species, value="Pipistrellus pipistrellus")],
    )

    se2 = data.SoundEventAnnotation(
        sound_event=data.SoundEvent(recording=recording, geometry=geometry),
        tags=[data.Tag(term=species, value="Myotis myotis")],
    )

    (time1, freq1), _ = targets.encode_roi(se1)
    (time2, freq2), _ = targets.encode_roi(se2)

    assert time1 == time2 == 0.1
    assert freq1 == 12_000
    assert freq2 == 18_000


# TODO: rename this test function
def test_roi_is_recovered_roundtrip_even_with_overriders(
    create_temp_yaml,
    sample_term_registry,
    recording,
):
    yaml_content = """
    roi:
        name: anchor_bbox
        anchor: bottom-left
    classes:
        classes:
          - name: pippip
            tags:
              - key: species
                value: Pipistrellus pipistrellus
          - name: myomyo
            tags:
              - key: species
                value: Myotis myotis
            roi:
                name: anchor_bbox
                anchor: top-left
        generic_class:
          - key: order
            value: Chiroptera
    """
    config_path = create_temp_yaml(yaml_content)

    config = load_target_config(config_path)
    targets = build_targets(config, term_registry=sample_term_registry)

    geometry = data.BoundingBox(coordinates=[0.1, 12_000, 0.2, 18_000])

    species = get_term_from_key("species", term_registry=sample_term_registry)
    se1 = data.SoundEventAnnotation(
        sound_event=data.SoundEvent(recording=recording, geometry=geometry),
        tags=[data.Tag(term=species, value="Pipistrellus pipistrellus")],
    )

    se2 = data.SoundEventAnnotation(
        sound_event=data.SoundEvent(recording=recording, geometry=geometry),
        tags=[data.Tag(term=species, value="Myotis myotis")],
    )

    position1, size1 = targets.encode_roi(se1)
    position2, size2 = targets.encode_roi(se2)

    class_name1 = targets.encode_class(se1)
    class_name2 = targets.encode_class(se2)

    recovered1 = targets.decode_roi(position1, size1, class_name=class_name1)
    recovered2 = targets.decode_roi(position2, size2, class_name=class_name2)

    assert recovered1 == geometry
    assert recovered2 == geometry
