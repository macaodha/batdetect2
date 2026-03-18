from collections.abc import Callable
from pathlib import Path

from soundevent import data, terms

from batdetect2.targets import TargetConfig, build_targets


def test_can_override_default_roi_mapper_per_class(
    create_temp_yaml: Callable[..., Path],
    recording: data.Recording,
):
    yaml_content = """
    detection_target:
        name: bat
        match_if:
            name: has_tag
            tag:
                key: order
                value: Chiroptera
        assign_tags:
            - key: order
              value: Chiroptera

    classification_targets:
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

    roi:
        name: anchor_bbox
        anchor: bottom-left
    """
    config_path = create_temp_yaml(yaml_content)

    config = TargetConfig.load(config_path)
    targets = build_targets(config)

    geometry = data.BoundingBox(coordinates=[0.1, 12_000, 0.2, 18_000])

    species = terms.get_term("species")
    assert species is not None

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
    recording,
):
    yaml_content = """
    detection_target:
        name: bat
        match_if:
            name: has_tag
            tag:
                key: order
                value: Chiroptera
        assign_tags:
            - key: order
              value: Chiroptera

    classification_targets:
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

    roi:
        name: anchor_bbox
        anchor: bottom-left
    """
    config_path = create_temp_yaml(yaml_content)

    config = TargetConfig.load(config_path)
    targets = build_targets(config)

    geometry = data.BoundingBox(coordinates=[0.1, 12_000, 0.2, 18_000])

    species = terms.get_term("species")
    assert species is not None
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
