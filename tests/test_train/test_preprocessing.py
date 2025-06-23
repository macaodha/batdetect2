import pytest
import torch
import xarray as xr
from soundevent import data

from batdetect2.models.types import ModelOutput
from batdetect2.postprocess import build_postprocessor, load_postprocess_config
from batdetect2.preprocess import build_preprocessor, load_preprocessing_config
from batdetect2.targets import build_targets, load_target_config
from batdetect2.targets.terms import get_term_from_key
from batdetect2.train.labels import build_clip_labeler, load_label_config
from batdetect2.train.preprocess import generate_train_example


@pytest.fixture
def build_from_config(
    create_temp_yaml,
    sample_term_registry,
):
    def build(yaml_content):
        config_path = create_temp_yaml(yaml_content)

        targets_config = load_target_config(config_path, field="targets")
        preprocessing_config = load_preprocessing_config(
            config_path,
            field="preprocessing",
        )
        labels_config = load_label_config(config_path, field="labels")
        postprocessing_config = load_postprocess_config(
            config_path,
            field="postprocessing",
        )

        targets = build_targets(
            targets_config, term_registry=sample_term_registry
        )
        preprocessor = build_preprocessor(preprocessing_config)
        labeller = build_clip_labeler(
            targets=targets,
            config=labels_config,
        )
        postprocessor = build_postprocessor(
            targets,
            config=postprocessing_config,
            min_freq=preprocessor.min_freq,
            max_freq=preprocessor.max_freq,
        )

        return targets, preprocessor, labeller, postprocessor

    return build


# TODO: better name
def test_generated_train_example_has_expected_outputs(
    build_from_config,
    sample_term_registry,
    recording,
):
    yaml_content = """
    labels:
    targets:
        roi:
            name: anchor_bbox
            anchor: bottom-left
        classes:
            classes:
              - name: pippip
                tags:
                  - key: species
                    value: Pipistrellus pipistrellus
            generic_class:
              - key: order
                value: Chiroptera
    preprocessing:
    postprocessing:
    """
    _, preprocessor, labeller, _ = build_from_config(yaml_content)

    geometry = data.BoundingBox(coordinates=[0.1, 12_000, 0.2, 18_000])
    species = get_term_from_key("species", term_registry=sample_term_registry)
    se1 = data.SoundEventAnnotation(
        sound_event=data.SoundEvent(recording=recording, geometry=geometry),
        tags=[data.Tag(term=species, value="Pipistrellus pipistrellus")],
    )
    clip_annotation = data.ClipAnnotation(
        clip=data.Clip(start_time=0, end_time=0.5, recording=recording),
        sound_events=[se1],
    )

    encoded = generate_train_example(clip_annotation, preprocessor, labeller)

    assert isinstance(encoded, xr.Dataset)
    assert "audio" in encoded
    assert "spectrogram" in encoded
    assert "detection" in encoded
    assert "class" in encoded
    assert "size" in encoded

    spec_shape = encoded["spectrogram"].shape
    assert len(spec_shape) == 2

    height, width = spec_shape
    assert encoded["detection"].shape == (height, width)
    assert encoded["class"].shape == (1, height, width)
    assert encoded["size"].shape == (2, height, width)


def test_encoding_decoding_roundtrip_recovers_object(
    build_from_config,
    sample_term_registry,
    recording,
):
    yaml_content = """
    labels:
    targets:
        roi:
            name: anchor_bbox
            anchor: bottom-left
        classes:
            classes:
              - name: pippip
                tags:
                  - key: species
                    value: Pipistrellus pipistrellus
            generic_class:
              - key: order
                value: Chiroptera
    preprocessing:
    """
    _, preprocessor, labeller, postprocessor = build_from_config(yaml_content)

    geometry = data.BoundingBox(coordinates=[0.1, 40_000, 0.2, 80_000])
    species = get_term_from_key("species", term_registry=sample_term_registry)
    se1 = data.SoundEventAnnotation(
        sound_event=data.SoundEvent(recording=recording, geometry=geometry),
        tags=[data.Tag(term=species, value="Pipistrellus pipistrellus")],
    )
    clip = data.Clip(start_time=0, end_time=0.5, recording=recording)
    clip_annotation = data.ClipAnnotation(clip=clip, sound_events=[se1])

    encoded = generate_train_example(clip_annotation, preprocessor, labeller)
    predictions = postprocessor.get_predictions(
        ModelOutput(
            detection_probs=torch.tensor([[encoded["detection"].data]]),
            size_preds=torch.tensor([encoded["size"].data]),
            class_probs=torch.tensor([encoded["class"].data]),
            features=torch.tensor([[encoded["spectrogram"].data]]),
        ),
        [clip],
    )[0]

    assert isinstance(predictions, data.ClipPrediction)
    assert len(predictions.sound_events) == 1

    recovered = predictions.sound_events[0]
    assert recovered.sound_event.geometry is not None
    assert isinstance(recovered.sound_event.geometry, data.BoundingBox)
    start_time_rec, low_freq_rec, end_time_rec, high_freq_rec = (
        recovered.sound_event.geometry.coordinates
    )
    start_time_or, low_freq_or, end_time_or, high_freq_or = (
        geometry.coordinates
    )

    assert start_time_rec == pytest.approx(start_time_or, abs=0.01)
    assert low_freq_rec == pytest.approx(low_freq_or, abs=1_000)
    assert end_time_rec == pytest.approx(end_time_or, abs=0.01)
    assert high_freq_rec == pytest.approx(high_freq_or, abs=1_000)

    assert len(recovered.tags) == 2

    predicted_species_tag = next(
        iter(t for t in recovered.tags if t.tag.term == species), None
    )
    assert predicted_species_tag is not None
    assert predicted_species_tag.score == 1
    assert predicted_species_tag.tag.value == "Pipistrellus pipistrellus"

    predicted_order_tag = next(
        iter(t for t in recovered.tags if t.tag.term.label == "order"), None
    )
    assert predicted_order_tag is not None
    assert predicted_order_tag.score == 1
    assert predicted_order_tag.tag.value == "Chiroptera"


def test_encoding_decoding_roundtrip_recovers_object_with_roi_override(
    build_from_config,
    sample_term_registry,
    recording,
):
    yaml_content = """
    labels:
    targets:
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
    preprocessing:
    """
    _, preprocessor, labeller, postprocessor = build_from_config(yaml_content)

    geometry = data.BoundingBox(coordinates=[0.1, 40_000, 0.2, 80_000])
    species = get_term_from_key("species", term_registry=sample_term_registry)
    se1 = data.SoundEventAnnotation(
        sound_event=data.SoundEvent(recording=recording, geometry=geometry),
        tags=[data.Tag(term=species, value="Myotis myotis")],
    )
    clip = data.Clip(start_time=0, end_time=0.5, recording=recording)
    clip_annotation = data.ClipAnnotation(clip=clip, sound_events=[se1])

    encoded = generate_train_example(clip_annotation, preprocessor, labeller)
    predictions = postprocessor.get_predictions(
        ModelOutput(
            detection_probs=torch.tensor([[encoded["detection"].data]]),
            size_preds=torch.tensor([encoded["size"].data]),
            class_probs=torch.tensor([encoded["class"].data]),
            features=torch.tensor([[encoded["spectrogram"].data]]),
        ),
        [clip],
    )[0]

    assert isinstance(predictions, data.ClipPrediction)
    assert len(predictions.sound_events) == 1

    recovered = predictions.sound_events[0]
    assert recovered.sound_event.geometry is not None
    assert isinstance(recovered.sound_event.geometry, data.BoundingBox)
    start_time_rec, low_freq_rec, end_time_rec, high_freq_rec = (
        recovered.sound_event.geometry.coordinates
    )
    start_time_or, low_freq_or, end_time_or, high_freq_or = (
        geometry.coordinates
    )

    assert start_time_rec == pytest.approx(start_time_or, abs=0.01)
    assert low_freq_rec == pytest.approx(low_freq_or, abs=1_000)
    assert end_time_rec == pytest.approx(end_time_or, abs=0.01)
    assert high_freq_rec == pytest.approx(high_freq_or, abs=1_000)

    assert len(recovered.tags) == 2

    predicted_species_tag = next(
        iter(t for t in recovered.tags if t.tag.term == species), None
    )
    assert predicted_species_tag is not None
    assert predicted_species_tag.score == 1
    assert predicted_species_tag.tag.value == "Myotis myotis"

    predicted_order_tag = next(
        iter(t for t in recovered.tags if t.tag.term.label == "order"), None
    )
    assert predicted_order_tag is not None
    assert predicted_order_tag.score == 1
    assert predicted_order_tag.tag.value == "Chiroptera"
