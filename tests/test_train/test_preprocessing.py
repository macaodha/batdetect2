import pytest
from soundevent import data
from soundevent.terms import get_term

from batdetect2.postprocess import build_postprocessor, load_postprocess_config
from batdetect2.preprocess import build_preprocessor, load_preprocessing_config
from batdetect2.targets import build_targets, load_target_config
from batdetect2.train.labels import build_clip_labeler, load_label_config
from batdetect2.train.preprocess import generate_train_example
from batdetect2.typing import ModelOutput
from batdetect2.typing.preprocess import AudioLoader


@pytest.fixture
def build_from_config(
    create_temp_yaml,
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

        targets = build_targets(targets_config)
        preprocessor = build_preprocessor(preprocessing_config)
        labeller = build_clip_labeler(
            targets=targets,
            config=labels_config,
            min_freq=preprocessor.min_freq,
            max_freq=preprocessor.max_freq,
        )
        postprocessor = build_postprocessor(
            targets,
            preprocessor=preprocessor,
            config=postprocessing_config,
        )

        return targets, preprocessor, labeller, postprocessor

    return build


def test_encoding_decoding_roundtrip_recovers_object(
    sample_audio_loader: AudioLoader,
    build_from_config,
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
    se1 = data.SoundEventAnnotation(
        sound_event=data.SoundEvent(recording=recording, geometry=geometry),
        tags=[
            data.Tag(key="species", value="Pipistrellus pipistrellus"),  # type: ignore
        ],
    )
    clip = data.Clip(start_time=0, end_time=0.5, recording=recording)
    clip_annotation = data.ClipAnnotation(clip=clip, sound_events=[se1])

    encoded = generate_train_example(
        clip_annotation,
        sample_audio_loader,
        preprocessor,
        labeller,
    )
    predictions = postprocessor.get_predictions(
        ModelOutput(
            detection_probs=encoded.detection_heatmap.unsqueeze(0).unsqueeze(
                0
            ),
            size_preds=encoded.size_heatmap.unsqueeze(0),
            class_probs=encoded.class_heatmap.unsqueeze(0),
            features=encoded.spectrogram.unsqueeze(0).unsqueeze(0),
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
        iter(t for t in recovered.tags if t.tag.term == get_term("species")),
        None,
    )
    assert predicted_species_tag is not None
    assert predicted_species_tag.score == 1
    assert predicted_species_tag.tag.value == "Pipistrellus pipistrellus"

    predicted_order_tag = next(
        iter(t for t in recovered.tags if t.tag.term == get_term("order")),
        None,
    )
    assert predicted_order_tag is not None
    assert predicted_order_tag.score == 1
    assert predicted_order_tag.tag.value == "Chiroptera"


def test_encoding_decoding_roundtrip_recovers_object_with_roi_override(
    sample_audio_loader: AudioLoader,
    build_from_config,
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
    se1 = data.SoundEventAnnotation(
        sound_event=data.SoundEvent(recording=recording, geometry=geometry),
        tags=[data.Tag(key="species", value="Myotis myotis")],  # type: ignore
    )
    clip = data.Clip(start_time=0, end_time=0.5, recording=recording)
    clip_annotation = data.ClipAnnotation(clip=clip, sound_events=[se1])

    encoded = generate_train_example(
        clip_annotation,
        sample_audio_loader,
        preprocessor,
        labeller,
    )
    predictions = postprocessor.get_predictions(
        ModelOutput(
            detection_probs=encoded.detection_heatmap.unsqueeze(0).unsqueeze(
                0
            ),
            size_preds=encoded.size_heatmap.unsqueeze(0),
            class_probs=encoded.class_heatmap.unsqueeze(0),
            features=encoded.spectrogram.unsqueeze(0).unsqueeze(0),
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
        iter(t for t in recovered.tags if t.tag.term == get_term("species")),
        None,
    )
    assert predicted_species_tag is not None
    assert predicted_species_tag.score == 1
    assert predicted_species_tag.tag.value == "Myotis myotis"

    predicted_order_tag = next(
        iter(t for t in recovered.tags if t.tag.term == get_term("order")),
        None,
    )
    assert predicted_order_tag is not None
    assert predicted_order_tag.score == 1
    assert predicted_order_tag.tag.value == "Chiroptera"
