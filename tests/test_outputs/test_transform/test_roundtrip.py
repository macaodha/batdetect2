import pytest
import torch
from soundevent import data
from soundevent.geometry import compute_bounds

from batdetect2.models.types import ModelOutput
from batdetect2.outputs import build_output_transform
from batdetect2.postprocess import build_postprocessor
from batdetect2.targets.types import TargetProtocol
from batdetect2.train.labels import build_clip_labeler


def test_annotation_roundtrip_through_postprocess_and_output_transform(
    create_recording,
    create_clip,
    sample_preprocessor,
    sample_targets: TargetProtocol,
    pippip_tag: data.Tag,
    bat_tag: data.Tag,
) -> None:
    recording = create_recording(duration=30, samplerate=256_000)
    clip = create_clip(recording=recording, start_time=10.0, end_time=10.5)

    annotation = data.SoundEventAnnotation(
        sound_event=data.SoundEvent(
            recording=recording,
            geometry=data.BoundingBox(
                coordinates=[10.2, 40_000, 10.26, 55_000]
            ),
        ),
        tags=[pippip_tag, bat_tag],
    )
    clip_annotation = data.ClipAnnotation(clip=clip, sound_events=[annotation])

    height = 128
    duration = clip.end_time - clip.start_time
    width = int(duration * sample_preprocessor.output_samplerate)
    spec = torch.zeros((1, height, width), dtype=torch.float32)

    labeler = build_clip_labeler(targets=sample_targets)
    heatmaps = labeler(clip_annotation, spec)

    output = ModelOutput(
        detection_probs=heatmaps.detection.unsqueeze(0),
        size_preds=heatmaps.size.unsqueeze(0),
        class_probs=heatmaps.classes.unsqueeze(0),
        features=torch.zeros((1, 1, height, width), dtype=torch.float32),
    )

    postprocessor = build_postprocessor(preprocessor=sample_preprocessor)
    clip_detection_tensors = postprocessor(output)
    assert len(clip_detection_tensors) == 1

    transform = build_output_transform(targets=sample_targets)
    clip_detections = transform.to_clip_detections(
        detections=clip_detection_tensors[0],
        clip=clip,
    )

    assert len(clip_detections.detections) == 1
    recovered = clip_detections.detections[0]

    recovered_bounds = compute_bounds(recovered.geometry)
    original_bounds = compute_bounds(annotation.sound_event.geometry)

    # 1 ms of tolerance (spectrogram resolution)
    assert recovered_bounds[0] == pytest.approx(original_bounds[0], abs=0.001)
    assert recovered_bounds[2] == pytest.approx(original_bounds[2], abs=0.001)

    # 1000 Hz of tolerance (spectrogram resolution)
    assert recovered_bounds[1] == pytest.approx(original_bounds[1], abs=1000)
    assert recovered_bounds[3] == pytest.approx(original_bounds[3], abs=1000)
