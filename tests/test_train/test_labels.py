from pathlib import Path

import torch
from soundevent import data

from batdetect2.targets import TargetConfig, build_targets
from batdetect2.targets.rois import AnchorBBoxMapperConfig
from batdetect2.train.labels import generate_heatmaps

recording = data.Recording(
    samplerate=256_000,
    duration=1,
    channels=1,
    time_expansion=1,
    hash="asdf98sdf",
    path=Path("/path/to/audio.wav"),
)

clip = data.Clip(
    recording=recording,
    start_time=0,
    end_time=100,
)


def test_generated_heatmap_are_non_zero_at_correct_positions(
    sample_target_config: TargetConfig,
    pippip_tag: data.Tag,
):
    config = sample_target_config.model_copy(
        update=dict(
            roi=AnchorBBoxMapperConfig(
                time_scale=1,
                frequency_scale=1,
            )
        )
    )

    targets = build_targets(config)

    clip_annotation = data.ClipAnnotation(
        clip=clip,
        sound_events=[
            data.SoundEventAnnotation(
                sound_event=data.SoundEvent(
                    recording=recording,
                    geometry=data.BoundingBox(
                        coordinates=[10, 10, 20, 30],
                    ),
                ),
                tags=[data.Tag(key=pippip_tag.key, value=pippip_tag.value)],  # type: ignore
            )
        ],
    )

    detection_heatmap, class_heatmap, size_heatmap = generate_heatmaps(
        clip_annotation,
        torch.rand([1, 100, 100]),
        min_freq=0,
        max_freq=100,
        targets=targets,
    )
    pippip_index = targets.class_names.index("pippip")
    myomyo_index = targets.class_names.index("myomyo")
    assert size_heatmap[0, 10, 10] == 10
    assert size_heatmap[1, 10, 10] == 20
    assert class_heatmap[pippip_index, 10, 10] == 1.0
    assert class_heatmap[myomyo_index, 10, 10] == 0.0
    assert detection_heatmap[0, 10, 10] == 1.0
