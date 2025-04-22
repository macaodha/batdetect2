from pathlib import Path

import numpy as np
import xarray as xr
from soundevent import data

from batdetect2.targets import TargetConfig, TargetProtocol, build_targets
from batdetect2.targets.rois import ROIConfig
from batdetect2.targets.terms import TagInfo, TermRegistry
from batdetect2.train.labels import generate_heatmaps
from tests.test_targets.test_transform import term_registry

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
    end_time=1,
)


def test_generated_heatmaps_have_correct_dimensions(
    sample_targets: TargetProtocol,
):
    spec = xr.DataArray(
        data=np.random.rand(100, 100),
        dims=["time", "frequency"],
        coords={
            "time": np.linspace(0, 100, 100, endpoint=False),
            "frequency": np.linspace(0, 100, 100, endpoint=False),
        },
    )

    clip_annotation = data.ClipAnnotation(
        clip=clip,
        sound_events=[
            data.SoundEventAnnotation(
                sound_event=data.SoundEvent(
                    recording=recording,
                    geometry=data.BoundingBox(
                        coordinates=[10, 10, 20, 20],
                    ),
                ),
            )
        ],
    )

    detection_heatmap, class_heatmap, size_heatmap = generate_heatmaps(
        clip_annotation.sound_events,
        spec,
        targets=sample_targets,
    )

    assert isinstance(detection_heatmap, xr.DataArray)
    assert detection_heatmap.shape == (100, 100)
    assert detection_heatmap.dims == ("time", "frequency")

    assert isinstance(class_heatmap, xr.DataArray)
    assert class_heatmap.shape == (2, 100, 100)
    assert class_heatmap.dims == ("category", "time", "frequency")
    assert class_heatmap.coords["category"].values.tolist() == [
        "pippip",
        "myomyo",
    ]

    assert isinstance(size_heatmap, xr.DataArray)
    assert size_heatmap.shape == (2, 100, 100)
    assert size_heatmap.dims == ("dimension", "time", "frequency")
    assert size_heatmap.coords["dimension"].values.tolist() == [
        "width",
        "height",
    ]


def test_generated_heatmap_are_non_zero_at_correct_positions(
    sample_target_config: TargetConfig,
    sample_term_registry: TermRegistry,
    pippip_tag: TagInfo,
):
    config = sample_target_config.model_copy(
        update=dict(
            roi=ROIConfig(
                time_scale=1,
                frequency_scale=1,
            )
        )
    )

    targets = build_targets(config, term_registry=sample_term_registry)

    spec = xr.DataArray(
        data=np.random.rand(100, 100),
        dims=["time", "frequency"],
        coords={
            "time": np.linspace(0, 100, 100, endpoint=False),
            "frequency": np.linspace(0, 100, 100, endpoint=False),
        },
    )

    clip_annotation = data.ClipAnnotation(
        clip=clip,
        sound_events=[
            data.SoundEventAnnotation(
                sound_event=data.SoundEvent(
                    recording=recording,
                    geometry=data.BoundingBox(
                        coordinates=[10, 10, 20, 20],
                    ),
                ),
                tags=[
                    data.Tag(
                        term=sample_term_registry[pippip_tag.key],
                        value=pippip_tag.value,
                    )
                ],
            )
        ],
    )

    detection_heatmap, class_heatmap, size_heatmap = generate_heatmaps(
        clip_annotation.sound_events,
        spec,
        targets=targets,
    )
    assert size_heatmap.sel(time=10, frequency=10, dimension="width") == 10
    assert size_heatmap.sel(time=10, frequency=10, dimension="height") == 10
    assert class_heatmap.sel(time=10, frequency=10, category="pippip") == 1.0
    assert class_heatmap.sel(time=10, frequency=10, category="myomyo") == 0.0
    assert detection_heatmap.sel(time=10, frequency=10) == 1.0
