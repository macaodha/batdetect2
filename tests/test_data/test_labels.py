from pathlib import Path

import numpy as np
import xarray as xr
from soundevent import data
from soundevent.types import ClassMapper

from batdetect2.data.labels import generate_heatmaps

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


class Mapper(ClassMapper):
    class_labels = ["bat", "cat"]

    def encode(self, sound_event_annotation: data.SoundEventAnnotation) -> str:
        return "bat"

    def decode(self, label: str) -> list:
        return [data.Tag(term=data.term_from_key("species"), value="bat")]


def test_generated_heatmaps_have_correct_dimensions():
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

    class_mapper = Mapper()

    detection_heatmap, class_heatmap, size_heatmap = generate_heatmaps(
        clip_annotation,
        spec,
        class_mapper,
    )

    assert isinstance(detection_heatmap, xr.DataArray)
    assert detection_heatmap.shape == (100, 100)
    assert detection_heatmap.dims == ("time", "frequency")

    assert isinstance(class_heatmap, xr.DataArray)
    assert class_heatmap.shape == (2, 100, 100)
    assert class_heatmap.dims == ("category", "time", "frequency")
    assert class_heatmap.coords["category"].values.tolist() == ["bat", "cat"]

    assert isinstance(size_heatmap, xr.DataArray)
    assert size_heatmap.shape == (2, 100, 100)
    assert size_heatmap.dims == ("dimension", "time", "frequency")
    assert size_heatmap.coords["dimension"].values.tolist() == [
        "width",
        "height",
    ]


def test_generated_heatmap_are_non_zero_at_correct_positions():
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

    class_mapper = Mapper()
    detection_heatmap, class_heatmap, size_heatmap = generate_heatmaps(
        clip_annotation,
        spec,
        class_mapper,
    )
    assert size_heatmap.sel(time=10, frequency=10, dimension="width") == 10
    assert size_heatmap.sel(time=10, frequency=10, dimension="height") == 10
    assert class_heatmap.sel(time=10, frequency=10, category="bat") == 1.0
    assert class_heatmap.sel(time=10, frequency=10, category="cat") == 0.0
    assert detection_heatmap.sel(time=10, frequency=10) == 1.0
