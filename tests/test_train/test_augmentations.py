from collections.abc import Callable

import numpy as np
import pytest
import xarray as xr
from soundevent import arrays, data

from batdetect2.preprocess.types import PreprocessorProtocol
from batdetect2.train.augmentations import (
    add_echo,
    mix_examples,
)
from batdetect2.train.clips import select_subclip
from batdetect2.train.preprocess import generate_train_example
from batdetect2.train.types import ClipLabeller


def test_mix_examples(
    sample_preprocessor: PreprocessorProtocol,
    sample_labeller: ClipLabeller,
    create_recording: Callable[..., data.Recording],
):
    recording1 = create_recording()
    recording2 = create_recording()

    clip1 = data.Clip(recording=recording1, start_time=0.2, end_time=0.7)
    clip2 = data.Clip(recording=recording2, start_time=0.3, end_time=0.8)

    clip_annotation_1 = data.ClipAnnotation(clip=clip1)
    clip_annotation_2 = data.ClipAnnotation(clip=clip2)

    example1 = generate_train_example(
        clip_annotation_1,
        preprocessor=sample_preprocessor,
        labeller=sample_labeller,
    )
    example2 = generate_train_example(
        clip_annotation_2,
        preprocessor=sample_preprocessor,
        labeller=sample_labeller,
    )

    mixed = mix_examples(example1, example2, preprocessor=sample_preprocessor)

    assert mixed["spectrogram"].shape == example1["spectrogram"].shape
    assert mixed["detection"].shape == example1["detection"].shape
    assert mixed["size"].shape == example1["size"].shape
    assert mixed["class"].shape == example1["class"].shape


@pytest.mark.parametrize("duration1", [0.1, 0.4, 0.7])
@pytest.mark.parametrize("duration2", [0.1, 0.4, 0.7])
def test_mix_examples_of_different_durations(
    sample_preprocessor: PreprocessorProtocol,
    sample_labeller: ClipLabeller,
    create_recording: Callable[..., data.Recording],
    duration1: float,
    duration2: float,
):
    recording1 = create_recording()
    recording2 = create_recording()

    clip1 = data.Clip(recording=recording1, start_time=0, end_time=duration1)
    clip2 = data.Clip(recording=recording2, start_time=0, end_time=duration2)

    clip_annotation_1 = data.ClipAnnotation(clip=clip1)
    clip_annotation_2 = data.ClipAnnotation(clip=clip2)

    example1 = generate_train_example(
        clip_annotation_1,
        preprocessor=sample_preprocessor,
        labeller=sample_labeller,
    )
    example2 = generate_train_example(
        clip_annotation_2,
        preprocessor=sample_preprocessor,
        labeller=sample_labeller,
    )

    mixed = mix_examples(example1, example2, preprocessor=sample_preprocessor)

    # Check the spectrogram has the expected duration
    step = arrays.get_dim_step(mixed["spectrogram"], "time")
    start, stop = arrays.get_dim_range(mixed["spectrogram"], "time")
    assert start == 0
    assert np.isclose(stop + step, duration1, atol=2 * step)


def test_add_echo(
    sample_preprocessor: PreprocessorProtocol,
    sample_labeller: ClipLabeller,
    create_recording: Callable[..., data.Recording],
):
    recording1 = create_recording()
    clip1 = data.Clip(recording=recording1, start_time=0.2, end_time=0.7)
    clip_annotation_1 = data.ClipAnnotation(clip=clip1)

    original = generate_train_example(
        clip_annotation_1,
        preprocessor=sample_preprocessor,
        labeller=sample_labeller,
    )
    with_echo = add_echo(original, preprocessor=sample_preprocessor)

    assert with_echo["spectrogram"].shape == original["spectrogram"].shape
    xr.testing.assert_identical(with_echo["size"], original["size"])
    xr.testing.assert_identical(with_echo["class"], original["class"])
    xr.testing.assert_identical(with_echo["detection"], original["detection"])


def test_selected_random_subclip_has_the_correct_width(
    sample_preprocessor: PreprocessorProtocol,
    sample_labeller: ClipLabeller,
    create_recording: Callable[..., data.Recording],
):
    recording1 = create_recording()
    clip1 = data.Clip(recording=recording1, start_time=0.2, end_time=0.7)
    clip_annotation_1 = data.ClipAnnotation(clip=clip1)
    original = generate_train_example(
        clip_annotation_1,
        preprocessor=sample_preprocessor,
        labeller=sample_labeller,
    )
    subclip = select_subclip(original, start=0, span=100)

    assert subclip["spectrogram"].shape[1] == 100


def test_add_echo_after_subclip(
    sample_preprocessor: PreprocessorProtocol,
    sample_labeller: ClipLabeller,
    create_recording: Callable[..., data.Recording],
):
    recording1 = create_recording(duration=2)
    clip1 = data.Clip(recording=recording1, start_time=0, end_time=1)
    clip_annotation_1 = data.ClipAnnotation(clip=clip1)
    original = generate_train_example(
        clip_annotation_1,
        preprocessor=sample_preprocessor,
        labeller=sample_labeller,
    )

    assert original.sizes["time"] > 512

    subclip = select_subclip(original, start=0, span=512)
    with_echo = add_echo(subclip, preprocessor=sample_preprocessor)

    assert with_echo.sizes["time"] == 512
