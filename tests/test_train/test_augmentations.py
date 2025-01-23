from collections.abc import Callable

import numpy as np
import xarray as xr
from soundevent import data

from batdetect2.train.augmentations import (
    add_echo,
    adjust_dataset_width,
    mix_examples,
    select_random_subclip,
)
from batdetect2.train.preprocess import (
    TrainPreprocessingConfig,
    generate_train_example,
)


def test_mix_examples(
    recording_factory: Callable[..., data.Recording],
):
    recording1 = recording_factory()
    recording2 = recording_factory()

    clip1 = data.Clip(recording=recording1, start_time=0.2, end_time=0.7)

    clip2 = data.Clip(recording=recording2, start_time=0.3, end_time=0.8)

    clip_annotation_1 = data.ClipAnnotation(clip=clip1)

    clip_annotation_2 = data.ClipAnnotation(clip=clip2)

    config = TrainPreprocessingConfig()

    example1 = generate_train_example(clip_annotation_1, config)
    example2 = generate_train_example(clip_annotation_2, config)

    mixed = mix_examples(example1, example2, config=config.preprocessing)

    assert mixed["spectrogram"].shape == example1["spectrogram"].shape
    assert mixed["detection"].shape == example1["detection"].shape
    assert mixed["size"].shape == example1["size"].shape
    assert mixed["class"].shape == example1["class"].shape


def test_add_echo(
    recording_factory: Callable[..., data.Recording],
):
    recording1 = recording_factory()
    clip1 = data.Clip(recording=recording1, start_time=0.2, end_time=0.7)
    clip_annotation_1 = data.ClipAnnotation(clip=clip1)
    config = TrainPreprocessingConfig()
    original = generate_train_example(clip_annotation_1, config)
    with_echo = add_echo(original, config=config.preprocessing)

    assert with_echo["spectrogram"].shape == original["spectrogram"].shape
    xr.testing.assert_identical(with_echo["size"], original["size"])
    xr.testing.assert_identical(with_echo["class"], original["class"])
    xr.testing.assert_identical(with_echo["detection"], original["detection"])


def test_selected_random_subclip_has_the_correct_width(
    recording_factory: Callable[..., data.Recording],
):
    recording1 = recording_factory()
    clip1 = data.Clip(recording=recording1, start_time=0.2, end_time=0.7)
    clip_annotation_1 = data.ClipAnnotation(clip=clip1)
    config = TrainPreprocessingConfig()
    original = generate_train_example(clip_annotation_1, config)
    subclip = select_random_subclip(original, width=100)

    assert subclip["spectrogram"].shape[1] == 100


def test_adjust_dataset_width():
    height = 128
    width = 512
    samplerate = 48_000

    times = np.linspace(0, 1, width)

    audio_times = np.linspace(0, 1, samplerate)
    frequency = np.linspace(0, 24_000, height)

    width_subset = 356
    audio_width_subset = int(samplerate * width_subset / width)

    times_subset = times[:width_subset]
    audio_times_subset = audio_times[:audio_width_subset]
    dimensions = ["width", "height"]
    class_names = [f"species_{i}" for i in range(17)]

    spectrogram = np.random.random([height, width_subset])
    sizes = np.random.random([len(dimensions), height, width_subset])
    classes = np.random.random([len(class_names), height, width_subset])
    audio = np.random.random([int(samplerate * width_subset / width)])

    dataset = xr.Dataset(
        data_vars={
            "audio": (("audio_time",), audio),
            "spectrogram": (("frequency", "time"), spectrogram),
            "sizes": (("dimension", "frequency", "time"), sizes),
            "classes": (("class", "frequency", "time"), classes),
        },
        coords={
            "audio_time": audio_times_subset,
            "time": times_subset,
            "frequency": frequency,
            "dimension": dimensions,
            "class": class_names,
        },
    )

    adjusted = adjust_dataset_width(dataset, width=width)

    # Spectrogram was adjusted correctly
    assert np.isclose(adjusted["spectrogram"].time, times).all()
    assert (adjusted["spectrogram"].frequency == frequency).all()

    # Sizes was adjusted correctly
    assert np.isclose(adjusted["sizes"].time, times).all()
    assert (adjusted["sizes"].frequency == frequency).all()
    assert list(adjusted["sizes"].dimension.values) == dimensions

    # Sizes was adjusted correctly
    assert np.isclose(adjusted["classes"].time, times).all()
    assert (adjusted["sizes"].frequency == frequency).all()
    assert list(adjusted["classes"]["class"].values) == class_names

    # Audio time was adjusted corretly
    assert np.isclose(
        len(adjusted["audio"].audio_time), len(audio_times), atol=2
    )
    assert np.isclose(
        adjusted["audio"].audio_time[-1], audio_times[-1], atol=1e-3
    )
