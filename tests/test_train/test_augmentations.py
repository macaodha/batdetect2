from collections.abc import Callable

import pytest
import torch
from soundevent import data

from batdetect2.train.augmentations import (
    add_echo,
    mix_audio,
)
from batdetect2.train.clips import select_subclip
from batdetect2.train.preprocess import generate_train_example
from batdetect2.typing import AudioLoader, ClipLabeller, PreprocessorProtocol


def test_mix_examples(
    sample_preprocessor: PreprocessorProtocol,
    sample_audio_loader: AudioLoader,
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
        audio_loader=sample_audio_loader,
        preprocessor=sample_preprocessor,
        labeller=sample_labeller,
    )
    example2 = generate_train_example(
        clip_annotation_2,
        audio_loader=sample_audio_loader,
        preprocessor=sample_preprocessor,
        labeller=sample_labeller,
    )

    mixed = mix_audio(
        example1,
        example2,
        weight=0.3,
        preprocessor=sample_preprocessor,
    )

    assert mixed.spectrogram.shape == example1.spectrogram.shape
    assert mixed.detection_heatmap.shape == example1.detection_heatmap.shape
    assert mixed.size_heatmap.shape == example1.size_heatmap.shape
    assert mixed.class_heatmap.shape == example1.class_heatmap.shape


@pytest.mark.parametrize("duration1", [0.1, 0.4, 0.7])
@pytest.mark.parametrize("duration2", [0.1, 0.4, 0.7])
def test_mix_examples_of_different_durations(
    sample_preprocessor: PreprocessorProtocol,
    sample_audio_loader: AudioLoader,
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
        audio_loader=sample_audio_loader,
        preprocessor=sample_preprocessor,
        labeller=sample_labeller,
    )
    example2 = generate_train_example(
        clip_annotation_2,
        audio_loader=sample_audio_loader,
        preprocessor=sample_preprocessor,
        labeller=sample_labeller,
    )

    mixed = mix_audio(
        example1,
        example2,
        weight=0.3,
        preprocessor=sample_preprocessor,
    )

    assert mixed.spectrogram.shape == example1.spectrogram.shape
    assert mixed.detection_heatmap.shape == example1.detection_heatmap.shape
    assert mixed.size_heatmap.shape == example1.size_heatmap.shape
    assert mixed.class_heatmap.shape == example1.class_heatmap.shape


def test_add_echo(
    sample_preprocessor: PreprocessorProtocol,
    sample_audio_loader: AudioLoader,
    sample_labeller: ClipLabeller,
    create_recording: Callable[..., data.Recording],
):
    recording1 = create_recording()
    clip1 = data.Clip(recording=recording1, start_time=0.2, end_time=0.7)
    clip_annotation_1 = data.ClipAnnotation(clip=clip1)

    original = generate_train_example(
        clip_annotation_1,
        audio_loader=sample_audio_loader,
        preprocessor=sample_preprocessor,
        labeller=sample_labeller,
    )
    with_echo = add_echo(
        original,
        preprocessor=sample_preprocessor,
        delay=0.1,
        weight=0.3,
    )

    assert with_echo.spectrogram.shape == original.spectrogram.shape
    torch.testing.assert_close(
        with_echo.size_heatmap,
        original.size_heatmap,
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(
        with_echo.class_heatmap,
        original.class_heatmap,
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(
        with_echo.detection_heatmap,
        original.detection_heatmap,
        atol=0,
        rtol=0,
    )


def test_selected_random_subclip_has_the_correct_width(
    sample_preprocessor: PreprocessorProtocol,
    sample_audio_loader: AudioLoader,
    sample_labeller: ClipLabeller,
    create_recording: Callable[..., data.Recording],
):
    recording1 = create_recording()
    clip1 = data.Clip(recording=recording1, start_time=0.2, end_time=0.7)
    clip_annotation_1 = data.ClipAnnotation(clip=clip1)

    original = generate_train_example(
        clip_annotation_1,
        audio_loader=sample_audio_loader,
        preprocessor=sample_preprocessor,
        labeller=sample_labeller,
    )

    subclip = select_subclip(
        original,
        input_samplerate=256_000,
        output_samplerate=1000,
        start=0,
        duration=0.512,
    )
    assert subclip.spectrogram.shape[1] == 512
