from soundevent import data

from batdetect2.train import generate_train_example
from batdetect2.typing import (
    AudioLoader,
    ClipLabeller,
    ClipperProtocol,
    PreprocessorProtocol,
)


def test_default_clip_size_is_correct(
    sample_clipper: ClipperProtocol,
    sample_labeller: ClipLabeller,
    sample_audio_loader: AudioLoader,
    clip_annotation: data.ClipAnnotation,
    sample_preprocessor: PreprocessorProtocol,
):
    example = generate_train_example(
        clip_annotation=clip_annotation,
        audio_loader=sample_audio_loader,
        preprocessor=sample_preprocessor,
        labeller=sample_labeller,
    )

    clip, _, _ = sample_clipper(example)
    assert clip.spectrogram.shape == (1, 128, 256)
