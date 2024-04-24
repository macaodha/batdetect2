from pathlib import Path

import numpy as np
import pytest
from soundevent import data

from batdetect2.data import preprocessing
from batdetect2.utils import audio_utils

ROOT_DIR = Path(__file__).parent.parent.parent
EXAMPLE_AUDIO = ROOT_DIR / "example_data" / "audio"
TEST_AUDIO = ROOT_DIR / "tests" / "data"


TEST_FILES = [
    EXAMPLE_AUDIO / "20170701_213954-MYOMYS-LR_0_0.5.wav",
    EXAMPLE_AUDIO / "20180530_213516-EPTSER-LR_0_0.5.wav",
    EXAMPLE_AUDIO / "20180627_215323-RHIFER-LR_0_0.5.wav",
    TEST_AUDIO / "20230322_172000_selec2.wav",
]


@pytest.mark.parametrize("audio_file", TEST_FILES)
@pytest.mark.parametrize("scale", [True, False])
def test_audio_loading_hasnt_changed(
    audio_file,
    scale,
):
    time_expansion = 1
    target_sampling_rate = 256_000
    recording = data.Recording.from_file(
        audio_file,
        time_expansion=time_expansion,
    )
    clip = data.Clip(
        recording=recording,
        start_time=0,
        end_time=recording.duration,
    )

    _, audio_original = audio_utils.load_audio(
        audio_file,
        time_expansion,
        target_samp_rate=target_sampling_rate,
        scale=scale,
    )
    audio_new = preprocessing.load_clip_audio(
        clip,
        target_sampling_rate=target_sampling_rate,
        scale=scale,
        dtype=np.float32,
    )

    assert audio_original.shape == audio_new.shape
    assert audio_original.dtype == audio_new.dtype
    assert np.isclose(audio_original, audio_new.data).all()


@pytest.mark.parametrize("audio_file", TEST_FILES)
@pytest.mark.parametrize("spec_scale", ["log", "pcen", "amplitude"])
@pytest.mark.parametrize("denoise_spec_avg", [True, False])
@pytest.mark.parametrize("max_scale_spec", [True, False])
@pytest.mark.parametrize("fft_win_length", [512 / 256_000, 1024 / 256_000])
def test_spectrogram_generation_hasnt_changed(
    audio_file,
    spec_scale,
    denoise_spec_avg,
    max_scale_spec,
    fft_win_length,
):
    time_expansion = 1
    target_sampling_rate = 256_000
    min_freq = 10_000
    max_freq = 120_000
    fft_overlap = 0.75
    recording = data.Recording.from_file(
        audio_file,
        time_expansion=time_expansion,
    )
    clip = data.Clip(
        recording=recording,
        start_time=0,
        end_time=recording.duration,
    )
    audio = preprocessing.load_clip_audio(
        clip,
        target_sampling_rate=target_sampling_rate,
    )

    spec_original, _ = audio_utils.generate_spectrogram(
        audio.data,
        sampling_rate=target_sampling_rate,
        params=dict(
            fft_win_length=fft_win_length,
            fft_overlap=fft_overlap,
            max_freq=max_freq,
            min_freq=min_freq,
            spec_scale=spec_scale,
            denoise_spec_avg=denoise_spec_avg,
            max_scale_spec=max_scale_spec,
        ),
    )

    new_spec = preprocessing.compute_spectrogram(
        audio,
        fft_win_length=fft_win_length,
        fft_overlap=fft_overlap,
        max_freq=max_freq,
        min_freq=min_freq,
        spec_scale=spec_scale,
        denoise_spec_avg=denoise_spec_avg,
        max_scale_spec=max_scale_spec,
        dtype=np.float32,
    )

    assert spec_original.shape == new_spec.shape
    assert spec_original.dtype == new_spec.dtype

    # NOTE: The original spectrogram is flipped vertically
    assert np.isclose(spec_original, np.flipud(new_spec.data)).all()
