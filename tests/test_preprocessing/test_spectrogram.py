import math
from pathlib import Path
from typing import Callable

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from soundevent import arrays

from batdetect2.preprocess.audio import AudioConfig, load_file_audio
from batdetect2.preprocess.spectrogram import (
    STFTConfig,
    FrequencyConfig,
    SpecSizeConfig,
    SpectrogramConfig,
    compute_spectrogram,
    duration_to_spec_width,
    get_spectrogram_resolution,
    spec_width_to_samples,
    stft,
)


@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    duration=st.floats(min_value=0.1, max_value=1.0),
    window_duration=st.floats(min_value=0.001, max_value=0.01),
    window_overlap=st.floats(min_value=0.2, max_value=0.9),
    samplerate=st.integers(min_value=256_000, max_value=512_000),
)
def test_can_estimate_correctly_spectrogram_width_from_duration(
    duration: float,
    window_duration: float,
    window_overlap: float,
    samplerate: int,
    wav_factory: Callable[..., Path],
):
    path = wav_factory(duration=duration, samplerate=samplerate)
    audio = load_file_audio(
        path,
        # NOTE: Dont resample nor adjust duration to test if the width
        # estimation works on all scenarios
        config=AudioConfig(resample=None, duration=None),
    )
    spectrogram = stft(audio, window_duration, window_overlap)

    spec_width = duration_to_spec_width(
        duration,
        samplerate=samplerate,
        window_duration=window_duration,
        window_overlap=window_overlap,
    )
    assert spectrogram.sizes["time"] == spec_width

    rebuilt_duration = (
        spec_width_to_samples(
            spec_width,
            samplerate=samplerate,
            window_duration=window_duration,
            window_overlap=window_overlap,
        )
        / samplerate
    )

    assert (
        abs(duration - rebuilt_duration)
        < (1 - window_overlap) * window_duration
    )


def test_can_estimate_spectrogram_resolution(
    wav_factory: Callable[..., Path],
):
    path = wav_factory(duration=0.2, samplerate=256_000)

    audio = load_file_audio(
        path,
        # NOTE: Dont resample nor adjust duration to test if the width
        # estimation works on all scenarios
        config=AudioConfig(resample=None, duration=None),
    )

    config = SpectrogramConfig(
        stft=STFTConfig(),
        size=SpecSizeConfig(height=256, resize_factor=0.5),
        frequencies=FrequencyConfig(min_freq=10_000, max_freq=120_000),
    )

    spec = compute_spectrogram(audio, config=config)

    freq_res, time_res = get_spectrogram_resolution(config)

    assert math.isclose(
        arrays.get_dim_step(spec, dim="frequency"),
        freq_res,
        rel_tol=0.1,
    )
    assert math.isclose(
        arrays.get_dim_step(spec, dim="time"),
        time_res,
        rel_tol=0.1,
    )
