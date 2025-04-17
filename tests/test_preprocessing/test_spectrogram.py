import math
from pathlib import Path
from typing import Callable

import numpy as np
import pytest
import xarray as xr
from soundevent import arrays

from batdetect2.preprocess.audio import AudioConfig, load_file_audio
from batdetect2.preprocess.spectrogram import (
    MAX_FREQ,
    MIN_FREQ,
    ConfigurableSpectrogramBuilder,
    FrequencyConfig,
    PcenConfig,
    SpecSizeConfig,
    SpectrogramConfig,
    STFTConfig,
    apply_pcen,
    build_spectrogram_builder,
    compute_spectrogram,
    crop_spectrogram_frequencies,
    get_spectrogram_resolution,
    remove_spectral_mean,
    resize_spectrogram,
    scale_log,
    scale_spectrogram,
    stft,
)

SAMPLERATE = 250_000
DURATION = 0.1
TEST_FREQ = 30_000
N_SAMPLES = int(SAMPLERATE * DURATION)
TIME_COORD = np.linspace(
    0, DURATION, N_SAMPLES, endpoint=False, dtype=np.float32
)


@pytest.fixture
def sine_wave_xr() -> xr.DataArray:
    """Generate a single sine wave as an xr.DataArray."""
    t = TIME_COORD
    wav_data = np.sin(2 * np.pi * TEST_FREQ * t, dtype=np.float32)
    return xr.DataArray(
        wav_data,
        coords={"time": t},
        dims=["time"],
        attrs={"samplerate": SAMPLERATE},
    )


@pytest.fixture
def constant_wave_xr() -> xr.DataArray:
    """Generate a constant signal as an xr.DataArray."""
    t = TIME_COORD
    wav_data = np.ones(N_SAMPLES, dtype=np.float32) * 0.5
    return xr.DataArray(
        wav_data,
        coords={"time": t},
        dims=["time"],
        attrs={"samplerate": SAMPLERATE},
    )


@pytest.fixture
def sample_spec(sine_wave_xr: xr.DataArray) -> xr.DataArray:
    """Generate a basic spectrogram for testing downstream functions."""
    config = SpectrogramConfig(
        stft=STFTConfig(window_duration=0.002, window_overlap=0.5),
        frequencies=FrequencyConfig(
            min_freq=0,
            max_freq=int(SAMPLERATE / 2),
        ),
        size=None,
        pcen=None,
        spectral_mean_substraction=False,
        peak_normalize=False,
        scale="amplitude",
    )
    spec = stft(
        sine_wave_xr,
        window_duration=config.stft.window_duration,
        window_overlap=config.stft.window_overlap,
        window_fn=config.stft.window_fn,
    )
    return spec


def test_stft_config_defaults():
    config = STFTConfig()
    assert config.window_duration == 0.002
    assert config.window_overlap == 0.75
    assert config.window_fn == "hann"


def test_frequency_config_defaults():
    config = FrequencyConfig()
    assert config.min_freq == MIN_FREQ
    assert config.max_freq == MAX_FREQ


def test_spec_size_config_defaults():
    config = SpecSizeConfig()
    assert config.height == 128
    assert config.resize_factor == 0.5


def test_pcen_config_defaults():
    config = PcenConfig()
    assert config.time_constant == 0.4
    assert config.gain == 0.98
    assert config.bias == 2
    assert config.power == 0.5


def test_spectrogram_config_defaults():
    config = SpectrogramConfig()
    assert isinstance(config.stft, STFTConfig)
    assert isinstance(config.frequencies, FrequencyConfig)
    assert isinstance(config.pcen, PcenConfig)
    assert config.scale == "amplitude"
    assert isinstance(config.size, SpecSizeConfig)
    assert config.spectral_mean_substraction is True
    assert config.peak_normalize is False


def test_stft_output_properties(sine_wave_xr: xr.DataArray):
    window_duration = 0.002
    window_overlap = 0.5
    samplerate = sine_wave_xr.attrs["samplerate"]
    nfft = int(window_duration * samplerate)
    hop_len = nfft - int(window_overlap * nfft)

    spec = stft(
        sine_wave_xr,
        window_duration=window_duration,
        window_overlap=window_overlap,
        window_fn="hann",
    )

    assert isinstance(spec, xr.DataArray)
    assert spec.dims == ("frequency", "time")
    assert spec.dtype == np.float32
    assert "frequency" in spec.coords
    assert "time" in spec.coords

    time_step = arrays.get_dim_step(spec, "time")
    freq_step = arrays.get_dim_step(spec, "frequency")
    freq_start, freq_end = arrays.get_dim_range(spec, "frequency")
    assert np.isclose(freq_step, samplerate / nfft)
    assert np.isclose(time_step, hop_len / samplerate)
    assert spec.frequency.min() >= 0
    assert freq_start == 0
    assert np.isclose(freq_end + freq_step, samplerate / 2, atol=5)
    assert spec.time.min() >= 0
    assert spec.time.max() < DURATION

    assert spec.attrs["original_samplerate"] == samplerate
    assert spec.attrs["nfft"] == nfft
    assert spec.attrs["noverlap"] == int(window_overlap * nfft)

    assert np.all(spec.data >= 0)


@pytest.mark.parametrize("window_fn", ["hann", "hamming"])
def test_stft_window_fn(sine_wave_xr: xr.DataArray, window_fn: str):
    spec = stft(
        sine_wave_xr,
        window_duration=0.002,
        window_overlap=0.5,
        window_fn=window_fn,
    )
    assert isinstance(spec, xr.DataArray)
    assert np.all(spec.data >= 0)


def test_crop_spectrogram_frequencies(sample_spec: xr.DataArray):
    min_f, max_f = 20_000, 80_000
    cropped_spec = crop_spectrogram_frequencies(
        sample_spec, min_freq=min_f, max_freq=max_f
    )

    assert cropped_spec.dims == sample_spec.dims
    assert cropped_spec.dtype == sample_spec.dtype
    assert cropped_spec.sizes["time"] == sample_spec.sizes["time"]
    assert cropped_spec.sizes["frequency"] < sample_spec.sizes["frequency"]
    assert cropped_spec.coords["frequency"].min() >= min_f

    assert np.isclose(cropped_spec.coords["frequency"].max(), max_f, rtol=0.1)


def test_crop_spectrogram_full_range(sample_spec: xr.DataArray):
    samplerate = sample_spec.attrs["original_samplerate"]
    min_f, max_f = 0, samplerate / 2
    cropped_spec = crop_spectrogram_frequencies(
        sample_spec, min_freq=min_f, max_freq=max_f
    )

    assert cropped_spec.sizes == sample_spec.sizes
    assert np.allclose(cropped_spec.data, sample_spec.data)


def test_apply_pcen(sample_spec: xr.DataArray):
    if "original_samplerate" not in sample_spec.attrs:
        sample_spec.attrs["original_samplerate"] = SAMPLERATE
    if "nfft" not in sample_spec.attrs:
        sample_spec.attrs["nfft"] = int(0.002 * SAMPLERATE)
    if "noverlap" not in sample_spec.attrs:
        sample_spec.attrs["noverlap"] = int(0.5 * sample_spec.attrs["nfft"])

    pcen_config = PcenConfig()
    pcen_spec = apply_pcen(
        sample_spec,
        time_constant=pcen_config.time_constant,
        gain=pcen_config.gain,
        bias=pcen_config.bias,
        power=pcen_config.power,
    )

    assert pcen_spec.dims == sample_spec.dims
    assert pcen_spec.sizes == sample_spec.sizes
    assert pcen_spec.dtype == sample_spec.dtype
    assert np.all(pcen_spec.data >= 0)

    assert not np.allclose(pcen_spec.data, sample_spec.data)


def test_scale_log(sample_spec: xr.DataArray):
    if "original_samplerate" not in sample_spec.attrs:
        sample_spec.attrs["original_samplerate"] = SAMPLERATE
    if "nfft" not in sample_spec.attrs:
        sample_spec.attrs["nfft"] = int(0.002 * SAMPLERATE)

    log_spec = scale_log(sample_spec, dtype=np.float32)

    assert log_spec.dims == sample_spec.dims
    assert log_spec.sizes == sample_spec.sizes
    assert log_spec.dtype == np.float32
    assert np.all(log_spec.data >= 0)
    assert not np.allclose(log_spec.data, sample_spec.data)


def test_scale_log_missing_attrs(sample_spec: xr.DataArray):
    spec_copy = sample_spec.copy()
    del spec_copy.attrs["original_samplerate"]
    with pytest.raises(KeyError):
        scale_log(spec_copy)

    spec_copy = sample_spec.copy()
    del spec_copy.attrs["nfft"]
    with pytest.raises(KeyError):
        scale_log(spec_copy)


def test_scale_spectrogram_amplitude(sample_spec: xr.DataArray):
    scaled_spec = scale_spectrogram(sample_spec, scale="amplitude")
    assert np.allclose(scaled_spec.data, sample_spec.data)
    assert scaled_spec.dtype == sample_spec.dtype


def test_scale_spectrogram_power(sample_spec: xr.DataArray):
    scaled_spec = scale_spectrogram(sample_spec, scale="power")
    assert np.allclose(scaled_spec.data, sample_spec.data**2)
    assert scaled_spec.dtype == sample_spec.dtype


def test_scale_spectrogram_db(sample_spec: xr.DataArray):
    if "original_samplerate" not in sample_spec.attrs:
        sample_spec.attrs["original_samplerate"] = SAMPLERATE
    if "nfft" not in sample_spec.attrs:
        sample_spec.attrs["nfft"] = int(0.002 * SAMPLERATE)

    scaled_spec = scale_spectrogram(sample_spec, scale="dB", dtype=np.float64)
    log_spec_expected = scale_log(sample_spec, dtype=np.float64)
    assert scaled_spec.dtype == np.float64
    assert np.allclose(scaled_spec.data, log_spec_expected.data)


def test_remove_spectral_mean(sample_spec: xr.DataArray):
    spec_noisy = sample_spec.copy() + 0.1
    denoised_spec = remove_spectral_mean(spec_noisy)

    assert denoised_spec.dims == spec_noisy.dims
    assert denoised_spec.sizes == spec_noisy.sizes
    assert denoised_spec.dtype == spec_noisy.dtype
    assert np.all(denoised_spec.data >= 0)


def test_remove_spectral_mean_constant(constant_wave_xr: xr.DataArray):
    const_spec = stft(constant_wave_xr, 0.002, 0.5)
    denoised_spec = remove_spectral_mean(const_spec)

    assert np.allclose(denoised_spec.data, 0, atol=1e-6)


@pytest.mark.parametrize(
    "height, resize_factor, expected_freq_size, expected_time_factor",
    [
        (128, 1.0, 128, 1.0),
        (64, 0.5, 64, 0.5),
        (256, None, 256, 1.0),
        (100, 2.0, 100, 2.0),
    ],
)
def test_resize_spectrogram(
    sample_spec: xr.DataArray,
    height: int,
    resize_factor: float | None,
    expected_freq_size: int,
    expected_time_factor: float,
):
    original_time_size = sample_spec.sizes["time"]
    resized_spec = resize_spectrogram(
        sample_spec,
        height=height,
        resize_factor=resize_factor,
    )

    assert resized_spec.dims == ("frequency", "time")
    assert resized_spec.sizes["frequency"] == expected_freq_size
    expected_time_size = int(original_time_size * expected_time_factor)

    assert abs(resized_spec.sizes["time"] - expected_time_size) <= 1

    assert resized_spec.dtype == np.float32


def test_compute_spectrogram_defaults(sine_wave_xr: xr.DataArray):
    config = SpectrogramConfig()
    spec = compute_spectrogram(sine_wave_xr, config=config)

    assert isinstance(spec, xr.DataArray)
    assert spec.dims == ("frequency", "time")
    assert spec.dtype == np.float32
    assert config.size is not None
    assert spec.sizes["frequency"] == config.size.height

    temp_stft = stft(
        sine_wave_xr, config.stft.window_duration, config.stft.window_overlap
    )
    assert config.size.resize_factor is not None
    expected_time_size = int(
        temp_stft.sizes["time"] * config.size.resize_factor
    )
    assert abs(spec.sizes["time"] - expected_time_size) <= 1

    assert spec.coords["frequency"].min() >= config.frequencies.min_freq
    assert np.isclose(
        spec.coords["frequency"].max(),
        config.frequencies.max_freq,
        rtol=0.1,
    )


def test_compute_spectrogram_no_pcen_no_mean_sub_no_resize(
    sine_wave_xr: xr.DataArray,
):
    config = SpectrogramConfig(
        pcen=None,
        spectral_mean_substraction=False,
        size=None,
        scale="power",
        frequencies=FrequencyConfig(min_freq=0, max_freq=int(SAMPLERATE / 2)),
    )
    spec = compute_spectrogram(sine_wave_xr, config=config)

    stft_direct = stft(
        sine_wave_xr, config.stft.window_duration, config.stft.window_overlap
    )
    expected_spec = scale_spectrogram(stft_direct, scale="power")

    assert spec.sizes == expected_spec.sizes
    assert np.allclose(spec.data, expected_spec.data)
    assert spec.dtype == expected_spec.dtype


def test_compute_spectrogram_peak_normalize(sine_wave_xr: xr.DataArray):
    config = SpectrogramConfig(peak_normalize=True)
    spec = compute_spectrogram(sine_wave_xr, config=config)
    assert np.isclose(spec.data.max(), 1.0, atol=1e-6)

    config = SpectrogramConfig(peak_normalize=False)
    spec_no_norm = compute_spectrogram(sine_wave_xr, config=config)
    assert not np.isclose(spec_no_norm.data.max(), 1.0, atol=1e-6)


def test_get_spectrogram_resolution_calculation():
    config = SpectrogramConfig(
        stft=STFTConfig(window_duration=0.002, window_overlap=0.75),
        size=SpecSizeConfig(height=100, resize_factor=0.5),
        frequencies=FrequencyConfig(min_freq=10_000, max_freq=110_000),
    )

    freq_res, time_res = get_spectrogram_resolution(config)

    expected_freq_res = (110_000 - 10_000) / 100
    expected_hop_duration = 0.002 * (1 - 0.75)
    expected_time_res = expected_hop_duration / 0.5

    assert np.isclose(freq_res, expected_freq_res)
    assert np.isclose(time_res, expected_time_res)


def test_get_spectrogram_resolution_no_resize_factor():
    config = SpectrogramConfig(
        stft=STFTConfig(window_duration=0.004, window_overlap=0.5),
        size=SpecSizeConfig(height=200, resize_factor=None),
        frequencies=FrequencyConfig(min_freq=20_000, max_freq=120_000),
    )
    freq_res, time_res = get_spectrogram_resolution(config)
    expected_freq_res = (120_000 - 20_000) / 200
    expected_hop_duration = 0.004 * (1 - 0.5)
    expected_time_res = expected_hop_duration / 1.0

    assert np.isclose(freq_res, expected_freq_res)
    assert np.isclose(time_res, expected_time_res)


def test_get_spectrogram_resolution_no_size_config():
    config = SpectrogramConfig(size=None)
    with pytest.raises(
        ValueError, match="Spectrogram size configuration is required"
    ):
        get_spectrogram_resolution(config)


def test_configurable_spectrogram_builder_init():
    config = SpectrogramConfig()
    builder = ConfigurableSpectrogramBuilder(config=config, dtype=np.float16)
    assert builder.config is config
    assert builder.dtype == np.float16


def test_configurable_spectrogram_builder_call_xr(sine_wave_xr: xr.DataArray):
    config = SpectrogramConfig()
    builder = ConfigurableSpectrogramBuilder(config=config)
    spec_builder = builder(sine_wave_xr)
    spec_direct = compute_spectrogram(sine_wave_xr, config=config)
    assert isinstance(spec_builder, xr.DataArray)
    assert np.allclose(spec_builder.data, spec_direct.data)
    assert spec_builder.dtype == spec_direct.dtype


def test_configurable_spectrogram_builder_call_np(sine_wave_xr: xr.DataArray):
    config = SpectrogramConfig()
    builder = ConfigurableSpectrogramBuilder(config=config)
    wav_np = sine_wave_xr.data
    samplerate = sine_wave_xr.attrs["samplerate"]

    spec_builder = builder(wav_np.astype(np.float32), samplerate=samplerate)
    spec_direct = compute_spectrogram(sine_wave_xr, config=config)

    assert isinstance(spec_builder, xr.DataArray)
    assert np.allclose(spec_builder.data, spec_direct.data, atol=1e-4)
    assert spec_builder.dtype == spec_direct.dtype


def test_configurable_spectrogram_builder_call_np_no_samplerate(
    sine_wave_xr: xr.DataArray,
):
    config = SpectrogramConfig()
    builder = ConfigurableSpectrogramBuilder(config=config)
    wav_np = sine_wave_xr.data
    with pytest.raises(ValueError, match="Samplerate must be provided"):
        builder(wav_np, samplerate=None)


def test_build_spectrogram_builder():
    config = SpectrogramConfig(peak_normalize=True)
    builder = build_spectrogram_builder(config=config, dtype=np.float64)
    assert isinstance(builder, ConfigurableSpectrogramBuilder)
    assert builder.config is config
    assert builder.dtype == np.float64


def test_can_estimate_spectrogram_resolution(
    wav_factory: Callable[..., Path],
):
    path = wav_factory(duration=0.2, samplerate=256_000)

    audio_data = load_file_audio(
        path,
        config=AudioConfig(resample=None, duration=None),
    )

    config = SpectrogramConfig(
        stft=STFTConfig(),
        size=SpecSizeConfig(height=256, resize_factor=0.5),
        frequencies=FrequencyConfig(min_freq=10_000, max_freq=120_000),
    )

    spec = compute_spectrogram(audio_data, config=config)

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
