import pathlib
import uuid
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import xarray as xr
from soundevent import data
from soundevent.arrays import Dimensions, create_time_dim_from_array

from batdetect2.preprocess import audio


def create_dummy_wave(
    samplerate: int,
    duration: float,
    num_channels: int = 1,
    freq: float = 440.0,
    amplitude: float = 0.5,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    """Generates a simple numpy waveform."""
    t = np.linspace(
        0.0, duration, int(samplerate * duration), endpoint=False, dtype=dtype
    )
    wave = amplitude * np.sin(2 * np.pi * freq * t)
    if num_channels > 1:
        wave = np.stack([wave] * num_channels, axis=0)
    return wave.astype(dtype)


def create_xr_wave(
    samplerate: int,
    duration: float,
    num_channels: int = 1,
    freq: float = 440.0,
    amplitude: float = 0.5,
    start_time: float = 0.0,
) -> xr.DataArray:
    """Generates a simple xarray waveform."""
    num_samples = int(samplerate * duration)
    times = np.linspace(
        start_time,
        start_time + duration,
        num_samples,
        endpoint=False,
    )
    coords = {
        Dimensions.time.value: create_time_dim_from_array(
            times, samplerate=samplerate, start_time=start_time
        )
    }
    dims = [Dimensions.time.value]

    wave_data = amplitude * np.sin(2 * np.pi * freq * times)

    if num_channels > 1:
        coords[Dimensions.channel.value] = np.arange(num_channels)
        dims = [Dimensions.channel.value] + dims
        wave_data = np.stack([wave_data] * num_channels, axis=0)

    return xr.DataArray(
        wave_data.astype(np.float32),
        coords=coords,
        dims=dims,
        attrs={"samplerate": samplerate},
    )


@pytest.fixture
def dummy_wav_path(tmp_path: pathlib.Path) -> pathlib.Path:
    """Creates a dummy WAV file and returns its path."""
    samplerate = 48000
    duration = 2.0
    num_channels = 2
    wave_data = create_dummy_wave(samplerate, duration, num_channels)
    file_path = tmp_path / f"{uuid.uuid4()}.wav"
    sf.write(file_path, wave_data.T, samplerate, format="WAV", subtype="FLOAT")
    return file_path


@pytest.fixture
def dummy_recording(dummy_wav_path: pathlib.Path) -> data.Recording:
    """Creates a Recording object pointing to the dummy WAV file."""
    return data.Recording.from_file(dummy_wav_path)


@pytest.fixture
def dummy_clip(dummy_recording: data.Recording) -> data.Clip:
    """Creates a Clip object from the dummy recording."""
    return data.Clip(
        recording=dummy_recording,
        start_time=0.5,
        end_time=1.5,
    )


@pytest.fixture
def default_audio_config() -> audio.AudioConfig:
    return audio.AudioConfig()


@pytest.fixture
def no_resample_config() -> audio.AudioConfig:
    return audio.AudioConfig(resample=None)


@pytest.fixture
def fixed_duration_config() -> audio.AudioConfig:
    return audio.AudioConfig(duration=0.5)


@pytest.fixture
def scale_config() -> audio.AudioConfig:
    return audio.AudioConfig(scale=True, center=False)


@pytest.fixture
def no_center_config() -> audio.AudioConfig:
    return audio.AudioConfig(center=False)


@pytest.fixture
def resample_fourier_config() -> audio.AudioConfig:
    return audio.AudioConfig(
        resample=audio.ResampleConfig(
            samplerate=audio.TARGET_SAMPLERATE_HZ // 2, method="fourier"
        )
    )


def test_resample_config_defaults():
    config = audio.ResampleConfig()
    assert config.samplerate == audio.TARGET_SAMPLERATE_HZ
    assert config.method == "poly"


def test_audio_config_defaults():
    config = audio.AudioConfig()
    assert config.resample is not None
    assert config.resample.samplerate == audio.TARGET_SAMPLERATE_HZ
    assert config.resample.method == "poly"
    assert config.scale == audio.SCALE_RAW_AUDIO
    assert config.center is True
    assert config.duration == audio.DEFAULT_DURATION


def test_audio_config_override():
    resample_cfg = audio.ResampleConfig(samplerate=44100, method="fourier")
    config = audio.AudioConfig(
        resample=resample_cfg,
        scale=True,
        center=False,
        duration=1.0,
    )
    assert config.resample == resample_cfg
    assert config.scale is True
    assert config.center is False
    assert config.duration == 1.0


def test_audio_config_no_resample():
    config = audio.AudioConfig(resample=None)
    assert config.resample is None


@pytest.mark.parametrize(
    "orig_sr, orig_dur, target_dur",
    [
        (256_000, 1.0, 0.5),
        (256_000, 0.5, 1.0),
        (256_000, 1.0, 1.0),
    ],
)
def test_adjust_audio_duration(orig_sr, orig_dur, target_dur):
    wave = create_xr_wave(samplerate=orig_sr, duration=orig_dur)
    adjusted_wave = audio.adjust_audio_duration(wave, duration=target_dur)
    expected_samples = int(target_dur * orig_sr)
    assert adjusted_wave.sizes["time"] == expected_samples
    assert adjusted_wave.coords["time"].attrs["step"] == 1 / orig_sr
    assert adjusted_wave.dtype == wave.dtype
    if orig_dur > 0 and target_dur > orig_dur:
        padding_start_index = int(orig_dur * orig_sr) + 1
        assert np.all(adjusted_wave.values[padding_start_index:] == 0)


def test_adjust_audio_duration_negative_target_raises():
    wave = create_xr_wave(1000, 1.0)
    with pytest.raises(ValueError):
        audio.adjust_audio_duration(wave, duration=-0.5)


@pytest.mark.parametrize(
    "orig_sr, target_sr, mode",
    [
        (48000, 96000, "poly"),
        (96000, 48000, "poly"),
        (48000, 96000, "fourier"),
        (96000, 48000, "fourier"),
        (48000, 44100, "poly"),
        (48000, 44100, "fourier"),
    ],
)
def test_resample_audio(orig_sr, target_sr, mode):
    duration = 0.1
    wave = create_xr_wave(orig_sr, duration)
    resampled_wave = audio.resample_audio(
        wave, samplerate=target_sr, method=mode, dtype=np.float32
    )
    expected_samples = int(wave.sizes["time"] * (target_sr / orig_sr))
    assert resampled_wave.sizes["time"] == expected_samples
    assert resampled_wave.coords["time"].attrs["step"] == 1 / target_sr
    assert np.isclose(
        resampled_wave.coords["time"].values[-1]
        - resampled_wave.coords["time"].values[0],
        duration,
        atol=2 / target_sr,
    )
    assert resampled_wave.dtype == np.float32


def test_resample_audio_same_samplerate():
    sr = 48000
    duration = 0.1
    wave = create_xr_wave(sr, duration)
    resampled_wave = audio.resample_audio(
        wave, samplerate=sr, dtype=np.float64
    )
    xr.testing.assert_equal(wave.astype(np.float64), resampled_wave)


def test_resample_audio_invalid_mode_raises():
    wave = create_xr_wave(48000, 0.1)
    with pytest.raises(NotImplementedError):
        audio.resample_audio(wave, samplerate=96000, method="invalid_mode")


def test_resample_audio_no_time_dim_raises():
    wave = xr.DataArray(np.random.rand(100), dims=["samples"])
    with pytest.raises(ValueError, match="Audio must have a time dimension"):
        audio.resample_audio(wave, samplerate=96000)


def test_load_clip_audio_default_config(
    dummy_clip: data.Clip,
    default_audio_config: audio.AudioConfig,
    tmp_path: Path,
):
    assert default_audio_config.resample is not None
    target_sr = default_audio_config.resample.samplerate
    orig_duration = dummy_clip.duration
    expected_samples = int(orig_duration * target_sr)

    wav = audio.load_clip_audio(
        dummy_clip, config=default_audio_config, audio_dir=tmp_path
    )

    assert isinstance(wav, xr.DataArray)
    assert wav.dims == ("time",)
    assert wav.sizes["time"] == expected_samples
    assert wav.coords["time"].attrs["step"] == 1 / target_sr
    assert np.isclose(wav.mean(), 0.0, atol=1e-6)
    assert wav.dtype == np.float32


def test_load_clip_audio_no_resample(
    dummy_clip: data.Clip,
    no_resample_config: audio.AudioConfig,
    tmp_path: Path,
):
    orig_sr = dummy_clip.recording.samplerate
    orig_duration = dummy_clip.duration
    expected_samples = int(orig_duration * orig_sr)

    wav = audio.load_clip_audio(
        dummy_clip, config=no_resample_config, audio_dir=tmp_path
    )

    assert wav.coords["time"].attrs["step"] == 1 / orig_sr
    assert wav.sizes["time"] == expected_samples
    assert np.isclose(wav.mean(), 0.0, atol=1e-6)


def test_load_clip_audio_fixed_duration_crop(
    dummy_clip: data.Clip,
    fixed_duration_config: audio.AudioConfig,
    tmp_path: Path,
):
    target_sr = audio.TARGET_SAMPLERATE_HZ
    target_duration = fixed_duration_config.duration
    assert target_duration is not None
    expected_samples = int(target_duration * target_sr)

    assert dummy_clip.duration > target_duration

    wav = audio.load_clip_audio(
        dummy_clip, config=fixed_duration_config, audio_dir=tmp_path
    )

    assert wav.coords["time"].attrs["step"] == 1 / target_sr
    assert wav.sizes["time"] == expected_samples


def test_load_clip_audio_fixed_duration_pad(
    dummy_clip: data.Clip,
    tmp_path: Path,
):
    target_duration = dummy_clip.duration * 2
    config = audio.AudioConfig(duration=target_duration)

    assert config.resample is not None
    target_sr = config.resample.samplerate
    expected_samples = int(target_duration * target_sr)

    wav = audio.load_clip_audio(dummy_clip, config=config, audio_dir=tmp_path)

    assert wav.coords["time"].attrs["step"] == 1 / target_sr
    assert wav.sizes["time"] == expected_samples

    original_samples_after_resample = int(dummy_clip.duration * target_sr)
    assert np.allclose(
        wav.values[original_samples_after_resample:], 0.0, atol=1e-6
    )


def test_load_clip_audio_scale(
    dummy_clip: data.Clip, scale_config: audio.AudioConfig, tmp_path
):
    wav = audio.load_clip_audio(
        dummy_clip,
        config=scale_config,
        audio_dir=tmp_path,
    )

    assert np.isclose(np.max(np.abs(wav.values)), 1.0, atol=1e-5)


def test_load_clip_audio_no_center(
    dummy_clip: data.Clip, no_center_config: audio.AudioConfig, tmp_path
):
    wav = audio.load_clip_audio(
        dummy_clip, config=no_center_config, audio_dir=tmp_path
    )

    raw_wav, _ = sf.read(
        dummy_clip.recording.path,
        start=int(dummy_clip.start_time * dummy_clip.recording.samplerate),
        stop=int(dummy_clip.end_time * dummy_clip.recording.samplerate),
        dtype=np.float32,  # type: ignore
    )
    raw_wav_mono = raw_wav[:, 0]

    if not np.isclose(raw_wav_mono.mean(), 0.0, atol=1e-7):
        assert not np.isclose(wav.mean(), 0.0, atol=1e-6)


def test_load_clip_audio_resample_fourier(
    dummy_clip: data.Clip, resample_fourier_config: audio.AudioConfig, tmp_path
):
    assert resample_fourier_config.resample is not None
    target_sr = resample_fourier_config.resample.samplerate
    orig_duration = dummy_clip.duration
    expected_samples = int(orig_duration * target_sr)

    wav = audio.load_clip_audio(
        dummy_clip, config=resample_fourier_config, audio_dir=tmp_path
    )

    assert wav.coords["time"].attrs["step"] == 1 / target_sr
    assert wav.sizes["time"] == expected_samples


def test_load_clip_audio_dtype(
    dummy_clip: data.Clip, default_audio_config: audio.AudioConfig, tmp_path
):
    wav = audio.load_clip_audio(
        dummy_clip,
        config=default_audio_config,
        audio_dir=tmp_path,
        dtype=np.float64,
    )
    assert wav.dtype == np.float64


def test_load_clip_audio_file_not_found(
    dummy_clip: data.Clip, default_audio_config: audio.AudioConfig, tmp_path
):
    non_existent_path = tmp_path / "not_a_real_file.wav"
    dummy_clip.recording = data.Recording(
        path=non_existent_path,
        duration=1,
        channels=1,
        samplerate=256000,
    )
    with pytest.raises(FileNotFoundError):
        audio.load_clip_audio(
            dummy_clip, config=default_audio_config, audio_dir=tmp_path
        )


def test_load_recording_audio(
    dummy_recording: data.Recording,
    default_audio_config: audio.AudioConfig,
    tmp_path,
):
    assert default_audio_config.resample is not None
    target_sr = default_audio_config.resample.samplerate
    orig_duration = dummy_recording.duration
    expected_samples = int(orig_duration * target_sr)

    wav = audio.load_recording_audio(
        dummy_recording, config=default_audio_config, audio_dir=tmp_path
    )

    assert isinstance(wav, xr.DataArray)
    assert wav.dims == ("time",)
    assert wav.coords["time"].attrs["step"] == 1 / target_sr
    assert wav.sizes["time"] == expected_samples
    assert np.isclose(wav.mean(), 0.0, atol=1e-6)
    assert wav.dtype == np.float32


def test_load_recording_audio_file_not_found(
    dummy_recording: data.Recording,
    default_audio_config: audio.AudioConfig,
    tmp_path,
):
    non_existent_path = tmp_path / "not_a_real_file.wav"
    dummy_recording = data.Recording(
        path=non_existent_path,
        duration=1,
        channels=1,
        samplerate=256000,
    )
    with pytest.raises(FileNotFoundError):
        audio.load_recording_audio(
            dummy_recording, config=default_audio_config, audio_dir=tmp_path
        )


def test_load_file_audio(
    dummy_wav_path: pathlib.Path,
    default_audio_config: audio.AudioConfig,
    tmp_path,
):
    info = sf.info(dummy_wav_path)
    orig_duration = info.duration
    assert default_audio_config.resample is not None
    target_sr = default_audio_config.resample.samplerate
    expected_samples = int(orig_duration * target_sr)

    wav = audio.load_file_audio(
        dummy_wav_path, config=default_audio_config, audio_dir=tmp_path
    )

    assert isinstance(wav, xr.DataArray)
    assert wav.dims == ("time",)
    assert wav.coords["time"].attrs["step"] == 1 / target_sr
    assert wav.sizes["time"] == expected_samples
    assert np.isclose(wav.mean(), 0.0, atol=1e-6)
    assert wav.dtype == np.float32


def test_load_file_audio_file_not_found(
    default_audio_config: audio.AudioConfig, tmp_path
):
    non_existent_path = tmp_path / "not_a_real_file.wav"
    with pytest.raises(FileNotFoundError):
        audio.load_file_audio(
            non_existent_path, config=default_audio_config, audio_dir=tmp_path
        )


def test_build_audio_loader(default_audio_config: audio.AudioConfig):
    loader = audio.build_audio_loader(config=default_audio_config)
    assert isinstance(loader, audio.ConfigurableAudioLoader)
    assert loader.config == default_audio_config


def test_configurable_audio_loader_methods(
    default_audio_config: audio.AudioConfig,
    dummy_wav_path: pathlib.Path,
    dummy_recording: data.Recording,
    dummy_clip: data.Clip,
    tmp_path,
):
    loader = audio.build_audio_loader(config=default_audio_config)

    expected_wav_file = audio.load_file_audio(
        dummy_wav_path, config=default_audio_config, audio_dir=tmp_path
    )
    loaded_wav_file = loader.load_file(dummy_wav_path, audio_dir=tmp_path)
    xr.testing.assert_equal(expected_wav_file, loaded_wav_file)

    expected_wav_rec = audio.load_recording_audio(
        dummy_recording, config=default_audio_config, audio_dir=tmp_path
    )
    loaded_wav_rec = loader.load_recording(dummy_recording, audio_dir=tmp_path)
    xr.testing.assert_equal(expected_wav_rec, loaded_wav_rec)

    expected_wav_clip = audio.load_clip_audio(
        dummy_clip, config=default_audio_config, audio_dir=tmp_path
    )
    loaded_wav_clip = loader.load_clip(dummy_clip, audio_dir=tmp_path)
    xr.testing.assert_equal(expected_wav_clip, loaded_wav_clip)
