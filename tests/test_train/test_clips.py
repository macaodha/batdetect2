import numpy as np
import pytest
import xarray as xr

from batdetect2.train.clips import (
    Clipper,
    _compute_expected_width,
    select_subclip,
)

AUDIO_SAMPLERATE = 48000

SPEC_SAMPLERATE = 100
SPEC_FREQS = 64
CLIP_DURATION = 0.5


CLIP_WIDTH_SPEC = int(np.floor(CLIP_DURATION * SPEC_SAMPLERATE))
CLIP_WIDTH_AUDIO = int(np.floor(CLIP_DURATION * AUDIO_SAMPLERATE))
MAX_EMPTY = 0.2


def create_test_dataset(
    duration_sec: float,
    spec_samplerate: int = SPEC_SAMPLERATE,
    audio_samplerate: int = AUDIO_SAMPLERATE,
    num_freqs: int = SPEC_FREQS,
    start_time: float = 0.0,
) -> xr.Dataset:
    """Creates a sample xr.Dataset for testing."""
    time_step = 1 / spec_samplerate
    audio_time_step = 1 / audio_samplerate

    times = np.arange(start_time, start_time + duration_sec, step=time_step)
    freqs = np.linspace(0, audio_samplerate / 2, num_freqs)
    audio_times = np.arange(
        start_time,
        start_time + duration_sec,
        step=audio_time_step,
    )

    num_time_steps = len(times)
    num_audio_samples = len(audio_times)
    spec_shape = (num_freqs, num_time_steps)

    spectrogram_data = np.arange(num_time_steps).reshape(1, -1) * np.ones(
        (num_freqs, 1)
    )

    spectrogram = xr.DataArray(
        spectrogram_data.astype(np.float32),
        coords=[("frequency", freqs), ("time", times)],
        name="spectrogram",
    )

    detection = xr.DataArray(
        np.ones(spec_shape, dtype=np.float32) * 0.5,
        coords=spectrogram.coords,
        name="detection",
    )

    classes = xr.DataArray(
        np.ones((3, *spec_shape), dtype=np.float32),
        coords=[
            ("category", ["A", "B", "C"]),
            ("frequency", freqs),
            ("time", times),
        ],
        name="class",
    )

    size = xr.DataArray(
        np.ones((2, *spec_shape), dtype=np.float32),
        coords=[
            ("dimension", ["height", "width"]),
            ("frequency", freqs),
            ("time", times),
        ],
        name="size",
    )

    audio_data = np.arange(num_audio_samples)
    audio = xr.DataArray(
        audio_data.astype(np.float32),
        coords=[("audio_time", audio_times)],
        name="audio",
    )

    metadata = xr.DataArray([1, 2, 3], dims=["other_dim"], name="metadata")

    return xr.Dataset(
        {
            "audio": audio,
            "spectrogram": spectrogram,
            "detection": detection,
            "class": classes,
            "size": size,
            "metadata": metadata,
        }
    ).assign_attrs(
        samplerate=audio_samplerate,
        spec_samplerate=spec_samplerate,
    )


@pytest.fixture
def long_dataset() -> xr.Dataset:
    """Dataset longer than the clip duration."""
    return create_test_dataset(duration_sec=2.0)


@pytest.fixture
def short_dataset() -> xr.Dataset:
    """Dataset shorter than the clip duration."""
    return create_test_dataset(duration_sec=0.3)


@pytest.fixture
def exact_dataset() -> xr.Dataset:
    """Dataset exactly the clip duration."""
    return create_test_dataset(duration_sec=CLIP_DURATION - 1e-9)


@pytest.fixture
def offset_dataset() -> xr.Dataset:
    """Dataset starting at a non-zero time."""
    return create_test_dataset(duration_sec=1.0, start_time=0.5)


def test_select_subclip_within_bounds(long_dataset):
    start_time = 0.5
    subclip = select_subclip(
        long_dataset, span=CLIP_DURATION, start=start_time, dim="time"
    )
    expected_width = _compute_expected_width(
        long_dataset, CLIP_DURATION, "time"
    )

    assert "time" in subclip.dims
    assert subclip.dims["time"] == expected_width
    assert subclip.spectrogram.dims == ("frequency", "time")
    assert subclip.spectrogram.shape == (SPEC_FREQS, expected_width)
    assert subclip.detection.shape == (SPEC_FREQS, expected_width)
    assert subclip["class"].shape == (3, SPEC_FREQS, expected_width)
    assert subclip.size.shape == (2, SPEC_FREQS, expected_width)
    assert subclip.time.min() >= start_time
    assert (
        subclip.time.max() <= start_time + CLIP_DURATION + 1 / SPEC_SAMPLERATE
    )

    assert "metadata" in subclip
    xr.testing.assert_equal(subclip.metadata, long_dataset.metadata)


def test_select_subclip_pad_start(long_dataset):
    start_time = -0.1
    subclip = select_subclip(
        long_dataset, span=CLIP_DURATION, start=start_time, dim="time"
    )
    expected_width = _compute_expected_width(
        long_dataset, CLIP_DURATION, "time"
    )
    step = 1 / SPEC_SAMPLERATE
    expected_pad_samples = int(np.floor(abs(start_time) / step))

    assert subclip.dims["time"] == expected_width
    assert subclip.spectrogram.shape[1] == expected_width

    assert np.all(
        subclip.spectrogram.isel(time=slice(0, expected_pad_samples)) == 0
    )

    assert np.any(
        subclip.spectrogram.isel(time=slice(expected_pad_samples, None)) != 0
    )
    assert subclip.time.min() >= start_time
    assert subclip.time.max() < start_time + CLIP_DURATION + step


def test_select_subclip_pad_end(long_dataset):
    original_duration = long_dataset.time.max() - long_dataset.time.min()
    start_time = original_duration - 0.1
    subclip = select_subclip(
        long_dataset, span=CLIP_DURATION, start=start_time, dim="time"
    )
    expected_width = _compute_expected_width(
        long_dataset, CLIP_DURATION, "time"
    )
    step = 1 / SPEC_SAMPLERATE
    original_width = long_dataset.dims["time"]
    expected_pad_samples = expected_width - (
        original_width - int(np.floor(start_time / step))
    )

    assert subclip.sizes["time"] == expected_width
    assert subclip.spectrogram.shape[1] == expected_width

    assert np.all(
        subclip.spectrogram.isel(
            time=slice(expected_width - expected_pad_samples, None)
        )
        == 0
    )

    assert np.any(
        subclip.spectrogram.isel(
            time=slice(0, expected_width - expected_pad_samples)
        )
        != 0
    )
    assert subclip.time.min() >= start_time
    assert subclip.time.max() < start_time + CLIP_DURATION + step


def test_select_subclip_pad_both_short_dataset(short_dataset):
    start_time = -0.1
    subclip = select_subclip(
        short_dataset, span=CLIP_DURATION, start=start_time, dim="time"
    )
    expected_width = _compute_expected_width(
        short_dataset, CLIP_DURATION, "time"
    )
    step = 1 / SPEC_SAMPLERATE

    assert subclip.dims["time"] == expected_width
    assert subclip.spectrogram.shape[1] == expected_width

    assert subclip.spectrogram.coords["time"][0] == pytest.approx(
        start_time,
        abs=step,
    )
    assert subclip.spectrogram.coords["time"][-1] == pytest.approx(
        start_time + CLIP_DURATION - step,
        abs=2 * step,
    )


def test_select_subclip_width_consistency(long_dataset):
    expected_width = _compute_expected_width(
        long_dataset, CLIP_DURATION, "time"
    )
    step = 1 / SPEC_SAMPLERATE

    subclip_aligned = select_subclip(
        long_dataset.copy(deep=True),
        span=CLIP_DURATION,
        start=5 * step,
        dim="time",
    )

    subclip_offset = select_subclip(
        long_dataset.copy(deep=True),
        span=CLIP_DURATION,
        start=5.3 * step,
        dim="time",
    )

    assert subclip_aligned.sizes["time"] == expected_width
    assert subclip_offset.sizes["time"] == expected_width
    assert subclip_aligned.spectrogram.shape[1] == expected_width
    assert subclip_offset.spectrogram.shape[1] == expected_width


def test_select_subclip_different_dimension(long_dataset):
    freq_coords = long_dataset.frequency.values
    freq_min, freq_max = freq_coords.min(), freq_coords.max()
    freq_span = (freq_max - freq_min) / 2
    start_freq = freq_min + freq_span / 2

    subclip = select_subclip(
        long_dataset, span=freq_span, start=start_freq, dim="frequency"
    )

    assert "frequency" in subclip.dims
    assert subclip.spectrogram.shape[0] < long_dataset.spectrogram.shape[0]
    assert subclip.detection.shape[0] < long_dataset.detection.shape[0]
    assert subclip["class"].shape[1] < long_dataset["class"].shape[1]
    assert subclip.size.shape[1] < long_dataset.size.shape[1]

    assert subclip.dims["time"] == long_dataset.dims["time"]
    assert subclip.spectrogram.shape[1] == long_dataset.spectrogram.shape[1]

    xr.testing.assert_equal(subclip.audio, long_dataset.audio)
    assert subclip.dims["audio_time"] == long_dataset.dims["audio_time"]


def test_select_subclip_fill_value(short_dataset):
    fill_value = -999.0
    subclip = select_subclip(
        short_dataset,
        span=CLIP_DURATION,
        start=0,
        dim="time",
        fill_value=fill_value,
    )

    expected_width = _compute_expected_width(
        short_dataset,
        CLIP_DURATION,
        "time",
    )

    assert subclip.dims["time"] == expected_width
    assert np.all(subclip.spectrogram.sel(time=slice(0.3, None)) == fill_value)


def test_select_subclip_no_overlap_raises_error(long_dataset):
    original_duration = long_dataset.time.max() - long_dataset.time.min()

    with pytest.raises(ValueError, match="does not overlap"):
        select_subclip(
            long_dataset,
            span=CLIP_DURATION,
            start=original_duration + 1.0,
            dim="time",
        )

    with pytest.raises(ValueError, match="does not overlap"):
        select_subclip(
            long_dataset,
            span=CLIP_DURATION,
            start=-1.0 * CLIP_DURATION - 1.0,
            dim="time",
        )


def test_clipper_non_random(long_dataset, exact_dataset, short_dataset):
    clipper = Clipper(duration=CLIP_DURATION, random=False)

    for ds in [long_dataset, exact_dataset, short_dataset]:
        clip, _, _ = clipper.extract_clip(ds)
        expected_spec_width = _compute_expected_width(
            ds, CLIP_DURATION, "time"
        )
        expected_audio_width = _compute_expected_width(
            ds, CLIP_DURATION, "audio_time"
        )

        assert clip.dims["time"] == expected_spec_width
        assert clip.dims["audio_time"] == expected_audio_width
        assert clip.spectrogram.shape[1] == expected_spec_width
        assert clip.audio.shape[0] == expected_audio_width

        assert clip.time.min() >= -1 / SPEC_SAMPLERATE
        assert clip.audio_time.min() >= -1 / AUDIO_SAMPLERATE

        time_span = clip.time.max() - clip.time.min()
        audio_span = clip.audio_time.max() - clip.audio_time.min()
        assert np.isclose(time_span, CLIP_DURATION, atol=1 / SPEC_SAMPLERATE)
        assert np.isclose(audio_span, CLIP_DURATION, atol=1 / AUDIO_SAMPLERATE)


def test_clipper_random(long_dataset):
    seed = 42
    np.random.seed(seed)
    clipper = Clipper(duration=CLIP_DURATION, random=True, max_empty=MAX_EMPTY)
    clip1, _, _ = clipper.extract_clip(long_dataset)

    np.random.seed(seed + 1)
    clip2, _, _ = clipper.extract_clip(long_dataset)

    expected_spec_width = _compute_expected_width(
        long_dataset, CLIP_DURATION, "time"
    )
    expected_audio_width = _compute_expected_width(
        long_dataset, CLIP_DURATION, "audio_time"
    )

    for clip in [clip1, clip2]:
        assert clip.dims["time"] == expected_spec_width
        assert clip.dims["audio_time"] == expected_audio_width
        assert clip.spectrogram.shape[1] == expected_spec_width
        assert clip.audio.shape[0] == expected_audio_width

    assert not np.isclose(clip1.time.min(), clip2.time.min())
    assert not np.isclose(clip1.audio_time.min(), clip2.audio_time.min())

    for clip in [clip1, clip2]:
        time_span = clip.time.max() - clip.time.min()
        audio_span = clip.audio_time.max() - clip.audio_time.min()
        assert np.isclose(time_span, CLIP_DURATION, atol=1 / SPEC_SAMPLERATE)
        assert np.isclose(audio_span, CLIP_DURATION, atol=1 / AUDIO_SAMPLERATE)

    max_start_time = (
        (long_dataset.time.max() - long_dataset.time.min())
        - CLIP_DURATION
        + MAX_EMPTY
    )
    assert clip1.time.min() <= max_start_time + 1 / SPEC_SAMPLERATE
    assert clip2.time.min() <= max_start_time + 1 / SPEC_SAMPLERATE


def test_clipper_random_max_empty_effect(long_dataset):
    """Check that max_empty influences the possible start times."""
    seed = 123
    data_duration = long_dataset.time.max() - long_dataset.time.min()

    np.random.seed(seed)
    clipper0 = Clipper(duration=CLIP_DURATION, random=True, max_empty=0.0)
    max_start_time0 = data_duration - CLIP_DURATION
    start_times0 = []

    for _ in range(20):
        clip, _, _ = clipper0.extract_clip(long_dataset)
        start_times0.append(clip.time.min().item())

    assert all(
        st <= max_start_time0 + 1 / SPEC_SAMPLERATE for st in start_times0
    )
    assert any(st > 0.1 for st in start_times0)

    np.random.seed(seed)
    clipper_pos = Clipper(duration=CLIP_DURATION, random=True, max_empty=0.2)
    max_start_time_pos = data_duration - CLIP_DURATION + 0.2
    start_times_pos = []
    for _ in range(20):
        clip, _, _ = clipper_pos.extract_clip(long_dataset)
        start_times_pos.append(clip.time.min().item())
    assert all(
        st <= max_start_time_pos + 1 / SPEC_SAMPLERATE
        for st in start_times_pos
    )

    assert any(st > max_start_time0 + 1e-6 for st in start_times_pos)


def test_clipper_short_dataset_random(short_dataset):
    clipper = Clipper(duration=CLIP_DURATION, random=True, max_empty=MAX_EMPTY)
    clip, _, _ = clipper.extract_clip(short_dataset)

    expected_spec_width = _compute_expected_width(
        short_dataset, CLIP_DURATION, "time"
    )
    expected_audio_width = _compute_expected_width(
        short_dataset, CLIP_DURATION, "audio_time"
    )

    assert clip.sizes["time"] == expected_spec_width
    assert clip.sizes["audio_time"] == expected_audio_width
    assert clip["spectrogram"].shape[1] == expected_spec_width
    assert clip["audio"].shape[0] == expected_audio_width

    assert np.any(clip.spectrogram == 0)
    assert np.any(clip.audio == 0)


def test_clipper_exact_dataset_random(exact_dataset):
    clipper = Clipper(duration=CLIP_DURATION, random=True, max_empty=MAX_EMPTY)
    clip, _, _ = clipper.extract_clip(exact_dataset)

    expected_spec_width = _compute_expected_width(
        exact_dataset, CLIP_DURATION, "time"
    )
    expected_audio_width = _compute_expected_width(
        exact_dataset, CLIP_DURATION, "audio_time"
    )

    assert clip.dims["time"] == expected_spec_width
    assert clip.dims["audio_time"] == expected_audio_width
    assert clip.spectrogram.shape[1] == expected_spec_width
    assert clip.audio.shape[0] == expected_audio_width

    time_span = clip.time.max() - clip.time.min()
    audio_span = clip.audio_time.max() - clip.audio_time.min()
    assert np.isclose(time_span, CLIP_DURATION, atol=1 / SPEC_SAMPLERATE)
    assert np.isclose(audio_span, CLIP_DURATION, atol=1 / AUDIO_SAMPLERATE)
