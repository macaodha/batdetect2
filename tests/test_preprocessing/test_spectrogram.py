
import numpy as np
import pytest
import xarray as xr

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
