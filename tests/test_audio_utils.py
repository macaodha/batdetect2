import numpy as np
import torch
import torch.nn.functional as F
from hypothesis import given
from hypothesis import strategies as st

from batdetect2.detector import parameters
from batdetect2.utils import audio_utils, detector_utils
import io
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

@given(duration=st.floats(min_value=0.1, max_value=2))
def test_can_compute_correct_spectrogram_width(duration: float):
    samplerate = parameters.TARGET_SAMPLERATE_HZ
    params = parameters.DEFAULT_SPECTROGRAM_PARAMETERS

    length = int(duration * samplerate)
    audio = np.random.rand(length)

    spectrogram, _ = audio_utils.generate_spectrogram(
        audio,
        samplerate,
        params,
    )

    # convert to pytorch
    spectrogram = torch.from_numpy(spectrogram)

    # add batch and channel dimensions
    spectrogram = spectrogram.unsqueeze(0).unsqueeze(0)

    # resize the spec
    resize_factor = params["resize_factor"]
    spec_op_shape = (
        int(params["spec_height"] * resize_factor),
        int(spectrogram.shape[-1] * resize_factor),
    )
    spectrogram = F.interpolate(
        spectrogram,
        size=spec_op_shape,
        mode="bilinear",
        align_corners=False,
    )

    expected_width = audio_utils.compute_spectrogram_width(
        length,
        samplerate=parameters.TARGET_SAMPLERATE_HZ,
        window_duration=params["fft_win_length"],
        window_overlap=params["fft_overlap"],
        resize_factor=params["resize_factor"],
    )

    assert spectrogram.shape[-1] == expected_width


@given(duration=st.floats(min_value=0.1, max_value=2))
def test_pad_audio_without_fixed_size(duration: float):
    # Test the pad_audio function
    # This function is used to pad audio with zeros to a specific length
    # It is used in the generate_spectrogram function
    # The function is tested with a simplepas
    samplerate = parameters.TARGET_SAMPLERATE_HZ
    params = parameters.DEFAULT_SPECTROGRAM_PARAMETERS

    length = int(duration * samplerate)
    audio = np.random.rand(length)

    # pad the audio to be divisible by divide factor
    padded_audio = audio_utils.pad_audio(
        audio,
        samplerate=samplerate,
        window_duration=params["fft_win_length"],
        window_overlap=params["fft_overlap"],
        resize_factor=params["resize_factor"],
        divide_factor=params["spec_divide_factor"],
    )

    # check that the padded audio is divisible by the divide factor
    expected_width = audio_utils.compute_spectrogram_width(
        len(padded_audio),
        samplerate=parameters.TARGET_SAMPLERATE_HZ,
        window_duration=params["fft_win_length"],
        window_overlap=params["fft_overlap"],
        resize_factor=params["resize_factor"],
    )

    assert expected_width % params["spec_divide_factor"] == 0


@given(duration=st.floats(min_value=0.1, max_value=2))
def test_computed_spectrograms_are_actually_divisible_by_the_spec_divide_factor(
    duration: float,
):
    samplerate = parameters.TARGET_SAMPLERATE_HZ
    params = parameters.DEFAULT_SPECTROGRAM_PARAMETERS
    length = int(duration * samplerate)
    audio = np.random.rand(length)
    _, spectrogram, _ = detector_utils.compute_spectrogram(
        audio,
        samplerate,
        params,
        torch.device("cpu"),
    )
    assert spectrogram.shape[-1] % params["spec_divide_factor"] == 0


@given(
    duration=st.floats(min_value=0.1, max_value=2),
    width=st.integers(min_value=128, max_value=1024),
)
def test_pad_audio_with_fixed_width(duration: float, width: int):
    samplerate = parameters.TARGET_SAMPLERATE_HZ
    params = parameters.DEFAULT_SPECTROGRAM_PARAMETERS

    length = int(duration * samplerate)
    audio = np.random.rand(length)

    # pad the audio to be divisible by divide factor
    padded_audio = audio_utils.pad_audio(
        audio,
        samplerate=samplerate,
        window_duration=params["fft_win_length"],
        window_overlap=params["fft_overlap"],
        resize_factor=params["resize_factor"],
        divide_factor=params["spec_divide_factor"],
        fixed_width=width,
    )

    # check that the padded audio is divisible by the divide factor
    expected_width = audio_utils.compute_spectrogram_width(
        len(padded_audio),
        samplerate=parameters.TARGET_SAMPLERATE_HZ,
        window_duration=params["fft_win_length"],
        window_overlap=params["fft_overlap"],
        resize_factor=params["resize_factor"],
    )
    assert expected_width == width
    

def test_load_audio_using_bytesio():
    basename = "20230322_172000_selec2.wav"
    path = os.path.join(DATA_DIR, basename)

    with open(path, "rb") as f:
        data = io.BytesIO(f.read())
    
    sample_rate, audio_data, file_sample_rate = audio_utils.load_audio_and_samplerate(data, time_exp_fact=1, target_samp_rate=parameters.TARGET_SAMPLERATE_HZ)

    expected_sample_rate, expected_audio_data, exp_file_sample_rate = audio_utils.load_audio_and_samplerate(path, time_exp_fact=1, target_samp_rate=parameters.TARGET_SAMPLERATE_HZ)

    assert expected_sample_rate == sample_rate
    assert exp_file_sample_rate == file_sample_rate

    assert np.array_equal(audio_data, expected_audio_data)