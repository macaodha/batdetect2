import warnings
from typing import Optional, Tuple

import librosa
import librosa.core.spectrum
import numpy as np
import torch

from . import wavfile

__all__ = [
    "load_audio",
    "generate_spectrogram",
    "pad_audio",
]


def time_to_x_coords(time_in_file, sampling_rate, fft_win_length, fft_overlap):
    nfft = np.floor(fft_win_length * sampling_rate)  # int() uses floor
    noverlap = np.floor(fft_overlap * nfft)
    return (time_in_file * sampling_rate - noverlap) / (nfft - noverlap)


# NOTE this is also defined in post_process
def x_coords_to_time(x_pos, sampling_rate, fft_win_length, fft_overlap):
    nfft = np.floor(fft_win_length * sampling_rate)
    noverlap = np.floor(fft_overlap * nfft)
    return ((x_pos * (nfft - noverlap)) + noverlap) / sampling_rate
    # return (1.0 - fft_overlap) * fft_win_length * (x_pos + 0.5)  # 0.5 is for center of temporal window


def generate_spectrogram(
    audio,
    sampling_rate,
    params,
    return_spec_for_viz=False,
    check_spec_size=True,
):
    # generate spectrogram
    spec = gen_mag_spectrogram(
        audio,
        sampling_rate,
        params["fft_win_length"],
        params["fft_overlap"],
    )

    # crop to min/max freq
    max_freq = round(params["max_freq"] * params["fft_win_length"])
    min_freq = round(params["min_freq"] * params["fft_win_length"])
    if spec.shape[0] < max_freq:
        freq_pad = max_freq - spec.shape[0]
        spec = np.vstack(
            (np.zeros((freq_pad, spec.shape[1]), dtype=spec.dtype), spec)
        )
    spec_cropped = spec[-max_freq : spec.shape[0] - min_freq, :]

    if params["spec_scale"] == "log":
        log_scaling = (
            2.0
            * (1.0 / sampling_rate)
            * (
                1.0
                / (
                    np.abs(
                        np.hanning(
                            int(params["fft_win_length"] * sampling_rate)
                        )
                    )
                    ** 2
                ).sum()
            )
        )
        # log_scaling = (1.0 / sampling_rate)*0.1
        # log_scaling = (1.0 / sampling_rate)*10e4
        spec = np.log1p(log_scaling * spec_cropped)
    elif params["spec_scale"] == "pcen":
        spec = pcen(spec_cropped, sampling_rate)

    elif params["spec_scale"] == "none":
        pass

    if params["denoise_spec_avg"]:
        spec = spec - np.mean(spec, 1)[:, np.newaxis]
        spec.clip(min=0, out=spec)

    if params["max_scale_spec"]:
        spec = spec / (spec.max() + 10e-6)

    # needs to be divisible by specific factor - if not it should have been padded
    # if check_spec_size:
    # assert((int(spec.shape[0]*params['resize_factor']) % params['spec_divide_factor']) == 0)
    # assert((int(spec.shape[1]*params['resize_factor']) % params['spec_divide_factor']) == 0)

    # for visualization purposes - use log scaled spectrogram
    if return_spec_for_viz:
        log_scaling = (
            2.0
            * (1.0 / sampling_rate)
            * (
                1.0
                / (
                    np.abs(
                        np.hanning(
                            int(params["fft_win_length"] * sampling_rate)
                        )
                    )
                    ** 2
                ).sum()
            )
        )
        spec_for_viz = np.log1p(log_scaling * spec_cropped).astype(np.float32)
    else:
        spec_for_viz = None

    return spec, spec_for_viz


def load_audio(
    audio_file: str,
    time_exp_fact: float,
    target_samp_rate: int,
    scale: bool = False,
    max_duration: Optional[float] = None,
) -> Tuple[int, np.ndarray]:
    """Load an audio file and resample it to the target sampling rate.

    The audio is also scaled to [-1, 1] and clipped to the maximum duration.
    Only mono files are supported.

    Args:
        audio_file (str): Path to the audio file.
        target_samp_rate (int): Target sampling rate.
        scale (bool): Whether to scale the audio to [-1, 1].
        max_duration (float): Maximum duration of the audio in seconds.

    Returns:
        sampling_rate: The sampling rate of the audio.
        audio_raw: The audio signal in a numpy array.

    Raises:
        ValueError: If the audio file is stereo.

    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=wavfile.WavFileWarning)
        # sampling_rate, audio_raw = wavfile.read(audio_file)
        audio_raw, sampling_rate = librosa.load(
            audio_file,
            sr=None,
            dtype=np.float32,
        )

    if len(audio_raw.shape) > 1:
        raise ValueError("Currently does not handle stereo files")

    sampling_rate = sampling_rate * time_exp_fact

    # resample - need to do this after correcting for time expansion
    sampling_rate_old = sampling_rate
    sampling_rate = target_samp_rate
    if sampling_rate_old != sampling_rate:
        audio_raw = librosa.resample(
            audio_raw,
            orig_sr=sampling_rate_old,
            target_sr=sampling_rate,
            res_type="polyphase",
        )

    # clipping maximum duration
    if max_duration is not None:
        max_duration = int(
            np.minimum(
                int(sampling_rate * max_duration),
                audio_raw.shape[0],
            )
        )
        audio_raw = audio_raw[:max_duration]

    # scale to [-1, 1]
    if scale:
        audio_raw = audio_raw - audio_raw.mean()
        audio_raw = audio_raw / (np.abs(audio_raw).max() + 10e-6)

    return sampling_rate, audio_raw


def pad_audio(
    audio_raw,
    fs,
    ms,
    overlap_perc,
    resize_factor,
    divide_factor,
    fixed_width=None,
):
    # Adds zeros to the end of the raw data so that the generated sepctrogram
    # will be evenly divisible by `divide_factor`
    # Also deals with very short audio clips and fixed_width during training

    # This code could be clearer, clean up
    nfft = int(ms * fs)
    noverlap = int(overlap_perc * nfft)
    step = nfft - noverlap
    min_size = int(divide_factor * (1.0 / resize_factor))
    spec_width = (audio_raw.shape[0] - noverlap) // step
    spec_width_rs = spec_width * resize_factor

    if fixed_width is not None and spec_width < fixed_width:
        # too small
        # used during training to ensure all the batches are the same size
        diff = fixed_width * step + noverlap - audio_raw.shape[0]
        audio_raw = np.hstack(
            (audio_raw, np.zeros(diff, dtype=audio_raw.dtype))
        )

    elif fixed_width is not None and spec_width > fixed_width:
        # too big
        # used during training to ensure all the batches are the same size
        diff = fixed_width * step + noverlap - audio_raw.shape[0]
        audio_raw = audio_raw[:diff]

    elif (
        spec_width_rs < min_size
        or (np.floor(spec_width_rs) % divide_factor) != 0
    ):
        # need to be at least min_size
        div_amt = np.ceil(spec_width_rs / float(divide_factor))
        div_amt = np.maximum(1, div_amt)
        target_size = int(div_amt * divide_factor * (1.0 / resize_factor))
        diff = target_size * step + noverlap - audio_raw.shape[0]
        audio_raw = np.hstack(
            (audio_raw, np.zeros(diff, dtype=audio_raw.dtype))
        )

    return audio_raw


def gen_mag_spectrogram(x, fs, ms, overlap_perc):
    # Computes magnitude spectrogram by specifying time.

    x = x.astype(np.float32)
    nfft = int(ms * fs)
    noverlap = int(overlap_perc * nfft)

    # window data
    step = nfft - noverlap

    # compute spec
    spec, _ = librosa.core.spectrum._spectrogram(
        y=x, power=1, n_fft=nfft, hop_length=step, center=False
    )

    # remove DC component and flip vertical orientation
    spec = np.flipud(spec[1:, :])

    return spec.astype(np.float32)


def gen_mag_spectrogram_pt(x, fs, ms, overlap_perc):
    nfft = int(ms * fs)
    nstep = round((1.0 - overlap_perc) * nfft)

    han_win = torch.hann_window(nfft, periodic=False).to(x.device)

    complex_spec = torch.stft(x, nfft, nstep, window=han_win, center=False)
    spec = complex_spec.pow(2.0).sum(-1)

    # remove DC component and flip vertically
    spec = torch.flipud(spec[0, 1:, :])

    return spec


def pcen(spec_cropped, sampling_rate):
    # TODO should be passing hop_length too i.e. step
    spec = librosa.pcen(spec_cropped * (2**31), sr=sampling_rate / 10).astype(
        np.float32
    )
    return spec
