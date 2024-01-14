import warnings
from typing import Optional, Tuple, Union, overload

import librosa
import librosa.core.spectrum
import numpy as np
import torch

__all__ = [
    "load_audio",
    "generate_spectrogram",
    "pad_audio",
]


@overload
def time_to_x_coords(
    time_in_file: np.ndarray,
    sampling_rate: float,
    fft_win_length: float,
    fft_overlap: float,
) -> np.ndarray:
    ...


@overload
def time_to_x_coords(
    time_in_file: float,
    sampling_rate: float,
    fft_win_length: float,
    fft_overlap: float,
) -> float:
    ...


def time_to_x_coords(
    time_in_file: Union[float, np.ndarray],
    sampling_rate: float,
    fft_win_length: float,
    fft_overlap: float,
) -> Union[float, np.ndarray]:
    nfft = np.floor(fft_win_length * sampling_rate)
    noverlap = np.floor(fft_overlap * nfft)
    return (time_in_file * sampling_rate - noverlap) / (nfft - noverlap)


# NOTE this is also defined in post_process
def x_coords_to_time(
    x_pos: float,
    sampling_rate: int,
    fft_win_length: float,
    fft_overlap: float,
) -> float:
    nfft = np.floor(fft_win_length * sampling_rate)
    noverlap = np.floor(fft_overlap * nfft)
    return ((x_pos * (nfft - noverlap)) + noverlap) / sampling_rate

    # return (1.0 - fft_overlap) * fft_win_length * (x_pos + 0.5)  # 0.5 is for
    # center of temporal window


def generate_spectrogram(
    audio: np.ndarray,
    sampling_rate: float,
    fft_win_length: float,
    fft_overlap: float,
    max_freq: float,
    min_freq: float,
    spec_scale: str,
    denoise_spec_avg: bool = False,
    max_scale_spec: bool = False,
) -> np.ndarray:
    # generate spectrogram
    spec = gen_mag_spectrogram(
        audio,
        sampling_rate,
        window_len=fft_win_length,
        overlap_perc=fft_overlap,
    )
    spec = crop_spectrogram(
        spec,
        fft_win_length=fft_win_length,
        max_freq=max_freq,
        min_freq=min_freq,
    )
    spec = scale_spectrogram(
        spec,
        sampling_rate,
        spec_scale=spec_scale,
        fft_win_length=fft_win_length,
    )

    if denoise_spec_avg:
        spec = denoise_spectrogram(spec)

    if max_scale_spec:
        spec = max_scale_spectrogram(spec)

    return spec


def crop_spectrogram(
    spec: np.ndarray,
    fft_win_length: float,
    max_freq: float,
    min_freq: float,
) -> np.ndarray:
    # crop to min/max freq
    max_freq = round(max_freq * fft_win_length)
    min_freq = round(min_freq * fft_win_length)
    if spec.shape[0] < max_freq:
        freq_pad = max_freq - spec.shape[0]
        spec = np.vstack(
            (np.zeros((freq_pad, spec.shape[1]), dtype=spec.dtype), spec)
        )
    return spec[-max_freq : spec.shape[0] - min_freq, :]


def denoise_spectrogram(spec: np.ndarray) -> np.ndarray:
    spec = spec - np.mean(spec, 1)[:, np.newaxis]
    return spec.clip(min=0)


def max_scale_spectrogram(spec: np.ndarray) -> np.ndarray:
    return spec / (spec.max() + 10e-6)


def log_scale(
    spec: np.ndarray,
    sampling_rate: float,
    fft_win_length: float,
) -> np.ndarray:
    log_scaling = (
        2.0
        * (1.0 / sampling_rate)
        * (
            1.0
            / (
                np.abs(np.hanning(int(fft_win_length * sampling_rate))) ** 2
            ).sum()
        )
    )
    return np.log1p(log_scaling * spec)


def scale_spectrogram(
    spec: np.ndarray,
    sampling_rate: float,
    spec_scale: str,
    fft_win_length: float,
) -> np.ndarray:
    if spec_scale == "log":
        return log_scale(spec, sampling_rate, fft_win_length)

    if spec_scale == "pcen":
        return pcen(spec, sampling_rate)

    return spec


def prepare_spec_for_viz(
    spec: np.ndarray,
    sampling_rate: int,
    fft_win_length: float,
) -> np.ndarray:
    # for visualization purposes - use log scaled spectrogram
    return log_scale(
        spec,
        sampling_rate,
        fft_win_length=fft_win_length,
    ).astype(np.float32)


def load_audio(
    audio_file: str,
    time_exp_fact: float,
    target_sampling_rate: int,
    scale: bool = False,
    max_duration: Optional[float] = None,
) -> Tuple[float, np.ndarray]:
    """Load an audio file and resample it to the target sampling rate.

    The audio is also scaled to [-1, 1] and clipped to the maximum duration.
    Only mono files are supported.

    Parameters
    ----------
    audio_file: str
        Path to the audio file.
    target_samp_rate: int
        Target sampling rate.
    scale: bool, optional
        Whether to scale the audio to [-1, 1]. Default: False.
    max_duration: float, optional
        Maximum duration of the audio in seconds. Defaults to None.
        If provided, the audio is clipped to this duration.

    Returns
    -------
    sampling_rate: int
        The sampling rate of the audio.
    audio_raw: np.ndarray
        The audio signal in a numpy array.

    Raises
    ------
    ValueError: If the audio file is stereo.

    """
    with warnings.catch_warnings():
        audio, sampling_rate = librosa.load(
            audio_file,
            sr=None,
            dtype=np.float32,
        )

    if len(audio.shape) > 1:
        raise ValueError("Currently does not handle stereo files")

    sampling_rate = sampling_rate * time_exp_fact

    # resample - need to do this after correcting for time expansion
    audio = resample_audio(audio, sampling_rate, target_sampling_rate)

    if max_duration is not None:
        audio = clip_audio(audio, target_sampling_rate, max_duration)

    # scale to [-1, 1]
    if scale:
        audio = scale_audio(audio)

    return target_sampling_rate, audio


def resample_audio(
    audio: np.ndarray,
    sr_orig: float,
    sr_target: float,
) -> np.ndarray:
    if sr_orig != sr_target:
        return librosa.resample(
            audio,
            orig_sr=sr_orig,
            target_sr=sr_target,
            res_type="polyphase",
        )

    return audio


def clip_audio(
    audio: np.ndarray,
    sampling_rate: float,
    max_duration: float,
) -> np.ndarray:
    max_duration = int(
        np.minimum(
            int(sampling_rate * max_duration),
            audio.shape[0],
        )
    )
    return audio[:max_duration]


def scale_audio(
    audio: np.ndarray,
    eps: float = 10e-6,
) -> np.ndarray:
    return (audio - audio.mean()) / (np.abs(audio).max() + eps)


def pad_audio(
    audio_raw: np.ndarray,
    sampling_rate: float,
    window_len: float,
    overlap_perc: float,
    resize_factor: float,
    divide_factor: float,
    fixed_width: Optional[int] = None,
) -> np.ndarray:
    # Adds zeros to the end of the raw data so that the generated sepctrogram
    # will be evenly divisible by `divide_factor`
    # Also deals with very short audio clips and fixed_width during training

    # This code could be clearer, clean up
    nfft = int(window_len * sampling_rate)
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


def gen_mag_spectrogram(
    audio: np.ndarray,
    sampling_rate: float,
    window_len: float,
    overlap_perc: float,
) -> np.ndarray:
    # Computes magnitude spectrogram by specifying time.
    audio = audio.astype(np.float32)
    nfft = int(window_len * sampling_rate)
    noverlap = int(overlap_perc * nfft)

    # compute spec
    spec, _ = librosa.core.spectrum._spectrogram(
        y=audio,
        power=1,
        n_fft=nfft,
        hop_length=nfft - noverlap,
        center=False,
    )

    # remove DC component and flip vertical orientation
    spec = np.flipud(spec[1:, :])

    return spec.astype(np.float32)


def gen_mag_spectrogram_pt(
    audio: torch.Tensor,
    sampling_rate: float,
    window_len: float,
    overlap_perc: float,
) -> torch.Tensor:
    nfft = int(window_len * sampling_rate)
    nstep = round((1.0 - overlap_perc) * nfft)
    han_win = torch.hann_window(nfft, periodic=False).to(audio.device)

    complex_spec = torch.stft(audio, nfft, nstep, window=han_win, center=False)
    spec = complex_spec.pow(2.0).sum(-1)

    # remove DC component and flip vertically
    return torch.flipud(spec[0, 1:, :])


def pcen(spec: np.ndarray, sampling_rate: float) -> np.ndarray:
    # TODO should be passing hop_length too i.e. step
    return librosa.pcen(spec * (2**31), sr=sampling_rate / 10).astype(
        np.float32
    )
