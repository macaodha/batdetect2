import warnings
from typing import Optional, Tuple, Union, Any, BinaryIO

import librosa
import librosa.core.spectrum
import numpy as np
import torch

import audioread
import os 
import soundfile as sf
import io

from batdetect2.detector import parameters

from . import wavfile

__all__ = [
    "load_audio",
    "load_audio_data",
    "generate_spectrogram",
    "pad_audio",
]


def time_to_x_coords(
    time_in_file: float,
    samplerate: float = parameters.TARGET_SAMPLERATE_HZ,
    window_duration: float = parameters.FFT_WIN_LENGTH_S,
    window_overlap: float = parameters.FFT_OVERLAP,
) -> float:
    nfft = np.floor(window_duration * samplerate)  # int() uses floor
    noverlap = np.floor(window_overlap * nfft)
    return (time_in_file * samplerate - noverlap) / (nfft - noverlap)


def x_coords_to_time(
    x_pos: int,
    samplerate: float = parameters.TARGET_SAMPLERATE_HZ,
    window_duration: float = parameters.FFT_WIN_LENGTH_S,
    window_overlap: float = parameters.FFT_OVERLAP,
) -> float:
    n_fft = np.floor(window_duration * samplerate)
    n_overlap = np.floor(window_overlap * n_fft)
    n_step = n_fft - n_overlap
    return ((x_pos * n_step) + n_overlap) / samplerate
    # return (1.0 - fft_overlap) * fft_win_length * (x_pos + 0.5)  # 0.5 is for center of temporal window


def x_coord_to_sample(
    x_pos: int,
    samplerate: float = parameters.TARGET_SAMPLERATE_HZ,
    window_duration: float = parameters.FFT_WIN_LENGTH_S,
    window_overlap: float = parameters.FFT_OVERLAP,
    resize_factor: float = parameters.RESIZE_FACTOR,
) -> int:
    n_fft = np.floor(window_duration * samplerate)
    n_overlap = np.floor(window_overlap * n_fft)
    n_step = n_fft - n_overlap
    x_pos = int(x_pos / resize_factor)
    return int((x_pos * n_step) + n_overlap)


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
    path:  Union[
        str, int, os.PathLike[Any], sf.SoundFile, audioread.AudioFile, BinaryIO
    ],
    time_exp_fact: float,
    target_samp_rate: int,
    scale: bool = False,
    max_duration: Optional[float] = None,
) -> Tuple[int, np.ndarray ]:
    """Load an audio file and resample it to the target sampling rate.

    The audio is also scaled to [-1, 1] and clipped to the maximum duration.
    Only mono files are supported.

    Args:
        path (string, int, pathlib.Path, soundfile.SoundFile, audioread object, or file-like object): path to the input file.
        target_samp_rate (int): Target sampling rate.
        scale (bool): Whether to scale the audio to [-1, 1].
        max_duration (float): Maximum duration of the audio in seconds.

    Returns:
        sampling_rate: The sampling rate of the audio.
        audio_raw: The audio signal in a numpy array.

    Raises:
        ValueError: If the audio file is stereo.

    """
    sample_rate, audio_data, _ = load_audio_data(path, time_exp_fact, target_samp_rate, scale, max_duration)
    return sample_rate, audio_data

def load_audio_data(
    path:  Union[
        str, int, os.PathLike[Any], sf.SoundFile, audioread.AudioFile, BinaryIO
    ],
    time_exp_fact: float,
    target_samp_rate: int,
    scale: bool = False,
    max_duration: Optional[float] = None,
) -> Tuple[int, np.ndarray, Union[float, int]]:
    """Load an audio file and resample it to the target sampling rate.

    The audio is also scaled to [-1, 1] and clipped to the maximum duration.
    Only mono files are supported.

    Args:
        path (string, int, pathlib.Path, soundfile.SoundFile, audioread object, or file-like object): path to the input file.
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
        audio_raw, file_sampling_rate = librosa.load(
            path,
            sr=None,
            dtype=np.float32,
        )
    
    if len(audio_raw.shape) > 1:
        raise ValueError("Currently does not handle stereo files")

    sampling_rate = file_sampling_rate * time_exp_fact

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

    return sampling_rate, audio_raw, file_sampling_rate


def compute_spectrogram_width(
    length: int,
    samplerate: int = parameters.TARGET_SAMPLERATE_HZ,
    window_duration: float = parameters.FFT_WIN_LENGTH_S,
    window_overlap: float = parameters.FFT_OVERLAP,
    resize_factor: float = parameters.RESIZE_FACTOR,
) -> int:
    n_fft = int(window_duration * samplerate)
    n_overlap = int(window_overlap * n_fft)
    n_step = n_fft - n_overlap
    width = (length - n_overlap) // n_step
    return int(width * resize_factor)


def pad_audio(
    audio: np.ndarray,
    samplerate: int = parameters.TARGET_SAMPLERATE_HZ,
    window_duration: float = parameters.FFT_WIN_LENGTH_S,
    window_overlap: float = parameters.FFT_OVERLAP,
    resize_factor: float = parameters.RESIZE_FACTOR,
    divide_factor: int = parameters.SPEC_DIVIDE_FACTOR,
    fixed_width: Optional[int] = None,
):
    """Pad audio to be evenly divisible by `divide_factor`.

    This function pads the audio signal with zeros to ensure that the
    generated spectrogram length will be evenly divisible by `divide_factor`.
    This is important for the model to work correctly.

    This `divide_factor` comes from the model architecture as it downscales
    the spectrogram by this factor, so the input must be divisible by this
    integer number.

    Parameters
    ----------
    audio : np.ndarray
        The audio signal.
    samplerate : int
        The sampling rate of the audio signal.
    window_size : float
        The window size in seconds used for the spectrogram computation.
    window_overlap : float
        The overlap between windows in the spectrogram computation.
    resize_factor : float
        This factor is used to resize the spectrogram after the STFT
        computation. Default is 0.5 which means that the spectrogram will be
        reduced by half. Important to take into account for the final size of
        the spectrogram.
    divide_factor : int
        The factor by which the spectrogram will be divided.
    fixed_width : int, optional
        If provided, the audio will be padded or cut so that the resulting
        spectrogram width will be equal to this value.

    Returns
    -------
    np.ndarray
        The padded audio signal.
    """
    spec_width = compute_spectrogram_width(
        audio.shape[0],
        samplerate=samplerate,
        window_duration=window_duration,
        window_overlap=window_overlap,
        resize_factor=resize_factor,
    )

    if fixed_width:
        target_samples = x_coord_to_sample(
            fixed_width,
            samplerate=samplerate,
            window_duration=window_duration,
            window_overlap=window_overlap,
            resize_factor=resize_factor,
        )

        if spec_width < fixed_width:
            # need to be at least min_size
            diff = target_samples - audio.shape[0]
            return np.hstack((audio, np.zeros(diff, dtype=audio.dtype)))

        if spec_width > fixed_width:
            return audio[:target_samples]

        return audio

    min_width = int(divide_factor / resize_factor)

    if spec_width < min_width:
        target_samples = x_coord_to_sample(
            min_width,
            samplerate=samplerate,
            window_duration=window_duration,
            window_overlap=window_overlap,
            resize_factor=resize_factor,
        )
        diff = target_samples - audio.shape[0]
        return np.hstack((audio, np.zeros(diff, dtype=audio.dtype)))

    if (spec_width % divide_factor) == 0:
        return audio

    target_width = int(np.ceil(spec_width / divide_factor)) * divide_factor
    target_samples = x_coord_to_sample(
        target_width,
        samplerate=samplerate,
        window_duration=window_duration,
        window_overlap=window_overlap,
        resize_factor=resize_factor,
    )
    diff = target_samples - audio.shape[0]
    return np.hstack((audio, np.zeros(diff, dtype=audio.dtype)))


def gen_mag_spectrogram(x, fs, ms, overlap_perc):
    # Computes magnitude spectrogram by specifying time.

    x = x.astype(np.float32)
    nfft = int(ms * fs)
    noverlap = int(overlap_perc * nfft)

    # window data
    step = nfft - noverlap

    # compute spec
    spec, _ = librosa.core.spectrum._spectrogram(
        y=x,
        power=1,
        n_fft=nfft,
        hop_length=step,
        center=False,
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
