import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

sys.path.append(os.path.join(".."))

import bat_detect.utils.audio_utils as au


def generate_spectrogram_data(
    audio, sampling_rate, params, norm_type="log", smooth_spec=False
):
    max_freq = round(params["max_freq"] * params["fft_win_length"])
    min_freq = round(params["min_freq"] * params["fft_win_length"])

    # create spectrogram - numpy
    spec = au.gen_mag_spectrogram(
        audio, sampling_rate, params["fft_win_length"], params["fft_overlap"]
    )
    # spec = au.gen_mag_spectrogram_pt(audio, sampling_rate, params['fft_win_length'], params['fft_overlap']).numpy()
    if spec.shape[0] < max_freq:
        freq_pad = max_freq - spec.shape[0]
        spec = np.vstack(
            (np.zeros((freq_pad, spec.shape[1]), dtype=np.float32), spec)
        )
    spec = spec[-max_freq : spec.shape[0] - min_freq, :]

    if norm_type == "log":
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
        ##log_scaling = 0.01
        spec = np.log(1.0 + log_scaling * spec).astype(np.float32)
    elif norm_type == "pcen":
        spec = au.pcen(spec, sampling_rate)
    else:
        pass

    if smooth_spec:
        spec = ndimage.gaussian_filter(spec, 1)

    return spec


def load_data(
    anns,
    params,
    class_names,
    smooth_spec=False,
    norm_type="log",
    extract_bg=False,
):
    specs = []
    labels = []
    coords = []
    audios = []
    sampling_rates = []
    file_names = []
    for cur_file in anns:
        sampling_rate, audio_orig = au.load_audio_file(
            cur_file["file_path"],
            cur_file["time_exp"],
            params["target_samp_rate"],
            params["scale_raw_audio"],
        )

        for ann in cur_file["annotation"]:
            if (
                ann["class"] not in params["classes_to_ignore"]
                and ann["class"] in class_names
            ):
                # clip out of bounds
                if ann["low_freq"] < params["min_freq"]:
                    ann["low_freq"] = params["min_freq"]
                if ann["high_freq"] > params["max_freq"]:
                    ann["high_freq"] = params["max_freq"]

                # load cropped audio
                start_samp_diff = int(sampling_rate * ann["start_time"]) - int(
                    sampling_rate * params["aud_pad"]
                )
                start_samp = np.maximum(0, start_samp_diff)
                end_samp = np.minimum(
                    audio_orig.shape[0],
                    int(sampling_rate * ann["end_time"]) * 2
                    + int(sampling_rate * params["aud_pad"]),
                )
                audio = audio_orig[start_samp:end_samp]
                if start_samp_diff < 0:
                    # need to pad at start if the call is at the very begining
                    audio = np.hstack(
                        (np.zeros(-start_samp_diff, dtype=np.float32), audio)
                    )

                nfft = int(params["fft_win_length"] * sampling_rate)
                noverlap = int(params["fft_overlap"] * nfft)
                max_samps = params["spec_width"] * (nfft - noverlap) + noverlap

                if max_samps > audio.shape[0]:
                    audio = np.hstack(
                        (audio, np.zeros(max_samps - audio.shape[0]))
                    )
                audio = audio[:max_samps].astype(np.float32)

                audio = au.pad_audio(
                    audio,
                    sampling_rate,
                    params["fft_win_length"],
                    params["fft_overlap"],
                    params["resize_factor"],
                    params["spec_divide_factor"],
                )

                # generate spectrogram
                spec = generate_spectrogram_data(
                    audio, sampling_rate, params, norm_type, smooth_spec
                )[:, : params["spec_width"]]

                specs.append(spec[np.newaxis, ...])
                labels.append(ann["class"])

                audios.append(audio)
                sampling_rates.append(sampling_rate)
                file_names.append(cur_file["file_path"])

                # position in crop
                x1 = int(
                    au.time_to_x_coords(
                        np.array(params["aud_pad"]),
                        sampling_rate,
                        params["fft_win_length"],
                        params["fft_overlap"],
                    )
                )
                y1 = (ann["low_freq"] - params["min_freq"]) * params[
                    "fft_win_length"
                ]
                coords.append((y1, x1))

    _, file_ids = np.unique(file_names, return_inverse=True)
    labels = np.array([class_names.index(ll) for ll in labels])

    # return np.vstack(specs), labels, coords, audios, sampling_rates, file_ids, file_names
    return np.vstack(specs), labels


def save_summary_image(
    specs,
    labels,
    species_names,
    params,
    op_file_name="plots/all_species.png",
    order=None,
):
    # takes the mean for each class and plots it on a grid
    mean_specs = []
    max_band = []
    for ii in range(len(species_names)):
        inds = np.where(labels == ii)[0]
        mu = specs[inds, :].mean(0)
        max_band.append(np.argmax(mu.sum(1)))
        mean_specs.append(mu)

    # control the order in which classes are printed
    if order is None:
        order = np.arange(len(species_names))

    max_cols = 6
    nrows = int(np.ceil(len(species_names) / max_cols))
    ncols = np.minimum(len(species_names), max_cols)

    fig, ax = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(ncols * 3.3, nrows * 6),
        gridspec_kw={"wspace": 0, "hspace": 0.2},
    )
    spec_min_max = (
        0,
        mean_specs[0].shape[1],
        params["min_freq"] / 1000,
        params["max_freq"] / 1000,
    )
    ii = 0
    for row in ax:

        if type(row) != np.ndarray:
            row = np.array([row])

        for col in row:
            if ii >= len(species_names):
                col.axis("off")
            else:
                inds = np.where(labels == order[ii])[0]
                col.imshow(
                    mean_specs[order[ii]],
                    extent=spec_min_max,
                    cmap="plasma",
                    aspect="equal",
                )
                col.grid(color="w", alpha=0.3, linewidth=0.3)
                col.set_xticks([])
                col.title.set_text(str(ii + 1) + " " + species_names[order[ii]])
                col.tick_params(axis="both", which="major", labelsize=7)
                ii += 1

    # plt.tight_layout()
    # plt.show()
    plt.savefig(op_file_name)
    plt.close("all")
