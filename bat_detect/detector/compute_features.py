import numpy as np


def convert_int_to_freq(spec_ind, spec_height, min_freq, max_freq):
    spec_ind = spec_height - spec_ind
    return round(
        (spec_ind / float(spec_height)) * (max_freq - min_freq) + min_freq, 2
    )


def extract_spec_slices(spec, pred_nms, params):
    """
    Extracts spectrogram slices from spectrogram based on detected call locations.
    """

    x_pos = pred_nms["x_pos"]
    y_pos = pred_nms["y_pos"]
    bb_width = pred_nms["bb_width"]
    bb_height = pred_nms["bb_height"]
    slices = []

    # add 20% padding either side of call
    pad = bb_width * 0.2
    x_pos_pad = x_pos - pad
    bb_width_pad = bb_width + 2 * pad

    for ff in range(len(pred_nms["det_probs"])):
        x_start = int(np.maximum(0, x_pos_pad[ff]))
        x_end = int(
            np.minimum(
                spec.shape[1] - 1, np.round(x_pos_pad[ff] + bb_width_pad[ff])
            )
        )
        slices.append(spec[:, x_start:x_end].astype(np.float16))
    return slices


def get_feature_names():
    feature_names = [
        "duration",
        "low_freq_bb",
        "high_freq_bb",
        "bandwidth",
        "max_power_bb",
        "max_power",
        "max_power_first",
        "max_power_second",
        "call_interval",
    ]
    return feature_names


def get_feats(spec, pred_nms, params):
    """
    Extracts features from spectrogram based on detected call locations.
    Condsider re-extracting spectrogram for this to get better temporal resolution.

    For more possible features check out:
    https://github.com/YvesBas/Tadarida-D/blob/master/Manual_Tadarida-D.odt
    """

    x_pos = pred_nms["x_pos"]
    y_pos = pred_nms["y_pos"]
    bb_width = pred_nms["bb_width"]
    bb_height = pred_nms["bb_height"]

    feature_names = get_feature_names()
    num_detections = len(pred_nms["det_probs"])
    features = (
        np.ones((num_detections, len(feature_names)), dtype=np.float32) * -1
    )

    for ff in range(num_detections):
        x_start = int(np.maximum(0, x_pos[ff]))
        x_end = int(
            np.minimum(spec.shape[1] - 1, np.round(x_pos[ff] + bb_width[ff]))
        )
        # y low is the lowest freq but it will have a higher value due to array starting at 0 at top
        y_low = int(np.minimum(spec.shape[0] - 1, y_pos[ff]))
        y_high = int(np.maximum(0, np.round(y_pos[ff] - bb_height[ff])))
        spec_slice = spec[:, x_start:x_end]

        if spec_slice.shape[1] > 1:
            features[ff, 0] = round(
                pred_nms["end_times"][ff] - pred_nms["start_times"][ff], 5
            )
            features[ff, 1] = int(pred_nms["low_freqs"][ff])
            features[ff, 2] = int(pred_nms["high_freqs"][ff])
            features[ff, 3] = int(
                pred_nms["high_freqs"][ff] - pred_nms["low_freqs"][ff]
            )
            features[ff, 4] = int(
                convert_int_to_freq(
                    y_high + spec_slice[y_high:y_low, :].sum(1).argmax(),
                    spec.shape[0],
                    params["min_freq"],
                    params["max_freq"],
                )
            )
            features[ff, 5] = int(
                convert_int_to_freq(
                    spec_slice.sum(1).argmax(),
                    spec.shape[0],
                    params["min_freq"],
                    params["max_freq"],
                )
            )
            hlf_val = spec_slice.shape[1] // 2

            features[ff, 6] = int(
                convert_int_to_freq(
                    spec_slice[:, :hlf_val].sum(1).argmax(),
                    spec.shape[0],
                    params["min_freq"],
                    params["max_freq"],
                )
            )
            features[ff, 7] = int(
                convert_int_to_freq(
                    spec_slice[:, hlf_val:].sum(1).argmax(),
                    spec.shape[0],
                    params["min_freq"],
                    params["max_freq"],
                )
            )

            if ff > 0:
                features[ff, 8] = round(
                    pred_nms["start_times"][ff]
                    - pred_nms["start_times"][ff - 1],
                    5,
                )

    return features
