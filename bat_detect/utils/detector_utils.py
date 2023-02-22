import json
import os
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import bat_detect.detector.compute_features as feats
import bat_detect.detector.post_process as pp
import bat_detect.utils.audio_utils as au
from bat_detect.detector import models

try:
    from typing import TypedDict
except ImportError:
    from typing_extensions import TypedDict


DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "models",
    "model.pth",
)

__all__ = ["load_model", "DEFAULT_MODEL_PATH"]


def get_default_bd_args():
    args = {}
    args["detection_threshold"] = 0.001
    args["time_expansion_factor"] = 1
    args["audio_dir"] = ""
    args["ann_dir"] = ""
    args["spec_slices"] = False
    args["chunk_size"] = 3
    args["spec_features"] = False
    args["cnn_features"] = False
    args["quiet"] = True
    args["save_preds_if_empty"] = True
    args["ann_dir"] = os.path.join(args["ann_dir"], "")
    return args


def get_audio_files(ip_dir: str) -> List[str]:
    """Get all audio files in directory.

    Args:
        ip_dir (str): Input directory.

    Returns:
        list: List of audio files. Only .wav files are returned. Paths are
        relative to ip_dir.

    Raises:
        FileNotFoundError: Input directory not found.

    """
    matches = []
    for root, _, filenames in os.walk(ip_dir):
        for filename in filenames:
            if filename.lower().endswith(".wav"):
                matches.append(os.path.join(root, filename))
    return matches


class ModelParameters(TypedDict):
    """Model parameters."""
    model_name: str
    num_filters: int
    emb_dim: int
    ip_height: int
    resize_factor: int
    class_names: List[str]
    device: torch.device


def load_model(
    model_path: str=DEFAULT_MODEL_PATH,
    load_weights: bool=True
) -> Tuple[torch.nn.Module, ModelParameters]:
    """Load model from file.

    Args:
        model_path (str): Path to model file. Defaults to DEFAULT_MODEL_PATH.
        load_weights (bool, optional): Load weights. Defaults to True.

    Returns:
        model, params: Model and parameters.

    Raises:
        FileNotFoundError: Model file not found.
        ValueError: Unknown model.
    """

    # load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.isfile(model_path):
        raise FileNotFoundError("Model file not found.")

    net_params = torch.load(model_path, map_location=device)

    params = net_params["params"]
    params["device"] = device

    if params["model_name"] == "Net2DFast":
        model = models.Net2DFast(
            params["num_filters"],
            num_classes=len(params["class_names"]),
            emb_dim=params["emb_dim"],
            ip_height=params["ip_height"],
            resize_factor=params["resize_factor"],
        )
    elif params["model_name"] == "Net2DFastNoAttn":
        model = models.Net2DFastNoAttn(
            params["num_filters"],
            num_classes=len(params["class_names"]),
            emb_dim=params["emb_dim"],
            ip_height=params["ip_height"],
            resize_factor=params["resize_factor"],
        )
    elif params["model_name"] == "Net2DFastNoCoordConv":
        model = models.Net2DFastNoCoordConv(
            params["num_filters"],
            num_classes=len(params["class_names"]),
            emb_dim=params["emb_dim"],
            ip_height=params["ip_height"],
            resize_factor=params["resize_factor"],
        )
    else:
        raise ValueError("Unknown model.")

    if load_weights:
        model.load_state_dict(net_params["state_dict"])

    model = model.to(params["device"])
    model.eval()

    return model, params


def merge_results(predictions, spec_feats, cnn_feats, spec_slices):

    predictions_m = {}
    num_preds = np.sum([len(pp["det_probs"]) for pp in predictions])

    if num_preds > 0:
        for kk in predictions[0].keys():
            predictions_m[kk] = np.hstack(
                [pp[kk] for pp in predictions if pp["det_probs"].shape[0] > 0]
            )
    else:
        # hack in case where no detected calls as we need some of the key names in dict
        predictions_m = predictions[0]

    if len(spec_feats) > 0:
        spec_feats = np.vstack(spec_feats)
    if len(cnn_feats) > 0:
        cnn_feats = np.vstack(cnn_feats)
    return predictions_m, spec_feats, cnn_feats, spec_slices


def convert_results(
    file_id,
    time_exp,
    duration,
    params,
    predictions,
    spec_feats,
    cnn_feats,
    spec_slices,
):

    # create a single dictionary - this is the format used by the annotation tool
    pred_dict = {}
    pred_dict["id"] = file_id
    pred_dict["annotated"] = False
    pred_dict["issues"] = False
    pred_dict["notes"] = "Automatically generated."
    pred_dict["time_exp"] = time_exp
    pred_dict["duration"] = round(duration, 4)
    pred_dict["annotation"] = []

    class_prob_best = predictions["class_probs"].max(0)
    class_ind_best = predictions["class_probs"].argmax(0)
    class_overall = pp.overall_class_pred(
        predictions["det_probs"], predictions["class_probs"]
    )
    pred_dict["class_name"] = params["class_names"][np.argmax(class_overall)]

    for ii in range(predictions["det_probs"].shape[0]):
        res = {}
        res["start_time"] = round(float(predictions["start_times"][ii]), 4)
        res["end_time"] = round(float(predictions["end_times"][ii]), 4)
        res["low_freq"] = int(predictions["low_freqs"][ii])
        res["high_freq"] = int(predictions["high_freqs"][ii])
        res["class"] = str(params["class_names"][int(class_ind_best[ii])])
        res["class_prob"] = round(float(class_prob_best[ii]), 3)
        res["det_prob"] = round(float(predictions["det_probs"][ii]), 3)
        res["individual"] = "-1"
        res["event"] = "Echolocation"
        pred_dict["annotation"].append(res)

    # combine into final results dictionary
    results = {}
    results["pred_dict"] = pred_dict
    if len(spec_feats) > 0:
        results["spec_feats"] = spec_feats
        results["spec_feat_names"] = feats.get_feature_names()
    if len(cnn_feats) > 0:
        results["cnn_feats"] = cnn_feats
        results["cnn_feat_names"] = [str(ii) for ii in range(cnn_feats.shape[1])]
    if len(spec_slices) > 0:
        results["spec_slices"] = spec_slices

    return results


def save_results_to_file(results, op_path):

    # make directory if it does not exist
    if not os.path.isdir(os.path.dirname(op_path)):
        os.makedirs(os.path.dirname(op_path))

    # save csv file - if there are predictions
    result_list = [res for res in results["pred_dict"]["annotation"]]
    df = pd.DataFrame(result_list)
    df["file_name"] = [results["pred_dict"]["id"]] * len(result_list)
    df.index.name = "id"
    if "class_prob" in df.columns:
        df = df[
            [
                "det_prob",
                "start_time",
                "end_time",
                "high_freq",
                "low_freq",
                "class",
                "class_prob",
            ]
        ]
        df.to_csv(op_path + ".csv", sep=",")

    # save features
    if "spec_feats" in results.keys():
        df = pd.DataFrame(results["spec_feats"], columns=results["spec_feat_names"])
        df.to_csv(
            op_path + "_spec_features.csv",
            sep=",",
            index=False,
            float_format="%.5f",
        )

    if "cnn_feats" in results.keys():
        df = pd.DataFrame(results["cnn_feats"], columns=results["cnn_feat_names"])
        df.to_csv(
            op_path + "_cnn_features.csv",
            sep=",",
            index=False,
            float_format="%.5f",
        )

    # save json file
    with open(op_path + ".json", "w") as da:
        json.dump(results["pred_dict"], da, indent=2, sort_keys=True)


def compute_spectrogram(audio, sampling_rate, params, return_np=False):
    # pad audio so it is evenly divisible by downsampling factors
    duration = audio.shape[0] / float(sampling_rate)
    audio = au.pad_audio(
        audio,
        sampling_rate,
        params["fft_win_length"],
        params["fft_overlap"],
        params["resize_factor"],
        params["spec_divide_factor"],
    )

    # generate spectrogram
    spec, _ = au.generate_spectrogram(audio, sampling_rate, params)

    # convert to pytorch
    spec = torch.from_numpy(spec).to(params["device"])
    spec = spec.unsqueeze(0).unsqueeze(0)

    # resize the spec
    rs = params["resize_factor"]
    spec_op_shape = (int(params["spec_height"] * rs), int(spec.shape[-1] * rs))
    spec = F.interpolate(spec, size=spec_op_shape, mode="bilinear", align_corners=False)

    if return_np:
        spec_np = spec[0, 0, :].cpu().data.numpy()
    else:
        spec_np = None

    return duration, spec, spec_np


def process_file(
    audio_file,
    model,
    params,
    args,
    time_exp=None,
    top_n=5,
    return_raw_preds=False,
    max_duration=False,
):

    # store temporary results here
    predictions = []
    spec_feats = []
    cnn_feats = []
    spec_slices = []

    # get time expansion  factor
    if time_exp is None:
        time_exp = args["time_expansion_factor"]

    params["detection_threshold"] = args["detection_threshold"]

    # load audio file
    sampling_rate, audio_full = au.load_audio_file(
        audio_file,
        time_exp,
        params["target_samp_rate"],
        params["scale_raw_audio"],
    )

    # clipping maximum duration
    if max_duration is not False:
        max_duration = np.minimum(
            int(sampling_rate * max_duration), audio_full.shape[0]
        )
        audio_full = audio_full[:max_duration]

    duration_full = audio_full.shape[0] / float(sampling_rate)

    return_np_spec = args["spec_features"] or args["spec_slices"]

    # loop through larger file and split into chunks
    # TODO fix so that it overlaps correctly and takes care of duplicate detections at borders
    num_chunks = int(np.ceil(duration_full / args["chunk_size"]))
    for chunk_id in range(num_chunks):

        # chunk
        chunk_time = args["chunk_size"] * chunk_id
        chunk_length = int(sampling_rate * args["chunk_size"])
        start_sample = chunk_id * chunk_length
        end_sample = np.minimum((chunk_id + 1) * chunk_length, audio_full.shape[0])
        audio = audio_full[start_sample:end_sample]

        # load audio file and compute spectrogram
        duration, spec, spec_np = compute_spectrogram(
            audio, sampling_rate, params, return_np_spec
        )

        # evaluate model
        with torch.no_grad():
            outputs = model(spec, return_feats=args["cnn_features"])

        # run non-max suppression
        pred_nms, features = pp.run_nms(
            outputs, params, np.array([float(sampling_rate)])
        )
        pred_nms = pred_nms[0]
        pred_nms["start_times"] += chunk_time
        pred_nms["end_times"] += chunk_time

        # if we have a background class
        if pred_nms["class_probs"].shape[0] > len(params["class_names"]):
            pred_nms["class_probs"] = pred_nms["class_probs"][:-1, :]

        predictions.append(pred_nms)

        # extract features - if there are any calls detected
        if pred_nms["det_probs"].shape[0] > 0:
            if args["spec_features"]:
                spec_feats.append(feats.get_feats(spec_np, pred_nms, params))

            if args["cnn_features"]:
                cnn_feats.append(features[0])

            if args["spec_slices"]:
                spec_slices.extend(feats.extract_spec_slices(spec_np, pred_nms, params))

    # convert the predictions into output dictionary
    file_id = os.path.basename(audio_file)
    predictions, spec_feats, cnn_feats, spec_slices = merge_results(
        predictions, spec_feats, cnn_feats, spec_slices
    )
    results = convert_results(
        file_id,
        time_exp,
        duration_full,
        params,
        predictions,
        spec_feats,
        cnn_feats,
        spec_slices,
    )

    # summarize results
    if not args["quiet"]:
        num_detections = len(results["pred_dict"]["annotation"])
        print("{}".format(num_detections) + " call(s) detected above the threshold.")

    # print results for top n classes
    if not args["quiet"] and (num_detections > 0):
        class_overall = pp.overall_class_pred(
            predictions["det_probs"], predictions["class_probs"]
        )
        print("species name".ljust(30) + "probablity present")
        for cc in np.argsort(class_overall)[::-1][:top_n]:
            print(
                params["class_names"][cc].ljust(30) + str(round(class_overall[cc], 3))
            )

    if return_raw_preds:
        return predictions
    else:
        return results
