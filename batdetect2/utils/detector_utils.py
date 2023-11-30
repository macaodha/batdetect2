import json
import os
from typing import Any, Iterator, List, Optional, Tuple, Union

import librosa
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import batdetect2.detector.compute_features as feats
import batdetect2.detector.post_process as pp
import batdetect2.utils.audio_utils as au
from batdetect2.detector import models
from batdetect2.detector.parameters import DEFAULT_MODEL_PATH
from batdetect2.types import (
    Annotation,
    DetectionModel,
    FileAnnotations,
    ModelOutput,
    ModelParameters,
    PredictionResults,
    ProcessingConfiguration,
    ResultParams,
    RunResults,
    SpectrogramParameters,
)

__all__ = [
    "load_model",
    "list_audio_files",
    "format_single_result",
    "save_results_to_file",
    "iterate_over_chunks",
    "process_spectrogram",
    "process_audio_array",
    "process_file",
    "get_default_bd_args",
]


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


def list_audio_files(ip_dir: str) -> List[str]:
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


def load_model(
    model_path: str = DEFAULT_MODEL_PATH,
    load_weights: bool = True,
    device: Optional[torch.device] = None,
) -> Tuple[DetectionModel, ModelParameters]:
    """Load model from file.

    Args:
        model_path (str): Path to model file. Defaults to DEFAULT_MODEL_PATH.
        load_weights (bool, optional): Load weights. Defaults to True.

    Returns:
        model, params: Model and parameters.

    Raises:
        FileNotFoundError: Model file not found.
        ValueError: Unknown model name.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.isfile(model_path):
        raise FileNotFoundError("Model file not found.")

    net_params = torch.load(model_path, map_location=device)

    params = net_params["params"]

    model: DetectionModel

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

    model = model.to(device)
    model.eval()

    return model, params


def _merge_results(predictions, spec_feats, cnn_feats, spec_slices):
    predictions_m = {
        "det_probs": np.array([]),
        "x_pos": np.array([]),
        "y_pos": np.array([]),
        "bb_widths": np.array([]),
        "bb_heights": np.array([]),
        "start_times": np.array([]),
        "end_times": np.array([]),
        "low_freqs": np.array([]),
        "high_freqs": np.array([]),
        "class_probs": np.array([]),
    }

    num_preds = np.sum([len(pp["det_probs"]) for pp in predictions])

    if num_preds > 0:
        for key in predictions[0].keys():
            predictions_m[key] = np.hstack(
                [pp[key] for pp in predictions if pp["det_probs"].shape[0] > 0]
            )

    if len(spec_feats) > 0:
        spec_feats = np.vstack(spec_feats)

    if len(cnn_feats) > 0:
        cnn_feats = np.vstack(cnn_feats)

    return predictions_m, spec_feats, cnn_feats, spec_slices


def get_annotations_from_preds(
    predictions: PredictionResults,
    class_names: List[str],
) -> List[Annotation]:
    """Get list of annotations from predictions."""
    # Get the best class prediction probability and index for each detection
    class_prob_best = predictions["class_probs"].max(0)
    class_ind_best = predictions["class_probs"].argmax(0)

    # Pack the results into a list of dictionaries
    annotations: List[Annotation] = [
        {
            "start_time": round(float(start_time), 4),
            "end_time": round(float(end_time), 4),
            "low_freq": int(low_freq),
            "high_freq": int(high_freq),
            "class": str(class_names[class_index]),
            "class_prob": round(float(class_prob), 3),
            "det_prob": round(float(det_prob), 3),
            "individual": "-1",
            "event": "Echolocation",
        }
        for (
            start_time,
            end_time,
            low_freq,
            high_freq,
            class_index,
            class_prob,
            det_prob,
        ) in zip(
            predictions["start_times"],
            predictions["end_times"],
            predictions["low_freqs"],
            predictions["high_freqs"],
            class_ind_best,
            class_prob_best,
            predictions["det_probs"],
        )
    ]
    return annotations


def format_single_result(
    file_id: str,
    time_exp: float,
    duration: float,
    predictions: PredictionResults,
    class_names: List[str],
) -> FileAnnotations:
    """Format results into the format expected by the annotation tool.

    Args:
        file_id (str): File ID.
        time_exp (float): Time expansion factor.
        duration (float): Duration of audio file.
        predictions (dict): Predictions.

    Returns:
        dict: Results in the format expected by the annotation tool.
    """
    try:
        # Get a single class prediction for the file
        class_overall = pp.overall_class_pred(
            predictions["det_probs"],
            predictions["class_probs"],
        )
        class_name = class_names[np.argmax(class_overall)]
        annotations = get_annotations_from_preds(predictions, class_names)
    except (np.AxisError, ValueError):
        # No detections
        class_overall = np.zeros(len(class_names))
        class_name = "None"
        annotations = []

    return {
        "id": file_id,
        "annotated": False,
        "issues": False,
        "notes": "Automatically generated.",
        "time_exp": time_exp,
        "duration": round(float(duration), 4),
        "annotation": annotations,
        "class_name": class_name,
    }


def convert_results(
    file_id: str,
    time_exp: float,
    duration: float,
    params: ResultParams,
    predictions,
    spec_feats,
    cnn_feats,
    spec_slices,
    nyquist_freq: Optional[float] = None,
) -> RunResults:
    """Convert results to dictionary as expected by the annotation tool.

    Args:
        file_id (str): File ID.
        time_exp (float): Time expansion factor.
        duration (float): Duration of audio file.
        params (dict): Model parameters.
        predictions (dict): Predictions.
        spec_feats (np.ndarray): Spectral features.
        cnn_feats (np.ndarray): CNN features.
        spec_slices (list): Spectrogram slices.

    Returns:
        dict: Dictionary with results.
    """

    pred_dict = format_single_result(
        file_id,
        time_exp,
        duration,
        predictions,
        params["class_names"],
    )

    # Remove high frequency detections
    if nyquist_freq is not None:
        pred_dict["annotation"] = [
            pred
            for pred in pred_dict["annotation"]
            if pred["high_freq"] <= nyquist_freq
        ]

    # combine into final results dictionary
    results: RunResults = {
        "pred_dict": pred_dict,
    }

    # add spectrogram features if they exist
    if len(spec_feats) > 0 and params["spec_features"]:
        results["spec_feats"] = spec_feats
        results["spec_feat_names"] = feats.get_feature_names()

    # add CNN features if they exist
    if len(cnn_feats) > 0 and params["cnn_features"]:
        results["cnn_feats"] = cnn_feats
        results["cnn_feat_names"] = [
            str(ii) for ii in range(cnn_feats.shape[1])
        ]

    # add spectrogram slices if they exist
    if len(spec_slices) > 0 and params["spec_slices"]:
        results["spec_slices"] = spec_slices

    return results


def save_results_to_file(results, op_path: str) -> None:
    """Save results to file.

    Will create the output directory if it does not exist.

    Args:
        results (dict): Results.
        op_path (str): Output path.
    """
    # make directory if it does not exist
    if not os.path.isdir(os.path.dirname(op_path)):
        os.makedirs(os.path.dirname(op_path))

    # save csv file - if there are predictions
    result_list = results["pred_dict"]["annotation"]

    results_df = pd.DataFrame(result_list)

    # add file name as a column
    results_df["file_name"] = results["pred_dict"]["id"]

    # rename index column
    results_df.index.name = "id"

    # create a csv file with predicted events
    if "class_prob" in results_df.columns:
        preds_df = results_df[
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
        preds_df.to_csv(op_path + ".csv", sep=",")

    if "spec_feats" in results.keys():
        # create csv file with spectrogram features
        spec_feats_df = pd.DataFrame(
            results["spec_feats"],
            columns=results["spec_feat_names"],
        )
        spec_feats_df.to_csv(
            op_path + "_spec_features.csv",
            sep=",",
            index=False,
            float_format="%.5f",
        )

    if "cnn_feats" in results.keys():
        # create csv file with cnn extracted features
        cnn_feats_df = pd.DataFrame(
            results["cnn_feats"],
            columns=results["cnn_feat_names"],
        )
        cnn_feats_df.to_csv(
            op_path + "_cnn_features.csv",
            sep=",",
            index=False,
            float_format="%.5f",
        )

    # save json file
    with open(op_path + ".json", "w", encoding="utf-8") as jsonfile:
        json.dump(results["pred_dict"], jsonfile, indent=2, sort_keys=True)


def compute_spectrogram(
    audio: np.ndarray,
    sampling_rate: int,
    params: SpectrogramParameters,
    device: torch.device,
    return_np: bool = False,
) -> Tuple[float, torch.Tensor, Optional[np.ndarray]]:
    """Compute a spectrogram from an audio array.

    Will pad the audio array so that it is evenly divisible by the
    downsampling factors.

    Parameters
    ----------
    audio : np.ndarray

    sampling_rate : int

    params : SpectrogramParameters
        The parameters to use for generating the spectrogram.

    return_np : bool, optional
        Whether to return the spectrogram as a numpy array as well as a
        torch tensor. The default is False.

    Returns
    -------
    duration : float
        The duration of the spectrgram in seconds.

    spec : torch.Tensor
        The spectrogram as a torch tensor.

    spec_np : np.ndarray, optional
        The spectrogram as a numpy array. Only returned if `return_np` is
        True, otherwise None.
    """
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
    spec = torch.from_numpy(spec).to(device)

    # add batch and channel dimensions
    spec = spec.unsqueeze(0).unsqueeze(0)

    # resize the spec
    resize_factor = params["resize_factor"]
    spec_op_shape = (
        int(params["spec_height"] * resize_factor),
        int(spec.shape[-1] * resize_factor),
    )
    spec = F.interpolate(
        spec,
        size=spec_op_shape,
        mode="bilinear",
        align_corners=False,
    )

    if return_np:
        spec_np = spec[0, 0, :].cpu().data.numpy()
    else:
        spec_np = None

    return duration, spec, spec_np


def iterate_over_chunks(
    audio: np.ndarray,
    samplerate: int,
    chunk_size: float,
) -> Iterator[Tuple[float, np.ndarray]]:
    """Iterate over audio in chunks of size chunk_size.

    Parameters
    ----------
    audio : np.ndarray

    samplerate : int

    chunk_size : float
        Size of chunks in seconds.

    Yields
    ------
    chunk_start : float
        Start time of chunk in seconds.
    chunk : np.ndarray
    """
    nsamples = audio.shape[0]
    duration_full = nsamples / samplerate
    num_chunks = int(np.ceil(duration_full / chunk_size))
    for chunk_id in range(num_chunks):
        chunk_start = chunk_size * chunk_id
        chunk_length = int(samplerate * chunk_size)
        start_sample = chunk_id * chunk_length
        end_sample = np.minimum((chunk_id + 1) * chunk_length, nsamples)
        yield chunk_start, audio[start_sample:end_sample]


def _process_spectrogram(
    spec: torch.Tensor,
    samplerate: int,
    model: DetectionModel,
    config: ProcessingConfiguration,
) -> Tuple[PredictionResults, np.ndarray]:
    # evaluate model
    with torch.no_grad():
        outputs = model(spec)

    # run non-max suppression
    pred_nms_list, features = pp.run_nms(
        outputs,
        {
            "nms_kernel_size": config["nms_kernel_size"],
            "max_freq": config["max_freq"],
            "min_freq": config["min_freq"],
            "fft_win_length": config["fft_win_length"],
            "fft_overlap": config["fft_overlap"],
            "resize_factor": config["resize_factor"],
            "nms_top_k_per_sec": config["nms_top_k_per_sec"],
            "detection_threshold": config["detection_threshold"],
        },
        np.array([float(samplerate)]),
    )

    pred_nms = pred_nms_list[0]

    # if we have a background class
    class_probs = pred_nms.get("class_probs")
    if (class_probs is not None) and (
        class_probs.shape[0] > len(config["class_names"])
    ):
        pred_nms["class_probs"] = class_probs[:-1, :]

    return pred_nms, np.concatenate(features, axis=0)


def postprocess_model_outputs(
    outputs: ModelOutput,
    samp_rate: int,
    config: ProcessingConfiguration,
) -> Tuple[List[Annotation], np.ndarray]:
    # run non-max suppression
    pred_nms_list, features = pp.run_nms(
        outputs,
        {
            "nms_kernel_size": config["nms_kernel_size"],
            "max_freq": config["max_freq"],
            "min_freq": config["min_freq"],
            "fft_win_length": config["fft_win_length"],
            "fft_overlap": config["fft_overlap"],
            "resize_factor": config["resize_factor"],
            "nms_top_k_per_sec": config["nms_top_k_per_sec"],
            "detection_threshold": config["detection_threshold"],
        },
        np.array([float(samp_rate)]),
    )

    pred_nms = pred_nms_list[0]

    # if we have a background class
    class_probs = pred_nms.get("class_probs")
    if (class_probs is not None) and (
        class_probs.shape[0] > len(config["class_names"])
    ):
        pred_nms["class_probs"] = class_probs[:-1, :]

    annotations = get_annotations_from_preds(
        pred_nms,
        config["class_names"],
    )

    return annotations, features[0]


def process_spectrogram(
    spec: torch.Tensor,
    samplerate: int,
    model: DetectionModel,
    config: ProcessingConfiguration,
) -> Tuple[List[Annotation], np.ndarray]:
    """Process a spectrogram with detection model.

    Will run non-maximum suppression on the output of the model.

    Parameters
    ----------
    spec : torch.Tensor

    samplerate : int

    model : torch.nn.Module
        Detection model.

    config : pp.NonMaximumSuppressionConfig
        Parameters for non-maximum suppression.

    Returns
    -------
    detections: List[Annotation]
        List of detections predicted by the model.
    features : np.ndarray
        An array of CNN features associated with each annotation.
        The array is of shape (num_detections, num_features).
        Is empty if `config["cnn_features"]` is False.
    """
    pred_nms, features = _process_spectrogram(
        spec,
        samplerate,
        model,
        config,
    )

    detections = get_annotations_from_preds(
        pred_nms,
        config["class_names"],
    )

    return detections, features


def _process_audio_array(
    audio: np.ndarray,
    sampling_rate: int,
    model: DetectionModel,
    config: ProcessingConfiguration,
    device: torch.device,
) -> Tuple[PredictionResults, np.ndarray, torch.Tensor]:
    # load audio file and compute spectrogram
    _, spec, _ = compute_spectrogram(
        audio,
        sampling_rate,
        {
            "fft_win_length": config["fft_win_length"],
            "fft_overlap": config["fft_overlap"],
            "spec_height": config["spec_height"],
            "resize_factor": config["resize_factor"],
            "spec_divide_factor": config["spec_divide_factor"],
            "max_freq": config["max_freq"],
            "min_freq": config["min_freq"],
            "spec_scale": config["spec_scale"],
            "denoise_spec_avg": config["denoise_spec_avg"],
            "max_scale_spec": config["max_scale_spec"],
        },
        device,
        return_np=False,
    )

    # process spectrogram with model
    pred_nms, features = _process_spectrogram(
        spec,
        sampling_rate,
        model,
        config,
    )

    return pred_nms, features, spec


def process_audio_array(
    audio: np.ndarray,
    sampling_rate: int,
    model: DetectionModel,
    config: ProcessingConfiguration,
    device: torch.device,
) -> Tuple[List[Annotation], np.ndarray, torch.Tensor]:
    """Process a single audio array with detection model.

    Parameters
    ----------
    audio : np.ndarray

    sampling_rate : int

    model : torch.nn.Module
        Detection model.

    config : ProcessingConfiguration
        Configuration for processing.

    device : torch.device
        Device to use for processing.

    Returns
    -------
    annotations : List[Annotation]
        List of annotations predicted by the model.
    features : np.ndarray
        Array of CNN features associated with each annotation.
        The array is of shape (num_detections, num_features).
    spec : torch.Tensor
        Spectrogram of the audio used as input.
    """
    pred_nms, features, spec = _process_audio_array(
        audio,
        sampling_rate,
        model,
        config,
        device,
    )

    annotations = get_annotations_from_preds(
        pred_nms,
        config["class_names"],
    )

    return annotations, features, spec


def process_file(
    audio_file: str,
    model: DetectionModel,
    config: ProcessingConfiguration,
    device: torch.device,
) -> Union[RunResults, Any]:
    """Process a single audio file with detection model.

    Will split the audio file into chunks if it is too long and
    process each chunk separately.

    Parameters
    ----------
    audio_file : str
        Path to audio file.

    model : torch.nn.Module
        Detection model.

    config : ProcessingConfiguration
        Configuration for processing.

    Returns
    -------
    results : Results or Any
        Results of processing audio file with the given detection model.
        Will be a dictionary if `config["return_raw_preds"]` is `True`,
    """
    # store temporary results here
    predictions = []
    spec_feats = []
    cnn_feats = []
    spec_slices = []

    # Get original sampling rate
    file_samp_rate = librosa.get_samplerate(audio_file)
    orig_samp_rate = file_samp_rate * config.get("time_expansion", 1) or 1

    # load audio file
    sampling_rate, audio_full = au.load_audio(
        audio_file,
        time_exp_fact=config.get("time_expansion", 1) or 1,
        target_samp_rate=config["target_samp_rate"],
        scale=config["scale_raw_audio"],
        max_duration=config.get("max_duration"),
    )

    # loop through larger file and split into chunks
    # TODO: fix so that it overlaps correctly and takes care of
    # duplicate detections at borders
    for chunk_time, audio in iterate_over_chunks(
        audio_full,
        sampling_rate,
        config["chunk_size"],
    ):
        # Run detection model on chunk
        pred_nms, features, spec = _process_audio_array(
            audio,
            sampling_rate,
            model,
            config,
            device,
        )

        # convert to numpy
        spec_np = spec.detach().cpu().numpy().squeeze()

        # add chunk time to start and end times
        pred_nms["start_times"] += chunk_time
        pred_nms["end_times"] += chunk_time

        predictions.append(pred_nms)

        # extract features - if there are any calls detected
        if pred_nms["det_probs"].shape[0] == 0:
            continue

        if config["spec_features"]:
            spec_feats.append(feats.get_feats(spec_np, pred_nms, config))

        if config["cnn_features"]:
            cnn_feats.append(features[0])

        if config["spec_slices"]:
            # FIX: This is not currently working. Returns empty slices
            spec_slices.extend(feats.extract_spec_slices(spec_np, pred_nms))

    # Merge results from chunks
    predictions, spec_feats, cnn_feats, spec_slices = _merge_results(
        predictions,
        spec_feats,
        cnn_feats,
        spec_slices,
    )

    # convert results to a dictionary in the right format
    results = convert_results(
        file_id=os.path.basename(audio_file),
        time_exp=config.get("time_expansion", 1) or 1,
        duration=audio_full.shape[0] / float(sampling_rate),
        params=config,
        predictions=predictions,
        spec_feats=spec_feats,
        cnn_feats=cnn_feats,
        spec_slices=spec_slices,
        nyquist_freq=orig_samp_rate / 2,
    )

    # summarize results
    if not config["quiet"]:
        summarize_results(results, predictions, config)

    if config["return_raw_preds"]:
        return predictions

    return results


def summarize_results(results, predictions, config):
    """Print summary of results."""
    num_detections = len(results["pred_dict"]["annotation"])
    print(f"{num_detections} call(s) detected above the threshold.")

    # print results for top n classes
    if num_detections > 0:
        class_overall = pp.overall_class_pred(
            predictions["det_probs"],
            predictions["class_probs"],
        )
        print("species name".ljust(30) + "probablity present")

        for class_index in np.argsort(class_overall)[::-1][: config["top_n"]]:
            print(
                config["class_names"][class_index].ljust(30)
                + str(round(class_overall[class_index], 3))
            )
