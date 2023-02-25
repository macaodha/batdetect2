import json
import os
from typing import Any, Iterator, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import bat_detect.detector.compute_features as feats
import bat_detect.detector.post_process as pp
import bat_detect.utils.audio_utils as au
from bat_detect.detector import models
from bat_detect.detector.parameters import (
    DENOISE_SPEC_AVG,
    DETECTION_THRESHOLD,
    FFT_OVERLAP,
    FFT_WIN_LENGTH_S,
    MAX_FREQ_HZ,
    MAX_SCALE_SPEC,
    MIN_FREQ_HZ,
    NMS_KERNEL_SIZE,
    NMS_TOP_K_PER_SEC,
    RESIZE_FACTOR,
    SCALE_RAW_AUDIO,
    SPEC_DIVIDE_FACTOR,
    SPEC_HEIGHT,
    SPEC_SCALE,
    TARGET_SAMPLERATE_HZ,
)

try:
    from typing import TypedDict
except ImportError:
    from typing_extensions import TypedDict


DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "models",
    "Net2DFast_UK_same.pth.tar",
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
    "DEFAULT_MODEL_PATH",
    "DEFAULT_PROCESSING_CONFIGURATIONS",
]


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


class ModelParameters(TypedDict):
    """Model parameters."""

    model_name: str
    """Model name."""

    num_filters: int
    """Number of filters."""

    emb_dim: int
    """Embedding dimension."""

    ip_height: int
    """Input height in pixels."""

    resize_factor: float
    """Resize factor."""

    class_names: List[str]
    """Class names. The model is trained to detect these classes."""


def load_model(
    model_path: str = DEFAULT_MODEL_PATH,
    load_weights: bool = True,
    device: Optional[torch.device] = None,
) -> Tuple[models.DetectionModel, ModelParameters]:
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
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.isfile(model_path):
        raise FileNotFoundError("Model file not found.")

    net_params = torch.load(model_path, map_location=device)

    params = net_params["params"]

    model: models.DetectionModel

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
    predictions_m = {}
    num_preds = np.sum([len(pp["det_probs"]) for pp in predictions])

    if num_preds > 0:
        for key in predictions[0].keys():
            predictions_m[key] = np.hstack(
                [pp[key] for pp in predictions if pp["det_probs"].shape[0] > 0]
            )
    else:
        # hack in case where no detected calls as we need some of the key names in dict
        predictions_m = predictions[0]

    if len(spec_feats) > 0:
        spec_feats = np.vstack(spec_feats)

    if len(cnn_feats) > 0:
        cnn_feats = np.vstack(cnn_feats)

    return predictions_m, spec_feats, cnn_feats, spec_slices


DictWithClass = TypedDict("DictWithClass", {"class": str})


class Annotation(DictWithClass):
    """Format of annotations.

    This is the format of a single annotation as  expected by the annotation
    tool.
    """

    start_time: float
    """Start time in seconds."""

    end_time: float
    """End time in seconds."""

    low_freq: int
    """Low frequency in Hz."""

    high_freq: int
    """High frequency in Hz."""

    class_prob: float
    """Probability of class assignment."""

    det_prob: float
    """Probability of detection."""

    individual: str
    """Individual ID."""

    event: str
    """Type of detected event."""


class FileAnnotations(TypedDict):
    """Format of results.

    This is the format of the results expected by the annotation tool.
    """

    id: str
    """File ID."""

    annotated: bool
    """Whether file has been annotated."""

    duration: float
    """Duration of audio file."""

    issues: bool
    """Whether file has issues."""

    time_exp: float
    """Time expansion factor."""

    class_name: str
    """Class predicted at file level"""

    notes: str
    """Notes of file."""

    annotation: List[Annotation]
    """List of annotations."""


class RunResults(TypedDict):
    """Run results."""

    pred_dict: FileAnnotations
    """Predictions in the format expected by the annotation tool."""

    spec_feats: Optional[List[np.ndarray]]
    """Spectrogram features."""

    spec_feat_names: Optional[List[str]]
    """Spectrogram feature names."""

    cnn_feats: Optional[List[np.ndarray]]
    """CNN features."""

    cnn_feat_names: Optional[List[str]]
    """CNN feature names."""

    spec_slices: Optional[List[np.ndarray]]
    """Spectrogram slices."""


class ResultParams(TypedDict):
    """Result parameters."""

    class_names: List[str]
    """Class names."""


def get_annotations_from_preds(
    predictions,
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
    predictions,
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
    # Get a single class prediction for the file
    class_overall = pp.overall_class_pred(
        predictions["det_probs"],
        predictions["class_probs"],
    )

    return {
        "id": file_id,
        "annotated": False,
        "issues": False,
        "notes": "Automatically generated.",
        "time_exp": time_exp,
        "duration": round(float(duration), 4),
        "annotation": get_annotations_from_preds(predictions, class_names),
        "class_name": class_names[np.argmax(class_overall)],
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

    # combine into final results dictionary
    results: RunResults = {
        "pred_dict": pred_dict,
        "spec_feats": None,
        "spec_feat_names": None,
        "cnn_feats": None,
        "cnn_feat_names": None,
        "spec_slices": None,
    }

    # add spectrogram features if they exist
    if len(spec_feats) > 0:
        results["spec_feats"] = spec_feats
        results["spec_feat_names"] = feats.get_feature_names()

    # add CNN features if they exist
    if len(cnn_feats) > 0:
        results["cnn_feats"] = cnn_feats
        results["cnn_feat_names"] = [
            str(ii) for ii in range(cnn_feats.shape[1])
        ]

    # add spectrogram slices if they exist
    if len(spec_slices) > 0:
        results["spec_slices"] = spec_slices

    return results


def save_results_to_file(results, op_path: str) -> None:
    """Save results to file.

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
    params: au.SpectrogramParameters,
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


class ProcessingConfiguration(TypedDict):
    """Parameters for processing audio files."""

    # audio parameters
    target_samp_rate: int
    """Target sampling rate of the audio."""

    fft_win_length: float
    """Length of the FFT window in seconds."""

    fft_overlap: float
    """Length of the FFT window in samples."""

    resize_factor: float
    """Factor to resize the spectrogram by."""

    spec_divide_factor: int
    """Factor to divide the spectrogram by."""

    spec_height: int
    """Height of the spectrogram in pixels."""

    spec_scale: str
    """Scale to use for the spectrogram."""

    denoise_spec_avg: bool
    """Whether to denoise the spectrogram by averaging."""

    max_scale_spec: bool
    """Whether to scale the spectrogram so that its max is 1."""

    scale_raw_audio: bool
    """Whether to scale the raw audio to be between -1 and 1."""

    class_names: List[str]
    """Names of the classes the model can detect."""

    detection_threshold: float
    """Threshold for detection probability."""

    time_expansion: Optional[float]
    """Time expansion factor of the processed recordings."""

    top_n: int
    """Number of top detections to keep."""

    return_raw_preds: bool
    """Whether to return raw predictions."""

    max_duration: Optional[float]
    """Maximum duration of audio file to process in seconds."""

    nms_kernel_size: int
    """Size of the kernel for non-maximum suppression."""

    max_freq: int
    """Maximum frequency to consider in Hz."""

    min_freq: int
    """Minimum frequency to consider in Hz."""

    nms_top_k_per_sec: float
    """Number of top detections to keep per second."""

    quiet: bool
    """Whether to suppress output."""

    chunk_size: float
    """Size of chunks to process in seconds."""

    cnn_features: bool
    """Whether to return CNN features."""

    spec_features: bool
    """Whether to return spectrogram features."""

    spec_slices: bool
    """Whether to return spectrogram slices."""


def _process_spectrogram(
    spec: torch.Tensor,
    samplerate: int,
    model: models.DetectionModel,
    config: ProcessingConfiguration,
) -> Tuple[List[Annotation], List[np.ndarray]]:
    # evaluate model
    with torch.no_grad():
        outputs = model(spec, return_feats=config["cnn_features"])

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
            "max_scale_spec": config["max_scale_spec"],
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

    return pred_nms, features


def process_spectrogram(
    spec: torch.Tensor,
    samplerate: int,
    model: models.DetectionModel,
    config: ProcessingConfiguration,
) -> Tuple[List[Annotation], List[np.ndarray]]:
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
    annotations : List[Annotation]
        List of annotations predicted by the model.
    features : List[np.ndarray]
        List of CNN features associated with each annotation.
        Is empty if `config["cnn_features"]` is False.
    """
    pred_nms, features = _process_spectrogram(
        spec,
        samplerate,
        model,
        config,
    )

    annotations = get_annotations_from_preds(
        pred_nms,
        config["class_names"],
    )

    return annotations, features


def _process_audio_array(
    audio: np.ndarray,
    sampling_rate: int,
    model: torch.nn.Module,
    config: ProcessingConfiguration,
    device: torch.device,
) -> Tuple[List[Annotation], List[np.ndarray], torch.Tensor]:
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
    model: torch.nn.Module,
    config: ProcessingConfiguration,
    device: torch.device,
) -> Tuple[List[Annotation], List[np.ndarray], torch.Tensor]:
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

    features : List[np.ndarray]
        List of CNN features associated with each annotation.

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
    model: torch.nn.Module,
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
        pred_nms, features, spec_np = _process_audio_array(
            audio,
            sampling_rate,
            model,
            config,
            device,
        )

        # add chunk time to start and end times
        pred_nms["start_times"] += chunk_time
        pred_nms["end_times"] += chunk_time

        predictions.append(pred_nms)

        # extract features - if there are any calls detected
        if pred_nms["det_probs"].shape[0] > 0:
            if config["spec_features"]:
                spec_feats.append(feats.get_feats(spec_np, pred_nms, config))

            if config["cnn_features"]:
                cnn_feats.append(features[0])

            if config["spec_slices"]:
                spec_slices.extend(
                    feats.extract_spec_slices(spec_np, pred_nms, config)
                )

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


DEFAULT_PROCESSING_CONFIGURATIONS: ProcessingConfiguration = {
    "detection_threshold": DETECTION_THRESHOLD,
    "spec_slices": False,
    "chunk_size": 3,
    "spec_features": False,
    "cnn_features": False,
    "quiet": True,
    "target_samp_rate": TARGET_SAMPLERATE_HZ,
    "fft_win_length": FFT_WIN_LENGTH_S,
    "fft_overlap": FFT_OVERLAP,
    "resize_factor": RESIZE_FACTOR,
    "spec_divide_factor": SPEC_DIVIDE_FACTOR,
    "spec_height": SPEC_HEIGHT,
    "scale_raw_audio": SCALE_RAW_AUDIO,
    "class_names": [],
    "time_expansion": 1,
    "top_n": 3,
    "return_raw_preds": False,
    "max_duration": None,
    "nms_kernel_size": NMS_KERNEL_SIZE,
    "max_freq": MAX_FREQ_HZ,
    "min_freq": MIN_FREQ_HZ,
    "nms_top_k_per_sec": NMS_TOP_K_PER_SEC,
    "spec_scale": SPEC_SCALE,
    "denoise_spec_avg": DENOISE_SPEC_AVG,
    "max_scale_spec": MAX_SCALE_SPEC,
}
