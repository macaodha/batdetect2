"""
Visualize predctions on top of spectrogram.

Will save images with:
1) raw spectrogram
2) spectrogram with GT boxes
3) spectrogram with predicted boxes
"""

import argparse
import json
import os
import sys

import torch
import matplotlib.pyplot as plt
import numpy as np

import batdetect2.evaluate.evaluate_models as evlm
import batdetect2.utils.audio_utils as au
import batdetect2.utils.detector_utils as du
import batdetect2.utils.plot_utils as viz


def filter_anns(anns, start_time, stop_time):
    anns_op = []
    for aa in anns:
        if (aa["start_time"] >= start_time) and (
            aa["start_time"] < stop_time - 0.02
        ):
            anns_op.append(aa)
    return anns_op


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("audio_file", type=str, help="Path to audio file")
    parser.add_argument("model_path", type=str, help="Path to BatDetect model")
    parser.add_argument(
        "--ann_file", type=str, default="", help="Path to annotation file"
    )
    parser.add_argument(
        "--op_dir",
        type=str,
        default="plots/",
        help="Output directory for plots",
    )
    parser.add_argument(
        "--file_type",
        type=str,
        default="png",
        help="Type of image to save png or pdf",
    )
    parser.add_argument(
        "--title_text",
        type=str,
        default="",
        help="Text to add as title of plots",
    )
    parser.add_argument(
        "--detection_threshold",
        type=float,
        default=0.2,
        help="Threshold for output detections",
    )
    parser.add_argument(
        "--start_time",
        type=float,
        default=0.0,
        help="Start time for cropped file",
    )
    parser.add_argument(
        "--stop_time",
        type=float,
        default=0.5,
        help="End time for cropped file",
    )
    parser.add_argument(
        "--time_expansion_factor",
        type=int,
        default=1,
        help="Time expansion factor",
    )

    args_cmd = vars(parser.parse_args())

    # load the model
    bd_args = du.get_default_bd_args()
    model, params_bd = du.load_model(args_cmd["model_path"])
    bd_args["detection_threshold"] = args_cmd["detection_threshold"]
    bd_args["time_expansion_factor"] = args_cmd["time_expansion_factor"]

    # load the annotation if it exists
    gt_present = False
    if args_cmd["ann_file"] != "":
        if os.path.isfile(args_cmd["ann_file"]):
            with open(args_cmd["ann_file"]) as da:
                gt_anns = json.load(da)
            gt_anns = filter_anns(
                gt_anns["annotation"],
                args_cmd["start_time"],
                args_cmd["stop_time"],
            )
            gt_present = True
        else:
            print("Annotation file not found: ", args_cmd["ann_file"])

    # load the audio file
    if not os.path.isfile(args_cmd["audio_file"]):
        print("Audio file not found: ", args_cmd["audio_file"])
        sys.exit()

    # load audio and crop
    print("\nProcessing: " + os.path.basename(args_cmd["audio_file"]))
    print("\nOutput directory: " + args_cmd["op_dir"])
    sampling_rate, audio = au.load_audio(
        args_cmd["audio_file"],
        args_cmd["time_exp"],
        params_bd["target_samp_rate"],
        params_bd["scale_raw_audio"],
    )
    st_samp = int(sampling_rate * args_cmd["start_time"])
    en_samp = int(sampling_rate * args_cmd["stop_time"])
    if en_samp > audio.shape[0]:
        audio = np.hstack(
            (audio, np.zeros((en_samp) - audio.shape[0], dtype=audio.dtype))
        )
    audio = audio[st_samp:en_samp]

    duration = audio.shape[0] / sampling_rate
    print("File duration: {} seconds".format(duration))

    # create spec for viz
    spec, _ = au.generate_spectrogram(
        audio, sampling_rate, params_bd, True, False
    )

    run_config = {
        **params_bd,
        **bd_args,
    }

    # run model and filter detections so only keep ones in relevant time range
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = du.process_file(
        args_cmd["audio_file"], model, run_config, device
    )
    pred_anns = filter_anns(
        results["pred_dict"]["annotation"],
        args_cmd["start_time"],
        args_cmd["stop_time"],
    )
    print(len(pred_anns), "Detections")

    # save output
    if not os.path.isdir(args_cmd["op_dir"]):
        os.makedirs(args_cmd["op_dir"])

    # create output file names
    op_path_clean = (
        os.path.basename(args_cmd["audio_file"])[:-4]
        + "_clean."
        + args_cmd["file_type"]
    )
    op_path_clean = os.path.join(args_cmd["op_dir"], op_path_clean)
    op_path_pred = (
        os.path.basename(args_cmd["audio_file"])[:-4]
        + "_pred."
        + args_cmd["file_type"]
    )
    op_path_pred = os.path.join(args_cmd["op_dir"], op_path_pred)

    # create and save iamges
    viz.save_ann_spec(
        op_path_clean,
        spec,
        params_bd["min_freq"],
        params_bd["max_freq"],
        duration,
        args_cmd["start_time"],
        "",
        None,
    )
    viz.save_ann_spec(
        op_path_pred,
        spec,
        params_bd["min_freq"],
        params_bd["max_freq"],
        duration,
        args_cmd["start_time"],
        "",
        pred_anns,
    )

    if gt_present:
        op_path_gt = (
            os.path.basename(args_cmd["audio_file"])[:-4]
            + "_gt."
            + args_cmd["file_type"]
        )
        op_path_gt = os.path.join(args_cmd["op_dir"], op_path_gt)
        viz.save_ann_spec(
            op_path_gt,
            spec,
            params_bd["min_freq"],
            params_bd["max_freq"],
            duration,
            args_cmd["start_time"],
            "",
            gt_anns,
        )
