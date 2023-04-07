"""
This script takes an audio file as input, runs the detector, and makes a video output

Notes:
    It needs ffmpeg installed to make the videos
    Sometimes conda can overwrite the default ffmpeg path set this to use system one.
    Check which one is being used with `which ffmpeg`. If conda version, can thow an error.
    Best to use system one - see ffmpeg_path.
"""

import argparse
import os
import shutil
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.io import wavfile

import batdetect2.detector.parameters as parameters
import batdetect2.utils.audio_utils as au
import batdetect2.utils.detector_utils as du
import batdetect2.utils.plot_utils as viz

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("audio_file", type=str, help="Path to input audio file")
    parser.add_argument(
        "model_path", type=str, help="Path to trained BatDetect model"
    )
    parser.add_argument(
        "--op_dir",
        type=str,
        default="generated_vids/",
        help="Path to output directory",
    )
    parser.add_argument(
        "--no_detector", action="store_true", help="Do not run detector"
    )
    parser.add_argument(
        "--plot_class_names_off",
        action="store_true",
        help="Do not plot class names",
    )
    parser.add_argument(
        "--disable_axis", action="store_true", help="Do not plot axis"
    )
    parser.add_argument(
        "--detection_threshold",
        type=float,
        default=0.2,
        help="Cut-off probability for detector",
    )
    parser.add_argument(
        "--time_expansion_factor",
        type=int,
        default=1,
        dest="time_expansion_factor",
        help="The time expansion factor used for all files (default is 1)",
    )
    args_cmd = vars(parser.parse_args())

    # file of interest
    audio_file = args_cmd["audio_file"]
    op_dir = args_cmd["op_dir"]
    op_str = "_output"
    ffmpeg_path = "/usr/bin/"

    if not os.path.isfile(audio_file):
        print("Audio file not found: ", audio_file)
        sys.exit()

    if not os.path.isfile(args_cmd["model_path"]):
        print("Model not found: ", args_cmd["model_path"])
        sys.exit()

    start_time = 0.0
    duration = 0.5
    reveal_boxes = True  # makes the boxes appear one at a time
    fps = 24
    dpi = 100

    op_dir_tmp = os.path.join(op_dir, "op_tmp_vids", "")
    if not os.path.isdir(op_dir_tmp):
        os.makedirs(op_dir_tmp)
    if not os.path.isdir(op_dir):
        os.makedirs(op_dir)

    params = parameters.get_params(False)
    args = du.get_default_bd_args()
    args["time_expansion_factor"] = args_cmd["time_expansion_factor"]
    args["detection_threshold"] = args_cmd["detection_threshold"]

    # load audio file
    print("\nProcessing: " + os.path.basename(audio_file))
    print("\nOutput directory: " + op_dir)
    sampling_rate, audio = au.load_audio(
        audio_file, args["time_expansion_factor"], params["target_samp_rate"]
    )
    audio = audio[
        int(sampling_rate * start_time) : int(
            sampling_rate * start_time + sampling_rate * duration
        )
    ]
    audio_orig = audio.copy()
    audio = au.pad_audio(
        audio,
        sampling_rate,
        params["fft_win_length"],
        params["fft_overlap"],
        params["resize_factor"],
        params["spec_divide_factor"],
    )

    # generate spectrogram
    spec, _ = au.generate_spectrogram(audio, sampling_rate, params, True)
    max_val = spec.max() * 1.1

    if not args_cmd["no_detector"]:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print("  Loading model and running detector on entire file ...")
        model, det_params = du.load_model(args_cmd["model_path"])
        det_params["detection_threshold"] = args["detection_threshold"]

        run_config = {
            **det_params,
            **args,
        }
        results = du.process_file(
            audio_file,
            model,
            run_config,
            device,
        )

        print("  Processing detections and plotting ...")
        detections = []
        for bb in results["pred_dict"]["annotation"]:
            if (bb["start_time"] >= start_time) and (
                bb["end_time"] < start_time + duration
            ):
                detections.append(bb)

        # plot boxes
        fig = plt.figure(
            1, figsize=(spec.shape[1] / dpi, spec.shape[0] / dpi), dpi=dpi
        )
        duration = au.x_coords_to_time(
            spec.shape[1],
            sampling_rate,
            params["fft_win_length"],
            params["fft_overlap"],
        )
        viz.create_box_image(
            spec,
            fig,
            detections,
            start_time,
            start_time + duration,
            duration,
            params,
            max_val,
            plot_class_names=not args_cmd["plot_class_names_off"],
        )
        op_im_file_boxes = os.path.join(
            op_dir, os.path.basename(audio_file)[:-4] + op_str + "_boxes.png"
        )
        fig.savefig(op_im_file_boxes, dpi=dpi)
        plt.close(1)
        spec_with_boxes = plt.imread(op_im_file_boxes)

    print("  Saving audio file ...")
    if args["time_expansion_factor"] == 1:
        sampling_rate_op = int(sampling_rate / 10.0)
    else:
        sampling_rate_op = sampling_rate
    op_audio_file = os.path.join(
        op_dir, os.path.basename(audio_file)[:-4] + op_str + ".wav"
    )
    wavfile.write(op_audio_file, sampling_rate_op, audio_orig)

    print("  Saving image ...")
    op_im_file = os.path.join(
        op_dir, os.path.basename(audio_file)[:-4] + op_str + ".png"
    )
    plt.imsave(op_im_file, spec, vmin=0, vmax=max_val, cmap="plasma")
    spec_blank = plt.imread(op_im_file)

    # create figure
    freq_scale = 1000  # turn Hz to kHz
    min_freq = params["min_freq"] // freq_scale
    max_freq = params["max_freq"] // freq_scale
    y_extent = [0, duration, min_freq, max_freq]

    print("  Saving video frames ...")
    # save images that will be combined into video
    # will either plot with or without boxes
    for ii, col in enumerate(
        np.linspace(0, spec.shape[1] - 1, int(fps * duration * 10))
    ):
        if not args_cmd["no_detector"]:
            spec_op = spec_with_boxes.copy()
            if ii > 0:
                spec_op[:, int(col), :] = 1.0
                if reveal_boxes:
                    spec_op[:, int(col) + 1 :, :] = spec_blank[
                        :, int(col) + 1 :, :
                    ]
            elif ii == 0 and reveal_boxes:
                spec_op = spec_blank

            if not args_cmd["disable_axis"]:
                plt.close("all")
                fig = plt.figure(
                    ii,
                    figsize=(
                        1.2 * (spec_op.shape[1] / dpi),
                        1.5 * (spec_op.shape[0] / dpi),
                    ),
                    dpi=dpi,
                )
                plt.xlabel("Time - seconds")
                plt.ylabel("Frequency - kHz")
                plt.imshow(
                    spec_op,
                    vmin=0,
                    vmax=1.0,
                    cmap="plasma",
                    extent=y_extent,
                    aspect="auto",
                )
                plt.tight_layout()
                fig.savefig(op_dir_tmp + str(ii).zfill(4) + ".png", dpi=dpi)
            else:
                plt.imsave(
                    op_dir_tmp + str(ii).zfill(4) + ".png",
                    spec_op,
                    vmin=0,
                    vmax=1.0,
                    cmap="plasma",
                )
        else:
            spec_op = spec.copy()
            if ii > 0:
                spec_op[:, int(col)] = max_val
            plt.imsave(
                op_dir_tmp + str(ii).zfill(4) + ".png",
                spec_op,
                vmin=0,
                vmax=max_val,
                cmap="plasma",
            )

    print("  Creating video ...")
    op_vid_file = os.path.join(
        op_dir, os.path.basename(audio_file)[:-4] + op_str + ".avi"
    )
    ffmpeg_cmd = (
        "ffmpeg -hide_banner -loglevel panic -y -r {} -f image2 -s {}x{} -i {}%04d.png -i {} -vcodec libx264 "
        "-crf 25  -pix_fmt yuv420p -acodec copy {}".format(
            fps,
            spec.shape[1],
            spec.shape[0],
            op_dir_tmp,
            op_audio_file,
            op_vid_file,
        )
    )
    ffmpeg_cmd = ffmpeg_path + ffmpeg_cmd
    os.system(ffmpeg_cmd)

    print("  Deleting temporary files ...")
    if os.path.isdir(op_dir_tmp):
        shutil.rmtree(op_dir_tmp)
