import argparse
import os

import bat_detect.utils.detector_utils as du


def main(args):

    print("Loading model: " + args["model_path"])
    model, params = du.load_model(args["model_path"])

    print("\nInput directory: " + args["audio_dir"])
    files = du.get_audio_files(args["audio_dir"])
    print("Number of audio files: {}".format(len(files)))
    print("\nSaving results to: " + args["ann_dir"])

    # process files
    error_files = []
    for ii, audio_file in enumerate(files):
        print("\n" + str(ii).ljust(6) + os.path.basename(audio_file))
        try:
            results = du.process_file(audio_file, model, params, args)
            if args["save_preds_if_empty"] or (
                len(results["pred_dict"]["annotation"]) > 0
            ):
                results_path = audio_file.replace(
                    args["audio_dir"], args["ann_dir"]
                )
                du.save_results_to_file(results, results_path)
        except:
            error_files.append(audio_file)
            print("Error processing file!")

    print("\nResults saved to: " + args["ann_dir"])

    if len(error_files) > 0:
        print("\nUnable to process the follow files:")
        for err in error_files:
            print("  " + err)


if __name__ == "__main__":

    info_str = (
        "\nBatDetect2 - Detection and Classification\n"
        + "  Assumes audio files are mono, not stereo.\n"
        + '  Spaces in the input paths will throw an error. Wrap in quotes "".\n'
        + "  Input files should be short in duration e.g. < 30 seconds.\n"
    )

    print(info_str)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "audio_dir", type=str, help="Input directory for audio"
    )
    parser.add_argument(
        "ann_dir",
        type=str,
        help="Output directory for where the predictions will be stored",
    )
    parser.add_argument(
        "detection_threshold",
        type=float,
        help="Cut-off probability for detector e.g. 0.1",
    )
    parser.add_argument(
        "--cnn_features",
        action="store_true",
        default=False,
        dest="cnn_features",
        help="Extracts CNN call features",
    )
    parser.add_argument(
        "--spec_features",
        action="store_true",
        default=False,
        dest="spec_features",
        help="Extracts low level call features",
    )
    parser.add_argument(
        "--time_expansion_factor",
        type=int,
        default=1,
        dest="time_expansion_factor",
        help="The time expansion factor used for all files (default is 1)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        default=False,
        dest="quiet",
        help="Minimize output printing",
    )
    parser.add_argument(
        "--save_preds_if_empty",
        action="store_true",
        default=False,
        dest="save_preds_if_empty",
        help="Save empty annotation file if no detections made.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/Net2DFast_UK_same.pth.tar",
        help="Path to trained BatDetect2 model",
    )
    args = vars(parser.parse_args())

    args["spec_slices"] = False  # used for visualization
    args[
        "chunk_size"
    ] = 2  # if files greater than this amount (seconds) they will be broken down into small chunks
    args["ann_dir"] = os.path.join(args["ann_dir"], "")

    main(args)
