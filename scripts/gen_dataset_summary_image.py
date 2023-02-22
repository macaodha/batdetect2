"""
Loads a set of annotations corresponding to a dataset and saves an image which
is the mean spectrogram for each class.
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import viz_helpers as vz

sys.path.append(os.path.join(".."))
import bat_detect.detector.parameters as parameters
import bat_detect.train.train_split as ts
import bat_detect.train.train_utils as tu
import bat_detect.utils.audio_utils as au

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "audio_path", type=str, help="Input directory for audio"
    )
    parser.add_argument(
        "op_dir",
        type=str,
        help="Path to where single annotation json file is stored",
    )
    parser.add_argument(
        "--ann_file",
        type=str,
        help="Path to where single annotation json file is stored",
    )
    parser.add_argument(
        "--uk_split", type=str, default="", help="Set as: diff or same"
    )
    parser.add_argument(
        "--file_type",
        type=str,
        default="png",
        help="Type of image to save png or pdf",
    )
    args = vars(parser.parse_args())

    if not os.path.isdir(args["op_dir"]):
        os.makedirs(args["op_dir"])

    params = parameters.get_params(False)
    params["smooth_spec"] = False
    params["spec_width"] = 48
    params["norm_type"] = "log"  # log, pcen
    params["aud_pad"] = 0.005
    classes_to_ignore = params["classes_to_ignore"] + params["generic_class"]

    # load train annotations
    if args["uk_split"] == "":
        print("\nLoading:", args["ann_file"], "\n")
        dataset_name = os.path.basename(args["ann_file"]).replace(".json", "")
        datasets = []
        datasets.append(
            tu.get_blank_dataset_dict(
                dataset_name, False, args["ann_file"], args["audio_path"]
            )
        )
    else:
        # load uk data - special case
        print("\nLoading:", args["uk_split"], "\n")
        dataset_name = "uk_" + args["uk_split"]  # should be uk_diff, or uk_same
        datasets, _ = ts.get_train_test_data(
            args["ann_file"],
            args["audio_path"],
            args["uk_split"],
            load_extra=False,
        )

    anns, class_names, _ = tu.load_set_of_anns(
        datasets, classes_to_ignore, params["events_of_interest"]
    )
    class_names_order = range(len(class_names))

    x_train, y_train = vz.load_data(
        anns,
        params,
        class_names,
        smooth_spec=params["smooth_spec"],
        norm_type=params["norm_type"],
    )

    op_file_name = os.path.join(
        args["op_dir"], dataset_name + "." + args["file_type"]
    )
    vz.save_summary_image(
        x_train, y_train, class_names, params, op_file_name, class_names_order
    )
    print("\nImage saved to:", op_file_name)
