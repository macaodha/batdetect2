import argparse
import json
import os

import numpy as np

import batdetect2.train.train_utils as tu


def print_dataset_stats(data, split_name, classes_to_ignore):
    print("\nSplit:", split_name)
    print("Num files:", len(data))

    class_cnts = {}
    for dd in data:
        for aa in dd["annotation"]:
            if aa["class"] not in classes_to_ignore:
                if aa["class"] in class_cnts:
                    class_cnts[aa["class"]] += 1
                else:
                    class_cnts[aa["class"]] = 1

    if len(class_cnts) == 0:
        class_names = []
    else:
        class_names = np.sort([*class_cnts]).tolist()
        print("Class count:")
        str_len = np.max([len(cc) for cc in class_names]) + 5

        for ii, cc in enumerate(class_names):
            print(str(ii).ljust(5) + cc.ljust(str_len) + str(class_cnts[cc]))

    return class_names


def load_file_names(file_name):
    if os.path.isfile(file_name):
        with open(file_name) as da:
            files = [line.rstrip() for line in da.readlines()]
        for ff in files:
            if ff.lower()[-3:] != "wav":
                print("Error: Filenames need to end in .wav - ", ff)
                assert False
    else:
        print("Error: Input file not found - ", file_name)
        assert False

    return files


if __name__ == "__main__":
    info_str = "\nBatDetect - Prepare Data for Finetuning\n"

    print(info_str)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset_name", type=str, help="Name to call your dataset"
    )
    parser.add_argument("audio_dir", type=str, help="Input directory for audio")
    parser.add_argument(
        "ann_dir",
        type=str,
        help="Input directory for where the audio annotations are stored",
    )
    parser.add_argument(
        "op_dir",
        type=str,
        help="Path where the train and test splits will be stored",
    )
    parser.add_argument(
        "--percent_val",
        type=float,
        default=0.20,
        help="Hold out this much data for validation. Should be number between 0 and 1",
    )
    parser.add_argument(
        "--rand_seed",
        type=int,
        default=2001,
        help="Random seed used for creating the validation split",
    )
    parser.add_argument(
        "--train_file",
        type=str,
        default="",
        help="Text file where each line is a wav file in train split",
    )
    parser.add_argument(
        "--test_file",
        type=str,
        default="",
        help="Text file where each line is a wav file in test split",
    )
    parser.add_argument(
        "--input_class_names",
        type=str,
        default="",
        help='Specify names of classes that you want to change. Separate with ";"',
    )
    parser.add_argument(
        "--output_class_names",
        type=str,
        default="",
        help='New class names to use instead. One to one mapping with "--input_class_names". \
                         Separate with ";"',
    )
    args = vars(parser.parse_args())

    np.random.seed(args["rand_seed"])

    classes_to_ignore = ["", " ", "Unknown", "Not Bat"]
    generic_class = ["Bat"]
    events_of_interest = ["Echolocation"]

    if args["input_class_names"] != "" and args["output_class_names"] != "":
        # change the names of the classes
        ip_names = args["input_class_names"].split(";")
        op_names = args["output_class_names"].split(";")
        name_dict = dict(zip(ip_names, op_names))
    else:
        name_dict = False

    # load annotations
    data_all, _, _ = tu.load_set_of_anns(
        {"ann_path": args["ann_dir"], "wav_path": args["audio_dir"]},
        classes_to_ignore,
        events_of_interest,
        False,
        False,
        list_of_anns=True,
        filter_issues=True,
        name_replace=name_dict,
    )

    print("Dataset name:         " + args["dataset_name"])
    print("Audio directory:      " + args["audio_dir"])
    print("Annotation directory: " + args["ann_dir"])
    print("Ouput directory:      " + args["op_dir"])
    print("Num annotated files:  " + str(len(data_all)))

    if args["train_file"] != "" and args["test_file"] != "":
        # user has specifed the train / test split
        train_files = load_file_names(args["train_file"])
        test_files = load_file_names(args["test_file"])
        file_names_all = [dd["id"] for dd in data_all]
        train_inds = [
            file_names_all.index(ff)
            for ff in train_files
            if ff in file_names_all
        ]
        test_inds = [
            file_names_all.index(ff)
            for ff in test_files
            if ff in file_names_all
        ]

    else:
        # split the data into train and test at the file level
        num_exs = len(data_all)
        test_inds = np.random.choice(
            np.arange(num_exs),
            int(num_exs * args["percent_val"]),
            replace=False,
        )
        test_inds = np.sort(test_inds)
        train_inds = np.setdiff1d(np.arange(num_exs), test_inds)

    data_train = [data_all[ii] for ii in train_inds]
    data_test = [data_all[ii] for ii in test_inds]

    if not os.path.isdir(args["op_dir"]):
        os.makedirs(args["op_dir"])
    op_name = os.path.join(args["op_dir"], args["dataset_name"])
    op_name_train = op_name + "_TRAIN.json"
    op_name_test = op_name + "_TEST.json"

    class_un_train = print_dataset_stats(data_train, "Train", classes_to_ignore)
    class_un_test = print_dataset_stats(data_test, "Test", classes_to_ignore)

    if len(data_train) > 0 and len(data_test) > 0:
        if class_un_train != class_un_test:
            print(
                '\nError: some classes are not in both the training and test sets.\
                   \nTry a different random seed "--rand_seed".'
            )
            assert False

    print("\n")
    if len(data_train) == 0:
        print("No train annotations to save")
    else:
        print("Saving: ", op_name_train)
        with open(op_name_train, "w") as da:
            json.dump(data_train, da, indent=2)

    if len(data_test) == 0:
        print("No test annotations to save")
    else:
        print("Saving: ", op_name_test)
        with open(op_name_test, "w") as da:
            json.dump(data_test, da, indent=2)
