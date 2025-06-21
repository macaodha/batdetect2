import argparse
import json
import os
from collections import Counter
from typing import List, Optional, Tuple

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

import batdetect2.train.train_utils as tu
from batdetect2 import types


def print_dataset_stats(
    data: List[types.FileAnnotation],
    classes_to_ignore: Optional[List[str]] = None,
) -> Counter[str]:
    print("Num files:", len(data))
    counts, _ = tu.get_class_names(data, classes_to_ignore)
    if len(counts) > 0:
        tu.report_class_counts(counts)
    return counts


def load_file_names(file_name: str) -> List[str]:
    if not os.path.isfile(file_name):
        raise FileNotFoundError(f"Input file not found - {file_name}")

    with open(file_name) as da:
        files = [line.rstrip() for line in da.readlines()]

    for path in files:
        if path.lower()[-3:] != "wav":
            raise ValueError(
                f"Invalid file name - {path}. Must be a .wav file"
            )

    return files


def parse_args():
    info_str = "\nBatDetect - Prepare Data for Finetuning\n"
    print(info_str)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset_name", type=str, help="Name to call your dataset"
    )
    parser.add_argument(
        "audio_dir", type=str, help="Input directory for audio"
    )
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
    return parser.parse_args()


def split_data(
    data: List[types.FileAnnotation],
    train_file: str,
    test_file: str,
    n_splits: int = 5,
    random_state: int = 0,
) -> Tuple[List[types.FileAnnotation], List[types.FileAnnotation]]:
    if train_file != "" and test_file != "":
        # user has specifed the train / test split
        mapping = {
            file_annotation["id"]: file_annotation for file_annotation in data
        }
        train_files = load_file_names(train_file)
        test_files = load_file_names(test_file)
        data_train = [
            mapping[file_id] for file_id in train_files if file_id in mapping
        ]
        data_test = [
            mapping[file_id] for file_id in test_files if file_id in mapping
        ]
        return data_train, data_test

    # NOTE: Using StratifiedGroupKFold to ensure that the same file does not
    # appear in both the training and test sets and trying to keep the
    # distribution of classes the same in both sets.
    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )
    anns = np.array(
        [
            [dd["id"], ann["class"], ann["event"]]
            for dd in data
            for ann in dd["annotation"]
        ]
    )
    y = anns[:, 1]
    group = anns[:, 0]

    train_idx, test_idx = next(splitter.split(X=anns, y=y, groups=group))
    train_ids = set(anns[train_idx, 0])
    test_ids = set(anns[test_idx, 0])

    assert not (train_ids & test_ids)
    data_train = [dd for dd in data if dd["id"] in train_ids]
    data_test = [dd for dd in data if dd["id"] in test_ids]
    return data_train, data_test


def main():
    args = parse_args()

    np.random.seed(args.rand_seed)

    classes_to_ignore = ["", " ", "Unknown", "Not Bat"]
    events_of_interest = ["Echolocation"]

    name_dict = None
    if args.input_class_names != "" and args.output_class_names != "":
        # change the names of the classes
        ip_names = args.input_class_names.split(";")
        op_names = args.output_class_names.split(";")
        name_dict = dict(zip(ip_names, op_names))

    # load annotations
    data_all = tu.load_set_of_anns(
        [
            {
                "dataset_name": args.dataset_name,
                "ann_path": args.ann_dir,
                "wav_path": args.audio_dir,
                "is_test": False,
                "is_binary": False,
            }
        ],
        classes_to_ignore=classes_to_ignore,
        events_of_interest=events_of_interest,
        convert_to_genus=False,
        filter_issues=True,
        name_replace=name_dict,
    )

    print("Dataset name:         " + args.dataset_name)
    print("Audio directory:      " + args.audio_dir)
    print("Annotation directory: " + args.ann_dir)
    print("Ouput directory:      " + args.op_dir)
    print("Num annotated files:  " + str(len(data_all)))

    data_train, data_test = split_data(
        data=data_all,
        train_file=args.train_file,
        test_file=args.test_file,
        n_splits=5,
        random_state=args.rand_seed,
    )

    if not os.path.isdir(args.op_dir):
        os.makedirs(args.op_dir)
    op_name = os.path.join(args.op_dir, args.dataset_name)
    op_name_train = op_name + "_TRAIN.json"
    op_name_test = op_name + "_TEST.json"

    print("\nSplit: Train")
    class_un_train = print_dataset_stats(data_train, classes_to_ignore)

    print("\nSplit: Test")
    class_un_test = print_dataset_stats(data_test, classes_to_ignore)

    if len(data_train) > 0 and len(data_test) > 0:
        if set(class_un_train.keys()) != set(class_un_test.keys()):
            raise RuntimeError(
                "Error: some classes are not in both the training and test sets."
                'Try a different random seed "--rand_seed".'
            )

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


if __name__ == "__main__":
    main()
