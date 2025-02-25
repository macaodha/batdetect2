import sys
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple

import numpy as np

from batdetect2 import types

if sys.version_info >= (3, 9):
    StringCounter = Counter[str]
else:
    from typing import Counter as StringCounter


def write_notes_file(file_name: str, text: str):
    with open(file_name, "a") as da:
        da.write(text + "\n")


def get_blank_dataset_dict(
    dataset_name: str,
    is_test: bool,
    ann_path: str,
    wav_path: str,
) -> types.DatasetDict:
    return {
        "dataset_name": dataset_name,
        "is_test": is_test,
        "is_binary": False,
        "ann_path": ann_path,
        "wav_path": wav_path,
    }


def get_short_class_names(
    class_names: List[str],
    str_len: int = 3,
) -> List[str]:
    class_names_short = []
    for cc in class_names:
        class_names_short.append(
            " ".join([sp[:str_len] for sp in cc.split(" ")])
        )
    return class_names_short


def remove_dupes(
    data_train: List[types.FileAnnotation],
    data_test: List[types.FileAnnotation],
) -> List[types.FileAnnotation]:
    test_ids = [dd["id"] for dd in data_test]
    data_train_prune = []
    for aa in data_train:
        if aa["id"] not in test_ids:
            data_train_prune.append(aa)
    diff = len(data_train) - len(data_train_prune)
    if diff != 0:
        print(diff, "items removed from train set")
    return data_train_prune


def get_genus_mapping(class_names: List[str]) -> Tuple[List[str], List[int]]:
    genus_names, genus_mapping = np.unique(
        [cc.split(" ")[0] for cc in class_names], return_inverse=True
    )
    return genus_names.tolist(), genus_mapping.tolist()


def standardize_low_freq(
    data: List[types.FileAnnotation],
    class_of_interest: str,
) -> List[types.FileAnnotation]:
    # address the issue of highly variable low frequency annotations
    # this often happens for contstant frequency calls
    # for the class of interest sets the low and high freq to be the dataset mean
    low_freqs = []
    high_freqs = []
    for dd in data:
        for aa in dd["annotation"]:
            if aa["class"] == class_of_interest:
                low_freqs.append(aa["low_freq"])
                high_freqs.append(aa["high_freq"])

    low_mean = float(np.mean(low_freqs))
    high_mean = float(np.mean(high_freqs))
    assert low_mean < high_mean

    print("\nStandardizing low and high frequency for:")
    print(class_of_interest)
    print("low:  ", round(low_mean, 2))
    print("high: ", round(high_mean, 2))

    # only set the low freq, high stays the same
    # assumes that low_mean < high_mean
    for dd in data:
        for aa in dd["annotation"]:
            if aa["class"] == class_of_interest:
                aa["low_freq"] = low_mean
                if aa["high_freq"] < low_mean:
                    aa["high_freq"] = high_mean

    return data


def format_annotation(
    annotation: types.FileAnnotation,
    events_of_interest: Optional[List[str]] = None,
    name_replace: Optional[Dict[str, str]] = None,
    convert_to_genus: bool = False,
    classes_to_ignore: Optional[List[str]] = None,
) -> types.FileAnnotation:
    formated = []
    for aa in annotation["annotation"]:
        if (
            events_of_interest is not None
            and aa["event"] not in events_of_interest
        ):
            # Omit files with annotation issues
            continue

        # remove leading and trailing spaces
        class_name = aa["class"].strip()

        if name_replace is not None:
            # replace_names will be a dictionary mapping input name to output
            class_name = name_replace.get(class_name, class_name)

        if convert_to_genus:
            # convert everything to genus name
            class_name = class_name.split(" ")[0]

        # NOTE: It is important to acknowledge that the class names filtering
        # is done after the name replacement and the conversion to
        # genus name. This allows filtering converted genus names and names
        # that were replaced with a name that should be ignored.
        if classes_to_ignore is not None and class_name in classes_to_ignore:
            # Omit annotations with ignored classes
            continue

        formated.append(
            {
                **aa,
                "class": class_name,
            }
        )

    return {
        **annotation,
        "annotation": formated,
    }


def get_class_names(
    data: List[types.FileAnnotation],
    classes_to_ignore: Optional[List[str]] = None,
) -> Tuple[StringCounter, List[float]]:
    """Extracts class names and their inverse frequencies.

    Parameters
    ----------
    data
        A list of file annotations, where each annotation contains a list of
        sound events with associated class names.
    classes_to_ignore
        A list of class names to ignore.

    Returns:
    --------
    class_names
        A list of unique class names extracted from the annotations.
    class_inv_freq
        List of inverse frequencies of each class name in the provided data.
    """
    if classes_to_ignore is None:
        classes_to_ignore = []

    class_names_list: List[str] = []
    for annotation in data:
        for sound_event in annotation["annotation"]:
            if sound_event["class"] in classes_to_ignore:
                continue

            class_names_list.append(sound_event["class"])

    counts = Counter(class_names_list)
    mean_counts = float(np.mean(list(counts.values())))
    return counts, [mean_counts / counts[cc] for cc in class_names_list]


def report_class_counts(class_names: StringCounter):
    print("Class count:")
    str_len = np.max([len(cc) for cc in class_names]) + 5
    for index, (class_name, count) in enumerate(class_names.most_common()):
        print(f"{index:<5}{class_name:<{str_len}}{count}")


def load_set_of_anns(
    data: List[types.DatasetDict],
    *,
    convert_to_genus: bool = False,
    filter_issues: bool = False,
    events_of_interest: Optional[List[str]] = None,
    classes_to_ignore: Optional[List[str]] = None,
    name_replace: Optional[Dict[str, str]] = None,
) -> List[types.FileAnnotation]:
    # load the annotations
    anns = []

    # dictionary of datasets
    for dataset in data:
        for ann in load_anns(dataset["ann_path"], dataset["wav_path"]):
            if not ann["annotated"]:
                # Omit unannotated files
                continue

            if filter_issues and ann["issues"]:
                # Omit files with annotation issues
                continue

            anns.append(
                format_annotation(
                    ann,
                    events_of_interest=events_of_interest,
                    name_replace=name_replace,
                    convert_to_genus=convert_to_genus,
                    classes_to_ignore=classes_to_ignore,
                )
            )

    return anns


def load_anns(
    ann_dir: str,
    raw_audio_dir: str,
) -> Generator[types.FileAnnotation, None, None]:
    for path in Path(ann_dir).rglob("*.json"):
        with open(path) as fp:
            file_annotation = json.load(fp)

        file_annotation["file_path"] = raw_audio_dir + file_annotation["id"]
        yield file_annotation


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
