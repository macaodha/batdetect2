import glob
import json
import os
import random

import numpy as np


def write_notes_file(file_name, text):
    with open(file_name, "a") as da:
        da.write(text + "\n")


def get_blank_dataset_dict(dataset_name, is_test, ann_path, wav_path):
    ddict = {
        "dataset_name": dataset_name,
        "is_test": is_test,
        "is_binary": False,
        "ann_path": ann_path,
        "wav_path": wav_path,
    }
    return ddict


def get_short_class_names(class_names, str_len=3):
    class_names_short = []
    for cc in class_names:
        class_names_short.append(
            " ".join([sp[:str_len] for sp in cc.split(" ")])
        )
    return class_names_short


def remove_dupes(data_train, data_test):
    test_ids = [dd["id"] for dd in data_test]
    data_train_prune = []
    for aa in data_train:
        if aa["id"] not in test_ids:
            data_train_prune.append(aa)
    diff = len(data_train) - len(data_train_prune)
    if diff != 0:
        print(diff, "items removed from train set")
    return data_train_prune


def get_genus_mapping(class_names):
    genus_names, genus_mapping = np.unique(
        [cc.split(" ")[0] for cc in class_names], return_inverse=True
    )
    return genus_names.tolist(), genus_mapping.tolist()


def standardize_low_freq(data, class_of_interest):
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

    low_mean = np.mean(low_freqs)
    high_mean = np.mean(high_freqs)
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


def load_set_of_anns(
    data,
    classes_to_ignore=[],
    events_of_interest=None,
    convert_to_genus=False,
    verbose=True,
    list_of_anns=False,
    filter_issues=False,
    name_replace=False,
):

    # load the annotations
    anns = []
    if list_of_anns:
        # path to list of individual json files
        anns.extend(load_anns_from_path(data["ann_path"], data["wav_path"]))
    else:
        # dictionary of datasets
        for dd in data:
            anns.extend(load_anns(dd["ann_path"], dd["wav_path"]))

    # discarding unannoated files
    anns = [aa for aa in anns if aa["annotated"] is True]

    # filter files that have annotation issues - is the input is a dictionary of
    # datasets, this will lilely have already been done
    if filter_issues:
        anns = [aa for aa in anns if aa["issues"] is False]

    # check for some basic formatting errors with class names
    for ann in anns:
        for aa in ann["annotation"]:
            aa["class"] = aa["class"].strip()

    # only load specified events - i.e. types of calls
    if events_of_interest is not None:
        for ann in anns:
            filtered_events = []
            for aa in ann["annotation"]:
                if aa["event"] in events_of_interest:
                    filtered_events.append(aa)
            ann["annotation"] = filtered_events

    # change class names
    # replace_names will be a dictionary mapping input name to output
    if type(name_replace) is dict:
        for ann in anns:
            for aa in ann["annotation"]:
                if aa["class"] in name_replace:
                    aa["class"] = name_replace[aa["class"]]

    # convert everything to genus name
    if convert_to_genus:
        for ann in anns:
            for aa in ann["annotation"]:
                aa["class"] = aa["class"].split(" ")[0]

    # get unique class names
    class_names_all = []
    for ann in anns:
        for aa in ann["annotation"]:
            if aa["class"] not in classes_to_ignore:
                class_names_all.append(aa["class"])

    class_names, class_cnts = np.unique(class_names_all, return_counts=True)
    class_inv_freq = class_cnts.sum() / (
        len(class_names) * class_cnts.astype(np.float32)
    )

    if verbose:
        print("Class count:")
        str_len = np.max([len(cc) for cc in class_names]) + 5
        for cc in range(len(class_names)):
            print(
                str(cc).ljust(5)
                + class_names[cc].ljust(str_len)
                + str(class_cnts[cc])
            )

    if len(classes_to_ignore) == 0:
        return anns
    else:
        return anns, class_names.tolist(), class_inv_freq.tolist()


def load_anns(ann_file_name, raw_audio_dir):
    with open(ann_file_name) as da:
        anns = json.load(da)

    for aa in anns:
        aa["file_path"] = raw_audio_dir + aa["id"]

    return anns


def load_anns_from_path(ann_file_dir, raw_audio_dir):
    files = glob.glob(ann_file_dir + "*.json")
    anns = []
    for ff in files:
        with open(ff) as da:
            ann = json.load(da)
        ann["file_path"] = raw_audio_dir + ann["id"]
        anns.append(ann)

    return anns


class AverageMeter(object):
    """Computes and stores the average and current value"""

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
