# Batdetect2 Datasets

This document describes the datasets used to train and evaluate the `batdetect2` model for acoustic bat detection and classification.

`batdetect2` was trained using a combination of datasets, primarily collected and annotated in partnership with the Bat Conservation Trust (BCT).
The data sources include:

- **Dedicated recordings**: The majority of the data comprises focal recordings of individual bats with known species identifications.
    These were collected specifically for this project in collaboration with the BCT and partners.
- **BatDetective project**: A subset of recordings was selected from the BatDetective project and re-annotated for this project.
- **Queen Elizabeth Olympic Park**: Recordings were sourced from the acoustic monitoring network in East London.

## Annotation Approach

Most datasets contain annotations with species-level identification.
However, the `BatDetective` and `bat_logger*` datasets lack species identification for individual calls.
This is because these recordings were collected passively without on-site species verification.
Annotations for these datasets include bounding boxes around echolocation calls but no species labels.

## Dataset Summary

The following table provides an overview of each dataset used in training and evaluating `batdetect2`.
It includes the location of the dataset files and whether the annotations include species identification ("Species ID").

| Dataset               | Dataset Path             | Species ID |
| --------------------- | ------------------------ | ---------- |
| BatDetective          | bat_detective_batdetect2 | no         |
| bat_logger_qeop_empty | bat_logger_qeop_empty    | no         |
| bat_logger_2016_empty | bat_logger_2016          | no         |
| echobank              | echobank_batdetect2      | yes        |
| sn_scot_nor           | sn_scot_nor              | yes        |
| bct_1_sec             | bct_1_sec                | yes        |
| bcireland             | bcireland                | yes        |
| rhinolophus_bct       | rhinolophus_bct          | yes        |
| bat_data_2018         | bat_data_2018            | yes        |
| bat_data_2018_test    | bat_data_2018_test       | yes        |
| bat_data_2019         | bat_data_2019            | yes        |
| bat_data_2019_test    | bat_data_2019_test       | yes        |

## Train Test splits

To ensure a robust and unbiased evaluation of `batdetect2`, we carefully considered how to split the data into training and test sets.
Due to potential dependencies within datasets (e.g., recordings from the same site, date, or even individual), we implemented two distinct splitting strategies:

1. Split Diff:

   This strategy assigns entire datasets to either the training or test set.
      This maximizes the independence between training and testing data, reducing the chance of the model encountering similar recordings in both sets.

2. Split Same:

   In this approach, each dataset is individually split into training and test subsets.
      This means that recordings in the test set might share similarities with those in the training set (e.g., same site, methodology).
      This split helps assess the model's ability to generalize to new recordings within familiar contexts.

For each split, the corresponding annotations were organized into JSON files.
The following tables list the annotation sets used for training and testing, along with their associated dataset:

### Split Diff Summary

| Dataset               | Annotation Set Name                           | Is Test |
| --------------------- | --------------------------------------------- | ------- |
| BatDetective          | train_set_bulgaria_batdetective_with_bbs.json | no      |
| bat_logger_qeop_empty | bat_logger_qeop_empty.json                    | no      |
| bat_logger_2016       | train_set_bat_logger_2016_empty.json          | no      |
| echobank_batdetect2   | Echobank_train_expert.json                    | no      |
| sn_scot_nor           | sn_scot_nor_0.5_expert.json                   | no      |
| bct_1_sec             | bct_1_sec_train_expert.json                   | no      |
| bcireland             | bcireland_expert.json                         | no      |
| rhinolophus_bct       | rhinolophus_BCT_expert.json                   | no      |
| bat_data_2018         | BritishBatCalls_2018_1_sec_train_expert.json  | yes     |
| bat_data_2018_test    | BritishBatCalls_2018_1_sec_test_expert.json   | yes     |
| bat_data_2019         | BritishBatCalls_2019_1_sec_test_expert.json   | yes     |
| bat_data_2019_test    | BritishBatCalls_2019_1_sec_test_expert.json   | yes     |

### Split Same Summary

| Dataset               | Annotation Set Name                                | Is Test |
| --------------------- | -------------------------------------------------- | ------- |
| BatDetective          | train_set_bulgaria_batdetective_with_bbs.json      | no      |
| bat_logger_qeop_empty | bat_logger_qeop_empty.json                         | no      |
| bat_logger_2016       | train_set_bat_logger_2016_empty.json               | no      |
| echobank              | Echobank_train_expert_TRAIN.json                   | no      |
| sn_scot_nor           | sn_scot_nor_0.5_expert_TRAIN.json                  | no      |
| bct_1_sec             | BCT_1_sec_train_expert_TRAIN.json                  | no      |
| bcireland             | bcireland_expert_TRAIN.json                        | no      |
| rhinolophus_bct       | rhinolophus_BCT_expert_TRAIN.json                  | no      |
| bat_data_2018         | BritishBatCalls_2018_1_sec_train_expert_TRAIN.json | no      |
| bat_data_2018_test    | BritishBatCalls_2018_1_sec_test_expert_TRAIN.json  | no      |
| bat_data_2019         | BritishBatCalls_2019_1_sec_train_expert_TRAIN.json | no      |
| bat_data_2019_test    | BritishBatCalls_2019_1_sec_test_expert_TRAIN.json  | no      |
| echobank              | Echobank_train_expert_TEST.json                    | yes     |
| sn_scot_nor           | sn_scot_nor_0.5_expert_TEST.json                   | yes     |
| BCT_1_sec             | BCT_1_sec_train_expert_TEST.json                   | yes     |
| bcireland             | bcireland_expert_TEST.json                         | yes     |
| rhinolophus_bct       | rhinolophus_BCT_expert_TEST.json                   | yes     |
| bat_data_2018         | BritishBatCalls_2018_1_sec_train_expert_TEST.json  | yes     |
| bat_data_2018_test    | BritishBatCalls_2018_1_sec_test_expert_TEST.json   | yes     |
| bat_data_2019         | BritishBatCalls_2019_1_sec_train_expert_TEST.json  | yes     |
| bat_data_2019_test    | BritishBatCalls_2019_1_sec_test_expert_TEST.json   | yes     |

## File structure

Each dataset is organized within a separate folder, as listed in the Dataset Summary table.
Within each dataset folder, you'll find the following subfolders:

- `audio`: Contains all the raw WAV audio recordings in a flat structure.
- `annotation_sets`: Contains JSON files that gather all annotations for the different train-test splits.
    For example, the `rhinolophus_BCT_expert_TRAIN.json` annotation set for the `rhinolophus_bct` dataset would be located at `rhinolophus_bct/annotation_sets/rhinolophus_BCT_expert_TRAIN.json`.
- `annotations`: (Optional) This folder contains individual JSON files for each recording in the dataset, storing all annotations for the corresponding recording.
