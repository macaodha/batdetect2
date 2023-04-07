"""
Run scripts/extract_anns.py to generate these json files.
"""


def get_train_test_data(ann_dir, wav_dir, split_name, load_extra=True):
    if split_name == "diff":
        train_sets, test_sets = split_diff(ann_dir, wav_dir, load_extra)
    elif split_name == "same":
        train_sets, test_sets = split_same(ann_dir, wav_dir, load_extra)
    else:
        print("Split not defined")
        assert False

    return train_sets, test_sets


def split_diff(ann_dir, wav_dir, load_extra=True):

    train_sets = []
    if load_extra:
        train_sets.append(
            {
                "dataset_name": "BatDetective",
                "is_test": False,
                "is_binary": True,  # just a bat / not bat dataset ie no classes
                "ann_path": ann_dir
                + "train_set_bulgaria_batdetective_with_bbs.json",
                "wav_path": wav_dir + "batdetect2ive/audio/",
            }
        )
        train_sets.append(
            {
                "dataset_name": "bat_logger_qeop_empty",
                "is_test": False,
                "is_binary": True,
                "ann_path": ann_dir + "bat_logger_qeop_empty.json",
                "wav_path": wav_dir + "bat_logger_qeop_empty/audio/",
            }
        )
        train_sets.append(
            {
                "dataset_name": "bat_logger_2016_empty",
                "is_test": False,
                "is_binary": True,
                "ann_path": ann_dir + "train_set_bat_logger_2016_empty.json",
                "wav_path": wav_dir + "bat_logger_2016/audio/",
            }
        )
        # train_sets.append({'dataset_name': 'brazil_data_binary',
        #      'is_test': False,
        #      'ann_path': ann_dir + 'brazil_data_binary.json',
        #      'wav_path': wav_dir + 'brazil_data/audio/'})

    train_sets.append(
        {
            "dataset_name": "echobank",
            "is_test": False,
            "is_binary": False,
            "ann_path": ann_dir + "Echobank_train_expert.json",
            "wav_path": wav_dir + "echobank/audio/",
        }
    )
    train_sets.append(
        {
            "dataset_name": "sn_scot_nor",
            "is_test": False,
            "is_binary": False,
            "ann_path": ann_dir + "sn_scot_nor_0.5_expert.json",
            "wav_path": wav_dir + "sn_scot_nor/audio/",
        }
    )
    train_sets.append(
        {
            "dataset_name": "BCT_1_sec",
            "is_test": False,
            "is_binary": False,
            "ann_path": ann_dir + "BCT_1_sec_train_expert.json",
            "wav_path": wav_dir + "BCT_1_sec/audio/",
        }
    )
    train_sets.append(
        {
            "dataset_name": "bcireland",
            "is_test": False,
            "is_binary": False,
            "ann_path": ann_dir + "bcireland_expert.json",
            "wav_path": wav_dir + "bcireland/audio/",
        }
    )
    train_sets.append(
        {
            "dataset_name": "rhinolophus_steve_BCT",
            "is_test": False,
            "is_binary": False,
            "ann_path": ann_dir + "rhinolophus_steve_BCT_expert.json",
            "wav_path": wav_dir + "rhinolophus_steve_BCT/audio/",
        }
    )

    test_sets = []
    test_sets.append(
        {
            "dataset_name": "bat_data_martyn_2018",
            "is_test": True,
            "is_binary": False,
            "ann_path": ann_dir
            + "BritishBatCalls_MartynCooke_2018_1_sec_train_expert.json",
            "wav_path": wav_dir + "bat_data_martyn_2018/audio/",
        }
    )
    test_sets.append(
        {
            "dataset_name": "bat_data_martyn_2018_test",
            "is_test": True,
            "is_binary": False,
            "ann_path": ann_dir
            + "BritishBatCalls_MartynCooke_2018_1_sec_test_expert.json",
            "wav_path": wav_dir + "bat_data_martyn_2018_test/audio/",
        }
    )
    test_sets.append(
        {
            "dataset_name": "bat_data_martyn_2019",
            "is_test": True,
            "is_binary": False,
            "ann_path": ann_dir
            + "BritishBatCalls_MartynCooke_2019_1_sec_train_expert.json",
            "wav_path": wav_dir + "bat_data_martyn_2019/audio/",
        }
    )
    test_sets.append(
        {
            "dataset_name": "bat_data_martyn_2019_test",
            "is_test": True,
            "is_binary": False,
            "ann_path": ann_dir
            + "BritishBatCalls_MartynCooke_2019_1_sec_test_expert.json",
            "wav_path": wav_dir + "bat_data_martyn_2019_test/audio/",
        }
    )

    return train_sets, test_sets


def split_same(ann_dir, wav_dir, load_extra=True):

    train_sets = []
    if load_extra:
        train_sets.append(
            {
                "dataset_name": "BatDetective",
                "is_test": False,
                "is_binary": True,
                "ann_path": ann_dir
                + "train_set_bulgaria_batdetective_with_bbs.json",
                "wav_path": wav_dir + "batdetect2ive/audio/",
            }
        )
        train_sets.append(
            {
                "dataset_name": "bat_logger_qeop_empty",
                "is_test": False,
                "is_binary": True,
                "ann_path": ann_dir + "bat_logger_qeop_empty.json",
                "wav_path": wav_dir + "bat_logger_qeop_empty/audio/",
            }
        )
        train_sets.append(
            {
                "dataset_name": "bat_logger_2016_empty",
                "is_test": False,
                "is_binary": True,
                "ann_path": ann_dir + "train_set_bat_logger_2016_empty.json",
                "wav_path": wav_dir + "bat_logger_2016/audio/",
            }
        )
        # train_sets.append({'dataset_name': 'brazil_data_binary',
        #      'is_test': False,
        #      'ann_path': ann_dir + 'brazil_data_binary.json',
        #      'wav_path': wav_dir + 'brazil_data/audio/'})

    train_sets.append(
        {
            "dataset_name": "echobank",
            "is_test": False,
            "is_binary": False,
            "ann_path": ann_dir + "Echobank_train_expert_TRAIN.json",
            "wav_path": wav_dir + "echobank/audio/",
        }
    )
    train_sets.append(
        {
            "dataset_name": "sn_scot_nor",
            "is_test": False,
            "is_binary": False,
            "ann_path": ann_dir + "sn_scot_nor_0.5_expert_TRAIN.json",
            "wav_path": wav_dir + "sn_scot_nor/audio/",
        }
    )
    train_sets.append(
        {
            "dataset_name": "BCT_1_sec",
            "is_test": False,
            "is_binary": False,
            "ann_path": ann_dir + "BCT_1_sec_train_expert_TRAIN.json",
            "wav_path": wav_dir + "BCT_1_sec/audio/",
        }
    )
    train_sets.append(
        {
            "dataset_name": "bcireland",
            "is_test": False,
            "is_binary": False,
            "ann_path": ann_dir + "bcireland_expert_TRAIN.json",
            "wav_path": wav_dir + "bcireland/audio/",
        }
    )
    train_sets.append(
        {
            "dataset_name": "rhinolophus_steve_BCT",
            "is_test": False,
            "is_binary": False,
            "ann_path": ann_dir + "rhinolophus_steve_BCT_expert_TRAIN.json",
            "wav_path": wav_dir + "rhinolophus_steve_BCT/audio/",
        }
    )
    train_sets.append(
        {
            "dataset_name": "bat_data_martyn_2018",
            "is_test": False,
            "is_binary": False,
            "ann_path": ann_dir
            + "BritishBatCalls_MartynCooke_2018_1_sec_train_expert_TRAIN.json",
            "wav_path": wav_dir + "bat_data_martyn_2018/audio/",
        }
    )
    train_sets.append(
        {
            "dataset_name": "bat_data_martyn_2018_test",
            "is_test": False,
            "is_binary": False,
            "ann_path": ann_dir
            + "BritishBatCalls_MartynCooke_2018_1_sec_test_expert_TRAIN.json",
            "wav_path": wav_dir + "bat_data_martyn_2018_test/audio/",
        }
    )
    train_sets.append(
        {
            "dataset_name": "bat_data_martyn_2019",
            "is_test": False,
            "is_binary": False,
            "ann_path": ann_dir
            + "BritishBatCalls_MartynCooke_2019_1_sec_train_expert_TRAIN.json",
            "wav_path": wav_dir + "bat_data_martyn_2019/audio/",
        }
    )
    train_sets.append(
        {
            "dataset_name": "bat_data_martyn_2019_test",
            "is_test": False,
            "is_binary": False,
            "ann_path": ann_dir
            + "BritishBatCalls_MartynCooke_2019_1_sec_test_expert_TRAIN.json",
            "wav_path": wav_dir + "bat_data_martyn_2019_test/audio/",
        }
    )

    # train_sets.append({'dataset_name': 'bat_data_martyn_2021_train',
    #     'is_test': False,
    #     'is_binary': False,
    #     'ann_path': ann_dir + 'bat_data_martyn_2021_TRAIN.json',
    #     'wav_path': wav_dir + 'bat_data_martyn_2021/audio/'})
    # train_sets.append({'dataset_name': 'volunteers_2021_train',
    #     'is_test': False,
    #     'is_binary': False,
    #     'ann_path': ann_dir + 'volunteers_2021_TRAIN.json',
    #     'wav_path': wav_dir + 'volunteers_2021/audio/'})

    test_sets = []
    test_sets.append(
        {
            "dataset_name": "echobank",
            "is_test": True,
            "is_binary": False,
            "ann_path": ann_dir + "Echobank_train_expert_TEST.json",
            "wav_path": wav_dir + "echobank/audio/",
        }
    )
    test_sets.append(
        {
            "dataset_name": "sn_scot_nor",
            "is_test": True,
            "is_binary": False,
            "ann_path": ann_dir + "sn_scot_nor_0.5_expert_TEST.json",
            "wav_path": wav_dir + "sn_scot_nor/audio/",
        }
    )
    test_sets.append(
        {
            "dataset_name": "BCT_1_sec",
            "is_test": True,
            "is_binary": False,
            "ann_path": ann_dir + "BCT_1_sec_train_expert_TEST.json",
            "wav_path": wav_dir + "BCT_1_sec/audio/",
        }
    )
    test_sets.append(
        {
            "dataset_name": "bcireland",
            "is_test": True,
            "is_binary": False,
            "ann_path": ann_dir + "bcireland_expert_TEST.json",
            "wav_path": wav_dir + "bcireland/audio/",
        }
    )
    test_sets.append(
        {
            "dataset_name": "rhinolophus_steve_BCT",
            "is_test": True,
            "is_binary": False,
            "ann_path": ann_dir + "rhinolophus_steve_BCT_expert_TEST.json",
            "wav_path": wav_dir + "rhinolophus_steve_BCT/audio/",
        }
    )
    test_sets.append(
        {
            "dataset_name": "bat_data_martyn_2018",
            "is_test": True,
            "is_binary": False,
            "ann_path": ann_dir
            + "BritishBatCalls_MartynCooke_2018_1_sec_train_expert_TEST.json",
            "wav_path": wav_dir + "bat_data_martyn_2018/audio/",
        }
    )
    test_sets.append(
        {
            "dataset_name": "bat_data_martyn_2018_test",
            "is_test": True,
            "is_binary": False,
            "ann_path": ann_dir
            + "BritishBatCalls_MartynCooke_2018_1_sec_test_expert_TEST.json",
            "wav_path": wav_dir + "bat_data_martyn_2018_test/audio/",
        }
    )
    test_sets.append(
        {
            "dataset_name": "bat_data_martyn_2019",
            "is_test": True,
            "is_binary": False,
            "ann_path": ann_dir
            + "BritishBatCalls_MartynCooke_2019_1_sec_train_expert_TEST.json",
            "wav_path": wav_dir + "bat_data_martyn_2019/audio/",
        }
    )
    test_sets.append(
        {
            "dataset_name": "bat_data_martyn_2019_test",
            "is_test": True,
            "is_binary": False,
            "ann_path": ann_dir
            + "BritishBatCalls_MartynCooke_2019_1_sec_test_expert_TEST.json",
            "wav_path": wav_dir + "bat_data_martyn_2019_test/audio/",
        }
    )

    # test_sets.append({'dataset_name': 'bat_data_martyn_2021_test',
    #     'is_test': True,
    #     'is_binary': False,
    #     'ann_path': ann_dir + 'bat_data_martyn_2021_TEST.json',
    #     'wav_path': wav_dir + 'bat_data_martyn_2021/audio/'})
    # test_sets.append({'dataset_name': 'volunteers_2021_test',
    #     'is_test': True,
    #     'is_binary': False,
    #     'ann_path': ann_dir + 'volunteers_2021_TEST.json',
    #     'wav_path': wav_dir + 'volunteers_2021/audio/'})

    return train_sets, test_sets
