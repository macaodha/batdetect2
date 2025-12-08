from typing import Tuple

from sklearn.model_selection import train_test_split

from batdetect2.data.datasets import Dataset
from batdetect2.data.summary import (
    extract_recordings_df,
    extract_sound_events_df,
)
from batdetect2.typing.targets import TargetProtocol


def split_dataset_by_recordings(
    dataset: Dataset,
    targets: TargetProtocol,
    train_size: float = 0.75,
    random_state: int | None = None,
) -> Tuple[Dataset, Dataset]:
    recordings = extract_recordings_df(dataset)

    sound_events = extract_sound_events_df(
        dataset,
        targets,
        exclude_non_target=True,
        exclude_generic=True,
    )

    majority_class = (
        sound_events.groupby("recording_id")
        .apply(
            lambda group: group["class_name"]  # type: ignore
            .value_counts()
            .sort_values(ascending=False)
            .index[0],
            include_groups=False,  # type: ignore
        )
        .rename("class_name")
        .to_frame()
        .reset_index()
    )

    train, test = train_test_split(
        majority_class["recording_id"],
        stratify=majority_class["class_name"],
        train_size=train_size,
        random_state=random_state,
    )

    train_ids_set = set(train.values)  # type: ignore
    test_ids_set = set(test.values)  # type: ignore

    extra = set(recordings["recording_id"]) - train_ids_set - test_ids_set

    if extra:
        train_extra, test_extra = train_test_split(
            list(extra),
            train_size=train_size,
            random_state=random_state,
        )
        train_ids_set.update(train_extra)
        test_ids_set.update(test_extra)

    train_dataset = [
        clip_annotation
        for clip_annotation in dataset
        if str(clip_annotation.clip.recording.uuid) in train_ids_set
    ]

    test_dataset = [
        clip_annotation
        for clip_annotation in dataset
        if str(clip_annotation.clip.recording.uuid) in test_ids_set
    ]

    return train_dataset, test_dataset
