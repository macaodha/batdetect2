import pandas as pd
from soundevent.geometry import compute_bounds

from batdetect2.data.datasets import Dataset
from batdetect2.typing.targets import TargetProtocol

__all__ = [
    "extract_recordings_df",
    "extract_sound_events_df",
    "compute_class_summary",
]


def extract_recordings_df(dataset: Dataset) -> pd.DataFrame:
    """Extract recording metadata into a pandas DataFrame.

    Parameters
    ----------
    dataset : List[data.ClipAnnotation]
        A list of clip annotations from which to extract recording information.

    Returns
    -------
    pd.DataFrame
        A DataFrame where each row corresponds to a recording, containing
        metadata such as duration, path, sample rate, and other properties.
    """
    recordings = []

    for clip_annotation in dataset:
        clip = clip_annotation.clip
        recording = clip.recording
        recordings.append(
            {
                "clip_annotation_id": str(clip_annotation.uuid),
                "recording_id": str(recording.uuid),
                "duration": clip.duration,
                "filename": recording.path.name,
                **recording.model_dump(
                    mode="json",
                    include={
                        "samplerate",
                        "hash",
                        "path",
                        "date",
                        "time",
                        "latitude",
                        "longitude",
                    },
                ),
            }
        )

    return pd.DataFrame(recordings)


def extract_sound_events_df(
    dataset: Dataset,
    targets: TargetProtocol,
    exclude_non_target: bool = True,
    exclude_generic: bool = True,
) -> pd.DataFrame:
    """Extract sound event data into a pandas DataFrame.

    This function iterates through all sound events in the provided dataset,
    applies filtering and classification logic based on the `targets`
    protocol, and compiles the results into a structured DataFrame.

    Parameters
    ----------
    dataset : List[data.ClipAnnotation]
        The dataset containing clip annotations with sound events.
    targets : TargetProtocol
        An object that provides methods to filter (`filter`) and classify
        (`encode_class`) sound events.
    exclude_non_target : bool, default=True
        If True, sound events that do not pass the `targets.filter()` check
        are excluded from the output.
    exclude_generic : bool, default=True
        If True, sound events that are classified with a `None` class name
        by `targets.encode_class()` are excluded.

    Returns
    -------
    pd.DataFrame
        A DataFrame where each row represents a single sound event, including
        its bounding box, class name, and other relevant attributes.
    """
    sound_events = []

    for clip_annotation in dataset:
        for sound_event in clip_annotation.sound_events:
            is_target = targets.filter(sound_event)

            if not is_target and exclude_non_target:
                continue

            if sound_event.sound_event.geometry is None:
                continue

            class_name = targets.encode_class(sound_event)

            if class_name is None:
                if exclude_generic:
                    continue
                else:
                    class_name = targets.detection_class_name

            start_time, low_freq, end_time, high_freq = compute_bounds(
                sound_event.sound_event.geometry
            )

            sound_events.append(
                {
                    "clip_annotation_id": str(clip_annotation.uuid),
                    "sound_event_id": str(sound_event.uuid),
                    "recording_id": str(
                        sound_event.sound_event.recording.uuid
                    ),
                    "start_time": start_time,
                    "end_time": end_time,
                    "low_freq": low_freq,
                    "high_freq": high_freq,
                    "is_target": is_target,
                    "class_name": class_name,
                }
            )

    return pd.DataFrame(sound_events)


def compute_class_summary(
    dataset: Dataset,
    targets: TargetProtocol,
) -> pd.DataFrame:
    """Compute a summary of sound event statistics grouped by class.

    This function generates a high-level summary DataFrame that provides
    key metrics for each class identified in the dataset. It calculates
    the total number of calls, the number of unique recordings containing
    each class, the total duration of those recordings, and the call rate.

    Parameters
    ----------
    dataset : List[data.ClipAnnotation]
        The dataset to be summarized.
    targets : TargetProtocol
        An object providing the classification logic for sound events.

    Returns
    -------
    pd.DataFrame
        A DataFrame indexed by class name, with columns for 'num calls',
        'num recordings', 'duration', and 'call_rate'.
    """
    sound_events = extract_sound_events_df(
        dataset,
        targets,
        exclude_generic=False,
        exclude_non_target=True,
    )

    recordings = extract_recordings_df(dataset)

    num_calls = (
        sound_events.groupby("class_name")
        .size()
        .sort_values(ascending=False)
        .rename("num calls")
    )
    num_recs = (
        sound_events.groupby("class_name")["clip_annotation_id"]
        .nunique()
        .sort_values(ascending=False)
        .rename("num recordings")
    )
    durations = (
        sound_events.groupby("class_name")
        .apply(
            lambda group: recordings[
                recordings["clip_annotation_id"].isin(
                    group["clip_annotation_id"]  # type: ignore
                )
            ]["duration"].sum(),
            include_groups=False,  # type: ignore
        )
        .sort_values(ascending=False)
        .rename("duration")
    )
    return (
        num_calls.to_frame()
        .join(num_recs)
        .join(durations)
        .sort_values("num calls", ascending=False)
        .assign(call_rate=lambda df: df["num calls"] / df["duration"])
    )
