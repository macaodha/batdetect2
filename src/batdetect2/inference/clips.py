from typing import List, Sequence
from uuid import uuid5

import numpy as np
from soundevent import data


def get_clips_from_files(
    paths: Sequence[data.PathLike],
    duration: float,
    overlap: float = 0.0,
    max_empty: float = 0.0,
    discard_empty: bool = True,
    compute_hash: bool = False,
) -> List[data.Clip]:
    clips: List[data.Clip] = []

    for path in paths:
        recording = data.Recording.from_file(path, compute_hash=compute_hash)
        clips.extend(
            get_recording_clips(
                recording,
                duration,
                overlap=overlap,
                max_empty=max_empty,
                discard_empty=discard_empty,
            )
        )

    return clips


def get_recording_clips(
    recording: data.Recording,
    duration: float,
    overlap: float = 0.0,
    max_empty: float = 0.0,
    discard_empty: bool = True,
) -> Sequence[data.Clip]:
    start_time = 0
    recording_duration = recording.duration
    hop = duration * (1 - overlap)

    num_clips = int(np.ceil(recording_duration / hop))

    if num_clips == 0:
        # This should only happen if the clip's duration is zero,
        # which should never happen in practice, but just in case...
        return []

    clips = []
    for i in range(num_clips):
        start = start_time + i * hop
        end = start + duration

        if end > recording_duration:
            empty_duration = end - recording_duration

            if empty_duration > max_empty and discard_empty:
                # Discard clips that contain too much empty space
                continue

        clips.append(
            data.Clip(
                uuid=uuid5(recording.uuid, f"{start}_{end}"),
                recording=recording,
                start_time=start,
                end_time=end,
            )
        )

    if discard_empty:
        clips = [clip for clip in clips if clip.duration > max_empty]

    return clips
