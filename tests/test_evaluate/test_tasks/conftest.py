from typing import Literal

import numpy as np
import pytest
from soundevent import data

from batdetect2.typing import Detection


@pytest.fixture
def clip(recording: data.Recording) -> data.Clip:
    return data.Clip(recording=recording, start_time=0, end_time=100)


@pytest.fixture
def create_detection():
    def factory(
        detection_score: float = 0.5,
        start_time: float = 0.1,
        duration: float = 0.01,
        low_freq: float = 40_000,
        bandwidth: float = 30_000,
        pip_score: float = 0,
        myo_score: float = 0,
    ):
        return Detection(
            detection_score=detection_score,
            class_scores=np.array(
                [
                    pip_score,
                    myo_score,
                ]
            ),
            features=np.zeros([32]),
            geometry=data.BoundingBox(
                coordinates=[
                    start_time,
                    low_freq,
                    start_time + duration,
                    low_freq + bandwidth,
                ]
            ),
        )

    return factory


@pytest.fixture
def create_annotation(
    clip: data.Clip,
    bat_tag: data.Tag,
    myomyo_tag: data.Tag,
    pippip_tag: data.Tag,
):
    def factory(
        start_time: float = 0.1,
        duration: float = 0.01,
        low_freq: float = 40_000,
        bandwidth: float = 30_000,
        is_target: bool = True,
        class_name: Literal["pippip", "myomyo"] | None = None,
    ):
        tags = [bat_tag] if is_target else []

        if class_name is not None:
            if class_name == "pippip":
                tags.append(pippip_tag)
            elif class_name == "myomyo":
                tags.append(myomyo_tag)

        return data.SoundEventAnnotation(
            sound_event=data.SoundEvent(
                geometry=data.BoundingBox(
                    coordinates=[
                        start_time,
                        low_freq,
                        start_time + duration,
                        low_freq + bandwidth,
                    ]
                ),
                recording=clip.recording,
            ),
            tags=tags,
        )

    return factory
