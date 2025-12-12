from pathlib import Path
from typing import List, Literal, Sequence

import numpy as np
from soundevent import data, io

from batdetect2.core import BaseConfig
from batdetect2.data.predictions.base import (
    prediction_formatters,
)
from batdetect2.typing import (
    ClipDetections,
    Detection,
    OutputFormatterProtocol,
    TargetProtocol,
)


class SoundEventOutputConfig(BaseConfig):
    name: Literal["soundevent"] = "soundevent"
    top_k: int | None = 1
    min_score: float | None = None


class SoundEventOutputFormatter(OutputFormatterProtocol[data.ClipPrediction]):
    def __init__(
        self,
        targets: TargetProtocol,
        top_k: int | None = 1,
        min_score: float | None = 0,
    ):
        self.targets = targets
        self.top_k = top_k
        self.min_score = min_score

    def format(
        self,
        predictions: Sequence[ClipDetections],
    ) -> List[data.ClipPrediction]:
        return [
            self.format_prediction(prediction) for prediction in predictions
        ]

    def save(
        self,
        predictions: Sequence[data.ClipPrediction],
        path: data.PathLike,
        audio_dir: data.PathLike | None = None,
    ) -> None:
        run = data.PredictionSet(clip_predictions=list(predictions))

        path = Path(path)

        if not path.suffix == ".json":
            path = Path(path).with_suffix(".json")

        io.save(run, path, audio_dir=audio_dir)

    def load(self, path: data.PathLike) -> List[data.ClipPrediction]:
        path = Path(path)
        run = io.load(path, type="prediction_set")
        return run.clip_predictions

    def format_prediction(
        self,
        prediction: ClipDetections,
    ) -> data.ClipPrediction:
        recording = prediction.clip.recording
        return data.ClipPrediction(
            clip=prediction.clip,
            sound_events=[
                self.format_sound_event_prediction(pred, recording)
                for pred in prediction.detections
            ],
        )

    def format_sound_event_prediction(
        self,
        prediction: Detection,
        recording: data.Recording,
    ) -> data.SoundEventPrediction:
        return data.SoundEventPrediction(
            sound_event=data.SoundEvent(
                recording=recording,
                geometry=prediction.geometry,
            ),
            score=prediction.detection_score,
            tags=self.get_sound_event_tags(prediction),
        )

    def get_sound_event_tags(
        self, prediction: Detection
    ) -> List[data.PredictedTag]:
        sorted_indices = np.argsort(prediction.class_scores)[::-1]

        tags = [
            data.PredictedTag(
                tag=tag,
                score=prediction.detection_score,
            )
            for tag in self.targets.detection_class_tags
        ]

        top_k = self.top_k or len(sorted_indices)

        for ind in sorted_indices[:top_k]:
            score = float(prediction.class_scores[ind])

            if self.min_score is not None and score < self.min_score:
                break

            class_name = self.targets.class_names[ind]
            class_tags = self.targets.decode_class(class_name)
            tags.extend(
                data.PredictedTag(
                    tag=tag,
                    score=score,
                )
                for tag in class_tags
            )

        return tags

    @prediction_formatters.register(SoundEventOutputConfig)
    @staticmethod
    def from_config(config: SoundEventOutputConfig, targets: TargetProtocol):
        return SoundEventOutputFormatter(
            targets,
            top_k=config.top_k,
            min_score=config.min_score,
        )
