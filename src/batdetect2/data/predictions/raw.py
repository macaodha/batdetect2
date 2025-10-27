from collections import defaultdict
from pathlib import Path
from typing import List, Literal, Optional, Sequence
from uuid import UUID, uuid4

import numpy as np
import xarray as xr
from soundevent import data
from soundevent.geometry import compute_bounds

from batdetect2.core import BaseConfig
from batdetect2.data.predictions.base import (
    make_path_relative,
    prediction_formatters,
)
from batdetect2.typing import (
    BatDetect2Prediction,
    OutputFormatterProtocol,
    RawPrediction,
    TargetProtocol,
)


class RawOutputConfig(BaseConfig):
    name: Literal["raw"] = "raw"

    include_class_scores: bool = True
    include_features: bool = True
    include_geometry: bool = True


class RawFormatter(OutputFormatterProtocol[BatDetect2Prediction]):
    def __init__(
        self,
        targets: TargetProtocol,
        include_class_scores: bool = True,
        include_features: bool = True,
        include_geometry: bool = True,
    ):
        self.targets = targets
        self.include_class_scores = include_class_scores
        self.include_features = include_features
        self.include_geometry = include_geometry

    def format(
        self,
        predictions: Sequence[BatDetect2Prediction],
    ) -> List[BatDetect2Prediction]:
        return list(predictions)

    def save(
        self,
        predictions: Sequence[BatDetect2Prediction],
        path: data.PathLike,
        audio_dir: Optional[data.PathLike] = None,
    ) -> None:
        num_features = 0

        tree = xr.DataTree()

        for prediction in predictions:
            clip = prediction.clip
            recording = clip.recording

            if audio_dir is not None:
                recording = recording.model_copy(
                    update=dict(
                        path=make_path_relative(recording.path, audio_dir)
                    )
                )

            clip_data = defaultdict(list)

            for pred in prediction.predictions:
                detection_id = str(uuid4())

                clip_data["detection_id"].append(detection_id)
                clip_data["detection_score"].append(pred.detection_score)

                start_time, low_freq, end_time, high_freq = compute_bounds(
                    pred.geometry
                )

                clip_data["start_time"].append(start_time)
                clip_data["end_time"].append(end_time)
                clip_data["low_freq"].append(low_freq)
                clip_data["high_freq"].append(high_freq)

                clip_data["geometry"].append(pred.geometry.model_dump_json())

                top_class_index = int(np.argmax(pred.class_scores))
                top_class_score = float(pred.class_scores[top_class_index])
                top_class = self.targets.class_names[top_class_index]

                clip_data["top_class"].append(top_class)
                clip_data["top_class_score"].append(top_class_score)

                clip_data["class_scores"].append(pred.class_scores)
                clip_data["features"].append(pred.features)

                num_features = len(pred.features)

            data_vars = {
                "score": (["detection"], clip_data["detection_score"]),
                "start_time": (["detection"], clip_data["start_time"]),
                "end_time": (["detection"], clip_data["end_time"]),
                "low_freq": (["detection"], clip_data["low_freq"]),
                "high_freq": (["detection"], clip_data["high_freq"]),
                "top_class": (["detection"], clip_data["top_class"]),
                "top_class_score": (
                    ["detection"],
                    clip_data["top_class_score"],
                ),
            }

            coords = {
                "detection": ("detection", clip_data["detection_id"]),
                "clip_start": clip.start_time,
                "clip_end": clip.end_time,
                "clip_id": str(clip.uuid),
            }

            if self.include_class_scores:
                data_vars["class_scores"] = (
                    ["detection", "classes"],
                    clip_data["class_scores"],
                )
                coords["classes"] = ("classes", self.targets.class_names)

            if self.include_features:
                data_vars["features"] = (
                    ["detection", "feature"],
                    clip_data["features"],
                )
                coords["feature"] = ("feature", np.arange(num_features))

            if self.include_geometry:
                data_vars["geometry"] = (["detection"], clip_data["geometry"])

            dataset = xr.Dataset(
                data_vars=data_vars,
                coords=coords,
                attrs={
                    "recording": recording.model_dump_json(exclude_none=True),
                },
            )

            tree = tree.assign(
                {
                    str(clip.uuid): xr.DataTree(
                        dataset=dataset,
                        name=str(clip.uuid),
                    )
                }
            )

        path = Path(path)

        if not path.suffix == ".nc":
            path = Path(path).with_suffix(".nc")

        tree.to_netcdf(path)

    def load(self, path: data.PathLike) -> List[BatDetect2Prediction]:
        path = Path(path)

        root = xr.load_datatree(path)

        predictions: List[BatDetect2Prediction] = []

        for _, clip_data in root.items():
            recording = data.Recording.model_validate_json(
                clip_data.attrs["recording"]
            )

            clip_id = clip_data.clip_id.item()
            clip = data.Clip(
                recording=recording,
                uuid=UUID(clip_id),
                start_time=clip_data.clip_start,
                end_time=clip_data.clip_end,
            )

            sound_events = []

            for detection in clip_data.detection:
                score = clip_data.score.sel(detection=detection).item()

                if "geometry" in clip_data:
                    geometry = data.geometry_validate(
                        clip_data.geometry.sel(detection=detection).item()
                    )
                else:
                    start_time = clip_data.start_time.sel(detection=detection)
                    end_time = clip_data.end_time.sel(detection=detection)
                    low_freq = clip_data.low_freq.sel(detection=detection)
                    high_freq = clip_data.high_freq.sel(detection=detection)
                    geometry = data.BoundingBox(
                        coordinates=[start_time, low_freq, end_time, high_freq]
                    )

                if "class_scores" in clip_data:
                    class_scores = clip_data.class_scores.sel(
                        detection=detection
                    ).data
                else:
                    class_scores = np.zeros(len(self.targets.class_names))
                    class_index = self.targets.class_names.index(
                        clip_data.top_class.sel(detection=detection).item()
                    )
                    class_scores[class_index] = clip_data.top_class_score.sel(
                        detection=detection
                    ).item()

                if "features" in clip_data:
                    features = clip_data.features.sel(detection=detection).data
                else:
                    features = np.zeros(0)

                sound_events.append(
                    RawPrediction(
                        geometry=geometry,
                        detection_score=score,
                        class_scores=class_scores,
                        features=features,
                    )
                )

            predictions.append(
                BatDetect2Prediction(
                    clip=clip,
                    predictions=sound_events,
                )
            )

        return predictions

    @prediction_formatters.register(RawOutputConfig)
    @staticmethod
    def from_config(config: RawOutputConfig, targets: TargetProtocol):
        return RawFormatter(
            targets,
            include_class_scores=config.include_class_scores,
            include_features=config.include_features,
            include_geometry=config.include_geometry,
        )
