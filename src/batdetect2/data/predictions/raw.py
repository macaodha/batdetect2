from collections import defaultdict
from pathlib import Path
from typing import List, Literal, Optional, Sequence
from uuid import UUID, uuid4

import numpy as np
import xarray as xr
from loguru import logger
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
        parse_full_geometry: bool = False,
    ):
        self.targets = targets
        self.include_class_scores = include_class_scores
        self.include_features = include_features
        self.include_geometry = include_geometry
        self.parse_full_geometry = parse_full_geometry

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
        path = Path(path)

        if not path.exists():
            path.mkdir(parents=True)

        for prediction in predictions:
            logger.debug(f"Saving clip predictions {prediction.clip.uuid}")
            clip = prediction.clip
            dataset = self.pred_to_xr(prediction, audio_dir)
            dataset.to_netcdf(path / f"{clip.uuid}.nc")

    def load(self, path: data.PathLike) -> List[BatDetect2Prediction]:
        path = Path(path)
        files = list(path.glob("*.nc"))
        predictions: List[BatDetect2Prediction] = []

        for filepath in files:
            logger.debug(f"Loading clip predictions {filepath}")
            clip_data = xr.load_dataset(filepath)
            prediction = self.pred_from_xr(clip_data)
            predictions.append(prediction)

        return predictions

    def pred_to_xr(
        self,
        prediction: BatDetect2Prediction,
        audio_dir: Optional[data.PathLike] = None,
    ) -> xr.Dataset:
        clip = prediction.clip
        recording = clip.recording
        num_features = 0

        if audio_dir is not None:
            recording = recording.model_copy(
                update=dict(path=make_path_relative(recording.path, audio_dir))
            )

        data = defaultdict(list)

        for pred in prediction.predictions:
            detection_id = str(uuid4())

            data["detection_id"].append(detection_id)
            data["detection_score"].append(pred.detection_score)

            start_time, low_freq, end_time, high_freq = compute_bounds(
                pred.geometry
            )

            data["start_time"].append(start_time)
            data["end_time"].append(end_time)
            data["low_freq"].append(low_freq)
            data["high_freq"].append(high_freq)

            data["geometry"].append(pred.geometry.model_dump_json())

            top_class_index = int(np.argmax(pred.class_scores))
            top_class_score = float(pred.class_scores[top_class_index])
            top_class = self.targets.class_names[top_class_index]

            data["top_class"].append(top_class)
            data["top_class_score"].append(top_class_score)

            data["class_scores"].append(pred.class_scores)
            data["features"].append(pred.features)

            num_features = len(pred.features)

        data_vars = {
            "score": (["detection"], data["detection_score"]),
            "start_time": (["detection"], data["start_time"]),
            "end_time": (["detection"], data["end_time"]),
            "low_freq": (["detection"], data["low_freq"]),
            "high_freq": (["detection"], data["high_freq"]),
            "top_class": (["detection"], data["top_class"]),
            "top_class_score": (["detection"], data["top_class_score"]),
        }

        coords = {
            "detection": ("detection", data["detection_id"]),
            "clip_start": clip.start_time,
            "clip_end": clip.end_time,
            "clip_id": str(clip.uuid),
        }

        if self.include_class_scores:
            class_scores = np.stack(data["class_scores"], axis=0)
            data_vars["class_scores"] = (
                ["detection", "classes"],
                class_scores,
            )
            coords["classes"] = ("classes", self.targets.class_names)

        if self.include_features:
            features = np.stack(data["features"], axis=0)
            data_vars["features"] = (["detection", "feature"], features)
            coords["feature"] = ("feature", np.arange(num_features))

        if self.include_geometry:
            data_vars["geometry"] = (["detection"], data["geometry"])

        return xr.Dataset(
            data_vars=data_vars,
            coords=coords,
            attrs={
                "recording": recording.model_dump_json(exclude_none=True),
            },
        )

    def pred_from_xr(self, dataset: xr.Dataset) -> BatDetect2Prediction:
        clip_data = dataset
        clip_id = clip_data.clip_id.item()

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

        for detection in clip_data.coords["detection"]:
            detection_data = clip_data.sel(detection=detection)
            score = detection_data.score.item()

            if "geometry" in clip_data and self.parse_full_geometry:
                geometry = data.geometry_validate(
                    detection_data.geometry.item()
                )
            else:
                start_time = detection_data.start_time.item()
                end_time = detection_data.end_time.item()
                low_freq = detection_data.low_freq.item()
                high_freq = detection_data.high_freq.item()
                geometry = data.BoundingBox.model_construct(
                    coordinates=[start_time, low_freq, end_time, high_freq]
                )

            if "class_scores" in detection_data:
                class_scores = detection_data.class_scores.data
            else:
                class_scores = np.zeros(len(self.targets.class_names))
                class_index = self.targets.class_names.index(
                    detection_data.top_class.item()
                )
                class_scores[class_index] = (
                    detection_data.top_class_score.item()
                )

            if "features" in detection_data:
                features = detection_data.features.data
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

        return BatDetect2Prediction(
            clip=clip,
            predictions=sound_events,
        )

    @prediction_formatters.register(RawOutputConfig)
    @staticmethod
    def from_config(config: RawOutputConfig, targets: TargetProtocol):
        return RawFormatter(
            targets,
            include_class_scores=config.include_class_scores,
            include_features=config.include_features,
            include_geometry=config.include_geometry,
        )
