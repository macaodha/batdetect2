import json
from pathlib import Path
from typing import List, Literal, Optional, Sequence
from uuid import UUID

import numpy as np
import pandas as pd
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


class ParquetOutputConfig(BaseConfig):
    name: Literal["parquet"] = "parquet"

    include_class_scores: bool = True
    include_features: bool = True
    include_geometry: bool = True


class ParquetFormatter(OutputFormatterProtocol[BatDetect2Prediction]):
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
        path = Path(path)

        if not path.parent.exists():
            path.parent.mkdir(parents=True)

        # Ensure the file has .parquet extension if it's a file path
        if path.suffix != ".parquet":
             # If it's a directory, we might want to save as a partitioned dataset or a single file inside
             # For now, let's assume the user provides a full file path or a directory where we save 'predictions.parquet'
             if path.is_dir() or not path.suffix:
                 path = path / "predictions.parquet"

        rows = []
        for prediction in predictions:
            clip = prediction.clip
            recording = clip.recording
            
            if audio_dir is not None:
                recording = recording.model_copy(
                    update=dict(path=make_path_relative(recording.path, audio_dir))
                )

            recording_json = recording.model_dump_json(exclude_none=True)

            for pred in prediction.predictions:
                row = {
                    "clip_uuid": str(clip.uuid),
                    "clip_start_time": clip.start_time,
                    "clip_end_time": clip.end_time,
                    "recording_info": recording_json,
                    "detection_score": pred.detection_score,
                }

                if self.include_geometry:
                    # Store geometry as [start_time, low_freq, end_time, high_freq]
                    start_time, low_freq, end_time, high_freq = compute_bounds(
                        pred.geometry
                    )
                    row["start_time"] = start_time
                    row["low_freq"] = low_freq
                    row["end_time"] = end_time
                    row["high_freq"] = high_freq
                    
                    # Store full geometry as JSON
                    row["geometry"] = pred.geometry.model_dump_json()

                if self.include_class_scores:
                    row["class_scores"] = pred.class_scores.tolist()
                
                if self.include_features:
                    row["features"] = pred.features.tolist()

                rows.append(row)
        
        if not rows:
            logger.warning("No predictions to save.")
            return

        df = pd.DataFrame(rows)
        logger.info(f"Saving {len(df)} predictions to {path}")
        df.to_parquet(path, index=False)

    def load(self, path: data.PathLike) -> List[BatDetect2Prediction]:
        path = Path(path)
        if path.is_dir():
             # Try to find parquet files
             files = list(path.glob("*.parquet"))
             if not files:
                 return []
             # Read all and concatenate
             dfs = [pd.read_parquet(f) for f in files]
             df = pd.concat(dfs, ignore_index=True)
        else:
            df = pd.read_parquet(path)

        predictions_by_clip = {}

        for _, row in df.iterrows():
            clip_uuid = row["clip_uuid"]
            
            if clip_uuid not in predictions_by_clip:
                recording = data.Recording.model_validate_json(row["recording_info"])
                clip = data.Clip(
                    uuid=UUID(clip_uuid),
                    recording=recording,
                    start_time=row["clip_start_time"],
                    end_time=row["clip_end_time"],
                )
                predictions_by_clip[clip_uuid] = {
                    "clip": clip,
                    "preds": []
                }
            
            # Reconstruct geometry
            if "geometry" in row and row["geometry"]:
                geometry = data.geometry_validate(row["geometry"])
            else:
                geometry = data.BoundingBox.model_construct(
                    coordinates=[
                        row["start_time"],
                        row["low_freq"],
                        row["end_time"],
                        row["high_freq"]
                    ]
                )

            class_scores = np.array(row["class_scores"]) if "class_scores" in row else np.zeros(len(self.targets.class_names))
            features = np.array(row["features"]) if "features" in row else np.zeros(0)

            pred = RawPrediction(
                geometry=geometry,
                detection_score=row["detection_score"],
                class_scores=class_scores,
                features=features,
            )
            predictions_by_clip[clip_uuid]["preds"].append(pred)

        results = []
        for clip_data in predictions_by_clip.values():
            results.append(
                BatDetect2Prediction(
                    clip=clip_data["clip"],
                    predictions=clip_data["preds"]
                )
            )
        
        return results

    @prediction_formatters.register(ParquetOutputConfig)
    @staticmethod
    def from_config(config: ParquetOutputConfig, targets: TargetProtocol):
        return ParquetFormatter(
            targets,
            include_class_scores=config.include_class_scores,
            include_features=config.include_features,
            include_geometry=config.include_geometry,
        )
