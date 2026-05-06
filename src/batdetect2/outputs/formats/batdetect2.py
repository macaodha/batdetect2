import json
from pathlib import Path
from typing import List, Literal, Sequence, TypedDict

import numpy as np
import pandas as pd
from soundevent import data
from soundevent import terms as soundevent_terms
from soundevent.geometry import compute_bounds

from batdetect2.core import BaseConfig
from batdetect2.outputs.formats.base import (
    make_path_relative,
    output_formatters,
)
from batdetect2.outputs.types import OutputFormatterProtocol
from batdetect2.postprocess.types import ClipDetections, Detection
from batdetect2.targets.types import TargetProtocol

try:
    from typing import NotRequired  # type: ignore
except ImportError:
    from typing_extensions import NotRequired

DictWithClass = TypedDict("DictWithClass", {"class": str})


class Annotation(DictWithClass, total=False):
    start_time: float
    end_time: float
    low_freq: float
    high_freq: float
    class_prob: float
    det_prob: float
    individual: str
    event: str
    cnn_features: NotRequired[list[float]]  # ty: ignore[invalid-type-form]


class FileAnnotation(TypedDict):
    id: str
    annotated: bool
    duration: float
    issues: bool
    time_exp: float
    class_name: str
    notes: str
    annotation: List[Annotation]
    file_path: NotRequired[str]  # ty: ignore[invalid-type-form]


class BatDetect2OutputConfig(BaseConfig):
    name: Literal["batdetect2"] = "batdetect2"

    event_name: str = "Echolocation"
    annotation_note: str = "Automatically generated."
    write_detection_csv: bool = True
    write_cnn_features_csv: bool = False
    save_if_empty: bool = False
    preserve_audio_tree: bool = True
    include_file_path: bool = False


class BatDetect2Formatter(OutputFormatterProtocol[FileAnnotation]):
    def __init__(
        self,
        targets: TargetProtocol,
        event_name: str,
        annotation_note: str,
        write_detection_csv: bool = True,
        write_cnn_features_csv: bool = False,
        save_if_empty: bool = False,
        preserve_audio_tree: bool = True,
        include_file_path: bool = False,
    ):
        self.targets = targets
        self.event_name = event_name
        self.annotation_note = annotation_note
        self.write_detection_csv = write_detection_csv
        self.write_cnn_features_csv = write_cnn_features_csv
        self.save_if_empty = save_if_empty
        self.preserve_audio_tree = preserve_audio_tree
        self.include_file_path = include_file_path

    def format(
        self, predictions: Sequence[ClipDetections]
    ) -> List[FileAnnotation]:
        return [
            self.format_prediction(prediction) for prediction in predictions
        ]

    def save(
        self,
        predictions: Sequence[FileAnnotation],
        path: data.PathLike,
        audio_dir: data.PathLike | None = None,
    ) -> None:
        path = Path(path)

        if not path.is_dir():
            path.mkdir(parents=True)

        for prediction in predictions:
            annotations = prediction["annotation"]

            if not annotations and not self.save_if_empty:
                continue

            pred_path = self.get_output_path(prediction, path, audio_dir)
            pred_path.parent.mkdir(parents=True, exist_ok=True)

            # make a copy of the prediction
            data = dict(prediction)

            raw_file_path = data.get("file_path")
            if audio_dir is not None and isinstance(raw_file_path, str):
                data["file_path"] = str(
                    make_path_relative(raw_file_path, audio_dir)
                )

            if not self.include_file_path:
                data.pop("file_path", None)

            data["annotation"] = [
                {
                    key: value
                    for key, value in annotation.items()
                    if key != "cnn_features"
                }
                for annotation in data["annotation"]
            ]

            pred_path.write_text(json.dumps(data, indent=2, sort_keys=True))

            if self.write_detection_csv:
                self.save_detection_csv(
                    prediction,
                    pred_path.with_suffix(".csv"),
                )

            if self.write_cnn_features_csv:
                self.save_cnn_features_csv(
                    prediction,
                    pred_path.with_name(pred_path.stem + "_cnn_features.csv"),
                )

    def load(self, path: data.PathLike) -> List[FileAnnotation]:
        path = Path(path)

        if path.is_file():
            files = [path] if path.suffix == ".json" else []
        else:
            files = sorted(path.rglob("*.json"))

        if not files:
            return []

        return [
            json.loads(file.read_text()) for file in files if file.is_file()
        ]

    def get_output_path(
        self,
        prediction: FileAnnotation,
        output_dir: Path,
        audio_dir: data.PathLike | None,
    ) -> Path:
        if (
            self.preserve_audio_tree
            and audio_dir is not None
            and "file_path" in prediction
        ):
            relative_path = make_path_relative(
                prediction["file_path"],
                audio_dir,
            )
            return (
                output_dir / relative_path.parent / f"{prediction['id']}.json"
            )

        return output_dir / f"{prediction['id']}.json"

    def save_detection_csv(
        self,
        prediction: FileAnnotation,
        path: Path,
    ) -> None:
        annotations = prediction["annotation"]
        if not annotations:
            return

        preds_df = pd.DataFrame(annotations)[
            [
                "det_prob",
                "start_time",
                "end_time",
                "high_freq",
                "low_freq",
                "class",
                "class_prob",
            ]
        ]
        preds_df.to_csv(path, sep=",")

    def save_cnn_features_csv(
        self, prediction: FileAnnotation, path: Path
    ) -> None:
        annotations = prediction["annotation"]

        if not annotations:
            return

        cnn_features = [
            annotation["cnn_features"]
            for annotation in annotations
            if "cnn_features" in annotation
        ]

        if not cnn_features:
            return

        cnn_feats_df = pd.DataFrame(
            cnn_features,
            columns=[str(ii) for ii in range(len(cnn_features[0]))],
        )

        cnn_feats_df.to_csv(
            path,
            sep=",",
            index=False,
            float_format="%.5f",
        )

    def get_class_name(self, class_index: int) -> str:
        class_name = self.targets.class_names[class_index]
        tags = self.targets.decode_class(class_name)
        return data.find_tag_value(
            tags,
            term=soundevent_terms.scientific_name,
            default=class_name,
        )  # type: ignore

    def get_recording_class(self, detections: Sequence[Detection]) -> str:
        if not detections:
            return "None"

        class_scores = np.stack(
            [detection.class_scores for detection in detections],
            axis=1,
        )
        detection_scores = np.array(
            [detection.detection_score for detection in detections],
            dtype=np.float32,
        )
        weighted_scores = (class_scores * detection_scores).sum(axis=1)

        total = weighted_scores.sum()

        if total <= 0:
            return "None"

        top_class_index = int(np.argmax(weighted_scores / total))
        return self.get_class_name(top_class_index)

    def format_prediction(self, prediction: ClipDetections) -> FileAnnotation:
        recording = prediction.clip.recording

        annotations = [
            self.format_sound_event_prediction(pred)
            for pred in prediction.detections
        ]

        file_annotation = FileAnnotation(
            id=recording.path.name,
            annotated=False,
            duration=round(float(recording.duration), 4),
            issues=False,
            time_exp=recording.time_expansion,
            class_name=self.get_recording_class(prediction.detections),
            notes=self.annotation_note,
            annotation=annotations,
            file_path=str(recording.path),
        )

        return file_annotation

    def format_sound_event_prediction(
        self, prediction: Detection
    ) -> Annotation:
        start_time, low_freq, end_time, high_freq = compute_bounds(
            prediction.geometry
        )

        top_class_index = int(np.argmax(prediction.class_scores))
        top_class_score = float(prediction.class_scores[top_class_index])
        top_class = self.get_class_name(top_class_index)
        annotation: Annotation = {
            "start_time": round(float(start_time), 4),
            "end_time": round(float(end_time), 4),
            "low_freq": int(low_freq),
            "high_freq": int(high_freq),
            "class_prob": round(top_class_score, 3),
            "det_prob": round(float(prediction.detection_score), 3),
            "individual": "-1",
            "event": self.event_name,
            "class": top_class,
        }

        if self.write_cnn_features_csv:
            annotation["cnn_features"] = prediction.features.tolist()  # type: ignore[index]

        return annotation

    @output_formatters.register(BatDetect2OutputConfig)
    @staticmethod
    def from_config(config: BatDetect2OutputConfig, targets: TargetProtocol):
        return BatDetect2Formatter(
            targets,
            event_name=config.event_name,
            annotation_note=config.annotation_note,
            write_detection_csv=config.write_detection_csv,
            write_cnn_features_csv=config.write_cnn_features_csv,
            save_if_empty=config.save_if_empty,
            preserve_audio_tree=config.preserve_audio_tree,
            include_file_path=config.include_file_path,
        )
