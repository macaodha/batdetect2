import json
from pathlib import Path
from typing import List, Literal, Sequence, TypedDict

import numpy as np
from soundevent import data
from soundevent.geometry import compute_bounds

from batdetect2.core import BaseConfig
from batdetect2.outputs.formats.base import (
    make_path_relative,
    output_formatters,
)
from batdetect2.outputs.types import OutputFormatterProtocol
from batdetect2.postprocess.types import ClipDetections, Detection
from batdetect2.targets import terms
from batdetect2.targets.types import TargetProtocol

try:
    from typing import NotRequired  # type: ignore
except ImportError:
    from typing_extensions import NotRequired

DictWithClass = TypedDict("DictWithClass", {"class": str})


class Annotation(DictWithClass):
    start_time: float
    end_time: float
    low_freq: float
    high_freq: float
    class_prob: float
    det_prob: float
    individual: str
    event: str


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


class BatDetect2Formatter(OutputFormatterProtocol[FileAnnotation]):
    def __init__(
        self,
        targets: TargetProtocol,
        event_name: str,
        annotation_note: str,
    ):
        self.targets = targets
        self.event_name = event_name
        self.annotation_note = annotation_note

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
            pred_path = path / (prediction["id"] + ".json")

            if audio_dir is not None and "file_path" in prediction:
                prediction["file_path"] = str(
                    make_path_relative(
                        prediction["file_path"],
                        audio_dir,
                    )
                )

            pred_path.write_text(json.dumps(prediction))

    def load(self, path: data.PathLike) -> List[FileAnnotation]:
        path = Path(path)

        files = list(path.glob("*.json"))

        if not files:
            return []

        return [
            json.loads(file.read_text()) for file in files if file.is_file()
        ]

    def get_recording_class(self, annotations: List[Annotation]) -> str:
        if not annotations:
            return ""

        highest_scoring = max(annotations, key=lambda x: x["class_prob"])
        return highest_scoring["class"]

    def format_prediction(self, prediction: ClipDetections) -> FileAnnotation:
        recording = prediction.clip.recording

        annotations = [
            self.format_sound_event_prediction(pred)
            for pred in prediction.detections
        ]

        return FileAnnotation(
            id=recording.path.name,
            file_path=str(recording.path),
            annotated=False,
            duration=recording.duration,
            issues=False,
            time_exp=recording.time_expansion,
            class_name=self.get_recording_class(annotations),
            notes=self.annotation_note,
            annotation=annotations,
        )

    def get_class_name(self, class_index: int) -> str:
        class_name = self.targets.class_names[class_index]
        tags = self.targets.decode_class(class_name)
        return data.find_tag_value(
            tags,
            term=terms.generic_class,
            default=class_name,
        )  # type: ignore

    def format_sound_event_prediction(
        self, prediction: Detection
    ) -> Annotation:
        start_time, low_freq, end_time, high_freq = compute_bounds(
            prediction.geometry
        )

        top_class_index = int(np.argmax(prediction.class_scores))
        top_class_score = float(prediction.class_scores[top_class_index])
        top_class = self.get_class_name(top_class_index)
        return Annotation(
            start_time=start_time,
            end_time=end_time,
            low_freq=low_freq,
            high_freq=high_freq,
            class_prob=top_class_score,
            det_prob=float(prediction.detection_score),
            individual="",
            event=self.event_name,
            **{"class": top_class},
        )

    @output_formatters.register(BatDetect2OutputConfig)
    @staticmethod
    def from_config(config: BatDetect2OutputConfig, targets: TargetProtocol):
        return BatDetect2Formatter(
            targets,
            event_name=config.event_name,
            annotation_note=config.annotation_note,
        )
