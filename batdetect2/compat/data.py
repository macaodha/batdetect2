"""Compatibility functions between old and new data structures."""

import os
import uuid
from pathlib import Path
from typing import Callable, List, Optional, Union

import numpy as np
from pydantic import BaseModel, Field
from soundevent import data
from soundevent.geometry import compute_bounds
from soundevent.types import ClassMapper

from batdetect2 import types

PathLike = Union[Path, str, os.PathLike]

__all__ = [
    "convert_to_annotation_group",
    "load_annotation_project",
]

SPECIES_TAG_KEY = "species"
ECHOLOCATION_EVENT = "Echolocation"
UNKNOWN_CLASS = "__UNKNOWN__"

NAMESPACE = uuid.UUID("97a9776b-c0fd-4c68-accb-0b0ecd719242")


EventFn = Callable[[data.SoundEventAnnotation], Optional[str]]

ClassFn = Callable[[data.Recording], int]

IndividualFn = Callable[[data.SoundEventAnnotation], int]


def get_recording_class_name(recording: data.Recording) -> str:
    """Get the class name for a recording."""
    tag = data.find_tag(recording.tags, SPECIES_TAG_KEY)
    if tag is None:
        return UNKNOWN_CLASS
    return tag.value


def get_annotation_notes(annotation: data.ClipAnnotation) -> str:
    """Get the notes for a ClipAnnotation."""
    all_notes = [
        *annotation.notes,
        *annotation.clip.recording.notes,
    ]
    messages = [note.message for note in all_notes if note.message is not None]
    return "\n".join(messages)


def convert_to_annotation_group(
    annotation: data.ClipAnnotation,
    class_mapper: ClassMapper,
    event_fn: EventFn = lambda _: ECHOLOCATION_EVENT,
    class_fn: ClassFn = lambda _: 0,
    individual_fn: IndividualFn = lambda _: 0,
) -> types.AudioLoaderAnnotationGroup:
    """Convert a ClipAnnotation to an AudioLoaderAnnotationGroup."""
    recording = annotation.clip.recording

    start_times = []
    end_times = []
    low_freqs = []
    high_freqs = []
    class_ids = []
    x_inds = []
    y_inds = []
    individual_ids = []
    annotations: List[types.Annotation] = []
    class_id_file = class_fn(recording)

    for sound_event in annotation.sound_events:
        geometry = sound_event.sound_event.geometry

        if geometry is None:
            continue

        start_time, low_freq, end_time, high_freq = compute_bounds(geometry)
        class_id = class_mapper.transform(sound_event) or -1
        event = event_fn(sound_event) or ""
        individual_id = individual_fn(sound_event) or -1

        start_times.append(start_time)
        end_times.append(end_time)
        low_freqs.append(low_freq)
        high_freqs.append(high_freq)
        class_ids.append(class_id)
        individual_ids.append(individual_id)

        # NOTE: This will be computed later so we just put a placeholder
        # here for now.
        x_inds.append(0)
        y_inds.append(0)

        annotations.append(
            {
                "start_time": start_time,
                "end_time": end_time,
                "low_freq": low_freq,
                "high_freq": high_freq,
                "class_prob": 1.0,
                "det_prob": 1.0,
                "individual": "0",
                "event": event,
                "class_id": class_id,  # type: ignore
            }
        )

    return {
        "id": str(recording.path),
        "duration": recording.duration,
        "issues": False,
        "file_path": str(recording.path),
        "time_exp": recording.time_expansion,
        "class_name": get_recording_class_name(recording),
        "notes": get_annotation_notes(annotation),
        "annotated": True,
        "start_times": np.array(start_times),
        "end_times": np.array(end_times),
        "low_freqs": np.array(low_freqs),
        "high_freqs": np.array(high_freqs),
        "class_ids": np.array(class_ids),
        "x_inds": np.array(x_inds),
        "y_inds": np.array(y_inds),
        "individual_ids": np.array(individual_ids),
        "annotation": annotations,
        "class_id_file": class_id_file,
    }


class Annotation(BaseModel):
    """Annotation class to hold batdetect annotations."""

    label: str = Field(alias="class")
    event: str
    individual: int = 0

    start_time: float
    end_time: float
    low_freq: float
    high_freq: float


class FileAnnotation(BaseModel):
    """FileAnnotation class to hold batdetect annotations for a file."""

    id: str
    duration: float
    time_exp: float = 1

    label: str = Field(alias="class_name")

    annotation: List[Annotation]

    annotated: bool = False
    issues: bool = False
    notes: str = ""


def load_file_annotation(path: PathLike) -> FileAnnotation:
    """Load annotation from batdetect format."""
    path = Path(path)
    return FileAnnotation.model_validate_json(path.read_text())


def annotation_to_sound_event(
    annotation: Annotation,
    recording: data.Recording,
    label_key: str = "class",
    event_key: str = "event",
    individual_key: str = "individual",
) -> data.SoundEventAnnotation:
    """Convert annotation to sound event annotation."""
    sound_event = data.SoundEvent(
        uuid=uuid.uuid5(
            NAMESPACE,
            f"{recording.hash}_{annotation.start_time}_{annotation.end_time}",
        ),
        recording=recording,
        geometry=data.BoundingBox(
            coordinates=[
                annotation.start_time,
                annotation.low_freq,
                annotation.end_time,
                annotation.high_freq,
            ],
        ),
    )

    return data.SoundEventAnnotation(
        uuid=uuid.uuid5(NAMESPACE, f"{sound_event.uuid}_annotation"),
        sound_event=sound_event,
        tags=[
            data.Tag(
                term=data.term_from_key(label_key),
                value=annotation.label,
            ),
            data.Tag(
                term=data.term_from_key(event_key),
                value=annotation.event,
            ),
            data.Tag(
                term=data.term_from_key(individual_key),
                value=str(annotation.individual),
            ),
        ],
    )


def file_annotation_to_clip(
    file_annotation: FileAnnotation,
    audio_dir: Optional[PathLike] = None,
) -> data.Clip:
    """Convert file annotation to recording."""
    audio_dir = audio_dir or Path.cwd()

    full_path = Path(audio_dir) / file_annotation.id

    if not full_path.exists():
        raise FileNotFoundError(f"File {full_path} not found.")

    recording = data.Recording.from_file(
        full_path,
        time_expansion=file_annotation.time_exp,
    )

    return data.Clip(
        uuid=uuid.uuid5(NAMESPACE, f"{file_annotation.id}_clip"),
        recording=recording,
        start_time=0,
        end_time=recording.duration,
    )


def file_annotation_to_clip_annotation(
    file_annotation: FileAnnotation,
    clip: data.Clip,
    label_key: str = "class",
    event_key: str = "event",
    individual_key: str = "individual",
) -> data.ClipAnnotation:
    """Convert file annotation to clip annotation."""
    notes = []
    if file_annotation.notes:
        notes.append(data.Note(message=file_annotation.notes))

    return data.ClipAnnotation(
        uuid=uuid.uuid5(NAMESPACE, f"{file_annotation.id}_clip_annotation"),
        clip=clip,
        notes=notes,
        tags=[
            data.Tag(
                term=data.term_from_key(label_key), value=file_annotation.label
            )
        ],
        sound_events=[
            annotation_to_sound_event(
                annotation,
                clip.recording,
                label_key=label_key,
                event_key=event_key,
                individual_key=individual_key,
            )
            for annotation in file_annotation.annotation
        ],
    )


def file_annotation_to_annotation_task(
    file_annotation: FileAnnotation,
    clip: data.Clip,
) -> data.AnnotationTask:
    status_badges = []

    if file_annotation.issues:
        status_badges.append(
            data.StatusBadge(state=data.AnnotationState.rejected)
        )
    elif file_annotation.annotated:
        status_badges.append(
            data.StatusBadge(state=data.AnnotationState.completed)
        )

    return data.AnnotationTask(
        uuid=uuid.uuid5(uuid.NAMESPACE_URL, f"{file_annotation.id}_task"),
        clip=clip,
        status_badges=status_badges,
    )


def list_file_annotations(path: PathLike) -> List[Path]:
    """List all annotations in a directory."""
    path = Path(path)
    return [file for file in path.glob("*.json")]


def load_annotation_project(
    path: PathLike,
    name: Optional[str] = None,
    audio_dir: Optional[PathLike] = None,
) -> data.AnnotationProject:
    """Convert annotations to annotation project."""
    audio_dir = audio_dir or Path.cwd()

    paths = list_file_annotations(path)

    if name is None:
        name = str(path)

    annotations = []
    tasks = []

    for p in paths:
        try:
            file_annotation = load_file_annotation(p)
        except FileNotFoundError:
            continue

        try:
            clip = file_annotation_to_clip(
                file_annotation,
                audio_dir=audio_dir,
            )
        except FileNotFoundError:
            continue

        annotations.append(
            file_annotation_to_clip_annotation(
                file_annotation,
                clip,
            )
        )

        tasks.append(
            file_annotation_to_annotation_task(
                file_annotation,
                clip,
            )
        )

    return data.AnnotationProject(
        name=name,
        clip_annotations=annotations,
        tasks=tasks,
    )
