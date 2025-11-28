"""Compatibility functions between old and new data structures."""

import os
import uuid
from pathlib import Path
from typing import Callable, List, Optional, Union

from pydantic import BaseModel, Field
from soundevent import data

PathLike = Union[Path, str, os.PathLike]

__all__ = []

SPECIES_TAG_KEY = "species"
ECHOLOCATION_EVENT = "Echolocation"
UNKNOWN_CLASS = "__UNKNOWN__"

NAMESPACE = uuid.UUID("97a9776b-c0fd-4c68-accb-0b0ecd719242")

CLIP_NAMESPACE = uuid.uuid5(NAMESPACE, "clip")
CLIP_ANNOTATION_NAMESPACE = uuid.uuid5(NAMESPACE, "clip_annotation")
RECORDING_NAMESPACE = uuid.uuid5(NAMESPACE, "recording")
SOUND_EVENT_NAMESPACE = uuid.uuid5(NAMESPACE, "sound_event")
SOUND_EVENT_ANNOTATION_NAMESPACE = uuid.uuid5(
    NAMESPACE, "sound_event_annotation"
)


EventFn = Callable[[data.SoundEventAnnotation], Optional[str]]

ClassFn = Callable[[data.Recording], int]

IndividualFn = Callable[[data.SoundEventAnnotation], int]


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
            SOUND_EVENT_NAMESPACE,
            f"{recording.uuid}_{annotation.start_time}_{annotation.end_time}",
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
        uuid=uuid.uuid5(
            SOUND_EVENT_ANNOTATION_NAMESPACE,
            f"{sound_event.uuid}",
        ),
        sound_event=sound_event,
        tags=get_sound_event_tags(
            annotation, label_key, event_key, individual_key
        ),
    )


def get_sound_event_tags(
    annotation: Annotation,
    label_key: str = "class",
    event_key: str = "event",
    individual_key: str = "individual",
) -> List[data.Tag]:
    """Get the tags for a sound event annotation."""
    tags = []

    if annotation.label:
        tags.append(data.Tag(key=label_key, value=annotation.label))

    if annotation.event:
        tags.append(data.Tag(key=event_key, value=annotation.event))

    if annotation.individual is not None:
        tags.append(
            data.Tag(key=individual_key, value=str(annotation.individual))
        )

    return tags


def file_annotation_to_clip(
    file_annotation: FileAnnotation,
    audio_dir: Optional[PathLike] = None,
    label_key: str = "class",
) -> data.Clip:
    """Convert file annotation to recording."""
    audio_dir = audio_dir or Path.cwd()

    full_path = Path(audio_dir) / file_annotation.id

    if not full_path.exists():
        raise FileNotFoundError(f"File {full_path} not found.")

    tags = []
    if file_annotation.label:
        tags.append(data.Tag(key=label_key, value=file_annotation.label))

    recording = data.Recording.from_file(
        full_path,
        time_expansion=file_annotation.time_exp,
        tags=tags,
    )
    recording.uuid = uuid.uuid5(RECORDING_NAMESPACE, f"{recording.hash}")

    start_time = 0
    end_time = recording.duration
    return data.Clip(
        uuid=uuid.uuid5(
            CLIP_NAMESPACE,
            f"{recording.uuid}_{start_time}_{end_time}",
        ),
        recording=recording,
        start_time=start_time,
        end_time=end_time,
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

    tags = []
    if file_annotation.label:
        tags.append(data.Tag(key=label_key, value=file_annotation.label))

    return data.ClipAnnotation(
        uuid=uuid.uuid5(CLIP_ANNOTATION_NAMESPACE, f"{clip.uuid}"),
        clip=clip,
        notes=notes,
        tags=tags,
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
