import uuid
from pathlib import Path
from typing import Callable, Optional, Sequence

import pytest
from pydantic import ValidationError
from soundevent import data, io
from soundevent.data.annotation_tasks import AnnotationState

from batdetect2.data.annotations import aoef


@pytest.fixture
def base_dir(tmp_path: Path) -> Path:
    path = tmp_path / "base_dir"
    path.mkdir(parents=True, exist_ok=True)
    return path


@pytest.fixture
def audio_dir(base_dir: Path) -> Path:
    path = base_dir / "audio"
    path.mkdir(parents=True, exist_ok=True)
    return path


@pytest.fixture
def anns_dir(base_dir: Path) -> Path:
    path = base_dir / "annotations"
    path.mkdir(parents=True, exist_ok=True)
    return path


def create_task(
    clip: data.Clip,
    badges: list[data.StatusBadge],
    task_id: Optional[uuid.UUID] = None,
) -> data.AnnotationTask:
    """Creates a simple AnnotationTask for testing."""
    return data.AnnotationTask(
        uuid=task_id or uuid.uuid4(),
        clip=clip,
        status_badges=badges,
    )


def test_annotation_task_filter_defaults():
    """Test default values of AnnotationTaskFilter."""
    f = aoef.AnnotationTaskFilter()
    assert f.only_completed is True
    assert f.only_verified is False
    assert f.exclude_issues is True


def test_annotation_task_filter_initialization():
    """Test initialization of AnnotationTaskFilter with non-default values."""
    f = aoef.AnnotationTaskFilter(
        only_completed=False,
        only_verified=True,
        exclude_issues=False,
    )
    assert f.only_completed is False
    assert f.only_verified is True
    assert f.exclude_issues is False


def test_aoef_annotations_defaults(
    audio_dir: Path,
    anns_dir: Path,
):
    """Test default values of AOEFAnnotations."""
    annotations_path = anns_dir / "test.aoef"
    config = aoef.AOEFAnnotations(
        name="default_name",
        audio_dir=audio_dir,
        annotations_path=annotations_path,
    )
    assert config.format == "aoef"
    assert config.annotations_path == annotations_path
    assert config.audio_dir == audio_dir
    assert isinstance(config.filter, aoef.AnnotationTaskFilter)
    assert config.filter.only_completed is True
    assert config.filter.only_verified is False
    assert config.filter.exclude_issues is True


def test_aoef_annotations_initialization(tmp_path):
    """Test initialization of AOEFAnnotations with specific values."""
    annotations_path = tmp_path / "custom.json"
    audio_dir = Path("audio/files")
    custom_filter = aoef.AnnotationTaskFilter(
        only_completed=False, only_verified=True
    )
    config = aoef.AOEFAnnotations(
        name="custom_name",
        description="custom_desc",
        audio_dir=audio_dir,
        annotations_path=annotations_path,
        filter=custom_filter,
    )
    assert config.name == "custom_name"
    assert config.description == "custom_desc"
    assert config.format == "aoef"
    assert config.audio_dir == audio_dir
    assert config.annotations_path == annotations_path
    assert config.filter is custom_filter


def test_aoef_annotations_initialization_no_filter(tmp_path):
    """Test initialization of AOEFAnnotations with filter=None."""
    annotations_path = tmp_path / "no_filter.aoef"
    audio_dir = tmp_path / "audio"
    config = aoef.AOEFAnnotations(
        name="no_filter_name",
        description="no_filter_desc",
        audio_dir=audio_dir,
        annotations_path=annotations_path,
        filter=None,
    )
    assert config.format == "aoef"
    assert config.annotations_path == annotations_path
    assert config.filter is None


def test_aoef_annotations_validation_error(tmp_path):
    """Test Pydantic validation for missing required fields."""
    with pytest.raises(ValidationError, match="annotations_path"):
        aoef.AOEFAnnotations(  # type: ignore
            name="test_name",
            audio_dir=tmp_path,
        )
    with pytest.raises(ValidationError, match="name"):
        aoef.AOEFAnnotations(  # type: ignore
            annotations_path=tmp_path / "dummy.aoef",
            audio_dir=tmp_path,
        )


@pytest.mark.parametrize(
    "badges, only_completed, only_verified, exclude_issues, expected",
    [
        ([], True, False, True, False),  # No badges -> not completed
        (
            [data.StatusBadge(state=AnnotationState.completed)],
            True,
            False,
            True,
            True,
        ),
        (
            [data.StatusBadge(state=AnnotationState.verified)],
            True,
            False,
            True,
            False,
        ),  # Not completed
        (
            [data.StatusBadge(state=AnnotationState.rejected)],
            True,
            False,
            True,
            False,
        ),  # Has issues
        (
            [
                data.StatusBadge(state=AnnotationState.completed),
                data.StatusBadge(state=AnnotationState.rejected),
            ],
            True,
            False,
            True,
            False,
        ),  # Completed but has issues
        (
            [
                data.StatusBadge(state=AnnotationState.completed),
                data.StatusBadge(state=AnnotationState.verified),
            ],
            True,
            False,
            True,
            True,
        ),  # Completed, verified doesn't matter
        # Verified only (completed=F, verified=T, exclude_issues=T)
        (
            [data.StatusBadge(state=AnnotationState.verified)],
            False,
            True,
            True,
            True,
        ),
        (
            [data.StatusBadge(state=AnnotationState.completed)],
            False,
            True,
            True,
            False,
        ),  # Not verified
        (
            [
                data.StatusBadge(state=AnnotationState.verified),
                data.StatusBadge(state=AnnotationState.rejected),
            ],
            False,
            True,
            True,
            False,
        ),  # Verified but has issues
        # Completed AND Verified (completed=T, verified=T, exclude_issues=T)
        (
            [
                data.StatusBadge(state=AnnotationState.completed),
                data.StatusBadge(state=AnnotationState.verified),
            ],
            True,
            True,
            True,
            True,
        ),
        (
            [data.StatusBadge(state=AnnotationState.completed)],
            True,
            True,
            True,
            False,
        ),  # Not verified
        (
            [data.StatusBadge(state=AnnotationState.verified)],
            True,
            True,
            True,
            False,
        ),  # Not completed
        # Include Issues (completed=T, verified=F, exclude_issues=F)
        (
            [
                data.StatusBadge(state=AnnotationState.completed),
                data.StatusBadge(state=AnnotationState.rejected),
            ],
            True,
            False,
            False,
            True,
        ),  # Completed, issues allowed
        (
            [data.StatusBadge(state=AnnotationState.rejected)],
            True,
            False,
            False,
            False,
        ),  # Has issues, but not completed
        # No filters (completed=F, verified=F, exclude_issues=F)
        ([], False, False, False, True),
        (
            [data.StatusBadge(state=AnnotationState.rejected)],
            False,
            False,
            False,
            True,
        ),
        (
            [data.StatusBadge(state=AnnotationState.completed)],
            False,
            False,
            False,
            True,
        ),
        (
            [data.StatusBadge(state=AnnotationState.verified)],
            False,
            False,
            False,
            True,
        ),
    ],
)
def test_select_task(
    badges: Sequence[data.StatusBadge],
    only_completed: bool,
    only_verified: bool,
    exclude_issues: bool,
    expected: bool,
    create_recording: Callable[..., data.Recording],
    create_clip: Callable[..., data.Clip],
):
    """Test select_task logic with various badge and filter combinations."""
    rec = create_recording()
    clip = create_clip(rec)
    task = create_task(clip, badges=list(badges))
    result = aoef.select_task(
        task,
        only_completed=only_completed,
        only_verified=only_verified,
        exclude_issues=exclude_issues,
    )
    assert result == expected


def test_filter_ready_clips_default(
    tmp_path: Path,
    create_recording: Callable[..., data.Recording],
    create_clip: Callable[..., data.Clip],
    create_clip_annotation: Callable[..., data.ClipAnnotation],
    create_annotation_project: Callable[..., data.AnnotationProject],
):
    """Test filter_ready_clips with default filtering."""
    rec = create_recording(path=tmp_path / "rec.wav")
    clip_completed = create_clip(rec, 0, 1)
    clip_verified = create_clip(rec, 1, 2)
    clip_rejected = create_clip(rec, 2, 3)
    clip_completed_rejected = create_clip(rec, 3, 4)
    clip_no_badges = create_clip(rec, 4, 5)

    task_completed = create_task(
        clip_completed, [data.StatusBadge(state=AnnotationState.completed)]
    )
    task_verified = create_task(
        clip_verified, [data.StatusBadge(state=AnnotationState.verified)]
    )
    task_rejected = create_task(
        clip_rejected, [data.StatusBadge(state=AnnotationState.rejected)]
    )
    task_completed_rejected = create_task(
        clip_completed_rejected,
        [
            data.StatusBadge(state=AnnotationState.completed),
            data.StatusBadge(state=AnnotationState.rejected),
        ],
    )
    task_no_badges = create_task(clip_no_badges, [])

    ann_completed = create_clip_annotation(clip_completed)
    ann_verified = create_clip_annotation(clip_verified)
    ann_rejected = create_clip_annotation(clip_rejected)
    ann_completed_rejected = create_clip_annotation(clip_completed_rejected)
    ann_no_badges = create_clip_annotation(clip_no_badges)

    project = create_annotation_project(
        name="FilterTestProject",
        description="Project for testing filters",
        tasks=[
            task_completed,
            task_verified,
            task_rejected,
            task_completed_rejected,
            task_no_badges,
        ],
        annotations=[
            ann_completed,
            ann_verified,
            ann_rejected,
            ann_completed_rejected,
            ann_no_badges,
        ],
    )

    filtered_set = aoef.filter_ready_clips(project)

    assert isinstance(filtered_set, data.AnnotationSet)
    assert filtered_set.name == project.name
    assert filtered_set.description == project.description
    assert len(filtered_set.clip_annotations) == 1
    assert filtered_set.clip_annotations[0].clip.uuid == clip_completed.uuid

    expected_uuid = uuid.uuid5(project.uuid, f"{True}_{False}_{True}")
    assert filtered_set.uuid == expected_uuid


def test_filter_ready_clips_custom_filter(
    tmp_path: Path,
    create_recording: Callable[..., data.Recording],
    create_clip: Callable[..., data.Clip],
    create_clip_annotation: Callable[..., data.ClipAnnotation],
    create_annotation_project: Callable[..., data.AnnotationProject],
):
    """Test filter_ready_clips with custom filtering (verified=T, issues=F)."""
    rec = create_recording(path=tmp_path / "rec.wav")
    clip_completed = create_clip(rec, 0, 1)
    clip_verified = create_clip(rec, 1, 2)
    clip_rejected = create_clip(rec, 2, 3)
    clip_completed_verified = create_clip(rec, 3, 4)
    clip_verified_rejected = create_clip(rec, 4, 5)

    task_completed = create_task(
        clip_completed, [data.StatusBadge(state=AnnotationState.completed)]
    )
    task_verified = create_task(
        clip_verified, [data.StatusBadge(state=AnnotationState.verified)]
    )
    task_rejected = create_task(
        clip_rejected, [data.StatusBadge(state=AnnotationState.rejected)]
    )
    task_completed_verified = create_task(
        clip_completed_verified,
        [
            data.StatusBadge(state=AnnotationState.completed),
            data.StatusBadge(state=AnnotationState.verified),
        ],
    )
    task_verified_rejected = create_task(
        clip_verified_rejected,
        [
            data.StatusBadge(state=AnnotationState.verified),
            data.StatusBadge(state=AnnotationState.rejected),
        ],
    )

    ann_completed = create_clip_annotation(clip_completed)
    ann_verified = create_clip_annotation(clip_verified)
    ann_rejected = create_clip_annotation(clip_rejected)
    ann_completed_verified = create_clip_annotation(clip_completed_verified)
    ann_verified_rejected = create_clip_annotation(clip_verified_rejected)

    project = create_annotation_project(
        tasks=[
            task_completed,
            task_verified,
            task_rejected,
            task_completed_verified,
            task_verified_rejected,
        ],
        annotations=[
            ann_completed,
            ann_verified,
            ann_rejected,
            ann_completed_verified,
            ann_verified_rejected,
        ],
    )

    filtered_set = aoef.filter_ready_clips(
        project, only_completed=False, only_verified=True, exclude_issues=False
    )

    assert len(filtered_set.clip_annotations) == 3
    filtered_clip_uuids = {
        ann.clip.uuid for ann in filtered_set.clip_annotations
    }
    assert clip_verified.uuid in filtered_clip_uuids
    assert clip_completed_verified.uuid in filtered_clip_uuids
    assert clip_verified_rejected.uuid in filtered_clip_uuids

    expected_uuid = uuid.uuid5(project.uuid, f"{False}_{True}_{False}")
    assert filtered_set.uuid == expected_uuid


def test_filter_ready_clips_no_filters(
    tmp_path: Path,
    create_recording: Callable[..., data.Recording],
    create_clip: Callable[..., data.Clip],
    create_clip_annotation: Callable[..., data.ClipAnnotation],
    create_annotation_project: Callable[..., data.AnnotationProject],
):
    """Test filter_ready_clips with all filters disabled."""
    rec = create_recording(path=tmp_path / "rec.wav")
    clip1 = create_clip(rec, 0, 1)
    clip2 = create_clip(rec, 1, 2)

    task1 = create_task(
        clip1, [data.StatusBadge(state=AnnotationState.rejected)]
    )
    task2 = create_task(clip2, [])
    ann1 = create_clip_annotation(clip1)
    ann2 = create_clip_annotation(clip2)

    project = create_annotation_project(
        tasks=[task1, task2], annotations=[ann1, ann2]
    )

    filtered_set = aoef.filter_ready_clips(
        project,
        only_completed=False,
        only_verified=False,
        exclude_issues=False,
    )

    assert len(filtered_set.clip_annotations) == 2
    filtered_clip_uuids = {
        ann.clip.uuid for ann in filtered_set.clip_annotations
    }
    assert clip1.uuid in filtered_clip_uuids
    assert clip2.uuid in filtered_clip_uuids

    expected_uuid = uuid.uuid5(project.uuid, f"{False}_{False}_{False}")
    assert filtered_set.uuid == expected_uuid


def test_filter_ready_clips_empty_project(
    create_annotation_project: Callable[..., data.AnnotationProject],
):
    """Test filter_ready_clips with an empty project."""
    project = create_annotation_project(tasks=[], annotations=[])
    filtered_set = aoef.filter_ready_clips(project)
    assert len(filtered_set.clip_annotations) == 0
    assert filtered_set.name == project.name
    assert filtered_set.description == project.description


def test_filter_ready_clips_no_matching_tasks(
    tmp_path: Path,
    create_recording: Callable[..., data.Recording],
    create_clip: Callable[..., data.Clip],
    create_clip_annotation: Callable[..., data.ClipAnnotation],
    create_annotation_project: Callable[..., data.AnnotationProject],
):
    """Test filter_ready_clips when no tasks match the criteria."""
    rec = create_recording(path=tmp_path / "rec.wav")
    clip_rejected = create_clip(rec, 0, 1)

    task_rejected = create_task(
        clip_rejected, [data.StatusBadge(state=AnnotationState.rejected)]
    )
    ann_rejected = create_clip_annotation(clip_rejected)

    project = create_annotation_project(
        tasks=[task_rejected], annotations=[ann_rejected]
    )

    filtered_set = aoef.filter_ready_clips(project)
    assert len(filtered_set.clip_annotations) == 0


def test_load_aoef_annotated_dataset_set(
    tmp_path: Path,
    create_recording: Callable[..., data.Recording],
    create_clip: Callable[..., data.Clip],
    create_clip_annotation: Callable[..., data.ClipAnnotation],
    create_annotation_set: Callable[..., data.AnnotationSet],
):
    """Test loading a standard AnnotationSet file."""
    rec_path = tmp_path / "audio" / "rec1.wav"
    rec_path.parent.mkdir()
    rec = create_recording(path=rec_path)
    clip = create_clip(rec)
    ann = create_clip_annotation(clip)
    original_set = create_annotation_set(annotations=[ann])

    annotations_file = tmp_path / "set.json"
    io.save(original_set, annotations_file)

    config = aoef.AOEFAnnotations(
        name="test_set_load",
        annotations_path=annotations_file,
        audio_dir=rec_path.parent,
    )

    loaded_set = aoef.load_aoef_annotated_dataset(config)

    assert isinstance(loaded_set, data.AnnotationSet)

    assert loaded_set.name == original_set.name
    assert len(loaded_set.clip_annotations) == len(
        original_set.clip_annotations
    )
    assert (
        loaded_set.clip_annotations[0].clip.uuid
        == original_set.clip_annotations[0].clip.uuid
    )
    assert (
        loaded_set.clip_annotations[0].clip.recording.path
        == rec_path.resolve()
    )


def test_load_aoef_annotated_dataset_project_with_filter(
    tmp_path: Path,
    create_recording: Callable[..., data.Recording],
    create_clip: Callable[..., data.Clip],
    create_clip_annotation: Callable[..., data.ClipAnnotation],
    create_annotation_project: Callable[..., data.AnnotationProject],
):
    """Test loading an AnnotationProject file with filtering enabled."""
    rec_path = tmp_path / "audio" / "rec.wav"
    rec_path.parent.mkdir()
    rec = create_recording(path=rec_path)

    clip_completed = create_clip(rec, 0, 1)
    clip_rejected = create_clip(rec, 1, 2)

    task_completed = create_task(
        clip_completed, [data.StatusBadge(state=AnnotationState.completed)]
    )
    task_rejected = create_task(
        clip_rejected, [data.StatusBadge(state=AnnotationState.rejected)]
    )

    ann_completed = create_clip_annotation(clip_completed)
    ann_rejected = create_clip_annotation(clip_rejected)

    project = create_annotation_project(
        name="ProjectToFilter",
        tasks=[task_completed, task_rejected],
        annotations=[ann_completed, ann_rejected],
    )

    annotations_file = tmp_path / "project.json"
    io.save(project, annotations_file)

    config = aoef.AOEFAnnotations(
        name="test_project_filter_load",
        annotations_path=annotations_file,
        audio_dir=rec_path.parent,
    )

    loaded_data = aoef.load_aoef_annotated_dataset(config)

    assert isinstance(loaded_data, data.AnnotationSet)
    assert loaded_data.name == project.name
    assert len(loaded_data.clip_annotations) == 1
    assert loaded_data.clip_annotations[0].clip.uuid == clip_completed.uuid
    assert (
        loaded_data.clip_annotations[0].clip.recording.path
        == rec_path.resolve()
    )


def test_load_aoef_annotated_dataset_project_no_filter(
    tmp_path: Path,
    create_recording: Callable[..., data.Recording],
    create_clip: Callable[..., data.Clip],
    create_clip_annotation: Callable[..., data.ClipAnnotation],
    create_annotation_project: Callable[..., data.AnnotationProject],
):
    """Test loading an AnnotationProject file with filtering disabled."""
    rec_path = tmp_path / "audio" / "rec.wav"
    rec_path.parent.mkdir()
    rec = create_recording(path=rec_path)
    clip1 = create_clip(rec, 0, 1)
    clip2 = create_clip(rec, 1, 2)

    task1 = create_task(
        clip1, [data.StatusBadge(state=AnnotationState.completed)]
    )
    task2 = create_task(
        clip2, [data.StatusBadge(state=AnnotationState.rejected)]
    )
    ann1 = create_clip_annotation(clip1)
    ann2 = create_clip_annotation(clip2)

    original_project = create_annotation_project(
        tasks=[task1, task2], annotations=[ann1, ann2]
    )

    annotations_file = tmp_path / "project_nofilter.json"
    io.save(original_project, annotations_file)

    config = aoef.AOEFAnnotations(
        name="test_project_nofilter_load",
        annotations_path=annotations_file,
        audio_dir=rec_path.parent,
        filter=None,
    )

    loaded_data = aoef.load_aoef_annotated_dataset(config)

    assert isinstance(loaded_data, data.AnnotationProject)
    assert loaded_data.uuid == original_project.uuid
    assert len(loaded_data.clip_annotations) == 2
    assert (
        loaded_data.clip_annotations[0].clip.recording.path
        == rec_path.resolve()
    )
    assert (
        loaded_data.clip_annotations[1].clip.recording.path
        == rec_path.resolve()
    )


def test_load_aoef_annotated_dataset_base_dir(
    tmp_path: Path,
    create_recording: Callable[..., data.Recording],
    create_clip: Callable[..., data.Clip],
    create_clip_annotation: Callable[..., data.ClipAnnotation],
    create_annotation_project: Callable[..., data.AnnotationProject],
):
    """Test loading with a base_dir specified."""
    base = tmp_path / "basedir"
    base.mkdir()
    audio_rel = Path("audio")
    ann_rel = Path("annotations/project.json")

    abs_audio_dir = base / audio_rel
    abs_ann_path = base / ann_rel
    abs_audio_dir.mkdir(parents=True)
    abs_ann_path.parent.mkdir(parents=True)

    rec = create_recording(path=abs_audio_dir / "rec.wav")
    rec_path = rec.path

    clip = create_clip(rec)

    task = create_task(
        clip, [data.StatusBadge(state=AnnotationState.completed)]
    )
    ann = create_clip_annotation(clip)
    project = create_annotation_project(tasks=[task], annotations=[ann])
    io.save(project, abs_ann_path)

    config = aoef.AOEFAnnotations(
        name="test_base_dir_load",
        annotations_path=ann_rel,
        audio_dir=audio_rel,
        filter=aoef.AnnotationTaskFilter(),
    )

    loaded_set = aoef.load_aoef_annotated_dataset(config, base_dir=base)

    assert isinstance(loaded_set, data.AnnotationSet)
    assert len(loaded_set.clip_annotations) == 1

    assert (
        loaded_set.clip_annotations[0].clip.recording.path
        == rec_path.resolve()
    )


def test_load_aoef_annotated_dataset_file_not_found(tmp_path):
    """Test FileNotFoundError when annotation file doesn't exist."""
    config = aoef.AOEFAnnotations(
        name="test_not_found",
        annotations_path=tmp_path / "nonexistent.aoef",
        audio_dir=tmp_path,
    )
    with pytest.raises(FileNotFoundError):
        aoef.load_aoef_annotated_dataset(config)


def test_load_aoef_annotated_dataset_file_not_found_with_base_dir(tmp_path):
    """Test FileNotFoundError with base_dir."""
    base = tmp_path / "base"
    base.mkdir()
    config = aoef.AOEFAnnotations(
        name="test_not_found_base",
        annotations_path=Path("nonexistent.aoef"),
        audio_dir=Path("audio"),
    )
    with pytest.raises(FileNotFoundError):
        aoef.load_aoef_annotated_dataset(config, base_dir=base)


def test_load_aoef_annotated_dataset_invalid_content(tmp_path):
    """Test ValueError when file contains invalid JSON or non-soundevent data."""
    invalid_file = tmp_path / "invalid.json"
    invalid_file.write_text("{invalid json")

    config = aoef.AOEFAnnotations(
        name="test_invalid_content",
        annotations_path=invalid_file,
        audio_dir=tmp_path,
    )
    with pytest.raises(ValidationError):
        aoef.load_aoef_annotated_dataset(config)


def test_load_aoef_annotated_dataset_wrong_object_type(
    tmp_path: Path,
    create_recording: Callable[..., data.Recording],
):
    """Test ValueError when file contains correct soundevent obj but wrong type."""
    rec_path = tmp_path / "audio" / "rec.wav"
    rec_path.parent.mkdir()
    rec = create_recording(path=rec_path)
    dataset = data.Dataset(
        name="test_wrong_type",
        description="Test for wrong type",
        recordings=[rec],
    )

    wrong_type_file = tmp_path / "wrong_type.json"
    io.save(dataset, wrong_type_file)  # type: ignore

    config = aoef.AOEFAnnotations(
        name="test_wrong_type",
        annotations_path=wrong_type_file,
        audio_dir=rec_path.parent,
    )

    with pytest.raises(ValueError) as excinfo:
        aoef.load_aoef_annotated_dataset(config)

    assert (
        "does not contain a soundevent AnnotationSet or AnnotationProject"
        in str(excinfo.value)
    )
