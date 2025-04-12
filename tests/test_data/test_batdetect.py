"""Test suite for loading batdetect annotations."""

from pathlib import Path

from soundevent import data

from batdetect2.data import BatDetect2FilesAnnotations, load_annotated_dataset

ROOT_DIR = Path(__file__).parent.parent.parent


def test_load_example_annotation_project():
    path = ROOT_DIR / "example_data" / "anns"
    audio_dir = ROOT_DIR / "example_data" / "audio"
    project = load_annotated_dataset(
        BatDetect2FilesAnnotations(
            name="test",
            audio_dir=audio_dir,
            annotations_dir=path,
        )
    )
    assert isinstance(project, data.AnnotationProject)
    assert project.name == "test"
    assert len(project.clip_annotations) == 3
    assert len(project.tasks) == 3
