"""Test suite for loading batdetect annotations."""

from pathlib import Path

from soundevent import data

from batdetect2.data.compat import load_annotation_project

ROOT_DIR = Path(__file__).parent.parent.parent


def test_load_example_annotation_project():
    path = ROOT_DIR / "example_data" / "anns"
    audio_dir = ROOT_DIR / "example_data" / "audio"
    project = load_annotation_project(path, audio_dir=audio_dir)
    assert isinstance(project, data.AnnotationProject)
    assert project.name == str(path)
    assert len(project.clip_annotations) == 3
    assert len(project.tasks) == 3
