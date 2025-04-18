import json
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest
from soundevent import data

from batdetect2.data.annotations.batdetect2 import (
    AnnotationFilter,
    BatDetect2FilesAnnotations,
    BatDetect2MergedAnnotations,
    load_batdetect2_files_annotated_dataset,
    load_batdetect2_merged_annotated_dataset,
)


def create_legacy_file_annotation(
    file_id: str,
    duration: float = 5.0,
    time_exp: float = 1.0,
    class_name: str = "Myotis",
    annotations: Optional[List[Dict[str, Any]]] = None,
    annotated: bool = True,
    issues: bool = False,
    notes: str = "",
) -> Dict[str, Any]:
    if annotations is None:
        annotations = [
            {
                "class": "Myotis",
                "event": "Echolocation",
                "individual": 0,
                "start_time": 1.1,
                "end_time": 1.2,
                "low_freq": 30000,
                "high_freq": 40000,
            },
            {
                "class": "Pipistrellus",
                "event": "Echolocation",
                "individual": 0,
                "start_time": 2.5,
                "end_time": 2.55,
                "low_freq": 50000,
                "high_freq": 55000,
            },
        ]
    return {
        "id": file_id,
        "duration": duration,
        "time_exp": time_exp,
        "class_name": class_name,
        "annotation": annotations,
        "annotated": annotated,
        "issues": issues,
        "notes": notes,
    }


@pytest.fixture
def batdetect2_files_test_setup(
    tmp_path: Path, wav_factory
) -> Tuple[Path, Path, List[Dict[str, Any]]]:
    """Sets up a directory structure for batdetect2 files format tests."""
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    anns_dir = tmp_path / "anns"
    anns_dir.mkdir()

    files_data = []

    # 1. File with single myotis annotation
    rec1_path = wav_factory(path=audio_dir / "rec1.wav", duration=5.0)
    ann1_data = create_legacy_file_annotation(
        file_id="rec1.wav",
        annotated=True,
        issues=False,
        notes="Standard notes.",
        class_name="Myotis",
        annotations=[
            {
                "class": "Myotis",
                "event": "Echolocation",
                "individual": 0,
                "start_time": 1.1,
                "end_time": 1.2,
                "low_freq": 30000,
                "high_freq": 40000,
            }
        ],
    )
    (anns_dir / f"{rec1_path.name}.json").write_text(json.dumps(ann1_data))
    files_data.append(ann1_data)

    # 2. File that has not been annotated
    rec2_path = wav_factory(path=audio_dir / "rec2.wav", duration=4.0)
    ann2_data = create_legacy_file_annotation(
        file_id="rec2.wav",
        annotated=False,
        issues=False,
        annotations=[],
        class_name="Unknown",
    )
    (anns_dir / f"{rec2_path.name}.json").write_text(json.dumps(ann2_data))
    files_data.append(ann2_data)

    # 3. File that has been annotated but has issues
    rec3_path = wav_factory(path=audio_dir / "rec3.wav", duration=6.0)
    ann3_data = create_legacy_file_annotation(
        file_id="rec3.wav",
        annotated=True,
        issues=True,
        notes="File has issues.",
        class_name="Pipistrellus",
    )
    (anns_dir / f"{rec3_path.name}.json").write_text(json.dumps(ann3_data))
    files_data.append(ann3_data)

    # 4. File that has been not been annotated and has issues
    rec4_path = wav_factory(path=audio_dir / "rec4.wav", duration=3.0)
    ann4_data = create_legacy_file_annotation(
        file_id="rec4.wav", annotated=False, issues=True, class_name="Nyctalus"
    )
    (anns_dir / f"{rec4_path.name}.json").write_text(json.dumps(ann4_data))
    files_data.append(ann4_data)

    # 5. File that has been annotated but is missing audio
    ann5_data = create_legacy_file_annotation(
        file_id="rec_missing_audio.wav", annotated=True, issues=False
    )
    (anns_dir / "rec_missing_audio.wav.json").write_text(json.dumps(ann5_data))

    # 6. File that has missing annotations
    wav_factory(path=audio_dir / "rec_missing_ann.wav", duration=2.0)

    # 7. A non -JSON file in the annotations directory
    (anns_dir / "not_a_json.txt").write_text("hello")

    return audio_dir, anns_dir, files_data


@pytest.fixture
def batdetect2_merged_test_setup(
    tmp_path: Path, batdetect2_files_test_setup
) -> Tuple[Path, Path, List[Dict[str, Any]]]:
    """Sets up a directory structure for batdetect2 merged file format tests."""
    audio_dir, _, files_data = batdetect2_files_test_setup
    merged_anns_path = tmp_path / "merged_anns.json"

    merged_data = [
        fd for fd in files_data if fd["id"] != "rec_missing_audio.wav"
    ]
    merged_anns_path.write_text(json.dumps(merged_data))

    return audio_dir, merged_anns_path, merged_data


def test_annotation_filter_defaults():
    """Test default values for AnnotationFilter."""
    filt = AnnotationFilter()
    assert filt.only_annotated is True
    assert filt.exclude_issues is True


def test_annotation_filter_custom():
    """Test custom values for AnnotationFilter."""
    filt = AnnotationFilter(only_annotated=False, exclude_issues=False)
    assert filt.only_annotated is False
    assert filt.exclude_issues is False


def test_batdetect2_files_annotations_config(tmp_path: Path):
    """Test initialization of BatDetect2FilesAnnotations."""
    anns_dir = tmp_path / "annotations"
    config = BatDetect2FilesAnnotations(
        name="test_files",
        description="Test Files Desc",
        audio_dir=tmp_path / "audio",
        annotations_dir=anns_dir,
    )
    assert config.format == "batdetect2"
    assert config.name == "test_files"
    assert config.description == "Test Files Desc"
    assert config.annotations_dir == anns_dir
    assert isinstance(config.filter, AnnotationFilter)
    assert config.filter.only_annotated is True
    assert config.filter.exclude_issues is True


def test_batdetect2_files_annotations_config_no_filter(tmp_path: Path):
    """Test BatDetect2FilesAnnotations with filter explicitly set to None."""
    anns_dir = tmp_path / "annotations"
    data = {
        "name": "test_files_no_filter",
        "audio_dir": str(tmp_path / "audio"),
        "annotations_dir": str(anns_dir),
        "filter": None,
    }
    config = BatDetect2FilesAnnotations.model_validate(data)
    assert config.filter is None


def test_batdetect2_merged_annotations_config(tmp_path: Path):
    """Test initialization of BatDetect2MergedAnnotations."""
    anns_path = tmp_path / "annotations.json"
    config = BatDetect2MergedAnnotations(
        name="test_merged",
        description="Test Merged Desc",
        audio_dir=tmp_path / "audio",
        annotations_path=anns_path,
        filter=AnnotationFilter(only_annotated=False, exclude_issues=True),
    )
    assert config.format == "batdetect2_file"
    assert config.name == "test_merged"
    assert config.description == "Test Merged Desc"
    assert config.annotations_path == anns_path
    assert isinstance(config.filter, AnnotationFilter)
    assert config.filter.only_annotated is False
    assert config.filter.exclude_issues is True


def test_batdetect2_merged_annotations_config_default_filter(tmp_path: Path):
    """Test BatDetect2MergedAnnotations uses default filter if not provided."""
    anns_path = tmp_path / "annotations.json"
    config = BatDetect2MergedAnnotations(
        name="test_merged_default",
        audio_dir=tmp_path / "audio",
        annotations_path=anns_path,
    )
    assert isinstance(config.filter, AnnotationFilter)
    assert config.filter.only_annotated is True
    assert config.filter.exclude_issues is True


class TestLoadBatDetect2Files:
    def test_load_default_filter(self, batdetect2_files_test_setup):
        """Test loading with default filter (annotated=True, issues=False)."""
        audio_dir, anns_dir, _ = batdetect2_files_test_setup
        config = BatDetect2FilesAnnotations(
            name="default_load",
            audio_dir=audio_dir,
            annotations_dir=anns_dir,
        )

        result_set = load_batdetect2_files_annotated_dataset(config)

        assert isinstance(result_set, data.AnnotationSet)
        assert result_set.name == "default_load"
        assert len(result_set.clip_annotations) == 1

        clip_ann = result_set.clip_annotations[0]
        assert clip_ann.clip.recording.path.name == "rec1.wav"
        assert clip_ann.clip.recording.duration == 5.0
        assert len(clip_ann.sound_events) == 1
        assert clip_ann.notes[0].message == "Standard notes."
        clip_tag = data.find_tag(clip_ann.tags, "class")
        assert clip_tag is not None
        assert clip_tag.value == "Myotis"

        recording_tag = data.find_tag(clip_ann.clip.recording.tags, "class")
        assert recording_tag is not None
        assert recording_tag.value == "Myotis"

        se_ann = clip_ann.sound_events[0]
        assert se_ann.sound_event.geometry is not None
        assert se_ann.sound_event.geometry.coordinates == [
            1.1,
            30000,
            1.2,
            40000,
        ]

        se_class_tag = data.find_tag(se_ann.tags, "class")
        assert se_class_tag is not None
        assert se_class_tag.value == "Myotis"

        se_event_tag = data.find_tag(se_ann.tags, "event")
        assert se_event_tag is not None
        assert se_event_tag.value == "Echolocation"

        se_individual_tag = data.find_tag(se_ann.tags, "individual")
        assert se_individual_tag is not None
        assert se_individual_tag.value == "0"

    def test_load_only_annotated_false(self, batdetect2_files_test_setup):
        """Test filter with only_annotated=False."""
        audio_dir, anns_dir, _ = batdetect2_files_test_setup
        config = BatDetect2FilesAnnotations(
            name="ann_false",
            audio_dir=audio_dir,
            annotations_dir=anns_dir,
            filter=AnnotationFilter(only_annotated=False, exclude_issues=True),
        )
        result_set = load_batdetect2_files_annotated_dataset(config)
        assert len(result_set.clip_annotations) == 2
        loaded_files = {
            ann.clip.recording.path.name for ann in result_set.clip_annotations
        }
        assert loaded_files == {"rec1.wav", "rec2.wav"}

    def test_load_exclude_issues_false(self, batdetect2_files_test_setup):
        """Test filter with exclude_issues=False."""
        audio_dir, anns_dir, _ = batdetect2_files_test_setup
        config = BatDetect2FilesAnnotations(
            name="iss_false",
            audio_dir=audio_dir,
            annotations_dir=anns_dir,
            filter=AnnotationFilter(only_annotated=True, exclude_issues=False),
        )
        result_set = load_batdetect2_files_annotated_dataset(config)
        assert len(result_set.clip_annotations) == 2
        loaded_files = {
            ann.clip.recording.path.name for ann in result_set.clip_annotations
        }
        assert loaded_files == {"rec1.wav", "rec3.wav"}

    def test_load_no_filter(self, batdetect2_files_test_setup):
        """Test loading with filtering disabled."""
        audio_dir, anns_dir, _ = batdetect2_files_test_setup
        config_data = {
            "name": "no_filter",
            "audio_dir": str(audio_dir),
            "annotations_dir": str(anns_dir),
            "filter": None,
        }
        config = BatDetect2FilesAnnotations.model_validate(config_data)

        result_set = load_batdetect2_files_annotated_dataset(config)
        assert len(result_set.clip_annotations) == 4
        loaded_files = {
            ann.clip.recording.path.name for ann in result_set.clip_annotations
        }
        assert loaded_files == {"rec1.wav", "rec2.wav", "rec3.wav", "rec4.wav"}

    def test_load_with_base_dir(self, tmp_path, batdetect2_files_test_setup):
        """Test loading with a base_dir."""
        audio_dir_abs, anns_dir_abs, _ = batdetect2_files_test_setup
        base_dir = tmp_path
        audio_dir_rel = audio_dir_abs.relative_to(base_dir)
        anns_dir_rel = anns_dir_abs.relative_to(base_dir)

        config = BatDetect2FilesAnnotations(
            name="base_dir_test",
            audio_dir=audio_dir_rel,
            annotations_dir=anns_dir_rel,
        )

        result_set = load_batdetect2_files_annotated_dataset(
            config, base_dir=base_dir
        )
        assert len(result_set.clip_annotations) == 1
        assert result_set.clip_annotations[0].clip.recording.path.is_absolute()
        assert (
            result_set.clip_annotations[0].clip.recording.path
            == audio_dir_abs / "rec1.wav"
        )

    def test_load_missing_annotations_dir(self, tmp_path):
        """Test error when annotations_dir does not exist."""
        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()
        anns_dir = tmp_path / "non_existent_anns"
        config = BatDetect2FilesAnnotations(
            name="missing_anns",
            audio_dir=audio_dir,
            annotations_dir=anns_dir,
        )
        result_set = load_batdetect2_files_annotated_dataset(config)
        assert len(result_set.clip_annotations) == 0

    def test_load_missing_audio_dir(self, batdetect2_files_test_setup):
        """Test error or skipping when audio_dir does not exist or files missing."""
        _, anns_dir, _ = batdetect2_files_test_setup
        missing_audio_dir = Path(
            "/tmp/non_existent_audio_dir_" + str(uuid.uuid4())
        )
        config = BatDetect2FilesAnnotations(
            name="missing_audio",
            audio_dir=missing_audio_dir,
            annotations_dir=anns_dir,
            filter=None,
        )
        result_set = load_batdetect2_files_annotated_dataset(config)
        assert len(result_set.clip_annotations) == 0

    def test_load_skips_invalid_json(self, batdetect2_files_test_setup):
        """Test that invalid JSON files are skipped."""
        audio_dir, anns_dir, _ = batdetect2_files_test_setup
        (anns_dir / "invalid.json").write_text(".invalid json")
        (anns_dir / "wrong_structure.json").write_text("[1, 2, 3]")

        config = BatDetect2FilesAnnotations(
            name="invalid_json_test",
            audio_dir=audio_dir,
            annotations_dir=anns_dir,
            filter=None,
        )
        result_set = load_batdetect2_files_annotated_dataset(config)
        assert len(result_set.clip_annotations) == 4

    def test_load_skips_missing_individual_audio(
        self, batdetect2_files_test_setup
    ):
        """Test skipping a file if its corresponding audio is missing."""
        audio_dir, anns_dir, _ = batdetect2_files_test_setup
        config = BatDetect2FilesAnnotations(
            name="skip_missing_audio",
            audio_dir=audio_dir,
            annotations_dir=anns_dir,
            filter=None,
        )
        result_set = load_batdetect2_files_annotated_dataset(config)
        assert len(result_set.clip_annotations) == 4
        loaded_files = {
            ann.clip.recording.path.name for ann in result_set.clip_annotations
        }
        assert "rec_missing_audio.wav" not in loaded_files


class TestLoadBatDetect2Merged:
    def test_load_default_filter(self, batdetect2_merged_test_setup):
        """Test loading merged file with default filter."""
        audio_dir, anns_path, _ = batdetect2_merged_test_setup
        config = BatDetect2MergedAnnotations(
            name="merged_default",
            audio_dir=audio_dir,
            annotations_path=anns_path,
        )

        result_set = load_batdetect2_merged_annotated_dataset(config)

        assert isinstance(result_set, data.AnnotationSet)
        assert result_set.name == "merged_default"
        assert len(result_set.clip_annotations) == 1

        clip_ann = result_set.clip_annotations[0]
        assert clip_ann.clip.recording.path.name == "rec1.wav"
        assert clip_ann.clip.recording.duration == 5.0
        assert len(clip_ann.sound_events) == 1

        clip_class_tag = data.find_tag(clip_ann.tags, "class")
        assert clip_class_tag is not None
        assert clip_class_tag.value == "Myotis"

    def test_load_only_annotated_false(self, batdetect2_merged_test_setup):
        """Test merged filter with only_annotated=False."""
        audio_dir, anns_path, _ = batdetect2_merged_test_setup
        config = BatDetect2MergedAnnotations(
            name="merged_ann_false",
            audio_dir=audio_dir,
            annotations_path=anns_path,
            filter=AnnotationFilter(only_annotated=False, exclude_issues=True),
        )
        result_set = load_batdetect2_merged_annotated_dataset(config)
        assert len(result_set.clip_annotations) == 2
        loaded_files = {
            ann.clip.recording.path.name for ann in result_set.clip_annotations
        }
        assert loaded_files == {"rec1.wav", "rec2.wav"}

    def test_load_exclude_issues_false(self, batdetect2_merged_test_setup):
        """Test merged filter with exclude_issues=False."""
        audio_dir, anns_path, _ = batdetect2_merged_test_setup
        config = BatDetect2MergedAnnotations(
            name="merged_iss_false",
            audio_dir=audio_dir,
            annotations_path=anns_path,
            filter=AnnotationFilter(only_annotated=True, exclude_issues=False),
        )
        result_set = load_batdetect2_merged_annotated_dataset(config)
        assert len(result_set.clip_annotations) == 2
        loaded_files = {
            ann.clip.recording.path.name for ann in result_set.clip_annotations
        }
        assert loaded_files == {"rec1.wav", "rec3.wav"}

    def test_load_no_filter(self, batdetect2_merged_test_setup):
        """Test loading merged file with filtering disabled."""
        audio_dir, anns_path, _ = batdetect2_merged_test_setup
        config_data = {
            "name": "merged_no_filter",
            "audio_dir": str(audio_dir),
            "annotations_path": str(anns_path),
            "filter": None,
        }
        config = BatDetect2MergedAnnotations.model_validate(config_data)

        result_set = load_batdetect2_merged_annotated_dataset(config)
        assert len(result_set.clip_annotations) == 4
        loaded_files = {
            ann.clip.recording.path.name for ann in result_set.clip_annotations
        }
        assert loaded_files == {"rec1.wav", "rec2.wav", "rec3.wav", "rec4.wav"}

    def test_load_with_base_dir(self, tmp_path, batdetect2_merged_test_setup):
        """Test loading merged file with a base_dir."""
        audio_dir_abs, anns_path_abs, _ = batdetect2_merged_test_setup
        base_dir = tmp_path
        audio_dir_rel = audio_dir_abs.relative_to(base_dir)
        anns_path_rel = anns_path_abs.relative_to(base_dir)

        config = BatDetect2MergedAnnotations(
            name="merged_base_dir",
            audio_dir=audio_dir_rel,
            annotations_path=anns_path_rel,
        )

        result_set = load_batdetect2_merged_annotated_dataset(
            config, base_dir=base_dir
        )
        assert len(result_set.clip_annotations) == 1
        assert result_set.clip_annotations[0].clip.recording.path.is_absolute()
        assert (
            result_set.clip_annotations[0].clip.recording.path
            == audio_dir_abs / "rec1.wav"
        )

    def test_load_missing_annotations_path(self, tmp_path):
        """Test error when annotations_path does not exist."""
        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()
        anns_path = tmp_path / "non_existent_anns.json"
        config = BatDetect2MergedAnnotations(
            name="missing_anns_file",
            audio_dir=audio_dir,
            annotations_path=anns_path,
        )
        with pytest.raises(FileNotFoundError):
            load_batdetect2_merged_annotated_dataset(config)

    def test_load_missing_audio_dir(self, batdetect2_merged_test_setup):
        """Test error/skipping when audio_dir does not exist in merged."""
        _, anns_path, _ = batdetect2_merged_test_setup
        missing_audio_dir = Path(
            "/tmp/non_existent_audio_dir_merged_" + str(uuid.uuid4())
        )
        config = BatDetect2MergedAnnotations(
            name="missing_audio_merged",
            audio_dir=missing_audio_dir,
            annotations_path=anns_path,
            filter=None,
        )
        result_set = load_batdetect2_merged_annotated_dataset(config)
        assert len(result_set.clip_annotations) == 0

    def test_load_invalid_json_format(self, tmp_path):
        """Test error for malformed JSON file."""
        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()
        anns_path = tmp_path / "invalid.json"
        anns_path.write_text(".malformed json")
        config = BatDetect2MergedAnnotations(
            name="invalid_json",
            audio_dir=audio_dir,
            annotations_path=anns_path,
        )
        with pytest.raises(json.JSONDecodeError):
            load_batdetect2_merged_annotated_dataset(config)

    def test_load_json_not_a_list(self, tmp_path):
        """Test error if JSON root is not a list."""
        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()
        anns_path = tmp_path / "not_list.json"
        anns_path.write_text('{"not": "a list"}')
        config = BatDetect2MergedAnnotations(
            name="not_list", audio_dir=audio_dir, annotations_path=anns_path
        )
        with pytest.raises(TypeError):
            load_batdetect2_merged_annotated_dataset(config)

    def test_load_invalid_entry_in_list(self, batdetect2_merged_test_setup):
        """Test skipping entries that don't conform to FileAnnotation."""
        audio_dir, anns_path, merged_data = batdetect2_merged_test_setup
        invalid_entry = {"wrong_field": 123}
        merged_data_with_invalid = merged_data + [invalid_entry]
        anns_path.write_text(json.dumps(merged_data_with_invalid))

        config = BatDetect2MergedAnnotations(
            name="invalid_entry",
            audio_dir=audio_dir,
            annotations_path=anns_path,
            filter=None,
        )
        result_set = load_batdetect2_merged_annotated_dataset(config)
        assert len(result_set.clip_annotations) == 4

    def test_load_skips_missing_audio_in_merged(
        self, batdetect2_merged_test_setup
    ):
        """Test skipping an entry if its audio file is missing."""
        audio_dir, anns_path, merged_data = batdetect2_merged_test_setup
        missing_audio_entry = create_legacy_file_annotation(
            file_id="non_existent.wav", annotated=True, issues=False
        )
        merged_data_with_missing = merged_data + [missing_audio_entry]
        anns_path.write_text(json.dumps(merged_data_with_missing))

        config = BatDetect2MergedAnnotations(
            name="skip_missing_audio_merged",
            audio_dir=audio_dir,
            annotations_path=anns_path,
            filter=None,
        )
        result_set = load_batdetect2_merged_annotated_dataset(config)
        assert len(result_set.clip_annotations) == 4
        loaded_files = {
            ann.clip.recording.path.name for ann in result_set.clip_annotations
        }
        assert "non_existent.wav" not in loaded_files
