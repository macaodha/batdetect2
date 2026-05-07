"""Tests for the public load_annotated_dataset dispatcher.

Covers:
- load_annotated_dataset dispatches correctly to each of the three
  registered annotation format loaders.
- The base_dir keyword argument is forwarded to the loader.

The internal load functions, config models, and filtering logic for each
format are tested in their respective modules
"""

import json
from pathlib import Path

from soundevent import data, io

from batdetect2.data.annotations import (
    AOEFAnnotations,
    BatDetect2FilesAnnotations,
    BatDetect2MergedAnnotations,
    load_annotated_dataset,
)

ROOT_DIR = Path(__file__).parent.parent.parent.parent


class TestLoadAnnotatedDataset:
    def test_load_example_batdetect2_files_annotation_project(self):
        """load_annotated_dataset works end-to-end with BatDetect2Files format.

        Uses the committed example data so this test doubles as a smoke test
        against the real on-disk data.
        """
        audio_dir = ROOT_DIR / "example_data" / "audio"
        anns_dir = ROOT_DIR / "example_data" / "anns"
        config = BatDetect2FilesAnnotations(
            name="test",
            audio_dir=audio_dir,
            annotations_dir=anns_dir,
        )

        result = load_annotated_dataset(config)

        assert isinstance(result, data.AnnotationSet)
        assert result.name == "test"
        assert len(result.clip_annotations) == 3

    def test_dispatches_to_aoef_loader(
        self,
        tmp_path: Path,
        create_recording,
        create_clip,
        create_clip_annotation,
        create_annotation_set,
    ):
        """load_annotated_dataset returns an AnnotationSet for AOEFAnnotations."""
        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()
        rec = create_recording(path=audio_dir / "rec.wav")
        clip = create_clip(rec)
        ann = create_clip_annotation(clip)
        annotation_set = create_annotation_set(
            name="aoef_test", annotations=[ann]
        )
        anns_path = tmp_path / "anns.json"
        io.save(annotation_set, anns_path)

        config = AOEFAnnotations(
            name="aoef_test",
            audio_dir=audio_dir,
            annotations_path=anns_path,
        )

        result = load_annotated_dataset(config)

        assert isinstance(result, data.AnnotationSet)
        assert len(result.clip_annotations) == 1

    def test_dispatches_to_batdetect2_merged_loader(
        self,
        tmp_path: Path,
        wav_factory,
    ):
        """load_annotated_dataset returns an AnnotationSet for BatDetect2Merged."""
        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()
        wav_factory(path=audio_dir / "rec.wav", duration=2.0)

        merged = [
            {
                "id": "rec.wav",
                "duration": 2.0,
                "time_exp": 1.0,
                "class_name": "Myotis",
                "annotation": [],
                "annotated": True,
                "issues": False,
                "notes": "",
            }
        ]
        anns_path = tmp_path / "merged.json"
        anns_path.write_text(json.dumps(merged))

        config = BatDetect2MergedAnnotations(
            name="merged_test",
            audio_dir=audio_dir,
            annotations_path=anns_path,
        )

        result = load_annotated_dataset(config)

        assert isinstance(result, data.AnnotationSet)
        assert len(result.clip_annotations) == 1

    def test_passes_base_dir_to_loader(
        self,
        tmp_path: Path,
        wav_factory,
    ):
        """base_dir is forwarded to the loader so relative paths are resolved."""
        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()
        anns_dir = tmp_path / "anns"
        anns_dir.mkdir()

        wav_factory(path=audio_dir / "rec.wav", duration=2.0)
        ann_data = {
            "id": "rec.wav",
            "duration": 2.0,
            "time_exp": 1.0,
            "class_name": "Myotis",
            "annotation": [],
            "annotated": True,
            "issues": False,
            "notes": "",
        }
        (anns_dir / "rec.wav.json").write_text(json.dumps(ann_data))

        config = BatDetect2FilesAnnotations(
            name="base_dir_test",
            audio_dir=Path("audio"),
            annotations_dir=Path("anns"),
        )

        result = load_annotated_dataset(config, base_dir=tmp_path)

        assert isinstance(result, data.AnnotationSet)
        assert len(result.clip_annotations) == 1
        assert (
            result.clip_annotations[0].clip.recording.path
            == audio_dir / "rec.wav"
        )
