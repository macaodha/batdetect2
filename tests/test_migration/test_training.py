import json
from pathlib import Path
from typing import List

import numpy as np
import pytest

from batdetect2.compat.params import get_training_preprocessing_config
from batdetect2.data import BatDetect2FilesAnnotations, load_annotated_dataset
from batdetect2.train.preprocess import generate_train_example


@pytest.fixture
def regression_dir(data_dir: Path) -> Path:
    dir = data_dir / "regression"
    assert dir.exists()
    return dir


def test_can_generate_similar_training_inputs(
    example_audio_dir: Path,
    example_audio_files: List[Path],
    example_anns_dir: Path,
    regression_dir: Path,
):
    old_parameters = json.loads((regression_dir / "params.json").read_text())
    config = get_training_preprocessing_config(old_parameters)

    assert config is not None

    for audio_file in example_audio_files:
        example_file = regression_dir / f"{audio_file.name}.npz"

        dataset = np.load(example_file)

        spec = dataset["spec"][0]
        detection_mask = dataset["detection_mask"][0]
        size_mask = dataset["size_mask"]
        class_mask = dataset["class_mask"]

        project = load_annotated_dataset(
            BatDetect2FilesAnnotations(
                name="test",
                annotations_dir=example_anns_dir,
                audio_dir=example_audio_dir,
            )
        )

        clip_annotation = next(
            ann
            for ann in project.clip_annotations
            if ann.clip.recording.path == audio_file
        )

        new_dataset = generate_train_example(
            clip_annotation,
            preprocessing_config=config.preprocessing,
            target_config=config.target,
            label_config=config.labels,
        )
        new_spec = new_dataset["spectrogram"].values
        new_detection_mask = new_dataset["detection"].values
        new_size_mask = new_dataset["size"].values
        new_class_mask = new_dataset["class"].values

        assert spec.shape == new_spec.shape
        assert detection_mask.shape == new_detection_mask.shape
        assert size_mask.shape == new_size_mask.shape
        assert class_mask.shape[1:] == new_class_mask.shape[1:]
        assert class_mask.shape[0] == new_class_mask.shape[0] + 1

        x_new, y_new = np.nonzero(new_size_mask.max(axis=0))
        x_orig, y_orig = np.nonzero(np.flipud(size_mask.max(axis=0)))

        assert (x_new == x_orig).all()

        # NOTE: a difference of 1 pixel is due to discrepancies on how
        # frequency bins are interpreted. Shouldn't be an issue
        assert (y_new == y_orig + 1).all()

        width_new, height_new = new_size_mask[:, x_new, y_new]
        width_orig, height_orig = np.flip(size_mask, axis=1)[:, x_orig, y_orig]

        assert (np.floor(width_new) == width_orig).all()
        assert (np.ceil(height_new) == height_orig).all()
