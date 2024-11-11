from pathlib import Path
from typing import List

import pytest


@pytest.fixture
def example_data_dir() -> Path:
    pkg_dir = Path(__file__).parent.parent
    example_data_dir = pkg_dir / "example_data"
    assert example_data_dir.exists()
    return example_data_dir


@pytest.fixture
def example_audio_dir(example_data_dir: Path) -> Path:
    example_audio_dir = example_data_dir / "audio"
    assert example_audio_dir.exists()
    return example_audio_dir


@pytest.fixture
def example_audio_files(example_audio_dir: Path) -> List[Path]:
    audio_files = list(example_audio_dir.glob("*.[wW][aA][vV]"))
    assert len(audio_files) == 3
    return audio_files


@pytest.fixture
def data_dir() -> Path:
    dir = Path(__file__).parent / "data"
    assert dir.exists()
    return dir


@pytest.fixture
def contrib_dir(data_dir) -> Path:
    dir = data_dir / "contrib"
    assert dir.exists()
    return dir
