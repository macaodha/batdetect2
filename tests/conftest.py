from pathlib import Path

import pytest


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
