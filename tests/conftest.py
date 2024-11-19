import uuid
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
import pytest
import soundfile as sf
from soundevent import data


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
def example_anns_dir(example_data_dir: Path) -> Path:
    example_anns_dir = example_data_dir / "anns"
    assert example_anns_dir.exists()
    return example_anns_dir


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


@pytest.fixture
def wav_factory(tmp_path: Path):
    def _wav_factory(
        path: Optional[Path] = None,
        duration: float = 0.3,
        channels: int = 1,
        samplerate: int = 441_000,
        bit_depth: int = 16,
    ) -> Path:
        path = path or tmp_path / f"{uuid.uuid4()}.wav"
        frames = int(samplerate * duration)
        shape = (frames, channels)
        subtype = f"PCM_{bit_depth}"

        if bit_depth == 16:
            dtype = np.int16
        elif bit_depth == 32:
            dtype = np.int32
        else:
            raise ValueError(f"Unsupported bit depth: {bit_depth}")

        wav = np.random.uniform(
            low=np.iinfo(dtype).min,
            high=np.iinfo(dtype).max,
            size=shape,
        ).astype(dtype)
        sf.write(str(path), wav, samplerate, subtype=subtype)
        return path

    return _wav_factory


@pytest.fixture
def recording_factory(wav_factory: Callable[..., Path]):
    def _recording_factory(
        tags: Optional[list[data.Tag]] = None,
        path: Optional[Path] = None,
        recording_id: Optional[uuid.UUID] = None,
        duration: float = 1,
        channels: int = 1,
        samplerate: int = 256_000,
        time_expansion: float = 1,
    ) -> data.Recording:
        path = path or wav_factory(
            duration=duration,
            channels=channels,
            samplerate=samplerate,
        )
        return data.Recording.from_file(
            path=path,
            uuid=recording_id or uuid.uuid4(),
            time_expansion=time_expansion,
            tags=tags or [],
        )

    return _recording_factory
