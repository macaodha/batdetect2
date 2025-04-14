import uuid
from pathlib import Path
from typing import Callable, Iterable, List, Optional

import numpy as np
import pytest
import soundfile as sf
from soundevent import data, terms

from batdetect2.targets import (
    TargetConfig,
    build_target_encoder,
    call_type,
    get_class_names,
)


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


@pytest.fixture
def recording(
    recording_factory: Callable[..., data.Recording],
) -> data.Recording:
    return recording_factory()


@pytest.fixture
def clip(recording: data.Recording) -> data.Clip:
    return data.Clip(recording=recording, start_time=0, end_time=0.5)


@pytest.fixture
def sound_event(recording: data.Recording) -> data.SoundEvent:
    return data.SoundEvent(
        geometry=data.BoundingBox(coordinates=[0.1, 67_000, 0.11, 73_000]),
        recording=recording,
    )


@pytest.fixture
def echolocation_call(recording: data.Recording) -> data.SoundEventAnnotation:
    return data.SoundEventAnnotation(
        sound_event=data.SoundEvent(
            geometry=data.BoundingBox(coordinates=[0.1, 67_000, 0.11, 73_000]),
            recording=recording,
        ),
        tags=[
            data.Tag(term=terms.scientific_name, value="Myotis myotis"),
            data.Tag(term=call_type, value="Echolocation"),
        ],
    )


@pytest.fixture
def generic_call(recording: data.Recording) -> data.SoundEventAnnotation:
    return data.SoundEventAnnotation(
        sound_event=data.SoundEvent(
            geometry=data.BoundingBox(
                coordinates=[0.34, 35_000, 0.348, 62_000]
            ),
            recording=recording,
        ),
        tags=[
            data.Tag(term=terms.order, value="Chiroptera"),
            data.Tag(term=call_type, value="Echolocation"),
        ],
    )


@pytest.fixture
def non_relevant_sound_event(
    recording: data.Recording,
) -> data.SoundEventAnnotation:
    return data.SoundEventAnnotation(
        sound_event=data.SoundEvent(
            geometry=data.BoundingBox(
                coordinates=[0.22, 50_000, 0.24, 58_000]
            ),
            recording=recording,
        ),
        tags=[
            data.Tag(
                term=terms.scientific_name,
                value="Muscardinus avellanarius",
            ),
        ],
    )


@pytest.fixture
def clip_annotation(
    clip: data.Clip,
    echolocation_call: data.SoundEventAnnotation,
    generic_call: data.SoundEventAnnotation,
    non_relevant_sound_event: data.SoundEventAnnotation,
) -> data.ClipAnnotation:
    return data.ClipAnnotation(
        clip=clip,
        sound_events=[
            echolocation_call,
            generic_call,
            non_relevant_sound_event,
        ],
    )


@pytest.fixture
def target_config() -> TargetConfig:
    return TargetConfig()


@pytest.fixture
def class_names(target_config: TargetConfig) -> List[str]:
    return get_class_names(target_config.classes)


@pytest.fixture
def encoder(
    target_config: TargetConfig,
) -> Callable[[Iterable[data.Tag]], Optional[str]]:
    return build_target_encoder(
        classes=target_config.classes,
        replacement_rules=target_config.replace,
    )
