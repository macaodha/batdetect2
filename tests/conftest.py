import uuid
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
import pytest
import soundfile as sf
from soundevent import data, terms

from batdetect2.preprocess import build_preprocessor
from batdetect2.preprocess.types import PreprocessorProtocol
from batdetect2.targets import (
    TargetConfig,
    TermRegistry,
    build_targets,
    call_type,
)
from batdetect2.targets.classes import ClassesConfig, TargetClass
from batdetect2.targets.filtering import FilterConfig, FilterRule
from batdetect2.targets.terms import TagInfo
from batdetect2.targets.types import TargetProtocol
from batdetect2.train.labels import build_clip_labeler
from batdetect2.train.types import ClipLabeller


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
def create_recording(wav_factory: Callable[..., Path]):
    def factory(
        tags: Optional[list[data.Tag]] = None,
        path: Optional[Path] = None,
        recording_id: Optional[uuid.UUID] = None,
        duration: float = 1,
        channels: int = 1,
        samplerate: int = 256_000,
        time_expansion: float = 1,
    ) -> data.Recording:
        path = wav_factory(
            path=path,
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

    return factory


@pytest.fixture
def recording(
    create_recording: Callable[..., data.Recording],
) -> data.Recording:
    return create_recording()


@pytest.fixture
def create_clip():
    def factory(
        recording: data.Recording,
        start_time: float = 0,
        end_time: float = 0.5,
    ) -> data.Clip:
        return data.Clip(
            recording=recording,
            start_time=start_time,
            end_time=end_time,
        )

    return factory


@pytest.fixture
def clip(recording: data.Recording) -> data.Clip:
    return data.Clip(recording=recording, start_time=0, end_time=0.5)


@pytest.fixture
def create_sound_event():
    def factory(
        recording: data.Recording,
        coords: Optional[List[float]] = None,
    ) -> data.SoundEvent:
        coords = coords or [0.2, 60_000, 0.3, 70_000]

        return data.SoundEvent(
            geometry=data.BoundingBox(coordinates=coords),
            recording=recording,
        )

    return factory


@pytest.fixture
def sound_event(recording: data.Recording) -> data.SoundEvent:
    return data.SoundEvent(
        geometry=data.BoundingBox(coordinates=[0.1, 67_000, 0.11, 73_000]),
        recording=recording,
    )


@pytest.fixture
def create_sound_event_annotation():
    def factory(
        sound_event: data.SoundEvent,
        tags: Optional[List[data.Tag]] = None,
    ) -> data.SoundEventAnnotation:
        return data.SoundEventAnnotation(
            sound_event=sound_event,
            tags=tags or [],
        )

    return factory


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
def create_clip_annotation():
    def factory(
        clip: data.Clip,
        clip_tags: Optional[List[data.Tag]] = None,
        sound_events: Optional[List[data.SoundEventAnnotation]] = None,
    ) -> data.ClipAnnotation:
        return data.ClipAnnotation(
            clip=clip,
            tags=clip_tags or [],
            sound_events=sound_events or [],
        )

    return factory


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
def create_annotation_set():
    def factory(
        name: str = "test",
        description: str = "Test annotation set",
        annotations: Optional[List[data.ClipAnnotation]] = None,
    ) -> data.AnnotationSet:
        return data.AnnotationSet(
            name=name,
            description=description,
            clip_annotations=annotations or [],
        )

    return factory


@pytest.fixture
def create_annotation_project():
    def factory(
        name: str = "test_project",
        description: str = "Test Annotation Project",
        tasks: Optional[List[data.AnnotationTask]] = None,
        annotations: Optional[List[data.ClipAnnotation]] = None,
    ) -> data.AnnotationProject:
        return data.AnnotationProject(
            name=name,
            description=description,
            tasks=tasks or [],
            clip_annotations=annotations or [],
        )

    return factory


@pytest.fixture
def sample_term_registry() -> TermRegistry:
    """Fixture for a sample TermRegistry."""
    registry = TermRegistry()
    registry.add_custom_term("class")
    registry.add_custom_term("order")
    registry.add_custom_term("species")
    registry.add_custom_term("call_type")
    registry.add_custom_term("quality")
    return registry


@pytest.fixture
def sample_preprocessor() -> PreprocessorProtocol:
    return build_preprocessor()


@pytest.fixture
def bat_tag() -> TagInfo:
    return TagInfo(key="class", value="bat")


@pytest.fixture
def noise_tag() -> TagInfo:
    return TagInfo(key="class", value="noise")


@pytest.fixture
def myomyo_tag() -> TagInfo:
    return TagInfo(key="species", value="Myotis myotis")


@pytest.fixture
def pippip_tag() -> TagInfo:
    return TagInfo(key="species", value="Pipistrellus pipistrellus")


@pytest.fixture
def sample_target_config(
    sample_term_registry: TermRegistry,
    bat_tag: TagInfo,
    noise_tag: TagInfo,
    myomyo_tag: TagInfo,
    pippip_tag: TagInfo,
) -> TargetConfig:
    return TargetConfig(
        filtering=FilterConfig(
            rules=[FilterRule(match_type="exclude", tags=[noise_tag])]
        ),
        classes=ClassesConfig(
            classes=[
                TargetClass(name="pippip", tags=[pippip_tag]),
                TargetClass(name="myomyo", tags=[myomyo_tag]),
            ],
            generic_class=[bat_tag],
        ),
    )


@pytest.fixture
def sample_targets(
    sample_target_config: TargetConfig,
    sample_term_registry: TermRegistry,
) -> TargetProtocol:
    return build_targets(
        sample_target_config,
        term_registry=sample_term_registry,
    )


@pytest.fixture
def sample_labeller(
    sample_targets: TargetProtocol,
) -> ClipLabeller:
    return build_clip_labeler(sample_targets)
