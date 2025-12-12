from typing import TYPE_CHECKING, List, Optional, Sequence

from lightning import Trainer
from soundevent import data

from batdetect2.audio.loader import build_audio_loader
from batdetect2.inference.clips import get_clips_from_files
from batdetect2.inference.dataset import build_inference_loader
from batdetect2.inference.lightning import InferenceModule
from batdetect2.models import Model
from batdetect2.preprocess.preprocessor import build_preprocessor
from batdetect2.targets.targets import build_targets
from batdetect2.typing.postprocess import ClipDetections

if TYPE_CHECKING:
    from batdetect2.config import BatDetect2Config
    from batdetect2.typing import (
        AudioLoader,
        PreprocessorProtocol,
        TargetProtocol,
    )


def run_batch_inference(
    model,
    clips: Sequence[data.Clip],
    targets: Optional["TargetProtocol"] = None,
    audio_loader: Optional["AudioLoader"] = None,
    preprocessor: Optional["PreprocessorProtocol"] = None,
    config: Optional["BatDetect2Config"] = None,
    num_workers: int | None = None,
    batch_size: int | None = None,
) -> List[ClipDetections]:
    from batdetect2.config import BatDetect2Config

    config = config or BatDetect2Config()

    audio_loader = audio_loader or build_audio_loader()

    preprocessor = preprocessor or build_preprocessor(
        input_samplerate=audio_loader.samplerate,
    )

    targets = targets or build_targets()

    loader = build_inference_loader(
        clips,
        audio_loader=audio_loader,
        preprocessor=preprocessor,
        config=config.inference.loader,
        num_workers=num_workers,
        batch_size=batch_size,
    )

    module = InferenceModule(model)
    trainer = Trainer(enable_checkpointing=False, logger=False)
    outputs = trainer.predict(module, loader)
    return [
        clip_prediction
        for clip_predictions in outputs  # type: ignore
        for clip_prediction in clip_predictions
    ]


def process_file_list(
    model: Model,
    paths: Sequence[data.PathLike],
    config: "BatDetect2Config",
    targets: Optional["TargetProtocol"] = None,
    audio_loader: Optional["AudioLoader"] = None,
    preprocessor: Optional["PreprocessorProtocol"] = None,
    num_workers: int | None = None,
) -> List[ClipDetections]:
    clip_config = config.inference.clipping
    clips = get_clips_from_files(
        paths,
        duration=clip_config.duration,
        overlap=clip_config.overlap,
        max_empty=clip_config.max_empty,
        discard_empty=clip_config.discard_empty,
    )
    return run_batch_inference(
        model,
        clips,
        targets=targets,
        audio_loader=audio_loader,
        preprocessor=preprocessor,
        config=config,
        num_workers=num_workers,
    )
