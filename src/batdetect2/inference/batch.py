from typing import Sequence

from lightning import Trainer
from soundevent import data

from batdetect2.audio import AudioConfig
from batdetect2.audio.loader import build_audio_loader
from batdetect2.audio.types import AudioLoader
from batdetect2.inference.clips import get_clips_from_files
from batdetect2.inference.config import InferenceConfig
from batdetect2.inference.dataset import build_inference_loader
from batdetect2.inference.lightning import InferenceModule
from batdetect2.models import Model
from batdetect2.outputs import (
    OutputsConfig,
    OutputTransformProtocol,
    build_output_transform,
)
from batdetect2.postprocess.types import ClipDetections
from batdetect2.preprocess.types import PreprocessorProtocol
from batdetect2.targets.types import TargetProtocol


def run_batch_inference(
    model: Model,
    clips: Sequence[data.Clip],
    targets: TargetProtocol | None = None,
    audio_loader: AudioLoader | None = None,
    preprocessor: PreprocessorProtocol | None = None,
    audio_config: AudioConfig | None = None,
    output_transform: OutputTransformProtocol | None = None,
    output_config: OutputsConfig | None = None,
    inference_config: InferenceConfig | None = None,
    num_workers: int = 1,
    batch_size: int | None = None,
) -> list[ClipDetections]:
    audio_config = audio_config or AudioConfig(
        samplerate=model.preprocessor.input_samplerate,
    )
    output_config = output_config or OutputsConfig()
    inference_config = inference_config or InferenceConfig()

    audio_loader = audio_loader or build_audio_loader(config=audio_config)

    preprocessor = preprocessor or model.preprocessor
    targets = targets or model.targets

    output_transform = output_transform or build_output_transform(
        config=output_config.transform,
        targets=targets,
    )

    loader = build_inference_loader(
        clips,
        audio_loader=audio_loader,
        preprocessor=preprocessor,
        config=inference_config.loader,
        num_workers=num_workers,
        batch_size=batch_size,
    )

    module = InferenceModule(
        model,
        output_transform=output_transform,
    )
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
    targets: TargetProtocol | None = None,
    audio_loader: AudioLoader | None = None,
    audio_config: AudioConfig | None = None,
    preprocessor: PreprocessorProtocol | None = None,
    inference_config: InferenceConfig | None = None,
    output_config: OutputsConfig | None = None,
    output_transform: OutputTransformProtocol | None = None,
    batch_size: int | None = None,
    num_workers: int = 0,
) -> list[ClipDetections]:
    inference_config = inference_config or InferenceConfig()
    clip_config = inference_config.clipping
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
        batch_size=batch_size,
        num_workers=num_workers,
        output_config=output_config,
        audio_config=audio_config,
        output_transform=output_transform,
        inference_config=inference_config,
    )
