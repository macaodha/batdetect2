from pathlib import Path
from typing import Optional, Sequence

from soundevent import data

from batdetect2.audio import build_audio_loader
from batdetect2.config import BatDetect2Config
from batdetect2.evaluate import Evaluator, build_evaluator
from batdetect2.models import Model, build_model
from batdetect2.postprocess import build_postprocessor
from batdetect2.preprocess import build_preprocessor
from batdetect2.targets.targets import build_targets
from batdetect2.train import train
from batdetect2.train.lightning import load_model_from_checkpoint
from batdetect2.typing import (
    AudioLoader,
    PostprocessorProtocol,
    PreprocessorProtocol,
    TargetProtocol,
)


class BatDetect2API:
    def __init__(
        self,
        config: BatDetect2Config,
        targets: TargetProtocol,
        audio_loader: AudioLoader,
        preprocessor: PreprocessorProtocol,
        postprocessor: PostprocessorProtocol,
        evaluator: Evaluator,
        model: Model,
    ):
        self.config = config
        self.targets = targets
        self.audio_loader = audio_loader
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.evaluator = evaluator
        self.model = model

        self.model.eval()

    def train(
        self,
        train_annotations: Sequence[data.ClipAnnotation],
        val_annotations: Optional[Sequence[data.ClipAnnotation]] = None,
        train_workers: Optional[int] = None,
        val_workers: Optional[int] = None,
        checkpoint_dir: Optional[Path] = None,
        log_dir: Optional[Path] = None,
        experiment_name: Optional[str] = None,
        run_name: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        train(
            train_annotations=train_annotations,
            val_annotations=val_annotations,
            targets=self.targets,
            config=self.config,
            audio_loader=self.audio_loader,
            preprocessor=self.preprocessor,
            train_workers=train_workers,
            val_workers=val_workers,
            checkpoint_dir=checkpoint_dir,
            log_dir=log_dir,
            experiment_name=experiment_name,
            run_name=run_name,
            seed=seed,
        )
        return self

    @classmethod
    def from_config(cls, config: BatDetect2Config):
        targets = build_targets(config=config.targets)

        audio_loader = build_audio_loader(config=config.audio)

        preprocessor = build_preprocessor(
            input_samplerate=audio_loader.samplerate,
            config=config.preprocess,
        )

        postprocessor = build_postprocessor(
            preprocessor,
            config=config.postprocess,
        )

        evaluator = build_evaluator(
            config=config.evaluation,
            targets=targets,
        )

        # NOTE: Better to have a separate instance of
        # preprocessor and postprocessor as these may be moved
        # to another device.
        model = build_model(
            config=config.model,
            targets=targets,
            preprocessor=build_preprocessor(
                input_samplerate=audio_loader.samplerate,
                config=config.preprocess,
            ),
            postprocessor=build_postprocessor(
                preprocessor,
                config=config.postprocess,
            ),
        )

        return cls(
            config=config,
            targets=targets,
            audio_loader=audio_loader,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            evaluator=evaluator,
            model=model,
        )

    @classmethod
    def from_checkpoint(cls, path: data.PathLike):
        model, config = load_model_from_checkpoint(path)

        targets = build_targets(config=config.targets)

        audio_loader = build_audio_loader(config=config.audio)

        preprocessor = build_preprocessor(
            input_samplerate=audio_loader.samplerate,
            config=config.preprocess,
        )

        postprocessor = build_postprocessor(
            preprocessor,
            config=config.postprocess,
        )

        evaluator = build_evaluator(
            config=config.evaluation,
            targets=targets,
        )

        return cls(
            config=config,
            targets=targets,
            audio_loader=audio_loader,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            evaluator=evaluator,
            model=model,
        )
