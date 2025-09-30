from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import torch
from soundevent import data
from soundevent.audio.files import get_audio_files

from batdetect2.audio import build_audio_loader
from batdetect2.config import BatDetect2Config
from batdetect2.evaluate import DEFAULT_EVAL_DIR, build_evaluator, evaluate
from batdetect2.inference import process_file_list, run_batch_inference
from batdetect2.logging import DEFAULT_LOGS_DIR
from batdetect2.models import Model, build_model
from batdetect2.postprocess import build_postprocessor, to_raw_predictions
from batdetect2.preprocess import build_preprocessor
from batdetect2.targets import build_targets
from batdetect2.train import (
    DEFAULT_CHECKPOINT_DIR,
    load_model_from_checkpoint,
    train,
)
from batdetect2.typing import (
    AudioLoader,
    BatDetect2Prediction,
    EvaluatorProtocol,
    PostprocessorProtocol,
    PreprocessorProtocol,
    RawPrediction,
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
        evaluator: EvaluatorProtocol,
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
        checkpoint_dir: Optional[Path] = DEFAULT_CHECKPOINT_DIR,
        log_dir: Optional[Path] = DEFAULT_LOGS_DIR,
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

    def evaluate(
        self,
        test_annotations: Sequence[data.ClipAnnotation],
        num_workers: Optional[int] = None,
        output_dir: data.PathLike = DEFAULT_EVAL_DIR,
        experiment_name: Optional[str] = None,
        run_name: Optional[str] = None,
    ):
        return evaluate(
            self.model,
            test_annotations,
            targets=self.targets,
            audio_loader=self.audio_loader,
            preprocessor=self.preprocessor,
            config=self.config,
            num_workers=num_workers,
            output_dir=output_dir,
            experiment_name=experiment_name,
            run_name=run_name,
        )

    def load_audio(self, path: data.PathLike) -> np.ndarray:
        return self.audio_loader.load_file(path)

    def load_clip(self, clip: data.Clip) -> np.ndarray:
        return self.audio_loader.load_clip(clip)

    def generate_spectrogram(
        self,
        audio: np.ndarray,
    ) -> torch.Tensor:
        tensor = torch.tensor(audio).unsqueeze(0)
        return self.preprocessor(tensor)

    def process_file(self, audio_file: str) -> BatDetect2Prediction:
        recording = data.Recording.from_file(audio_file, compute_hash=False)
        wav = self.audio_loader.load_recording(recording)
        detections = self.process_audio(wav)
        return BatDetect2Prediction(
            clip=data.Clip(
                uuid=recording.uuid,
                recording=recording,
                start_time=0,
                end_time=recording.duration,
            ),
            predictions=detections,
        )

    def process_audio(
        self,
        audio: np.ndarray,
    ) -> List[RawPrediction]:
        spec = self.generate_spectrogram(audio)
        return self.process_spectrogram(spec)

    def process_spectrogram(
        self,
        spec: torch.Tensor,
        start_time: float = 0,
    ) -> List[RawPrediction]:
        if spec.ndim == 4 and spec.shape[0] > 1:
            raise ValueError("Batched spectrograms not supported.")

        if spec.ndim == 3:
            spec = spec.unsqueeze(0)

        outputs = self.model.detector(spec)

        detections = self.model.postprocessor(
            outputs,
            start_times=[start_time],
        )[0]

        return to_raw_predictions(detections.numpy(), targets=self.targets)

    def process_directory(
        self,
        audio_dir: data.PathLike,
    ) -> List[BatDetect2Prediction]:
        files = list(get_audio_files(audio_dir))
        return self.process_files(files)

    def process_files(
        self,
        audio_files: Sequence[data.PathLike],
        num_workers: Optional[int] = None,
    ) -> List[BatDetect2Prediction]:
        return process_file_list(
            self.model,
            audio_files,
            config=self.config,
            targets=self.targets,
            audio_loader=self.audio_loader,
            preprocessor=self.preprocessor,
            num_workers=num_workers,
        )

    def process_clips(
        self,
        clips: Sequence[data.Clip],
    ) -> List[BatDetect2Prediction]:
        return run_batch_inference(
            self.model,
            clips,
            targets=self.targets,
            audio_loader=self.audio_loader,
            preprocessor=self.preprocessor,
            config=self.config,
        )

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

        evaluator = build_evaluator(config=config.evaluation, targets=targets)

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
    def from_checkpoint(
        cls,
        path: data.PathLike,
        config: Optional[BatDetect2Config] = None,
    ):
        model, stored_config = load_model_from_checkpoint(path)

        config = config or stored_config

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

        evaluator = build_evaluator(config=config.evaluation, targets=targets)

        return cls(
            config=config,
            targets=targets,
            audio_loader=audio_loader,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            evaluator=evaluator,
            model=model,
        )
