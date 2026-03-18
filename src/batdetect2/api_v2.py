import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from soundevent import data
from soundevent.audio.files import get_audio_files

from batdetect2.audio import build_audio_loader
from batdetect2.audio.types import AudioLoader
from batdetect2.config import BatDetect2Config
from batdetect2.core import merge_configs
from batdetect2.data import (
    load_dataset_from_config,
)
from batdetect2.data.datasets import Dataset
from batdetect2.evaluate import DEFAULT_EVAL_DIR, build_evaluator, evaluate
from batdetect2.evaluate.types import EvaluatorProtocol
from batdetect2.inference import process_file_list, run_batch_inference
from batdetect2.logging import DEFAULT_LOGS_DIR
from batdetect2.models import Model, build_model
from batdetect2.outputs import (
    OutputFormatConfig,
    OutputTransformProtocol,
    build_output_formatter,
    build_output_transform,
    get_output_formatter,
)
from batdetect2.outputs.types import OutputFormatterProtocol
from batdetect2.postprocess import build_postprocessor, to_raw_predictions
from batdetect2.postprocess.types import (
    ClipDetections,
    Detection,
    PostprocessorProtocol,
)
from batdetect2.preprocess import build_preprocessor
from batdetect2.preprocess.types import PreprocessorProtocol
from batdetect2.targets import build_targets
from batdetect2.targets.types import TargetProtocol
from batdetect2.train import (
    DEFAULT_CHECKPOINT_DIR,
    load_model_from_checkpoint,
    run_train,
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
        formatter: OutputFormatterProtocol,
        output_transform: OutputTransformProtocol,
        model: Model,
    ):
        self.config = config
        self.targets = targets
        self.audio_loader = audio_loader
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.evaluator = evaluator
        self.model = model
        self.formatter = formatter
        self.output_transform = output_transform

        self.model.eval()

    def load_annotations(
        self,
        path: data.PathLike,
        base_dir: data.PathLike | None = None,
    ) -> Dataset:
        return load_dataset_from_config(path, base_dir=base_dir)

    def train(
        self,
        train_annotations: Sequence[data.ClipAnnotation],
        val_annotations: Sequence[data.ClipAnnotation] | None = None,
        train_workers: int | None = None,
        val_workers: int | None = None,
        checkpoint_dir: Path | None = DEFAULT_CHECKPOINT_DIR,
        log_dir: Path | None = DEFAULT_LOGS_DIR,
        experiment_name: str | None = None,
        num_epochs: int | None = None,
        run_name: str | None = None,
        seed: int | None = None,
    ):
        run_train(
            train_annotations=train_annotations,
            val_annotations=val_annotations,
            targets=self.targets,
            model_config=self.config.model,
            train_config=self.config.train,
            audio_config=self.config.audio,
            audio_loader=self.audio_loader,
            preprocessor=self.preprocessor,
            train_workers=train_workers,
            val_workers=val_workers,
            checkpoint_dir=checkpoint_dir,
            log_dir=log_dir,
            num_epochs=num_epochs,
            experiment_name=experiment_name,
            run_name=run_name,
            seed=seed,
        )
        return self

    def evaluate(
        self,
        test_annotations: Sequence[data.ClipAnnotation],
        num_workers: int | None = None,
        output_dir: data.PathLike = DEFAULT_EVAL_DIR,
        experiment_name: str | None = None,
        run_name: str | None = None,
        save_predictions: bool = True,
    ) -> tuple[dict[str, float], list[list[Detection]]]:
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
            formatter=self.formatter if save_predictions else None,
        )

    def evaluate_predictions(
        self,
        annotations: Sequence[data.ClipAnnotation],
        predictions: Sequence[ClipDetections],
        output_dir: data.PathLike | None = None,
    ):
        clip_evals = self.evaluator.evaluate(
            annotations,
            predictions,
        )

        metrics = self.evaluator.compute_metrics(clip_evals)

        if output_dir is not None:
            output_dir = Path(output_dir)

            if not output_dir.is_dir():
                output_dir.mkdir(parents=True)

            metrics_path = output_dir / "metrics.json"
            metrics_path.write_text(json.dumps(metrics))

            for figure_name, fig in self.evaluator.generate_plots(clip_evals):
                fig_path = output_dir / figure_name

                if not fig_path.parent.is_dir():
                    fig_path.parent.mkdir(parents=True)

                fig.savefig(fig_path)

        return metrics

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

    def process_file(self, audio_file: str) -> ClipDetections:
        recording = data.Recording.from_file(audio_file, compute_hash=False)
        wav = self.audio_loader.load_recording(recording)
        detections = self.process_audio(wav)
        return ClipDetections(
            clip=data.Clip(
                uuid=recording.uuid,
                recording=recording,
                start_time=0,
                end_time=recording.duration,
            ),
            detections=detections,
        )

    def process_audio(
        self,
        audio: np.ndarray,
    ) -> list[Detection]:
        spec = self.generate_spectrogram(audio)
        return self.process_spectrogram(spec)

    def process_spectrogram(
        self,
        spec: torch.Tensor,
        start_time: float = 0,
    ) -> list[Detection]:
        if spec.ndim == 4 and spec.shape[0] > 1:
            raise ValueError("Batched spectrograms not supported.")

        if spec.ndim == 3:
            spec = spec.unsqueeze(0)

        outputs = self.model.detector(spec)

        detections = self.model.postprocessor(
            outputs,
        )[0]
        raw_predictions = to_raw_predictions(
            detections.numpy(),
            targets=self.targets,
        )

        return self.output_transform.transform_detections(
            raw_predictions,
            start_time=start_time,
        )

    def process_directory(
        self,
        audio_dir: data.PathLike,
    ) -> list[ClipDetections]:
        files = list(get_audio_files(audio_dir))
        return self.process_files(files)

    def process_files(
        self,
        audio_files: Sequence[data.PathLike],
        num_workers: int | None = None,
    ) -> list[ClipDetections]:
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
        batch_size: int | None = None,
        num_workers: int | None = None,
    ) -> list[ClipDetections]:
        return run_batch_inference(
            self.model,
            clips,
            targets=self.targets,
            audio_loader=self.audio_loader,
            preprocessor=self.preprocessor,
            config=self.config,
            batch_size=batch_size,
            num_workers=num_workers,
        )

    def save_predictions(
        self,
        predictions: Sequence[ClipDetections],
        path: data.PathLike,
        audio_dir: data.PathLike | None = None,
        format: str | None = None,
        config: OutputFormatConfig | None = None,
    ):
        formatter = self.formatter

        if format is not None or config is not None:
            format = format or config.name  # type: ignore
            formatter = get_output_formatter(
                name=format,
                targets=self.targets,
                config=config,
            )

        outs = formatter.format(predictions)
        formatter.save(outs, audio_dir=audio_dir, path=path)

    def load_predictions(
        self,
        path: data.PathLike,
    ) -> list[ClipDetections]:
        return self.formatter.load(path)

    @classmethod
    def from_config(
        cls,
        config: BatDetect2Config,
    ):
        targets = build_targets(config=config.model.targets)

        audio_loader = build_audio_loader(config=config.audio)

        preprocessor = build_preprocessor(
            input_samplerate=audio_loader.samplerate,
            config=config.model.preprocess,
        )

        postprocessor = build_postprocessor(
            preprocessor,
            config=config.model.postprocess,
        )

        evaluator = build_evaluator(config=config.evaluation, targets=targets)

        # NOTE: Better to have a separate instance of preprocessor and
        # postprocessor as these may be moved to another device.
        model = build_model(config=config.model)

        formatter = build_output_formatter(
            targets,
            config=config.outputs.format,
        )
        output_transform = build_output_transform(
            config=config.outputs.transform
        )

        return cls(
            config=config,
            targets=targets,
            audio_loader=audio_loader,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            evaluator=evaluator,
            model=model,
            formatter=formatter,
            output_transform=output_transform,
        )

    @classmethod
    def from_checkpoint(
        cls,
        path: data.PathLike,
        config: BatDetect2Config | None = None,
    ):
        from batdetect2.audio import AudioConfig

        model, model_config = load_model_from_checkpoint(path)

        # Reconstruct a full BatDetect2Config from the checkpoint's
        # ModelConfig, then overlay any caller-supplied overrides.
        base = BatDetect2Config(
            model=model_config,
            audio=AudioConfig(samplerate=model_config.samplerate),
        )
        config = merge_configs(base, config) if config else base

        targets = build_targets(config=config.model.targets)

        audio_loader = build_audio_loader(config=config.audio)

        preprocessor = build_preprocessor(
            input_samplerate=audio_loader.samplerate,
            config=config.model.preprocess,
        )

        postprocessor = build_postprocessor(
            preprocessor,
            config=config.model.postprocess,
        )

        evaluator = build_evaluator(config=config.evaluation, targets=targets)

        formatter = build_output_formatter(
            targets,
            config=config.outputs.format,
        )
        output_transform = build_output_transform(
            config=config.outputs.transform
        )

        return cls(
            config=config,
            targets=targets,
            audio_loader=audio_loader,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            evaluator=evaluator,
            model=model,
            formatter=formatter,
            output_transform=output_transform,
        )
