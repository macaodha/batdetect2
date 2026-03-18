from pathlib import Path
from typing import Literal, Sequence, cast

import numpy as np
import torch
from soundevent import data
from soundevent.audio.files import get_audio_files

from batdetect2.audio import AudioLoader, build_audio_loader
from batdetect2.config import BatDetect2Config
from batdetect2.core import merge_configs
from batdetect2.data import Dataset, load_dataset_from_config
from batdetect2.evaluate import (
    DEFAULT_EVAL_DIR,
    EvaluatorProtocol,
    build_evaluator,
    run_evaluate,
    save_evaluation_results,
)
from batdetect2.inference import process_file_list, run_batch_inference
from batdetect2.logging import DEFAULT_LOGS_DIR
from batdetect2.models import Model, build_model, build_model_with_new_targets
from batdetect2.models.detectors import Detector
from batdetect2.outputs import (
    OutputFormatConfig,
    OutputFormatterProtocol,
    OutputTransformProtocol,
    build_output_formatter,
    build_output_transform,
    get_output_formatter,
)
from batdetect2.postprocess import (
    ClipDetections,
    Detection,
    PostprocessorProtocol,
    build_postprocessor,
)
from batdetect2.preprocess import PreprocessorProtocol, build_preprocessor
from batdetect2.targets import TargetProtocol, build_targets
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
        train_workers: int = 0,
        val_workers: int = 0,
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
            model=self.model,
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

    def finetune(
        self,
        train_annotations: Sequence[data.ClipAnnotation],
        val_annotations: Sequence[data.ClipAnnotation] | None = None,
        trainable: Literal[
            "all", "heads", "classifier_head", "bbox_head"
        ] = "heads",
        train_workers: int = 0,
        val_workers: int = 0,
        checkpoint_dir: Path | None = DEFAULT_CHECKPOINT_DIR,
        log_dir: Path | None = DEFAULT_LOGS_DIR,
        experiment_name: str | None = None,
        num_epochs: int | None = None,
        run_name: str | None = None,
        seed: int | None = None,
    ) -> "BatDetect2API":
        """Fine-tune the model with trainable-parameter selection."""

        self._set_trainable_parameters(trainable)

        run_train(
            train_annotations=train_annotations,
            val_annotations=val_annotations,
            model=self.model,
            targets=self.targets,
            model_config=self.config.model,
            train_config=self.config.train,
            preprocessor=self.preprocessor,
            audio_loader=self.audio_loader,
            train_workers=train_workers,
            val_workers=val_workers,
            checkpoint_dir=checkpoint_dir,
            log_dir=log_dir,
            experiment_name=experiment_name,
            num_epochs=num_epochs,
            run_name=run_name,
            seed=seed,
        )
        return self

    def evaluate(
        self,
        test_annotations: Sequence[data.ClipAnnotation],
        num_workers: int = 0,
        output_dir: data.PathLike = DEFAULT_EVAL_DIR,
        experiment_name: str | None = None,
        run_name: str | None = None,
        save_predictions: bool = True,
    ) -> tuple[dict[str, float], list[ClipDetections]]:
        return run_evaluate(
            self.model,
            test_annotations,
            targets=self.targets,
            audio_loader=self.audio_loader,
            preprocessor=self.preprocessor,
            audio_config=self.config.audio,
            evaluation_config=self.config.evaluation,
            output_config=self.config.outputs,
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
            save_evaluation_results(
                metrics=metrics,
                plots=self.evaluator.generate_plots(clip_evals),
                output_dir=output_dir,
            )

        return metrics

    def load_audio(self, path: data.PathLike) -> np.ndarray:
        return self.audio_loader.load_file(path)

    def load_recording(self, recording: data.Recording) -> np.ndarray:
        return self.audio_loader.load_recording(recording)

    def load_clip(self, clip: data.Clip) -> np.ndarray:
        return self.audio_loader.load_clip(clip)

    def get_top_class_name(self, detection: Detection) -> str:
        """Get highest-confidence class name for one detection."""

        top_index = int(np.argmax(detection.class_scores))
        return self.targets.class_names[top_index]

    def get_class_scores(
        self,
        detection: Detection,
        *,
        include_top_class: bool = True,
        sort_descending: bool = True,
    ) -> list[tuple[str, float]]:
        """Get class score list as ``(class_name, score)`` pairs."""

        scores = [
            (class_name, float(score))
            for class_name, score in zip(
                self.targets.class_names,
                detection.class_scores,
                strict=True,
            )
        ]

        if sort_descending:
            scores.sort(key=lambda item: item[1], reverse=True)

        if include_top_class:
            return scores

        top_class_name = self.get_top_class_name(detection)
        return [
            (class_name, score)
            for class_name, score in scores
            if class_name != top_class_name
        ]

    @staticmethod
    def get_detection_features(detection: Detection) -> np.ndarray:
        """Get extracted feature vector for one detection."""

        return detection.features

    def generate_spectrogram(
        self,
        audio: np.ndarray,
    ) -> torch.Tensor:
        tensor = torch.tensor(audio).unsqueeze(0)
        return self.preprocessor(tensor)

    def process_file(self, audio_file: data.PathLike) -> ClipDetections:
        recording = data.Recording.from_file(audio_file, compute_hash=False)

        predictions = self.process_files(
            [audio_file],
            batch_size=self.config.inference.loader.batch_size,
        )
        detections = [
            detection
            for prediction in predictions
            for detection in prediction.detections
        ]

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

        detections = self.postprocessor(
            outputs,
        )[0]
        return self.output_transform.to_detections(
            detections=detections,
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
        batch_size: int | None = None,
        num_workers: int = 0,
    ) -> list[ClipDetections]:
        return process_file_list(
            self.model,
            audio_files,
            targets=self.targets,
            audio_loader=self.audio_loader,
            audio_config=self.config.audio,
            preprocessor=self.preprocessor,
            inference_config=self.config.inference,
            output_config=self.config.outputs,
            output_transform=self.output_transform,
            batch_size=batch_size,
            num_workers=num_workers,
        )

    def process_clips(
        self,
        clips: Sequence[data.Clip],
        batch_size: int | None = None,
        num_workers: int = 0,
    ) -> list[ClipDetections]:
        return run_batch_inference(
            self.model,
            clips,
            targets=self.targets,
            audio_loader=self.audio_loader,
            audio_config=self.config.audio,
            preprocessor=self.preprocessor,
            inference_config=self.config.inference,
            output_config=self.config.outputs,
            output_transform=self.output_transform,
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
        format: str | None = None,
        config: OutputFormatConfig | None = None,
    ) -> list[object]:
        formatter = self.formatter

        if format is not None or config is not None:
            format = format or config.name  # type: ignore
            formatter = get_output_formatter(
                name=format,
                targets=self.targets,
                config=config,
            )

        return formatter.load(path)

    @classmethod
    def from_config(
        cls,
        config: BatDetect2Config,
    ) -> "BatDetect2API":
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

        formatter = build_output_formatter(
            targets,
            config=config.outputs.format,
        )
        output_transform = build_output_transform(
            config=config.outputs.transform,
            targets=targets,
        )

        evaluator = build_evaluator(
            config=config.evaluation,
            targets=targets,
            transform=output_transform,
        )

        model = build_model(
            config=config.model,
            targets=targets,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
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
        targets: TargetProtocol | None = None,
    ) -> "BatDetect2API":
        from batdetect2.audio import AudioConfig

        model, model_config = load_model_from_checkpoint(path)

        # Reconstruct a full BatDetect2Config from the checkpoint's
        # ModelConfig, then overlay any caller-supplied overrides.
        base = BatDetect2Config(
            model=model_config,
            audio=AudioConfig(samplerate=model_config.samplerate),
        )
        config = merge_configs(base, config) if config else base

        if targets is None:
            targets = build_targets(config=config.model.targets)
        else:
            target_config = getattr(targets, "config", None)
            if target_config is not None:
                config.model.targets = target_config

        audio_loader = build_audio_loader(config=config.audio)

        preprocessor = build_preprocessor(
            input_samplerate=audio_loader.samplerate,
            config=config.model.preprocess,
        )

        postprocessor = build_postprocessor(
            preprocessor,
            config=config.model.postprocess,
        )

        formatter = build_output_formatter(
            targets,
            config=config.outputs.format,
        )
        output_transform = build_output_transform(
            config=config.outputs.transform,
            targets=targets,
        )

        evaluator = build_evaluator(
            config=config.evaluation,
            targets=targets,
            transform=output_transform,
        )

        targets_changed = targets is not None or (
            config.model.targets.model_dump(mode="json")
            != model_config.targets.model_dump(mode="json")
        )
        if targets_changed:
            model = build_model_with_new_targets(
                model=model,
                targets=targets,
            )

        model.preprocessor = preprocessor
        model.postprocessor = postprocessor
        model.targets = targets

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

    def _set_trainable_parameters(
        self,
        trainable: Literal["all", "heads", "classifier_head", "bbox_head"],
    ) -> None:
        detector = cast(Detector, self.model.detector)

        for parameter in detector.parameters():
            parameter.requires_grad = False

        if trainable == "all":
            for parameter in detector.parameters():
                parameter.requires_grad = True
            return

        if trainable in {"heads", "classifier_head"}:
            for parameter in detector.classifier_head.parameters():
                parameter.requires_grad = True

        if trainable in {"heads", "bbox_head"}:
            for parameter in detector.bbox_head.parameters():
                parameter.requires_grad = True
