from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np
    import torch
    from soundevent import data

    from batdetect2.audio import AudioConfig, AudioLoader
    from batdetect2.data import Dataset
    from batdetect2.evaluate import EvaluationConfig, EvaluatorProtocol
    from batdetect2.inference import InferenceConfig
    from batdetect2.logging import (
        AppLoggingConfig,
        LoggerConfig,
        LoggingCallback,
    )
    from batdetect2.models import Model, ModelConfig
    from batdetect2.outputs import (
        OutputFormatConfig,
        OutputFormatterProtocol,
        OutputsConfig,
        OutputTransformProtocol,
    )
    from batdetect2.postprocess import (
        ClipDetections,
        Detection,
        PostprocessorProtocol,
    )
    from batdetect2.preprocess import PreprocessorProtocol
    from batdetect2.targets import (
        ROIMapperProtocol,
        TargetConfig,
        TargetProtocol,
    )
    from batdetect2.train import TrainingConfig
    from batdetect2.train.logging import TrainLoggingContext


DEFAULT_CHECKPOINT_DIR: Path = Path("outputs") / "checkpoints"
DEFAULT_LOGS_DIR: Path = Path("outputs") / "logs"
DEFAULT_EVAL_DIR: Path = Path("outputs") / "evaluations"


class BatDetect2API:
    def __init__(
        self,
        model_config: ModelConfig,
        audio_config: AudioConfig,
        train_config: TrainingConfig,
        evaluation_config: EvaluationConfig,
        inference_config: InferenceConfig,
        outputs_config: OutputsConfig,
        logging_config: AppLoggingConfig,
        targets: TargetProtocol,
        roi_mapper: ROIMapperProtocol,
        audio_loader: AudioLoader,
        preprocessor: PreprocessorProtocol,
        postprocessor: PostprocessorProtocol,
        evaluator: EvaluatorProtocol,
        formatter: OutputFormatterProtocol,
        output_transform: OutputTransformProtocol,
        model: Model,
    ):
        self.model_config = model_config
        self.audio_config = audio_config
        self.train_config = train_config
        self.evaluation_config = evaluation_config
        self.inference_config = inference_config
        self.outputs_config = outputs_config
        self.logging_config = logging_config
        self.targets = targets
        self.roi_mapper = roi_mapper
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
        from batdetect2.data import load_dataset_from_config

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
        model_config: ModelConfig | None = None,
        audio_config: AudioConfig | None = None,
        train_config: TrainingConfig | None = None,
        logger_config: LoggerConfig | None = None,
        logging_callbacks: Sequence[LoggingCallback[TrainLoggingContext]] = (),
    ):
        from batdetect2.train import run_train

        self.model.train()
        run_train(
            train_annotations=train_annotations,
            val_annotations=val_annotations,
            model=self.model,
            targets=self.targets,
            roi_mapper=self.roi_mapper,
            model_config=model_config or self.model_config,
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
            train_config=train_config or self.train_config,
            audio_config=audio_config or self.audio_config,
            logger_config=logger_config or self.logging_config.train,
            logging_callbacks=logging_callbacks,
        )
        self.model.eval()
        return self

    def finetune(
        self,
        train_annotations: Sequence[data.ClipAnnotation],
        targets_config: TargetConfig,
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
        audio_config: AudioConfig | None = None,
        train_config: TrainingConfig | None = None,
        logger_config: LoggerConfig | None = None,
        logging_callbacks: Sequence[LoggingCallback[TrainLoggingContext]] = (),
    ) -> "BatDetect2API":
        """Fine-tune from a checkpoint using a new target definition."""
        from batdetect2.evaluate import build_evaluator
        from batdetect2.models import build_model_with_new_targets
        from batdetect2.outputs import (
            build_output_formatter,
            build_output_transform,
        )
        from batdetect2.targets import (
            TargetConfig,
            build_roi_mapping,
            build_targets,
        )
        from batdetect2.train import run_train

        target_config = TargetConfig.model_validate(targets_config)
        targets = build_targets(config=target_config)
        roi_mapper = build_roi_mapping(config=target_config.roi)
        model = build_model_with_new_targets(
            model=self.model,
            targets=targets,
            roi_mapper=roi_mapper,
        )
        output_transform = build_output_transform(
            config=self.outputs_config.transform,
            targets=targets,
            roi_mapper=roi_mapper,
        )
        api = BatDetect2API(
            model_config=self.model_config,
            audio_config=audio_config or self.audio_config,
            train_config=train_config or self.train_config,
            evaluation_config=self.evaluation_config,
            inference_config=self.inference_config,
            outputs_config=self.outputs_config,
            logging_config=self.logging_config,
            targets=targets,
            roi_mapper=roi_mapper,
            audio_loader=self.audio_loader,
            preprocessor=self.preprocessor,
            postprocessor=self.postprocessor,
            evaluator=build_evaluator(
                config=self.evaluation_config,
                targets=targets,
                roi_mapper=roi_mapper,
                transform=output_transform,
            ),
            formatter=build_output_formatter(
                targets,
                config=self.outputs_config.format,
            ),
            output_transform=output_transform,
            model=model,
        )

        api._set_trainable_parameters(trainable)
        api.model.train()

        run_train(
            train_annotations=train_annotations,
            val_annotations=val_annotations,
            model=api.model,
            targets=api.targets,
            roi_mapper=api.roi_mapper,
            model_config=api.model_config,
            preprocessor=api.preprocessor,
            audio_loader=api.audio_loader,
            train_workers=train_workers,
            val_workers=val_workers,
            checkpoint_dir=checkpoint_dir,
            log_dir=log_dir,
            experiment_name=experiment_name,
            num_epochs=num_epochs,
            run_name=run_name,
            seed=seed,
            audio_config=api.audio_config,
            train_config=api.train_config,
            logger_config=logger_config or api.logging_config.train,
            logging_callbacks=logging_callbacks,
        )
        api.model.eval()
        return api

    def evaluate(
        self,
        test_annotations: Sequence[data.ClipAnnotation],
        num_workers: int = 0,
        output_dir: data.PathLike = DEFAULT_EVAL_DIR,
        experiment_name: str | None = None,
        run_name: str | None = None,
        save_predictions: bool = True,
        audio_config: AudioConfig | None = None,
        evaluation_config: EvaluationConfig | None = None,
        outputs_config: OutputsConfig | None = None,
        logger_config: LoggerConfig | None = None,
    ) -> tuple[dict[str, float], list[ClipDetections]]:
        from batdetect2.evaluate import run_evaluate

        return run_evaluate(
            self.model,
            test_annotations,
            targets=self.targets,
            roi_mapper=self.roi_mapper,
            audio_loader=self.audio_loader,
            preprocessor=self.preprocessor,
            audio_config=audio_config or self.audio_config,
            evaluation_config=evaluation_config or self.evaluation_config,
            output_config=outputs_config or self.outputs_config,
            logger_config=logger_config or self.logging_config.evaluation,
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
        from batdetect2.evaluate import save_evaluation_results

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

        import numpy as np

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
        import torch

        tensor = torch.tensor(audio).unsqueeze(0)
        return self.preprocessor(tensor)

    def process_file(
        self,
        audio_file: data.PathLike,
        batch_size: int | None = None,
        detection_threshold: float | None = None,
    ) -> ClipDetections:
        from soundevent import data

        from batdetect2.postprocess import ClipDetections

        recording = data.Recording.from_file(audio_file, compute_hash=False)

        predictions = self.process_files(
            [audio_file],
            batch_size=(
                batch_size
                if batch_size is not None
                else self.inference_config.loader.batch_size
            ),
            detection_threshold=detection_threshold,
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
        detection_threshold: float | None = None,
    ) -> list[Detection]:
        spec = self.generate_spectrogram(audio)
        return self.process_spectrogram(
            spec,
            detection_threshold=detection_threshold,
        )

    def process_spectrogram(
        self,
        spec: torch.Tensor,
        start_time: float = 0,
        detection_threshold: float | None = None,
    ) -> list[Detection]:
        if spec.ndim == 4 and spec.shape[0] > 1:
            raise ValueError("Batched spectrograms not supported.")

        if spec.ndim == 3:
            spec = spec.unsqueeze(0)

        outputs = self.model.detector(spec)

        detections = self.postprocessor(
            outputs,
            detection_threshold=detection_threshold,
        )[0]
        return self.output_transform.to_detections(
            detections=detections,
            start_time=start_time,
        )

    def process_directory(
        self,
        audio_dir: data.PathLike,
        detection_threshold: float | None = None,
    ) -> list[ClipDetections]:
        from soundevent.audio.files import get_audio_files

        files = list(get_audio_files(audio_dir))
        return self.process_files(
            files,
            detection_threshold=detection_threshold,
        )

    def process_files(
        self,
        audio_files: Sequence[data.PathLike],
        batch_size: int | None = None,
        num_workers: int = 0,
        audio_config: AudioConfig | None = None,
        inference_config: InferenceConfig | None = None,
        output_config: OutputsConfig | None = None,
        detection_threshold: float | None = None,
    ) -> list[ClipDetections]:
        from batdetect2.inference import process_file_list

        return process_file_list(
            self.model,
            audio_files,
            targets=self.targets,
            roi_mapper=self.roi_mapper,
            audio_loader=self.audio_loader,
            preprocessor=self.preprocessor,
            output_transform=self.output_transform,
            batch_size=batch_size,
            num_workers=num_workers,
            audio_config=audio_config or self.audio_config,
            inference_config=inference_config or self.inference_config,
            output_config=output_config or self.outputs_config,
            detection_threshold=detection_threshold,
        )

    def process_clips(
        self,
        clips: Sequence[data.Clip],
        batch_size: int | None = None,
        num_workers: int = 0,
        audio_config: AudioConfig | None = None,
        inference_config: InferenceConfig | None = None,
        output_config: OutputsConfig | None = None,
        detection_threshold: float | None = None,
    ) -> list[ClipDetections]:
        from batdetect2.inference import run_batch_inference

        return run_batch_inference(
            self.model,
            clips,
            targets=self.targets,
            roi_mapper=self.roi_mapper,
            audio_loader=self.audio_loader,
            preprocessor=self.preprocessor,
            output_transform=self.output_transform,
            batch_size=batch_size,
            num_workers=num_workers,
            audio_config=audio_config or self.audio_config,
            inference_config=inference_config or self.inference_config,
            output_config=output_config or self.outputs_config,
            detection_threshold=detection_threshold,
        )

    def save_predictions(
        self,
        predictions: Sequence[ClipDetections],
        path: data.PathLike,
        audio_dir: data.PathLike | None = None,
        format: str | None = None,
        config: OutputFormatConfig | None = None,
    ):
        from batdetect2.outputs import get_output_formatter

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
        from batdetect2.outputs import get_output_formatter

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
        model_config: ModelConfig | None = None,
        targets_config: TargetConfig | None = None,
        audio_config: AudioConfig | None = None,
        train_config: TrainingConfig | None = None,
        evaluation_config: EvaluationConfig | None = None,
        inference_config: InferenceConfig | None = None,
        outputs_config: OutputsConfig | None = None,
        logging_config: AppLoggingConfig | None = None,
    ) -> "BatDetect2API":
        from batdetect2.audio import AudioConfig, build_audio_loader
        from batdetect2.evaluate import EvaluationConfig, build_evaluator
        from batdetect2.inference import InferenceConfig
        from batdetect2.logging import AppLoggingConfig
        from batdetect2.models import ModelConfig, build_model
        from batdetect2.outputs import (
            OutputsConfig,
            build_output_formatter,
            build_output_transform,
        )
        from batdetect2.postprocess import build_postprocessor
        from batdetect2.preprocess import build_preprocessor
        from batdetect2.targets import (
            TargetConfig,
            build_roi_mapping,
            build_targets,
        )
        from batdetect2.train import TrainingConfig

        model_config = model_config or ModelConfig()
        targets_config = targets_config or TargetConfig()
        audio_config = audio_config or AudioConfig()
        train_config = train_config or TrainingConfig()
        evaluation_config = evaluation_config or EvaluationConfig()
        inference_config = inference_config or InferenceConfig()
        outputs_config = outputs_config or OutputsConfig()
        logging_config = logging_config or AppLoggingConfig()

        targets = build_targets(config=targets_config)
        roi_mapper = build_roi_mapping(config=targets_config.roi)

        audio_loader = build_audio_loader(config=audio_config)

        preprocessor = build_preprocessor(
            input_samplerate=audio_loader.samplerate,
            config=model_config.preprocess,
        )

        postprocessor = build_postprocessor(
            preprocessor,
            config=model_config.postprocess,
        )

        formatter = build_output_formatter(
            targets,
            config=outputs_config.format,
        )
        output_transform = build_output_transform(
            config=outputs_config.transform,
            targets=targets,
            roi_mapper=roi_mapper,
        )

        evaluator = build_evaluator(
            config=evaluation_config,
            targets=targets,
            roi_mapper=roi_mapper,
            transform=output_transform,
        )

        # NOTE: Build separate instances of preprocessor and postprocessor
        # to avoid device mismatch errors
        model = build_model(
            config=model_config,
            class_names=targets.class_names,
            dimension_names=roi_mapper.dimension_names,
            preprocessor=build_preprocessor(
                input_samplerate=audio_loader.samplerate,
                config=model_config.preprocess,
            ),
            postprocessor=build_postprocessor(
                preprocessor,
                config=model_config.postprocess,
            ),
        )

        return cls(
            model_config=model_config,
            audio_config=audio_config,
            train_config=train_config,
            evaluation_config=evaluation_config,
            inference_config=inference_config,
            outputs_config=outputs_config,
            logging_config=logging_config,
            targets=targets,
            roi_mapper=roi_mapper,
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
        path: data.PathLike | str | None = None,
        audio_config: AudioConfig | None = None,
        train_config: TrainingConfig | None = None,
        evaluation_config: EvaluationConfig | None = None,
        inference_config: InferenceConfig | None = None,
        outputs_config: OutputsConfig | None = None,
        logging_config: AppLoggingConfig | None = None,
    ) -> "BatDetect2API":
        from batdetect2.audio import AudioConfig, build_audio_loader
        from batdetect2.evaluate import EvaluationConfig, build_evaluator
        from batdetect2.inference import InferenceConfig
        from batdetect2.logging import AppLoggingConfig
        from batdetect2.outputs import (
            OutputsConfig,
            build_output_formatter,
            build_output_transform,
        )
        from batdetect2.postprocess import build_postprocessor
        from batdetect2.preprocess import build_preprocessor
        from batdetect2.targets import (
            build_roi_mapping,
            build_targets,
            check_target_compatibility,
        )
        from batdetect2.train import load_model_from_checkpoint

        model, configs = load_model_from_checkpoint(path)

        model_config = configs.model
        train_config = train_config or configs.train

        audio_config = audio_config or AudioConfig(
            samplerate=model_config.samplerate,
        )
        evaluation_config = evaluation_config or EvaluationConfig()
        inference_config = inference_config or InferenceConfig()
        outputs_config = outputs_config or OutputsConfig()
        logging_config = logging_config or AppLoggingConfig()
        targets_config = configs.targets

        targets = build_targets(config=targets_config)
        roi_mapper = build_roi_mapping(config=targets_config.roi)

        if not check_target_compatibility(targets, model.class_names):
            raise ValueError(
                "Provided targets_config is incompatible with the "
                "checkpoint model: missing one or more model classes."
            )

        if model.dimension_names != roi_mapper.dimension_names:
            raise ValueError(
                "Provided targets_config is incompatible with the "
                "checkpoint model: mismatched dimension names."
            )

        audio_loader = build_audio_loader(config=audio_config)

        preprocessor = build_preprocessor(
            input_samplerate=audio_loader.samplerate,
            config=model_config.preprocess,
        )

        postprocessor = build_postprocessor(
            preprocessor,
            config=model_config.postprocess,
        )

        formatter = build_output_formatter(
            targets,
            config=outputs_config.format,
        )

        output_transform = build_output_transform(
            config=outputs_config.transform,
            targets=targets,
            roi_mapper=roi_mapper,
        )

        evaluator = build_evaluator(
            config=evaluation_config,
            targets=targets,
            roi_mapper=roi_mapper,
            transform=output_transform,
        )

        return cls(
            model_config=model_config,
            audio_config=audio_config,
            train_config=train_config,
            evaluation_config=evaluation_config,
            inference_config=inference_config,
            outputs_config=outputs_config,
            logging_config=logging_config,
            targets=targets,
            roi_mapper=roi_mapper,
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
        detector = self.model.detector

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
