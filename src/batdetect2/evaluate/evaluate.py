from pathlib import Path
from typing import Sequence

from lightning import Trainer
from soundevent import data

from batdetect2.audio import AudioConfig, build_audio_loader
from batdetect2.audio.types import AudioLoader
from batdetect2.evaluate import EvaluationConfig
from batdetect2.evaluate.dataset import build_test_loader
from batdetect2.evaluate.evaluator import build_evaluator
from batdetect2.evaluate.lightning import EvaluationModule
from batdetect2.logging import CSVLoggerConfig, LoggerConfig, build_logger
from batdetect2.models import Model
from batdetect2.outputs import OutputsConfig, build_output_transform
from batdetect2.outputs.types import OutputFormatterProtocol
from batdetect2.postprocess.types import ClipDetections
from batdetect2.preprocess.types import PreprocessorProtocol
from batdetect2.targets.types import ROIMapperProtocol, TargetProtocol

DEFAULT_EVAL_DIR: Path = Path("outputs") / "evaluations"


def run_evaluate(
    model: Model,
    test_annotations: Sequence[data.ClipAnnotation],
    targets: TargetProtocol | None = None,
    roi_mapper: ROIMapperProtocol | None = None,
    audio_loader: AudioLoader | None = None,
    preprocessor: PreprocessorProtocol | None = None,
    audio_config: AudioConfig | None = None,
    evaluation_config: EvaluationConfig | None = None,
    output_config: OutputsConfig | None = None,
    logger_config: LoggerConfig | None = None,
    formatter: OutputFormatterProtocol | None = None,
    num_workers: int = 0,
    output_dir: data.PathLike = DEFAULT_EVAL_DIR,
    experiment_name: str | None = None,
    run_name: str | None = None,
) -> tuple[dict[str, float], list[ClipDetections]]:

    audio_config = audio_config or AudioConfig()
    evaluation_config = evaluation_config or EvaluationConfig()
    output_config = output_config or OutputsConfig()

    audio_loader = audio_loader or build_audio_loader(config=audio_config)

    preprocessor = preprocessor or model.preprocessor
    targets = targets or model.targets
    roi_mapper = roi_mapper or model.roi_mapper

    loader = build_test_loader(
        test_annotations,
        audio_loader=audio_loader,
        preprocessor=preprocessor,
        num_workers=num_workers,
    )

    output_transform = build_output_transform(
        config=output_config.transform,
        targets=targets,
        roi_mapper=roi_mapper,
    )
    evaluator = build_evaluator(
        config=evaluation_config,
        targets=targets,
        transform=output_transform,
    )

    logger = build_logger(
        logger_config or CSVLoggerConfig(),
        log_dir=Path(output_dir),
        experiment_name=experiment_name,
        run_name=run_name,
    )
    module = EvaluationModule(
        model,
        evaluator,
    )
    trainer = Trainer(logger=logger, enable_checkpointing=False)
    metrics = trainer.test(module, loader)

    if formatter is not None and logger.log_dir is not None:
        formatter.save(
            module.predictions,
            path=Path(logger.log_dir) / "predictions",
        )

    return metrics, module.predictions  # type: ignore
