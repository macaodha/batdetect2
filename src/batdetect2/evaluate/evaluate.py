from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple

from lightning import Trainer
from soundevent import data

from batdetect2.audio import build_audio_loader
from batdetect2.evaluate.dataset import build_test_loader
from batdetect2.evaluate.evaluator import build_evaluator
from batdetect2.evaluate.lightning import EvaluationModule
from batdetect2.logging import build_logger
from batdetect2.models import Model
from batdetect2.outputs import build_output_transform
from batdetect2.typing import Detection

if TYPE_CHECKING:
    from batdetect2.config import BatDetect2Config
    from batdetect2.typing import (
        AudioLoader,
        OutputFormatterProtocol,
        PreprocessorProtocol,
        TargetProtocol,
    )

DEFAULT_EVAL_DIR: Path = Path("outputs") / "evaluations"


def evaluate(
    model: Model,
    test_annotations: Sequence[data.ClipAnnotation],
    targets: Optional["TargetProtocol"] = None,
    audio_loader: Optional["AudioLoader"] = None,
    preprocessor: Optional["PreprocessorProtocol"] = None,
    config: Optional["BatDetect2Config"] = None,
    formatter: Optional["OutputFormatterProtocol"] = None,
    num_workers: int | None = None,
    output_dir: data.PathLike = DEFAULT_EVAL_DIR,
    experiment_name: str | None = None,
    run_name: str | None = None,
) -> Tuple[Dict[str, float], List[List[Detection]]]:
    from batdetect2.config import BatDetect2Config

    config = config or BatDetect2Config()

    audio_loader = audio_loader or build_audio_loader(config=config.audio)

    preprocessor = preprocessor or model.preprocessor
    targets = targets or model.targets

    loader = build_test_loader(
        test_annotations,
        audio_loader=audio_loader,
        preprocessor=preprocessor,
        num_workers=num_workers,
    )

    evaluator = build_evaluator(config=config.evaluation, targets=targets)

    logger = build_logger(
        config.evaluation.logger,
        log_dir=Path(output_dir),
        experiment_name=experiment_name,
        run_name=run_name,
    )
    output_transform = build_output_transform(config=config.outputs.transform)
    module = EvaluationModule(
        model,
        evaluator,
        output_transform=output_transform,
    )
    trainer = Trainer(logger=logger, enable_checkpointing=False)
    metrics = trainer.test(module, loader)

    if formatter is not None and logger.log_dir is not None:
        formatter.save(
            module.predictions,
            path=Path(logger.log_dir) / "predictions",
        )

    return metrics, module.predictions  # type: ignore
