from pathlib import Path
from typing import TYPE_CHECKING, Optional, Sequence

from lightning import Trainer
from soundevent import data

from batdetect2.audio import build_audio_loader
from batdetect2.evaluate.dataset import build_test_loader
from batdetect2.evaluate.evaluator import build_evaluator
from batdetect2.evaluate.lightning import EvaluationModule
from batdetect2.logging import build_logger
from batdetect2.models import Model
from batdetect2.preprocess import build_preprocessor
from batdetect2.targets import build_targets

if TYPE_CHECKING:
    from batdetect2.config import BatDetect2Config
    from batdetect2.typing import (
        AudioLoader,
        PreprocessorProtocol,
        TargetProtocol,
    )

DEFAULT_OUTPUT_DIR: Path = Path("outputs") / "evaluations"


def evaluate(
    model: Model,
    test_annotations: Sequence[data.ClipAnnotation],
    targets: Optional["TargetProtocol"] = None,
    audio_loader: Optional["AudioLoader"] = None,
    preprocessor: Optional["PreprocessorProtocol"] = None,
    config: Optional["BatDetect2Config"] = None,
    num_workers: Optional[int] = None,
    output_dir: data.PathLike = DEFAULT_OUTPUT_DIR,
    experiment_name: Optional[str] = None,
    run_name: Optional[str] = None,
):
    from batdetect2.config import BatDetect2Config

    config = config or BatDetect2Config()

    audio_loader = audio_loader or build_audio_loader()

    preprocessor = preprocessor or build_preprocessor(
        input_samplerate=audio_loader.samplerate,
    )

    targets = targets or build_targets()

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
    module = EvaluationModule(model, evaluator)
    trainer = Trainer(logger=logger, enable_checkpointing=False)
    return trainer.test(module, loader)
