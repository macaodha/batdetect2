from pathlib import Path
from typing import Optional

from lightning.pytorch.callbacks import Callback, ModelCheckpoint

from batdetect2.core import BaseConfig

__all__ = [
    "CheckpointConfig",
    "build_checkpoint_callback",
]

DEFAULT_CHECKPOINT_DIR: Path = Path("outputs") / "checkpoints"


class CheckpointConfig(BaseConfig):
    checkpoint_dir: Path = DEFAULT_CHECKPOINT_DIR
    monitor: str = "classification/mean_average_precision"
    mode: str = "max"
    save_top_k: int = 1
    filename: Optional[str] = None


def build_checkpoint_callback(
    config: Optional[CheckpointConfig] = None,
    checkpoint_dir: Optional[Path] = None,
    experiment_name: Optional[str] = None,
    run_name: Optional[str] = None,
) -> Callback:
    config = config or CheckpointConfig()

    if checkpoint_dir is None:
        checkpoint_dir = config.checkpoint_dir

    if experiment_name is not None:
        checkpoint_dir = checkpoint_dir / experiment_name

    if run_name is not None:
        checkpoint_dir = checkpoint_dir / run_name

    return ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        save_top_k=config.save_top_k,
        monitor=config.monitor,
        mode=config.mode,
        filename=config.filename,
    )
