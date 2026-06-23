from __future__ import annotations

import io
import sys
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Dict,
    Generic,
    Literal,
    Protocol,
    TypeVar,
)

from loguru import logger
from pydantic import Field

from batdetect2.core.configs import BaseConfig

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd
    from lightning.pytorch.loggers import Logger
    from matplotlib.figure import Figure
    from soundevent import data

DEFAULT_LOGS_DIR: Path = Path("outputs") / "logs"

__all__ = [
    "AppLoggingConfig",
    "BaseLoggerConfig",
    "CSVLoggerConfig",
    "DEFAULT_LOGS_DIR",
    "DVCLiveConfig",
    "LoggerConfig",
    "MLFlowLoggerConfig",
    "LoggingCallback",
    "TensorBoardLoggerConfig",
    "build_logger",
    "enable_logging",
    "get_image_logger",
    "get_table_logger",
    "log_artifact_file",
    "log_config_artifact",
    "log_csv_artifact",
]


def enable_logging(level: int, log_file: Path | None = None) -> None:
    logger.remove()

    if level == 0:
        log_level = "WARNING"
    elif level == 1:
        log_level = "INFO"
    else:
        log_level = "DEBUG"

    logger.add(sys.stderr, level=log_level)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        logger.add(log_file, level=log_level)

    logger.enable("batdetect2")


class BaseLoggerConfig(BaseConfig):
    log_dir: Path = DEFAULT_LOGS_DIR
    experiment_name: str | None = None
    run_name: str | None = None


class DVCLiveConfig(BaseLoggerConfig):
    name: Literal["dvclive"] = "dvclive"
    prefix: str = ""
    log_model: bool | Literal["all"] = False
    monitor_system: bool = False


class CSVLoggerConfig(BaseLoggerConfig):
    name: Literal["csv"] = "csv"
    flush_logs_every_n_steps: int = 100


class TensorBoardLoggerConfig(BaseLoggerConfig):
    name: Literal["tensorboard"] = "tensorboard"
    log_graph: bool = False


class MLFlowLoggerConfig(BaseLoggerConfig):
    name: Literal["mlflow"] = "mlflow"
    tracking_uri: str | None = "http://localhost:5000"
    tags: dict[str, Any] | None = None
    log_model: bool = False


LoggerConfig = Annotated[
    DVCLiveConfig
    | CSVLoggerConfig
    | TensorBoardLoggerConfig
    | MLFlowLoggerConfig,
    Field(discriminator="name"),
]


class AppLoggingConfig(BaseConfig):
    train: LoggerConfig = Field(default_factory=CSVLoggerConfig)
    evaluation: LoggerConfig = Field(default_factory=CSVLoggerConfig)
    inference: LoggerConfig = Field(default_factory=CSVLoggerConfig)


T = TypeVar("T", bound=LoggerConfig, contravariant=True)


class LoggerBuilder(Protocol, Generic[T]):
    def __call__(
        self,
        config: T,
        log_dir: Path | None = None,
        experiment_name: str | None = None,
        run_name: str | None = None,
    ) -> Logger: ...


LoggingContext = TypeVar("LoggingContext", contravariant=True)


class LoggingCallback(Protocol, Generic[LoggingContext]):
    def run(
        self,
        logger: Logger,
        artifact_path: Path,
        context: LoggingContext,
    ) -> None: ...


def create_dvclive_logger(
    config: DVCLiveConfig,
    log_dir: Path | None = None,
    experiment_name: str | None = None,
    run_name: str | None = None,
) -> Logger:
    try:
        from dvclive.lightning import DVCLiveLogger
    except ImportError as error:
        raise ValueError(
            "DVCLive is not installed and cannot be used for logging"
            "Make sure you have it installed by running `pip install dvclive`"
            "or `uv add dvclive`"
        ) from error

    return DVCLiveLogger(
        dir=log_dir if log_dir is not None else config.log_dir,
        run_name=run_name if run_name is not None else config.run_name,
        experiment=experiment_name
        if experiment_name is not None
        else config.experiment_name,
        prefix=config.prefix,
        log_model=config.log_model,
        monitor_system=config.monitor_system,
    )


def create_csv_logger(
    config: CSVLoggerConfig,
    log_dir: Path | None = None,
    experiment_name: str | None = None,
    run_name: str | None = None,
) -> Logger:
    from lightning.pytorch.loggers import CSVLogger

    if log_dir is None:
        log_dir = Path(config.log_dir)

    if run_name is None:
        run_name = config.run_name

    if experiment_name is None:
        experiment_name = config.experiment_name

    name = run_name

    if run_name is not None and experiment_name is not None:
        name = str(Path(experiment_name) / run_name)

    return CSVLogger(
        save_dir=str(log_dir),
        name=name,
        flush_logs_every_n_steps=config.flush_logs_every_n_steps,
    )


def create_tensorboard_logger(
    config: TensorBoardLoggerConfig,
    log_dir: Path | None = None,
    experiment_name: str | None = None,
    run_name: str | None = None,
) -> Logger:
    from lightning.pytorch.loggers import TensorBoardLogger

    if log_dir is None:
        log_dir = Path(config.log_dir)

    if run_name is None:
        run_name = config.run_name

    if experiment_name is None:
        experiment_name = config.experiment_name

    name = run_name

    if name is None:
        name = experiment_name

    if run_name is not None and experiment_name is not None:
        name = str(Path(experiment_name) / run_name)

    return TensorBoardLogger(
        save_dir=str(log_dir),
        name=name,
        log_graph=config.log_graph,
    )


def create_mlflow_logger(
    config: MLFlowLoggerConfig,
    log_dir: data.PathLike | None = None,
    experiment_name: str | None = None,
    run_name: str | None = None,
) -> Logger:
    try:
        from lightning.pytorch.loggers import MLFlowLogger
    except ImportError as error:
        raise ValueError(
            "MLFlow is not installed and cannot be used for logging. "
            "Make sure you have it installed by running `pip install mlflow` "
            "or `uv add mlflow`"
        ) from error

    if experiment_name is None:
        experiment_name = config.experiment_name or "Default"

    if log_dir is None:
        log_dir = config.log_dir

    return MLFlowLogger(
        experiment_name=experiment_name
        if experiment_name is not None
        else config.experiment_name,
        run_name=run_name if run_name is not None else config.run_name,
        save_dir=str(log_dir),
        tracking_uri=config.tracking_uri,
        tags=config.tags,
        log_model=config.log_model,
    )


LOGGER_FACTORY: Dict[str, LoggerBuilder] = {
    "dvclive": create_dvclive_logger,
    "csv": create_csv_logger,
    "tensorboard": create_tensorboard_logger,
    "mlflow": create_mlflow_logger,
}


def build_logger(
    config: LoggerConfig,
    log_dir: Path | None = None,
    experiment_name: str | None = None,
    run_name: str | None = None,
) -> Logger:
    logger.opt(lazy=True).debug(
        "Building logger with config: \n{}",
        lambda: config.to_yaml_string(),
    )

    logger_type = config.name
    if logger_type not in LOGGER_FACTORY:
        raise ValueError(f"Unknown logger type: {logger_type}")

    creation_func = LOGGER_FACTORY[logger_type]
    return creation_func(
        config,
        log_dir=log_dir,
        experiment_name=experiment_name,
        run_name=run_name,
    )


def log_artifact_file(
    runtime_logger: Logger,
    path: Path,
    artifact_path: str = "artifacts",
) -> None:
    from lightning.pytorch.loggers import (
        CSVLogger,
        MLFlowLogger,
        TensorBoardLogger,
    )

    if isinstance(runtime_logger, MLFlowLogger):
        runtime_logger.experiment.log_artifact(  # type: ignore[call-arg]
            local_path=str(path),
            artifact_path=artifact_path,
            run_id=runtime_logger.run_id,
        )
        return

    experiment = getattr(runtime_logger, "experiment", None)
    if experiment is not None and hasattr(experiment, "log_artifact"):
        experiment.log_artifact(path=path, name=path.name, copy=True)
        return

    if isinstance(runtime_logger, (CSVLogger, TensorBoardLogger)):
        return

    logger.warning(
        "Skipping artifact logging for unsupported logger type {logger_type}",
        logger_type=type(runtime_logger).__name__,
    )


def log_config_artifact(
    logger: Logger,
    config: BaseConfig,
    filename: str,
    artifact_path: Path,
) -> None:
    artifact_path.mkdir(parents=True, exist_ok=True)
    path = artifact_path / filename
    path.write_text(config.to_yaml_string())
    log_artifact_file(
        logger,
        path,
        artifact_path=artifact_path.name,
    )


def log_csv_artifact(
    logger: Logger,
    df: pd.DataFrame,
    filename: str,
    artifact_path: Path,
) -> None:
    artifact_path.mkdir(parents=True, exist_ok=True)
    path = artifact_path / filename
    df.to_csv(path, index=False)
    log_artifact_file(
        logger,
        path,
        artifact_path=artifact_path.name,
    )


PlotLogger = Callable[[str, "Figure", int], None]


def get_image_logger(logger: Logger) -> PlotLogger | None:
    from lightning.pytorch.loggers import (
        CSVLogger,
        MLFlowLogger,
        TensorBoardLogger,
    )

    if isinstance(logger, TensorBoardLogger):
        return logger.experiment.add_figure

    if isinstance(logger, MLFlowLogger):

        def plot_figure(name, figure, step):
            image = _convert_figure_to_array(figure)
            name = name.replace("/", "_")
            return logger.experiment.log_image(
                logger.run_id,
                image,
                key=name,
                step=step,
            )

        return plot_figure

    if isinstance(logger, CSVLogger):
        return partial(save_figure, dir=Path(logger.log_dir))


TableLogger = Callable[[str, "pd.DataFrame", int], None]


def get_table_logger(logger: Logger) -> TableLogger | None:
    from lightning.pytorch.loggers import (
        CSVLogger,
        MLFlowLogger,
        TensorBoardLogger,
    )

    if isinstance(logger, TensorBoardLogger):
        return partial(save_table, dir=Path(logger.log_dir))

    if isinstance(logger, MLFlowLogger):

        def plot_figure(name: str, df: pd.DataFrame, step: int):
            return logger.experiment.log_table(
                logger.run_id,
                data=df,
                artifact_file=f"{name}_step_{step}.json",
            )

        return plot_figure

    if isinstance(logger, CSVLogger):
        return partial(save_table, dir=Path(logger.log_dir))


def save_table(name: str, df: pd.DataFrame, step: int, dir: Path) -> None:
    path = dir / "tables" / f"{name}_step_{step}.csv"

    if not path.parent.exists():
        path.parent.mkdir(parents=True)

    df.to_csv(path, index=False)


def save_figure(name: str, fig: Figure, step: int, dir: Path) -> None:
    path = dir / "plots" / f"{name}_step_{step}.png"

    if not path.parent.exists():
        path.parent.mkdir(parents=True)

    fig.savefig(path, transparent=True, bbox_inches="tight")


def _convert_figure_to_array(figure: Figure) -> np.ndarray:
    import numpy as np

    with io.BytesIO() as buff:
        figure.savefig(buff, format="raw")
        buff.seek(0)
        data = np.frombuffer(buff.getvalue(), dtype=np.uint8)
    w, h = figure.canvas.get_width_height()
    im = data.reshape((int(h), int(w), -1))
    return im
