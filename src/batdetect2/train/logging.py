import io
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import (
    Annotated,
    Any,
    Dict,
    Generic,
    Literal,
    Optional,
    Protocol,
    TypeVar,
    Union,
)

import numpy as np
from lightning.pytorch.loggers import (
    CSVLogger,
    Logger,
    MLFlowLogger,
    TensorBoardLogger,
)
from loguru import logger
from matplotlib.figure import Figure
from pydantic import Field
from soundevent import data

from batdetect2.core.configs import BaseConfig

DEFAULT_LOGS_DIR: Path = Path("outputs") / "logs"


class BaseLoggerConfig(BaseConfig):
    log_dir: Path = DEFAULT_LOGS_DIR
    experiment_name: Optional[str] = None
    run_name: Optional[str] = None


class DVCLiveConfig(BaseLoggerConfig):
    name: Literal["dvclive"] = "dvclive"
    prefix: str = ""
    log_model: Union[bool, Literal["all"]] = False
    monitor_system: bool = False


class CSVLoggerConfig(BaseLoggerConfig):
    name: Literal["csv"] = "csv"
    flush_logs_every_n_steps: int = 100


class TensorBoardLoggerConfig(BaseLoggerConfig):
    name: Literal["tensorboard"] = "tensorboard"
    log_graph: bool = False


class MLFlowLoggerConfig(BaseLoggerConfig):
    name: Literal["mlflow"] = "mlflow"
    tracking_uri: Optional[str] = None
    tags: Optional[dict[str, Any]] = None
    log_model: bool = False


LoggerConfig = Annotated[
    Union[
        DVCLiveConfig,
        CSVLoggerConfig,
        TensorBoardLoggerConfig,
        MLFlowLoggerConfig,
    ],
    Field(discriminator="name"),
]


T = TypeVar("T", bound=LoggerConfig, contravariant=True)


class LoggerBuilder(Protocol, Generic[T]):
    def __call__(
        self,
        config: T,
        log_dir: Optional[Path] = None,
        experiment_name: Optional[str] = None,
        run_name: Optional[str] = None,
    ) -> Logger: ...


def create_dvclive_logger(
    config: DVCLiveConfig,
    log_dir: Optional[Path] = None,
    experiment_name: Optional[str] = None,
    run_name: Optional[str] = None,
) -> Logger:
    try:
        from dvclive.lightning import DVCLiveLogger  # type: ignore
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
    log_dir: Optional[Path] = None,
    experiment_name: Optional[str] = None,
    run_name: Optional[str] = None,
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
    log_dir: Optional[Path] = None,
    experiment_name: Optional[str] = None,
    run_name: Optional[str] = None,
) -> Logger:
    from lightning.pytorch.loggers import TensorBoardLogger

    if log_dir is None:
        log_dir = Path(config.log_dir)

    if run_name is None:
        run_name = config.run_name

    if experiment_name is None:
        experiment_name = config.experiment_name

    name = run_name

    if run_name is not None and experiment_name is not None:
        name = str(Path(experiment_name) / run_name)

    return TensorBoardLogger(
        save_dir=str(log_dir),
        name=name,
        log_graph=config.log_graph,
    )


def create_mlflow_logger(
    config: MLFlowLoggerConfig,
    log_dir: Optional[data.PathLike] = None,
    experiment_name: Optional[str] = None,
    run_name: Optional[str] = None,
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
    log_dir: Optional[Path] = None,
    experiment_name: Optional[str] = None,
    run_name: Optional[str] = None,
) -> Logger:
    """
    Creates a logger instance from a validated Pydantic config object.
    """
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


Plotter = Callable[[str, Figure, int], None]


def get_image_plotter(logger: Logger) -> Optional[Plotter]:
    if isinstance(logger, TensorBoardLogger):
        return logger.experiment.add_figure

    if isinstance(logger, MLFlowLogger):

        def plot_figure(name, figure, step):
            image = _convert_figure_to_array(figure)
            return logger.experiment.log_image(
                logger.run_id,
                image,
                key=name,
                step=step,
            )

        return plot_figure

    if isinstance(logger, CSVLogger):
        return partial(save_figure, dir=Path(logger.log_dir))


def save_figure(name: str, fig: Figure, step: int, dir: Path) -> None:
    path = dir / "plots" / f"{name}_step_{step}.png"

    if not path.parent.exists():
        path.parent.mkdir(parents=True)

    fig.savefig(path, transparent=True, bbox_inches="tight")


def _convert_figure_to_array(figure: Figure) -> np.ndarray:
    with io.BytesIO() as buff:
        figure.savefig(buff, format="raw")
        buff.seek(0)
        data = np.frombuffer(buff.getvalue(), dtype=np.uint8)
    w, h = figure.canvas.get_width_height()
    im = data.reshape((int(h), int(w), -1))
    return im
