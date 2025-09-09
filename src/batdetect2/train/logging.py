import io
from typing import Annotated, Any, Literal, Optional, Union

import numpy as np
from lightning.pytorch.loggers import Logger, MLFlowLogger, TensorBoardLogger
from loguru import logger
from pydantic import Field
from soundevent import data

from batdetect2.configs import BaseConfig

DEFAULT_LOGS_DIR: str = "outputs"


class DVCLiveConfig(BaseConfig):
    logger_type: Literal["dvclive"] = "dvclive"
    dir: str = DEFAULT_LOGS_DIR
    run_name: Optional[str] = None
    prefix: str = ""
    log_model: Union[bool, Literal["all"]] = False
    monitor_system: bool = False


class CSVLoggerConfig(BaseConfig):
    logger_type: Literal["csv"] = "csv"
    save_dir: str = DEFAULT_LOGS_DIR
    name: Optional[str] = "logs"
    version: Optional[str] = None
    flush_logs_every_n_steps: int = 100


class TensorBoardLoggerConfig(BaseConfig):
    logger_type: Literal["tensorboard"] = "tensorboard"
    save_dir: str = DEFAULT_LOGS_DIR
    name: Optional[str] = "logs"
    version: Optional[str] = None
    log_graph: bool = False


class MLFlowLoggerConfig(BaseConfig):
    logger_type: Literal["mlflow"] = "mlflow"
    experiment_name: str = "default"
    run_name: Optional[str] = None
    save_dir: Optional[str] = "./mlruns"
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
    Field(discriminator="logger_type"),
]


def create_dvclive_logger(
    config: DVCLiveConfig,
    log_dir: Optional[data.PathLike] = None,
    experiment_name: Optional[str] = None,
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
        dir=log_dir if log_dir is not None else config.dir,
        run_name=experiment_name
        if experiment_name is not None
        else config.run_name,
        prefix=config.prefix,
        log_model=config.log_model,
        monitor_system=config.monitor_system,
    )


def create_csv_logger(
    config: CSVLoggerConfig,
    log_dir: Optional[data.PathLike] = None,
    experiment_name: Optional[str] = None,
) -> Logger:
    from lightning.pytorch.loggers import CSVLogger

    return CSVLogger(
        save_dir=str(log_dir) if log_dir is not None else config.save_dir,
        name=experiment_name if experiment_name is not None else config.name,
        version=config.version,
        flush_logs_every_n_steps=config.flush_logs_every_n_steps,
    )


def create_tensorboard_logger(
    config: TensorBoardLoggerConfig,
    log_dir: Optional[data.PathLike] = None,
    experiment_name: Optional[str] = None,
) -> Logger:
    from lightning.pytorch.loggers import TensorBoardLogger

    return TensorBoardLogger(
        save_dir=str(log_dir) if log_dir is not None else config.save_dir,
        name=experiment_name if experiment_name is not None else config.name,
        version=config.version,
        log_graph=config.log_graph,
    )


def create_mlflow_logger(
    config: MLFlowLoggerConfig,
    log_dir: Optional[data.PathLike] = None,
    experiment_name: Optional[str] = None,
) -> Logger:
    try:
        from lightning.pytorch.loggers import MLFlowLogger
    except ImportError as error:
        raise ValueError(
            "MLFlow is not installed and cannot be used for logging. "
            "Make sure you have it installed by running `pip install mlflow` "
            "or `uv add mlflow`"
        ) from error

    return MLFlowLogger(
        experiment_name=experiment_name
        if experiment_name is not None
        else config.experiment_name,
        run_name=config.run_name,
        save_dir=str(log_dir) if log_dir is not None else config.save_dir,
        tracking_uri=config.tracking_uri,
        tags=config.tags,
        log_model=config.log_model,
    )


LOGGER_FACTORY = {
    "dvclive": create_dvclive_logger,
    "csv": create_csv_logger,
    "tensorboard": create_tensorboard_logger,
    "mlflow": create_mlflow_logger,
}


def build_logger(
    config: LoggerConfig,
    log_dir: Optional[data.PathLike] = None,
    experiment_name: Optional[str] = None,
) -> Logger:
    """
    Creates a logger instance from a validated Pydantic config object.
    """
    logger.opt(lazy=True).debug(
        "Building logger with config: \n{}",
        lambda: config.to_yaml_string(),
    )
    logger_type = config.logger_type

    if logger_type not in LOGGER_FACTORY:
        raise ValueError(f"Unknown logger type: {logger_type}")

    creation_func = LOGGER_FACTORY[logger_type]

    return creation_func(
        config,
        log_dir=log_dir,
        experiment_name=experiment_name,
    )


def get_image_plotter(logger: Logger):
    if isinstance(logger, TensorBoardLogger):

        def plot_figure(name, figure, step):
            return logger.experiment.add_figure(name, figure, step)

        return plot_figure

    if isinstance(logger, MLFlowLogger):

        def plot_figure(name, figure, step):
            image = _convert_figure_to_image(figure)
            return logger.experiment.log_image(
                run_id=logger.run_id,
                image=image,
                key=name,
                step=step,
            )

        return plot_figure


def _convert_figure_to_image(figure):
    with io.BytesIO() as buff:
        figure.savefig(buff, format="raw")
        buff.seek(0)
        data = np.frombuffer(buff.getvalue(), dtype=np.uint8)
    w, h = figure.canvas.get_width_height()
    im = data.reshape((int(h), int(w), -1))
    return im
