from typing import Annotated, Literal, Optional, Union

from lightning.pytorch.loggers import Logger
from pydantic import Field
from soundevent.data import PathLike

from batdetect2.configs import BaseConfig, load_config

DEFAULT_LOGS_DIR: str = "logs"


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
    name: Optional[str] = "default"
    version: Optional[str] = None
    log_graph: bool = False
    flush_logs_every_n_steps: Optional[int] = None


LoggerConfig = Annotated[
    Union[DVCLiveConfig, CSVLoggerConfig, TensorBoardLoggerConfig],
    Field(discriminator="logger_type"),
]


def create_dvclive_logger(config: DVCLiveConfig) -> Logger:
    try:
        from dvclive.lightning import DVCLiveLogger  # type: ignore
    except ImportError as error:
        raise ValueError(
            "DVCLive is not installed and cannot be used for logging"
            "Make sure you have it installed by running `pip install dvclive`"
            "or `uv add dvclive`"
        ) from error

    return DVCLiveLogger(
        dir=config.dir,
        run_name=config.run_name,
        prefix=config.prefix,
        log_model=config.log_model,
        monitor_system=config.monitor_system,
    )


def create_csv_logger(config: CSVLoggerConfig) -> Logger:
    from lightning.pytorch.loggers import CSVLogger

    return CSVLogger(
        save_dir=config.save_dir,
        name=config.name,
        version=config.version,
        flush_logs_every_n_steps=config.flush_logs_every_n_steps,
    )


def create_tensorboard_logger(config: TensorBoardLoggerConfig) -> Logger:
    from lightning.pytorch.loggers import TensorBoardLogger

    return TensorBoardLogger(
        save_dir=config.save_dir,
        name=config.name,
        version=config.version,
        log_graph=config.log_graph,
        flush_logs_every_n_steps=config.flush_logs_every_n_steps,
    )


LOGGER_FACTORY = {
    "dvclive": create_dvclive_logger,
    "csv": create_csv_logger,
    "tensorboard": create_tensorboard_logger,
}


def build_logger(config: LoggerConfig) -> Logger:
    """
    Creates a logger instance from a validated Pydantic config object.
    """
    logger_type = config.logger_type

    if logger_type not in LOGGER_FACTORY:
        raise ValueError(f"Unknown logger type: {logger_type}")

    creation_func = LOGGER_FACTORY[logger_type]

    return creation_func(config)
