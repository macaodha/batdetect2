from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from lightning.pytorch.loggers import Logger
from soundevent import data

from batdetect2.audio import AudioConfig
from batdetect2.core.configs import BaseConfig
from batdetect2.data import Dataset, compute_class_summary
from batdetect2.logging import log_config_artifact, log_csv_artifact
from batdetect2.models import ModelConfig
from batdetect2.targets import TargetConfig, TargetProtocol
from batdetect2.train.config import TrainingConfig

__all__ = [
    "ConfigHyperparameterLogging",
    "DataSummaryArtifactLogging",
    "DatasetConfigArtifact",
    "DatasetConfigArtifactLogging",
    "TargetConfigArtifactLogging",
    "TrainLoggingContext",
]


@dataclass(frozen=True)
class TrainLoggingContext:
    model_config: dict[str, Any]
    train_config: TrainingConfig
    audio_config: AudioConfig
    targets: TargetProtocol
    train_dataset: Dataset
    val_dataset: Dataset | None


@dataclass(frozen=True)
class DatasetConfigArtifact:
    filename: str
    config: BaseConfig


class ConfigHyperparameterLogging:
    def run(
        self,
        logger: Logger,
        artifact_path: Path,
        context: TrainLoggingContext,
    ) -> None:
        model_config = ModelConfig.model_validate(context.model_config)
        logger.log_hyperparams(
            {
                "model": model_config.model_dump(
                    mode="json",
                    exclude_none=True,
                ),
                "training": context.train_config.model_dump(
                    mode="json",
                    exclude_none=True,
                ),
                "audio": context.audio_config.model_dump(
                    mode="json",
                    exclude_none=True,
                ),
            }
        )


class TargetConfigArtifactLogging:
    def run(
        self,
        logger: Logger,
        artifact_path: Path,
        context: TrainLoggingContext,
    ) -> None:
        targets_config = TargetConfig.model_validate(
            context.targets.get_config()
        )
        log_config_artifact(
            logger,
            targets_config,
            filename="targets.yaml",
            artifact_path=artifact_path / "training_artifacts",
        )


class DatasetConfigArtifactLogging:
    def __init__(
        self,
        train_dataset_config: DatasetConfigArtifact,
        val_dataset_config: DatasetConfigArtifact | None = None,
    ):
        self.train_dataset_config = train_dataset_config
        self.val_dataset_config = val_dataset_config

    def run(
        self,
        logger: Logger,
        artifact_path: Path,
        context: TrainLoggingContext,
    ) -> None:
        training_artifact_path = artifact_path / "training_artifacts"

        log_config_artifact(
            logger,
            self.train_dataset_config.config,
            filename=self.train_dataset_config.filename,
            artifact_path=training_artifact_path,
        )

        if self.val_dataset_config is not None:
            log_config_artifact(
                logger,
                self.val_dataset_config.config,
                filename=self.val_dataset_config.filename,
                artifact_path=training_artifact_path,
            )


class DataSummaryArtifactLogging:
    def run(
        self,
        logger: Logger,
        artifact_path: Path,
        context: TrainLoggingContext,
    ) -> None:
        training_artifact_path = artifact_path / "training_artifacts"

        log_csv_artifact(
            logger,
            _compute_class_summary_or_empty(
                context.train_dataset,
                context.targets,
            ),
            filename="train_class_summary.csv",
            artifact_path=training_artifact_path,
        )

        if context.val_dataset is not None:
            log_csv_artifact(
                logger,
                _compute_class_summary_or_empty(
                    context.val_dataset,
                    context.targets,
                ),
                filename="val_class_summary.csv",
                artifact_path=training_artifact_path,
            )


def _compute_class_summary_or_empty(
    dataset: Sequence[data.ClipAnnotation],
    targets: TargetProtocol,
) -> pd.DataFrame:
    try:
        return compute_class_summary(dataset, targets)
    except KeyError as error:
        if error.args != ("class_name",):
            raise

        return pd.DataFrame(
            columns=["num calls", "num recordings", "duration", "call_rate"]
        )
