from typing import Optional

from pydantic import Field
from soundevent import data

from batdetect2.configs import BaseConfig, load_config
from batdetect2.evaluate.match import DEFAULT_MATCH_CONFIG, MatchConfig

__all__ = [
    "EvaluationConfig",
    "load_evaluation_config",
]


class EvaluationConfig(BaseConfig):
    match: MatchConfig = Field(
        default_factory=lambda: DEFAULT_MATCH_CONFIG.model_copy(),
    )


def load_evaluation_config(
    path: data.PathLike,
    field: Optional[str] = None,
) -> EvaluationConfig:
    return load_config(path, schema=EvaluationConfig, field=field)
