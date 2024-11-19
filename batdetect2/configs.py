from typing import Optional, Type, TypeVar

import yaml
from pydantic import BaseModel, ConfigDict
from soundevent.data import PathLike


class BaseConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")


T = TypeVar("T", bound=BaseModel)


def load_config(
    path: PathLike,
    schema: Type[T],
    field: Optional[str] = None,
) -> T:
    with open(path, "r") as file:
        config = yaml.safe_load(file)

    if field:
        config = config[field]

    return schema.model_validate(config)
