from pathlib import Path
from typing import Literal

from soundevent.data import PathLike

from batdetect2.core import ImportConfig, Registry, add_import_config
from batdetect2.outputs.types import OutputFormatterProtocol
from batdetect2.targets.types import TargetProtocol

__all__ = [
    "OutputFormatterProtocol",
    "PredictionFormatterImportConfig",
    "make_path_relative",
    "output_formatters",
]


def make_path_relative(path: PathLike, audio_dir: PathLike) -> Path:
    path = Path(path)
    audio_dir = Path(audio_dir)

    if path.is_absolute():
        if not path.is_relative_to(audio_dir):
            raise ValueError(
                f"Audio file {path} is not in audio_dir {audio_dir}"
            )

        return path.relative_to(audio_dir)

    return path


output_formatters: Registry[OutputFormatterProtocol, [TargetProtocol]] = (
    Registry(name="output_formatter")
)


@add_import_config(output_formatters)
class PredictionFormatterImportConfig(ImportConfig):
    """Use any callable as a prediction formatter.

    Set ``name="import"`` and provide a ``target`` pointing to any
    callable to use it instead of a built-in option.
    """

    name: Literal["import"] = "import"
