from pathlib import Path
from typing import Literal

from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from soundevent.data import PathLike

from batdetect2.core import BaseConfig

__all__ = [
    "CheckpointConfig",
    "build_checkpoint_callback",
    "resolve_checkpoint_path",
]

DEFAULT_CHECKPOINT_DIR: Path = Path("outputs") / "checkpoints"


class CheckpointConfig(BaseConfig):
    checkpoint_dir: Path = DEFAULT_CHECKPOINT_DIR
    monitor: str | None = None
    mode: str = "max"
    save_top_k: int = 1
    filename: str | None = None
    save_last: bool | Literal["link"] = "link"
    every_n_epochs: int | None = 1


def build_checkpoint_callback(
    config: CheckpointConfig | None = None,
    checkpoint_dir: Path | None = None,
    experiment_name: str | None = None,
    run_name: str | None = None,
) -> Callback:
    config = config or CheckpointConfig()

    if checkpoint_dir is None:
        checkpoint_dir = config.checkpoint_dir

    checkpoint_dir = Path(checkpoint_dir)

    if experiment_name is not None:
        checkpoint_dir = checkpoint_dir / experiment_name

    if run_name is not None:
        checkpoint_dir = checkpoint_dir / run_name

    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    return ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        save_top_k=config.save_top_k,
        monitor=config.monitor,
        mode=config.mode,
        filename=config.filename,
        save_last=config.save_last,
        every_n_epochs=config.every_n_epochs,
    )


def resolve_checkpoint_path(path: PathLike | str) -> Path:
    """Resolve a local path or Hugging Face checkpoint URI.

    Parameters
    ----------
    path : PathLike | str
        Local checkpoint path or a Hugging Face URI of the form
        ``hf://owner/repo/path/to/checkpoint.ckpt``.

    Returns
    -------
    Path
        Resolved local filesystem path to the checkpoint.
    """
    if isinstance(path, str) and path.startswith("hf://"):
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as error:
            raise ValueError(
                "Hugging Face checkpoint support is not installed. "
                "Install it with `uv sync --group huggingface` or "
                "`pip install huggingface-hub`."
            ) from error

        repo_id, filename = _parse_huggingface_uri(path)
        return Path(hf_hub_download(repo_id=repo_id, filename=filename))

    if not isinstance(path, Path):
        path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    return path


def _parse_huggingface_uri(uri: str) -> tuple[str, str]:
    prefix = "hf://"
    if not uri.startswith(prefix):
        raise ValueError(
            "Hugging Face checkpoint URIs must start with 'hf://'."
        )

    without_prefix = uri.removeprefix(prefix).strip("/")
    parts = without_prefix.split("/")

    if len(parts) < 3:
        raise ValueError(
            "Hugging Face checkpoint URIs must be in the form "
            "'hf://owner/repo/path/to/checkpoint.ckpt'."
        )

    repo_id = "/".join(parts[:2])
    filename = "/".join(parts[2:])
    return repo_id, filename
