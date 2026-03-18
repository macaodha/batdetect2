import json
from pathlib import Path
from typing import Iterable

from matplotlib.figure import Figure
from soundevent import data

__all__ = ["save_evaluation_results"]


def save_evaluation_results(
    metrics: dict[str, float],
    plots: Iterable[tuple[str, Figure]],
    output_dir: data.PathLike,
) -> None:
    """Save evaluation metrics and plots to disk."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    metrics_path = output_path / "metrics.json"
    metrics_path.write_text(json.dumps(metrics))

    for figure_name, figure in plots:
        figure_path = output_path / figure_name
        figure_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(figure_path)
