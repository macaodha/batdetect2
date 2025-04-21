from typing import Callable, NamedTuple

import xarray as xr
from soundevent import data

__all__ = [
    "Heatmaps",
    "ClipLabeller",
    "Augmentation",
]


class Heatmaps(NamedTuple):
    """Structure holding the generated heatmap targets.

    Attributes
    ----------
    detection : xr.DataArray
        Heatmap indicating the probability of sound event presence. Typically
        smoothed with a Gaussian kernel centered on event reference points.
        Shape matches the input spectrogram. Values normalized [0, 1].
    classes : xr.DataArray
        Heatmap indicating the probability of specific class presence. Has an
        additional 'category' dimension corresponding to the target class
        names. Each category slice is typically smoothed with a Gaussian
        kernel. Values normalized [0, 1] per category.
    size : xr.DataArray
        Heatmap encoding the size (width, height) of detected events. Has an
        additional 'dimension' coordinate ('width', 'height'). Values represent
        scaled dimensions placed at the event reference points.
    """

    detection: xr.DataArray
    classes: xr.DataArray
    size: xr.DataArray


ClipLabeller = Callable[[data.ClipAnnotation, xr.DataArray], Heatmaps]
"""Type alias for the final clip labelling function.

This function takes the complete annotations for a clip and the corresponding
spectrogram, applies all configured filtering, transformation, and encoding
steps, and returns the final `Heatmaps` used for model training.
"""

Augmentation = Callable[[xr.Dataset], xr.Dataset]


