import numpy as np
import xarray as xr
from soundevent.arrays import Dimensions

from batdetect2.models import ModelOutput
from batdetect2.preprocess import MAX_FREQ, MIN_FREQ


def to_xarray(
    output: ModelOutput,
    start_time: float,
    end_time: float,
    class_names: list[str],
    min_freq: float = MIN_FREQ,
    max_freq: float = MAX_FREQ,
):
    detection = output.detection_probs
    size = output.size_preds
    classes = output.class_probs
    features = output.features

    if len(detection.shape) == 4:
        if detection.shape[0] != 1:
            raise ValueError(
                "Expected a non-batched output or a batch of size 1, instead "
                f"got an input of shape {detection.shape}"
            )

        detection = detection.squeeze(dim=0)
        size = size.squeeze(dim=0)
        classes = classes.squeeze(dim=0)
        features = features.squeeze(dim=0)

    _, width, height = detection.shape

    times = np.linspace(start_time, end_time, width, endpoint=False)
    freqs = np.linspace(min_freq, max_freq, height, endpoint=False)

    if classes.shape[0] != len(class_names):
        raise ValueError(
            f"The number of classes does not coincide with the number of class names provided: ({classes.shape[0] = }) != ({len(class_names) = })"
        )

    return xr.Dataset(
        data_vars={
            "detection": (
                [Dimensions.time.value, Dimensions.frequency.value],
                detection.squeeze(dim=0).detach().numpy(),
            ),
            "size": (
                [
                    "dimension",
                    Dimensions.time.value,
                    Dimensions.frequency.value,
                ],
                detection.detach().numpy(),
            ),
            "classes": (
                [
                    "category",
                    Dimensions.time.value,
                    Dimensions.frequency.value,
                ],
                classes.detach().numpy(),
            ),
        },
        coords={
            Dimensions.time.value: times,
            Dimensions.frequency.value: freqs,
            "dimension": ["width", "height"],
            "category": class_names,
        },
    )
