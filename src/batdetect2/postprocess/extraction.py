"""Extracts associated data for detected points from model output arrays.

This module implements a Step 4 in the BatDetect2 postprocessing pipeline.
After candidate detection points (time, frequency, score) have been identified,
this module extracts the corresponding values from other raw model output
arrays, such as:

- Predicted bounding box sizes (width, height).
- Class probability scores for each defined target class.
- Intermediate feature vectors.

It uses coordinate-based indexing provided by `xarray` to ensure that the
correct values are retrieved from the original heatmaps/feature maps at the
precise time-frequency location of each detection. The final output aggregates
all extracted information into a structured `xarray.Dataset`.
"""

from typing import List, Optional, Tuple, Union

import torch

from batdetect2.postprocess.nms import NMS_KERNEL_SIZE, non_max_suppression
from batdetect2.typing.postprocess import Detections, ModelOutput

__all__ = [
    "extract_prediction_tensor",
]


def extract_prediction_tensor(
    output: ModelOutput,
    max_detections: int = 200,
    threshold: Optional[float] = None,
    nms_kernel_size: Union[int, Tuple[int, int]] = NMS_KERNEL_SIZE,
) -> List[Detections]:
    detection_heatmap = non_max_suppression(
        output.detection_probs.detach(),
        kernel_size=nms_kernel_size,
    )

    height = detection_heatmap.shape[-2]
    width = detection_heatmap.shape[-1]

    freqs, times = torch.meshgrid(
        torch.arange(height, dtype=torch.int32),
        torch.arange(width, dtype=torch.int32),
        indexing="ij",
    )

    freqs = freqs.flatten()
    times = times.flatten()

    output_size_preds = output.size_preds.detach()
    output_features = output.features.detach()
    output_class_probs = output.class_probs.detach()

    predictions = []
    for idx, item in enumerate(detection_heatmap):
        item = item.squeeze().flatten()  # Remove channel dim
        indices = torch.argsort(item, descending=True)[:max_detections]
        indices.to(detection_heatmap)

        detection_scores = item.take(indices)
        detection_freqs = freqs.take(indices)
        detection_times = times.take(indices)
        sizes = output_size_preds[idx, :, detection_freqs, detection_times].T
        features = output_features[idx, :, detection_freqs, detection_times].T
        class_scores = output_class_probs[
            idx, :, detection_freqs, detection_times
        ].T

        if threshold is not None:
            mask = detection_scores >= threshold
            detection_scores = detection_scores[mask]
            sizes = sizes[mask]
            detection_times = detection_times[mask]
            detection_freqs = detection_freqs[mask]
            features = features[mask]
            class_scores = class_scores[mask]

        predictions.append(
            Detections(
                scores=detection_scores,
                sizes=sizes,
                features=features,
                class_scores=class_scores,
                times=detection_times.to(torch.float32) / width,
                frequencies=(detection_freqs.to(torch.float32) / height),
            )
        )

    return predictions
