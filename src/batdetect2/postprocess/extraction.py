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

from typing import List, Optional

import torch

from batdetect2.typing.postprocess import ClipDetectionsTensor

__all__ = [
    "extract_detection_peaks",
]


def extract_detection_peaks(
    detection_heatmap: torch.Tensor,
    size_heatmap: torch.Tensor,
    feature_heatmap: torch.Tensor,
    classification_heatmap: torch.Tensor,
    max_detections: int = 200,
    threshold: Optional[float] = None,
) -> List[ClipDetectionsTensor]:
    height = detection_heatmap.shape[-2]
    width = detection_heatmap.shape[-1]

    freqs, times = torch.meshgrid(
        torch.arange(height, dtype=torch.int32),
        torch.arange(width, dtype=torch.int32),
        indexing="ij",
    )

    freqs = freqs.flatten().to(detection_heatmap.device)
    times = times.flatten().to(detection_heatmap.device)

    output_size_preds = size_heatmap.detach()
    output_features = feature_heatmap.detach()
    output_class_probs = classification_heatmap.detach()

    predictions = []
    for idx, item in enumerate(detection_heatmap):
        item = item.squeeze().flatten()  # Remove channel dim
        indices = torch.argsort(item, descending=True)[:max_detections]

        detection_scores = item.take(indices)
        detection_freqs = freqs.take(indices)
        detection_times = times.take(indices)

        if threshold is not None:
            mask = detection_scores >= threshold

            detection_scores = detection_scores[mask]
            detection_times = detection_times[mask]
            detection_freqs = detection_freqs[mask]

        sizes = output_size_preds[idx, :, detection_freqs, detection_times].T
        features = output_features[idx, :, detection_freqs, detection_times].T
        class_scores = output_class_probs[
            idx,
            :,
            detection_freqs,
            detection_times,
        ].T

        predictions.append(
            ClipDetectionsTensor(
                scores=detection_scores,
                sizes=sizes,
                features=features,
                class_scores=class_scores,
                times=detection_times.to(torch.float32) / width,
                frequencies=(detection_freqs.to(torch.float32) / height),
            )
        )

    return predictions
