"""Module for postprocessing model outputs."""

from typing import Callable, List, Tuple, Union
from pydantic import BaseModel, Field

import numpy as np
import torch
from soundevent import data
from torch import nn

from batdetect2.data.labels import ClassMapper
from batdetect2.models.typing import ModelOutput

__all__ = [
    "postprocess_model_outputs",
    "PostprocessConfig",
]

NMS_KERNEL_SIZE = 9
DETECTION_THRESHOLD = 0.01
TOP_K_PER_SEC = 200


class PostprocessConfig(BaseModel):
    """Configuration for postprocessing model outputs."""

    nms_kernel_size: int = Field(default=NMS_KERNEL_SIZE, gt=0)
    detection_threshold: float = Field(default=DETECTION_THRESHOLD, ge=0)
    min_freq: int = Field(default=10000, gt=0)
    max_freq: int = Field(default=120000, gt=0)
    top_k_per_sec: int = Field(default=TOP_K_PER_SEC, gt=0)


TagFunction = Callable[[int], List[data.Tag]]


def postprocess_model_outputs(
    outputs: ModelOutput,
    clips: List[data.Clip],
    class_mapper: ClassMapper,
    config: PostprocessConfig,
) -> List[data.ClipPrediction]:
    """Postprocesses model outputs to generate clip predictions.

    This function takes the output from the model, applies non-maximum suppression,
    selects the top-k scores, computes sound events from the outputs, and returns
    clip predictions based on these processed outputs.

    Parameters
    ----------
    outputs
        Output from the model containing detection probabilities, size
        predictions, class logits, and features. All tensors are expected
        to have a batch dimension.
    clips
        List of clips for which predictions are made. The number of clips
        must match the batch dimension of the model outputs.
    config
        Configuration for postprocessing model outputs.

    Returns
    -------
    predictions: List[data.ClipPrediction]
        List of clip predictions containing predicted sound events.

    Raises
    ------
    ValueError
        If the number of predictions does not match the number of clips.
    """
    num_predictions = len(outputs.detection_probs)

    if num_predictions == 0:
        return []

    if num_predictions != len(clips):
        raise ValueError(
            "Number of predictions must match the number of clips."
        )

    detection_probs = non_max_suppression(
        outputs.detection_probs,
        kernel_size=config.nms_kernel_size,
    )

    duration = clips[0].end_time - clips[0].start_time

    scores_batch, y_pos_batch, x_pos_batch = get_topk_scores(
        detection_probs,
        int(config.top_k_per_sec * duration / 2),
    )

    predictions: List[data.ClipPrediction] = []
    for scores, y_pos, x_pos, size_preds, class_probs, features, clip in zip(
        scores_batch,
        y_pos_batch,
        x_pos_batch,
        outputs.size_preds,
        outputs.class_probs,
        outputs.features,
        clips,
    ):
        sound_events = compute_sound_events_from_outputs(
            clip,
            scores,
            y_pos,
            x_pos,
            size_preds,
            class_probs,
            features,
            class_mapper=class_mapper,
            min_freq=config.min_freq,
            max_freq=config.max_freq,
            detection_threshold=config.detection_threshold,
        )

        predictions.append(
            data.ClipPrediction(
                clip=clip,
                sound_events=sound_events,
            )
        )

    return predictions


def compute_sound_events_from_outputs(
    clip: data.Clip,
    scores: torch.Tensor,
    y_pos: torch.Tensor,
    x_pos: torch.Tensor,
    size_preds: torch.Tensor,
    class_probs: torch.Tensor,
    features: torch.Tensor,
    class_mapper: ClassMapper,
    min_freq: int = 10000,
    max_freq: int = 120000,
    detection_threshold: float = DETECTION_THRESHOLD,
) -> List[data.SoundEventPrediction]:
    _, freq_bins, time_bins = size_preds.shape

    sorted_indices = torch.argsort(x_pos)
    valid_indices = sorted_indices[
        scores[sorted_indices] > detection_threshold
    ]

    scores = scores[valid_indices]
    x_pos = x_pos[valid_indices]
    y_pos = y_pos[valid_indices]

    predictions: List[data.SoundEventPrediction] = []
    for score, x, y in zip(scores, x_pos, y_pos):
        width, height = size_preds[:, y, x]
        class_prob = class_probs[:, y, x]
        feature = features[:, y, x]

        start_time = np.interp(
            x.item(),
            [0, time_bins],
            [clip.start_time, clip.end_time],
        )

        end_time = np.interp(
            x.item() + width.item(),
            [0, time_bins],
            [clip.start_time, clip.end_time],
        )

        low_freq = np.interp(
            y.item(),
            [0, freq_bins],
            [max_freq, min_freq],
        )

        high_freq = np.interp(
            y.item() - height.item(),
            [0, freq_bins],
            [max_freq, min_freq],
        )

        predicted_tags: List[data.PredictedTag] = []

        for label_id, class_score in enumerate(class_prob):
            corresponding_tags = class_mapper.inverse_transform(label_id)
            predicted_tags.extend(
                [
                    data.PredictedTag(
                        tag=tag,
                        score=class_score.item(),
                    )
                    for tag in corresponding_tags
                ]
            )

        start_time, end_time = sorted([float(start_time), float(end_time)])
        low_freq, high_freq = sorted([float(low_freq), float(high_freq)])

        sound_event = data.SoundEvent(
            recording=clip.recording,
            geometry=data.BoundingBox(
                coordinates=[
                    start_time,
                    low_freq,
                    end_time,
                    high_freq,
                ]
            ),
            features=[
                data.Feature(
                    name=f"batdetect2_{i}",
                    value=value.item(),
                )
                for i, value in enumerate(feature)
            ],
        )

        predictions.append(
            data.SoundEventPrediction(
                sound_event=sound_event,
                score=score.item(),
                tags=predicted_tags,
            )
        )

    return predictions


def non_max_suppression(
    tensor: torch.Tensor,
    kernel_size: Union[int, Tuple[int, int]] = NMS_KERNEL_SIZE,
) -> torch.Tensor:
    """Run non-maximum suppression on a tensor.

    This function removes values from the input tensor that are not local
    maxima in the neighborhood of the given kernel size.

    All non-maximum values are set to zero.

    Parameters
    ----------
    tensor : torch.Tensor
        Input tensor.
    kernel_size : Union[int, Tuple[int, int]], optional
        Size of the neighborhood to consider for non-maximum suppression.
        If an integer is given, the neighborhood will be a square of the
        given size. If a tuple is given, the neighborhood will be a
        rectangle with the given height and width.

    Returns
    -------
    torch.Tensor
        Tensor with non-maximum suppressed values.
    """
    if isinstance(kernel_size, int):
        kernel_size_h = kernel_size
        kernel_size_w = kernel_size
    else:
        kernel_size_h, kernel_size_w = kernel_size

    pad_h = (kernel_size_h - 1) // 2
    pad_w = (kernel_size_w - 1) // 2

    hmax = nn.functional.max_pool2d(
        tensor,
        (kernel_size_h, kernel_size_w),
        stride=1,
        padding=(pad_h, pad_w),
    )
    keep = (hmax == tensor).float()
    return tensor * keep


def get_topk_scores(
    scores: torch.Tensor,
    K: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Get the top-k scores and their indices.

    Parameters
    ----------
    scores : torch.Tensor
        Tensor with scores. Expects input of size: `batch x 1 x height x width`.
    K : int
        Number of top scores to return.

    Returns
    -------
    scores : torch.Tensor
        Top-k scores.
    ys : torch.Tensor
        Y coordinates of the top-k scores.
    xs : torch.Tensor
        X coordinates of the top-k scores.
    """
    batch, _, height, width = scores.size()
    topk_scores, topk_inds = torch.topk(scores.view(batch, -1), K)
    topk_inds = topk_inds % (height * width)
    topk_ys = torch.div(topk_inds, width, rounding_mode="floor").long()
    topk_xs = (topk_inds % width).long()
    return topk_scores, topk_ys, topk_xs
