"""Performs Non-Maximum Suppression (NMS) on detection heatmaps.

This module provides functionality to apply Non-Maximum Suppression, a common
technique used after model inference, particularly in object detection and peak
detection tasks.

In the context of BatDetect2 postprocessing, NMS is applied
to the raw detection heatmap output by the neural network. Its purpose is to
isolate distinct detection peaks by suppressing (setting to zero) nearby heatmap
activations that have lower scores than a local maximum. This helps prevent
multiple, overlapping detections originating from the same sound event.
"""

from typing import Tuple, Union

import torch

NMS_KERNEL_SIZE = 9
"""Default kernel size (pixels) for Non-Maximum Suppression.

Specifies the side length of the square neighborhood used by default in
`non_max_suppression` to find local maxima. A 9x9 neighborhood is often
a reasonable starting point for typical spectrogram resolutions used in
BatDetect2.
"""


def non_max_suppression(
    tensor: torch.Tensor,
    kernel_size: Union[int, Tuple[int, int]] = NMS_KERNEL_SIZE,
) -> torch.Tensor:
    """Apply Non-Maximum Suppression (NMS) to a tensor, typically a heatmap.

    This function identifies local maxima within a defined neighborhood for
    each point in the input tensor. Values that are *not* the maximum within
    their neighborhood are suppressed (set to zero). This is commonly used on
    detection probability heatmaps to isolate distinct peaks corresponding to
    individual detections and remove redundant lower scores nearby.

    The implementation uses efficient 2D max pooling to find the maximum value
    in the neighborhood of each point.

    Parameters
    ----------
    tensor : torch.Tensor
        Input tensor, typically representing a detection heatmap. Must be a
        3D (C, H, W) or 4D (N, C, H, W) tensor as required by the underlying
        `torch.nn.functional.max_pool2d` operation.
    kernel_size : Union[int, Tuple[int, int]], default=NMS_KERNEL_SIZE
        Size of the sliding window neighborhood used to find local maxima.
        If an integer `k` is provided, a square kernel of size `(k, k)` is used.
        If a tuple `(h, w)` is provided, a rectangular kernel of height `h`
        and width `w` is used. The kernel size should typically be odd to
        have a well-defined center.

    Returns
    -------
    torch.Tensor
        A tensor of the same shape as the input, where only local maxima within
        their respective neighborhoods (defined by `kernel_size`) retain their
        original values. All other values are set to zero.

    Raises
    ------
    TypeError
        If `kernel_size` is not an int or a tuple of two ints.
    RuntimeError
        If the input `tensor` does not have 3 or 4 dimensions (as required
        by `max_pool2d`).

    Notes
    -----
    - The function assumes higher values in the tensor indicate stronger peaks.
    - Choosing an appropriate `kernel_size` is important. It should be large
      enough to cover the typical "footprint" of a single detection peak plus
      some surrounding context, effectively preventing multiple detections for
      the same event. A size that is too large might suppress nearby distinct
      events.
    """
    if isinstance(kernel_size, int):
        kernel_size_h = kernel_size
        kernel_size_w = kernel_size
    else:
        kernel_size_h, kernel_size_w = kernel_size

    pad_h = (kernel_size_h - 1) // 2
    pad_w = (kernel_size_w - 1) // 2

    hmax = torch.nn.functional.max_pool2d(
        tensor,
        (kernel_size_h, kernel_size_w),
        stride=1,
        padding=(pad_h, pad_w),
    )
    keep = (hmax == tensor).float()
    return tensor * keep
