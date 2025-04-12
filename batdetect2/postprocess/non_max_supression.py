from typing import Tuple, Union

import torch

NMS_KERNEL_SIZE = 9


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

    hmax = torch.nn.functional.max_pool2d(
        tensor,
        (kernel_size_h, kernel_size_w),
        stride=1,
        padding=(pad_h, pad_w),
    )
    keep = (hmax == tensor).float()
    return tensor * keep
