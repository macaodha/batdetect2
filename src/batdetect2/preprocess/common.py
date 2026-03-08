"""Shared tensor primitives used across the preprocessing pipeline.

This module provides small, stateless helper functions that operate on
PyTorch tensors. They are used by both audio-level and spectrogram-level
transforms, and are kept here to avoid duplication.
"""

import torch

__all__ = [
    "center_tensor",
    "peak_normalize",
]


def center_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Subtract the mean of a tensor from all of its values.

    This centres the signal around zero, removing any constant DC offset.

    Parameters
    ----------
    tensor : torch.Tensor
        Input tensor of any shape.

    Returns
    -------
    torch.Tensor
        A new tensor of the same shape and dtype with the global mean
        subtracted from every element.
    """
    return tensor - tensor.mean()


def peak_normalize(tensor: torch.Tensor) -> torch.Tensor:
    """Scale a tensor so that its largest absolute value equals one.

    Divides the tensor by its peak absolute value. If the tensor is
    identically zero, it is returned unchanged (no division by zero).

    Parameters
    ----------
    tensor : torch.Tensor
        Input tensor of any shape.

    Returns
    -------
    torch.Tensor
        A new tensor of the same shape and dtype with values in the range
        ``[-1, 1]`` (or exactly ``[0, 0]`` for a zero tensor).
    """
    max_value = tensor.abs().max()

    denominator = torch.where(
        max_value == 0,
        torch.tensor(1.0, device=tensor.device, dtype=tensor.dtype),
        max_value,
    )

    return tensor / denominator
