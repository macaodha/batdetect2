import numpy as np
import torch

from batdetect2.utils.tensors import adjust_width, make_width_divisible


def test_width_is_divisible_after_adjustment():
    tensor = torch.rand([1, 1, 128, 374])
    adjusted = make_width_divisible(tensor, 32)
    assert adjusted.shape[-1] % 32 == 0
    assert adjusted.shape == (1, 1, 128, 384)


def test_non_last_axis_is_divisible_after_adjustment():
    tensor = torch.rand([1, 1, 77, 124])
    adjusted = make_width_divisible(tensor, 32, axis=-2)
    assert adjusted.shape[-2] % 32 == 0
    assert adjusted.shape == (1, 1, 96, 124)


def test_make_width_divisible_can_handle_numpy_array():
    array = np.random.random([1, 1, 128, 374])
    adjusted = make_width_divisible(array, 32)
    assert adjusted.shape[-1] % 32 == 0
    assert adjusted.shape == (1, 1, 128, 384)
    assert isinstance(adjusted, torch.Tensor)


def test_adjust_last_axis_width_by_default():
    tensor = torch.rand([1, 1, 128, 374])
    adjusted = adjust_width(tensor, 512)
    assert adjusted.shape == (1, 1, 128, 512)
    assert (tensor == adjusted[:, :, :, :374]).all()
    assert (adjusted[:, :, :, 374:] == 0).all()


def test_can_adjust_second_to_last_axis():
    tensor = torch.rand([1, 1, 89, 512])
    adjusted = adjust_width(tensor, 128, axis=-2)
    assert adjusted.shape == (1, 1, 128, 512)
    assert (tensor == adjusted[:, :, :89, :]).all()
    assert (adjusted[:, :, 89:, :] == 0).all()
